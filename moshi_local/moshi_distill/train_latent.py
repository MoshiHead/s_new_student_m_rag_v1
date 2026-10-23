"""
train_rag.py
============
Fine-tuning script for RAG-aware Moshi LM.
Usage
-----
    python train_rag.py \
        --train-data   data/train.jsonl \
        --eval-data    data/eval.jsonl \
        --moshi-weight /path/to/moshi.safetensors \
        --mimi-weight  /path/to/mimi.safetensors \
        --tokenizer    /path/to/tokenizer.model \
        --out-dir      checkpoints/rag_lora \
        --device       cuda

    # Resume from checkpoint
    python train_rag.py ... --resume checkpoints/rag_lora/step_1000

Hyper-parameters (matching provided config)
--------------------------------------------
    --lora-rank 128  --lora-scaling 2.0
    --lr 2e-6  --weight-decay 0.1
    --batch-size 16  --max-steps 2000
    --duration-sec 100
    --first-codebook-weight 100.0  --text-padding-weight 0.5
    --ref-token-weight 5.0  --body-weight 2.0  --filler-weight 0.5
    --contrastive-weight 0.3  --contrastive-margin 0.3
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
from functools import partial

import numpy as np
import sentencepiece
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import LinearLR, SequentialLR

from huggingface_hub import hf_hub_download

# Project imports — same package as offline inference
from .models import loaders
from .models.lm import LMModel
from .models.lm import LMModel, ScaledEmbedding

# Our pipeline modules
from .rag_loss_new import compute_rag_losses, compute_rag_trigger_loss, SegmentMeta, make_segment_weight_tensor
from .dataset import build_data_loader
from .interleaver import InterleavedTokenizer, Interleaver, Batch

logger = logging.getLogger(__name__)

def apply_lora_to_streaming_attention(lm_base, rank=32, alpha=64.0):
    """Freezes lm_base entirely, then attaches LoRA to in_proj_weight
    (via parametrize, since it's a bare Parameter with no owning Linear
    submodule peft-style matching could ever reach) and out_proj.weight
    (a real Linear, parametrized the same way for consistency) on every
    StreamingMultiheadAttention — covers both lm.transformer and
    lm.depformer. Returns the new trainable params for the optimizer."""
    from torch.nn.utils.parametrize import register_parametrization
    from .modules.transformer import StreamingMultiheadAttention, LoRAWeightDelta

    for p in lm_base.parameters():
        p.requires_grad_(False)

    lora_params = []
    n = 0
    for module in lm_base.modules():
        if isinstance(module, StreamingMultiheadAttention):
            module.in_proj_weight.requires_grad_(False)
            register_parametrization(
                module, "in_proj_weight",
                LoRAWeightDelta(module.in_proj_weight.shape, rank, alpha,
                                 device=module.in_proj_weight.device, dtype=module.in_proj_weight.dtype))
            module.out_proj.weight.requires_grad_(False)
            register_parametrization(
                module.out_proj, "weight",
                LoRAWeightDelta(module.out_proj.weight.shape, rank, alpha,
                                 device=module.out_proj.weight.device, dtype=module.out_proj.weight.dtype))
            n += 1

    for m in lm_base.modules():
        if hasattr(m, "parametrizations"):
            for pname in m.parametrizations:
                for p in m.parametrizations[pname][0].parameters():
                    lora_params.append(p)

    logger.info(f"LoRA attached to {n} attention modules — {sum(p.numel() for p in lora_params):,} trainable params")
    return lora_params

def collect_context_head_params(lm_base) -> list[nn.Parameter]:
    """context_gate_net and rag_head sit directly on LMModel, so
    apply_lora_to_streaming_attention's blanket freeze catches them too
    — must be explicitly re-enabled and added to the optimizer, or the
    gate's deliberately-zero-initialized last layer never leaves zero,
    making grounding injection a permanent no-op regardless of training."""
    params = []
    for name, p in lm_base.named_parameters():
        if name.startswith("context_gate_net.") or name.startswith("rag_head."):
            p.requires_grad_(True)
            params.append(p)
    logger.info(f"context_gate_net + rag_head: {sum(p.numel() for p in params):,} "
                f"trainable params (were frozen by apply_lora_to_streaming_attention)")
    return params



class ReferenceBuffer:
    """Holds recently-seen RAG reference strings so wrong-reference sampling
    for the contrastive loss isn't limited to the current (tiny) micro-batch."""
    def __init__(self, maxlen: int = 128):
        from collections import deque
        self.buf: deque[str] = deque(maxlen=maxlen)

    def add(self, refs: list[str]):
        self.buf.extend(r for r in refs if r)

    def sample_wrong(self, references: list[str], seg_metas, rng: random.Random) -> list[str]:
        wrong = [""] * len(references)
        pool = list(self.buf)
        for i in range(len(references)):
            if not references[i]:
                continue
            candidates = [r for r in pool if r != references[i]]
            if candidates:
                wrong[i] = rng.choice(candidates)
        return wrong

def _safe_id_to_piece(tokenizer, i: int, lm_model=None) -> str:
    """id_to_piece raises IndexError on sentinel ids that aren't real
    SentencePiece pieces (zero_token_id=-1, text_initial_token_id, etc).
    Diagnostic-only — never used for anything that affects training."""
    if i < 0:
        return f"<ZERO:{i}>"
    if lm_model is not None and i == lm_model.text_initial_token_id:
        return "<TEXT_INIT>"
    if i >= tokenizer.GetPieceSize():
        return f"<OOV:{i}>"
    return tokenizer.id_to_piece(i)
    
_lookup_diag_state = {"correct": 0, "total": 0}
def print_lookup_window(codes, seg_metas, tokenizer, step, window=6, max_print=2, lm_model=None):
    printed = 0
    T = codes.shape[-1]
    for meta in seg_metas:
        for span in meta.spans_of("lookup"):
            if printed >= max_print:
                return
            b = meta.batch_idx
            s = max(0, span.start_frame - window)
            e = min(T, span.start_frame + window)
            ids = codes[b, 0, s:e].tolist()
            pieces = [_safe_id_to_piece(tokenizer, i, lm_model) for i in ids]   # CHANGED
            marked = [f"[{p}]" if (s + j) == span.start_frame else p for j, p in enumerate(pieces)]
            logger.info(f"[diag-window] step {step} b={b} lookup_start={span.start_frame} "
                        f"window[{s}:{e}]={' '.join(marked)}")
            printed += 1        
def add_special_tokens_and_resize(
    lm_base: LMModel,
    tokenizer: sentencepiece.SentencePieceProcessor,
    new_tokens: tuple[str, ...] = ("<ref_start>", "<ref_end>", "<lookup>"),
) -> dict[str, int]:
    old_text_card = lm_base.text_card
    n_new = len(new_tokens)

    new_ids = []
    for i, tok in enumerate(new_tokens):
        tid = tokenizer.piece_to_id(tok)
        if tid == tokenizer.unk_id():
            raise ValueError(f"{tok!r} not in tokenizer vocab — run add_ref_tokens_to_spm.py first.")
        expected = old_text_card + i
        if tid != expected:
            raise ValueError(f"{tok!r} has id {tid}, expected {expected} (must be contiguous, appended last).")
        new_ids.append(tid)

    device = lm_base.text_emb.weight.device
    dtype = lm_base.text_emb.weight.dtype
    dim = lm_base.text_emb.embedding_dim
    new_text_card = old_text_card + n_new

    # --- text_emb (main transformer input) --- [unchanged from before]
    old_emb = lm_base.text_emb
    old_pad_row_emb = old_emb.weight.data[old_text_card].clone()
    norm_emb = old_emb.norm is not None

    new_emb = ScaledEmbedding(
        new_text_card + 1, dim, norm=norm_emb, zero_idx=lm_base.zero_token_id,
        device=device, dtype=dtype,
    )
    with torch.no_grad():
        new_emb.weight.data[:old_text_card] = old_emb.weight.data[:old_text_card]
        centroid = old_emb.weight.data[:old_text_card].mean(dim=0)
        for tid in new_ids:
            new_emb.weight.data[tid] = centroid + 0.02 * torch.randn_like(centroid)
        new_emb.weight.data[new_text_card] = old_pad_row_emb

    # --- text_linear (output projection) --- [unchanged from before]
    old_linear = lm_base.text_linear
    extra_text = 1 if lm_base.existing_text_padding_id is None else 0
    old_pad_row_out = old_linear.weight.data[old_text_card].clone() if extra_text else None
    old_pad_bias = (
        old_linear.bias.data[old_text_card].clone()
        if (extra_text and old_linear.bias is not None) else None
    )
    new_out_dim = new_text_card + extra_text
    new_linear = torch.nn.Linear(dim, new_out_dim, bias=old_linear.bias is not None, device=device, dtype=dtype)
    with torch.no_grad():
        new_linear.weight.data[:old_text_card] = old_linear.weight.data[:old_text_card]
        out_centroid = old_linear.weight.data[:old_text_card].mean(dim=0)
        for tid in new_ids:
            new_linear.weight.data[tid] = out_centroid + 0.02 * torch.randn_like(out_centroid)
        if extra_text:
            new_linear.weight.data[new_text_card] = old_pad_row_out
        if new_linear.bias is not None:
            new_linear.bias.data[:old_text_card] = old_linear.bias.data[:old_text_card]
            for tid in new_ids:
                new_linear.bias.data[tid] = 0.0
            if extra_text and old_pad_bias is not None:
                new_linear.bias.data[new_text_card] = old_pad_bias

    # --- NEW: depformer_text_emb — the sibling table forward_depformer_training
    # reads sequence[:, 0] (text codebook) against. Same ids, same resize
    # pattern, but its own dim (depformer_dim) and its own padding/initial
    # row semantics. This is what was missing and caused the OOB lookup.
    old_dep_emb = lm_base.depformer_text_emb
    dep_dim = old_dep_emb.embedding_dim
    dep_norm = old_dep_emb.norm is not None
    old_dep_pad_row = old_dep_emb.weight.data[old_text_card].clone()

    new_dep_emb = ScaledEmbedding(
        new_text_card + 1, dep_dim, norm=dep_norm, zero_idx=lm_base.zero_token_id,
        device=old_dep_emb.weight.device, dtype=old_dep_emb.weight.dtype,
    )
    with torch.no_grad():
        new_dep_emb.weight.data[:old_text_card] = old_dep_emb.weight.data[:old_text_card]
        dep_centroid = old_dep_emb.weight.data[:old_text_card].mean(dim=0)
        for tid in new_ids:
            new_dep_emb.weight.data[tid] = dep_centroid + 0.02 * torch.randn_like(dep_centroid)
        new_dep_emb.weight.data[new_text_card] = old_dep_pad_row

    lm_base.text_emb = new_emb
    lm_base.text_linear = new_linear
    lm_base.depformer_text_emb = new_dep_emb      # NEW
    lm_base.text_card = new_text_card

    # --- gradient masks: freeze everything except the new rows, in ALL THREE tables ---
    emb_mask = torch.zeros(new_text_card + 1, 1, device=device, dtype=dtype)
    emb_mask[new_ids] = 1.0
    new_emb.weight.register_hook(lambda g, m=emb_mask: g * m)

    lin_mask = torch.zeros(new_out_dim, 1, device=device, dtype=dtype)
    lin_mask[new_ids] = 1.0
    new_linear.weight.register_hook(lambda g, m=lin_mask: g * m)
    if new_linear.bias is not None:
        lin_bias_mask = torch.zeros(new_out_dim, device=device, dtype=dtype)
        lin_bias_mask[new_ids] = 1.0
        new_linear.bias.register_hook(lambda g, m=lin_bias_mask: g * m)

    dep_emb_mask = torch.zeros(new_text_card + 1, 1, device=old_dep_emb.weight.device, dtype=old_dep_emb.weight.dtype)
    dep_emb_mask[new_ids] = 1.0
    new_dep_emb.weight.register_hook(lambda g, m=dep_emb_mask: g * m)    # NEW

    logger.info(
        f"resized text vocab {old_text_card} -> {new_text_card} "
        f"(+{n_new}: {dict(zip(new_tokens, new_ids))}); padding/initial row moved "
        f"{old_text_card} -> {new_text_card}; also resized depformer_text_emb"
    )
    return dict(zip(new_tokens, new_ids))
# ---------------------------------------------------------------------------
# Reference adapter — replaces ARC encoder
# ---------------------------------------------------------------------------

from transformers import AutoModel, AutoTokenizer

class RefResampler(nn.Module):
    """Cross-attn (pull context into slots) + self-attn (slots reason with
    each other: reconcile across passages, prioritize, dedupe) + FFN,
    repeated n_layers deep. This is where grounding reasoning actually
    lives — everything downstream just reads its output as conditioning
    vectors, so getting this module's training signal right is the whole
    ballgame."""
    def __init__(self, encoder_name: str, d_model: int,
                 n_queries: int = 16, n_layers: int = 4, n_heads: int = 8,
                 device=None, dtype=None):
        super().__init__()
        object.__setattr__(self, "encoder", AutoModel.from_pretrained(encoder_name).to(device=device, dtype=dtype))
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        encoder_dim = self.encoder.config.hidden_size

        self.in_proj = nn.Linear(encoder_dim, d_model, device=device, dtype=dtype)
        self.n_queries = n_queries
        self.queries = nn.Parameter(torch.randn(n_queries, d_model, device=device, dtype=dtype) * 0.02)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "cross_attn": nn.MultiheadAttention(d_model, n_heads, batch_first=True, device=device, dtype=dtype),
                "cross_norm": nn.LayerNorm(d_model, device=device, dtype=dtype),
                "self_attn":  nn.MultiheadAttention(d_model, n_heads, batch_first=True, device=device, dtype=dtype),
                "self_norm":  nn.LayerNorm(d_model, device=device, dtype=dtype),
                "ff": nn.Sequential(
                    nn.Linear(d_model, d_model * 4, device=device, dtype=dtype), nn.GELU(),
                    nn.Linear(d_model * 4, d_model, device=device, dtype=dtype),
                ),
                "ff_norm": nn.LayerNorm(d_model, device=device, dtype=dtype),
            })
            for _ in range(n_layers)
        ])
        self.out_norm = nn.LayerNorm(d_model, device=device, dtype=dtype)
        self.pool_query = nn.Parameter(torch.randn(1, d_model, device=device, dtype=dtype) * 0.02)
        self.pool_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, device=device, dtype=dtype)

    def forward(self, ref_input_ids: torch.Tensor, ref_attn_mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            kv = self.encoder(input_ids=ref_input_ids, attention_mask=ref_attn_mask).last_hidden_state
        kv = self.in_proj(kv)
        B = ref_input_ids.shape[0]
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        key_padding_mask = ~ref_attn_mask.bool()
        for layer in self.layers:
            attn_out, _ = layer["cross_attn"](q, kv, kv, key_padding_mask=key_padding_mask)
            q = layer["cross_norm"](q + attn_out)
            self_out, _ = layer["self_attn"](q, q, q)
            q = layer["self_norm"](q + self_out)
            q = layer["ff_norm"](q + layer["ff"](q))
        B = q.shape[0]
        pool_q = self.pool_query.unsqueeze(0).expand(B, -1, -1)
        pooled, _ = self.pool_attn(pool_q, q, q)   # [B, 1, dim]
        return pooled.squeeze(1)

# ---------------------------------------------------------------------------
# Gradient checkpointing
# ---------------------------------------------------------------------------

def enable_gradient_checkpointing(lm: nn.Module) -> None:
    """
    Enable gradient checkpointing on the transformer layers inside LMModel.

    peft's get_peft_model exposes enable_input_require_grads which is needed
    for gradient checkpointing to work when inputs don't have requires_grad.
    Then we enable checkpointing on the underlying transformer modules.
    """
    # peft wrapper
    if hasattr(lm, "enable_input_require_grads"):
        lm.enable_input_require_grads()

    # Enable on the two StreamingTransformer modules
    base = lm.base_model if hasattr(lm, "base_model") else lm
    for name in ("transformer", "depformer"):
        module = getattr(base, name, None)
        if module is not None and hasattr(module, "gradient_checkpointing_enable"):
            module.gradient_checkpointing_enable()
            logger.info(f"gradient checkpointing enabled on lm.{name}")
        elif module is not None:
            # StreamingTransformer may not expose the method directly;
            # wrap its layers manually
            if hasattr(module, "layers"):
                module.gradient_checkpointing = True
                logger.info(f"set gradient_checkpointing=True on lm.{name}")


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------

def save_checkpoint(out_dir: Path, step: int, lm: nn.Module, resampler: RefResampler,
                     optimizer: torch.optim.Optimizer, scheduler, stats: dict) -> None:
    ckpt_dir = out_dir / f"step_{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
   
    torch.save(lm.state_dict(), ckpt_dir / "lm_state.pt")
    torch.save(resampler.state_dict(), ckpt_dir / "resampler.pt")
    torch.save({
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "step": step, "stats": stats,
    }, ckpt_dir / "optim.pt")
    logger.info(f"[ckpt] saved step {step} → {ckpt_dir}")


def load_checkpoint(ckpt_dir: Path, lm: nn.Module, resampler: RefResampler,
                     optimizer: torch.optim.Optimizer, scheduler) -> int:
    lm_state_path = ckpt_dir / "lm_state.pt"
    if lm_state_path.exists():
        missing, unexpected = lm.load_state_dict(torch.load(lm_state_path), strict=False)
        if missing or unexpected:
            logger.warning(f"[ckpt] lm state: {len(missing)} missing, {len(unexpected)} unexpected keys")

    resampler_path = ckpt_dir / "resampler.pt"
    if resampler_path.exists():
        resampler.load_state_dict(torch.load(resampler_path))
        logger.info(f"[ckpt] loaded resampler from {resampler_path}")

    optim_path = ckpt_dir / "optim.pt"
    if optim_path.exists():
        state = torch.load(optim_path)
        optimizer.load_state_dict(state["optimizer"])
        if scheduler and state.get("scheduler"):
            scheduler.load_state_dict(state["scheduler"])
        step = state["step"]
        logger.info(f"[ckpt] resumed from step {step}")
        return step
    return 0


# ---------------------------------------------------------------------------
# Wrong-reference construction for contrastive loss
# ---------------------------------------------------------------------------

def build_wrong_references(references, seg_metas, rng):
    rag_refs = [(i, references[i]) for i, m in enumerate(seg_metas) if m is not None and references[i]]
    if len(rag_refs) < 2:
        return [""] * len(references)
    wrong = list(references)
    for i, ref in rag_refs:
        others = [(j, r) for j, r in rag_refs if j != i]
        j, other_ref = rng.choice(others)
        wrong[i] = other_ref
    return wrong


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def _build_ref_injection_tensors(seg_metas, ref_texts, ref_tokenizer, resampler, device, args, audio_delay):
    """audio_delay is now delays[9] (user channel), not delays[0] (text) —
    same value (0) here, but pass it explicitly rather than hardcoding, in
    case delays config ever changes."""
    live = [(m, t) for m, t in zip(seg_metas, ref_texts)
            if m is not None and m.spans_of("ref_slot") and t]
    if not live:
        return None
    enc = ref_tokenizer([t for _, t in live], padding="max_length", truncation=True,
                         max_length=args.max_ref_source_tokens, return_tensors="pt")
    vecs = resampler(enc["input_ids"].to(device), enc["attention_mask"].to(device).bool())
    offset = 1 + audio_delay
    b_idx, t_idx, flat_vecs = [], [], []
    for i, (m, t) in enumerate(live):
        s, e = m.spans_of("ref_slot")[0].start_frame, m.spans_of("ref_slot")[0].end_frame
        s, e = s + offset, e + offset
        b_idx.extend([m.batch_idx] * (e - s))
        t_idx.extend(range(s, e))
        flat_vecs.append(vecs[i, : e - s])
    return (torch.tensor(b_idx, device=device, dtype=torch.long),
            torch.tensor(t_idx, device=device, dtype=torch.long),
            torch.cat(flat_vecs, dim=0))


def forward_train_checkpointed(self: LMModel, codes: torch.Tensor,
                                context_vecs: torch.Tensor | None = None,  # [B, dim] or None
                                body_starts: torch.Tensor | None = None,
                              body_ends: torch.Tensor | None = None):  # [B] frame index per sample, -1 if none
    from .models.lm import _delay_sequence, _undelay_sequence, LMOutput
    from torch.utils.checkpoint import checkpoint

    B, K, T = codes.shape
    initial = self._get_initial_token().expand(B, -1, -1)
    delayed = _delay_sequence(self.delays, codes, initial)
    delayed = torch.cat([initial, delayed], dim=2)

    def _embed_and_transform(seq, context_vecs, body_starts, body_ends):
        audio_sum = None
        for cb_index in range(self.num_audio_codebooks):
            audio_emb = self.emb[cb_index](seq[:, cb_index + self.audio_offset])
            audio_sum = audio_emb if audio_sum is None else audio_sum + audio_emb
        text_emb = self.text_emb(seq[:, 0])
        embedded = audio_sum + text_emb   # [B, S, dim]

        if context_vecs is not None and self.context_gate_net is not None:
            S = embedded.shape[1]
            frame_idx = torch.arange(S, device=embedded.device).view(1, S)
            upper = torch.where(body_ends.view(B, 1) >= 0, body_ends.view(B, 1), torch.full_like(body_ends.view(B,1), S))
            active_mask = (frame_idx >= body_starts.view(B, 1)) & (frame_idx < upper) & (body_starts.view(B, 1) >= 0)  
            ctx = context_vecs.unsqueeze(1).expand(-1, S, -1)
            gate_in = torch.cat([ctx, embedded], dim=-1)
            gated = self.context_gate_net(gate_in)
            embedded = embedded + gated * active_mask.unsqueeze(-1).to(embedded.dtype)

        return self.transformer(embedded)

    transformer_out = checkpoint(_embed_and_transform, delayed[:, :, :-1], context_vecs, body_starts, body_ends, use_reentrant=False)
    if self.out_norm:
        transformer_out = self.out_norm(transformer_out)
    text_logits = self.text_linear(transformer_out)[:, None]
    rag_logits = self.rag_head(transformer_out) if self.rag_head is not None else None   # [B, S, 1]

    logits = checkpoint(lambda seq, t_out: self.forward_depformer_training(seq, t_out),
                         delayed[:, :, 1:], transformer_out, use_reentrant=False)

    logits, logits_mask = _undelay_sequence(self.delays[self.audio_offset:self.audio_offset+self.dep_q], logits, fill_value=float("NaN"))
    logits_mask &= codes[:, self.audio_offset:self.audio_offset+self.dep_q] != self.zero_token_id
    text_logits, text_logits_mask = _undelay_sequence(self.delays[:1], text_logits, fill_value=float("NaN"))
    text_logits_mask &= codes[:, :1] != self.zero_token_id
    return LMOutput(logits, logits_mask, text_logits, text_logits_mask), rag_logits


def build_context_batch(seg_metas, ref_texts, ref_tokenizer, resampler, device, args):
    """(context_vecs [B,dim], body_starts [B] long) for the whole batch.
    context_vecs is zero for rows with no reference — harmless, since
    body_starts=-1 for those rows means the gate's active_mask excludes
    them entirely regardless of what context_vecs holds."""
    B = len(seg_metas)
    d_model = resampler.pool_query.shape[-1]
    body_starts = torch.full((B,), -1, dtype=torch.long, device=device)
    body_ends   = torch.full((B,), -1, dtype=torch.long, device=device)   # NEW
    live = []
    for i, (m, t) in enumerate(zip(seg_metas, ref_texts)):
        if m is not None and m.body_start >= 0 and t:
            live.append((i, t))
            body_starts[i] = m.body_start
            body_ends[i]   = m.body_end   # NEW

    context_vecs = torch.zeros(B, d_model, device=device, dtype=next(resampler.parameters()).dtype)
    if not live:
        return context_vecs, body_starts, body_ends   # CHANGED

    enc = ref_tokenizer([t for _, t in live], padding="max_length", truncation=True,
                         max_length=args.max_ref_source_tokens, return_tensors="pt")
    pooled = resampler(enc["input_ids"].to(device), enc["attention_mask"].to(device).bool())
    for j, (i, t) in enumerate(live):
        context_vecs[i] = pooled[j]
    return context_vecs, body_starts, body_ends   # CHANGED


def compute_contrastive_loss(base_lm, codes, seg_metas, resampler, ref_tokenizer,
                              correct_refs, wrong_refs, device, args):
    def body_logprob(ref_texts):
        context_vecs, body_starts, body_ends = build_context_batch(seg_metas, ref_texts, ref_tokenizer, resampler, device, args)  
        if not (body_starts >= 0).any():
            return {}
        (out, _rag_logits) = forward_train_checkpointed(base_lm, codes=codes,
                                                          context_vecs=context_vecs,
                                                          body_starts=body_starts,
                                                          body_ends=body_ends)   
        logprobs = torch.log_softmax(out.text_logits[:, 0], dim=-1)
        targets = codes[:, 0]
        gathered = logprobs.gather(-1, targets.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        result = {}
        for m in seg_metas:
            if m is None or m.body_start < 0:
                continue
            s = m.body_start
            e = m.body_end if m.body_end > s else min(s + 200, codes.shape[-1])   # CHANGED — real span, fallback only if malformed
            valid = targets[m.batch_idx, s:e] != base_lm.zero_token_id
            if valid.any():
                result[m.batch_idx] = gathered[m.batch_idx, s:e][valid].mean()
        return result
    correct_lp, wrong_lp = body_logprob(correct_refs), body_logprob(wrong_refs)
    gaps = [correct_lp[b].item() - wrong_lp[b].item() for b in correct_lp if b in wrong_lp]
    if gaps:
        logger.info(f"[contrastive] mean_gap={sum(gaps)/len(gaps):.4f} n={len(gaps)}")
    losses = [F.relu(args.contrastive_margin - (correct_lp[b] - wrong_lp[b])) for b in correct_lp if b in wrong_lp]
    return (torch.stack(losses).mean() * args.contrastive_weight) if losses else torch.tensor(0.0, device=device)


def training_step(batch, lm, resampler, ref_buffer, args, rng, device, tokenizer, ref_tokenizer, step=0, diag_every=20):
    codes = batch.codes.to(device)
    seg_metas = batch.segment_metas
    base_lm = lm

    ref_texts = [batch.reference_texts[m.batch_idx] if m is not None else "" for m in seg_metas]
    ref_buffer.add([t for t in ref_texts if t])
    wrong_texts = ref_buffer.sample_wrong(ref_texts, seg_metas, rng)

    context_vecs, body_starts, body_ends = build_context_batch(seg_metas, ref_texts, ref_tokenizer, resampler, device, args)   
    (output, rag_logits) = forward_train_checkpointed(base_lm, codes=codes,
                                                        context_vecs=context_vecs,
                                                        body_starts=body_starts,
                                                        body_ends=body_ends)   

    filtered_metas = [m for m in seg_metas if m is not None]

    text_loss, audio_loss = compute_rag_losses(model=base_lm, output=output, codes=codes,
                                                seg_metas=filtered_metas, args=args)

    rag_trigger_loss = (compute_rag_trigger_loss(rag_logits, filtered_metas, pos_weight=args.rag_trigger_pos_weight)
                         if rag_logits is not None else torch.tensor(0.0, device=device))

    contrastive_loss = torch.tensor(0.0, device=device)
    if (body_starts >= 0).any() and any(wrong_texts):
        contrastive_loss = compute_contrastive_loss(base_lm, codes, seg_metas, resampler, ref_tokenizer,
                                                      ref_texts, wrong_texts, device, args)

    loss = text_loss + audio_loss + contrastive_loss + args.rag_trigger_weight * rag_trigger_loss
    loss.backward()
    for name, p in base_lm.named_parameters():
        if name.startswith("context_gate_net."):
            if p.grad is None:
                logger.info(f"[GATE] {name}: grad=None")
            else:
                grad_norm = p.grad.norm().item()
                param_norm = p.data.norm().item()
                logger.info(
                    f"[GATE] {name}: "
                    f"grad={grad_norm:.3e}, "
                    f"param={param_norm:.3e}, "
                    f"ratio={grad_norm / (param_norm + 1e-12):.3e}"
                )
    return {"loss": loss.item(), "text_loss": text_loss.item(), "audio_loss": audio_loss.item(),
            "contrastive": contrastive_loss.item(), "rag_trigger": rag_trigger_loss.item()}
    


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    device = torch.device(args.device)

    # ── Load models ───────────────────────────────────────────────────────
    logger.info("loading tokenizer")
    if args.tokenizer is None:
        args.tokenizer = hf_hub_download(args.hf_repo, loaders.TEXT_TOKENIZER_NAME)
    tokenizer = sentencepiece.SentencePieceProcessor(args.tokenizer)

    logger.info("loading mimi (frozen)")
    if args.mimi_weight is None:
        args.mimi_weight = hf_hub_download(args.hf_repo, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(args.mimi_weight, device)
    mimi.eval()
    for p in mimi.parameters():
        p.requires_grad_(False)   # Mimi never trains

    logger.info("loading moshi LM")
    if args.moshi_weight is None:
        args.moshi_weight = hf_hub_download(args.hf_repo, loaders.MOSHI_NAME)
    lm_base: LMModel = loaders.get_moshi_lm(args.moshi_weight, device=device, cpu_offload=False)
    lm_base.eval()
    model_dtype = next(lm_base.parameters()).dtype
    lm_base = lm_base.to(dtype=model_dtype)
    
    lora_params = apply_lora_to_streaming_attention(lm_base, rank=args.lora_rank, alpha=args.lora_rank * args.lora_scaling)
    context_head_params = collect_context_head_params(lm_base)
    lm = lm_base
    
    
    d_model = lm_base.text_emb.embedding_dim
    ref_tokenizer = AutoTokenizer.from_pretrained(args.ref_encoder_name)
    resampler = RefResampler(
        encoder_name=args.ref_encoder_name, d_model=d_model,
        n_queries=args.n_ref_slots, n_layers=args.resampler_layers,
        device=device, dtype=model_dtype,
    )
    
    if args.gradient_checkpointing:
        enable_gradient_checkpointing(lm)
    lm.train()
    resampler.train()
    
    new_text_params = list(lm_base.text_emb.parameters()) + list(lm_base.text_linear.parameters()) \
                 + list(lm_base.depformer_text_emb.parameters())
    
    optimizer = AdamW([
        {"params": lora_params, "lr": args.lr, "weight_decay": args.weight_decay},
        {"params": new_text_params, "lr": args.lr, "weight_decay": 0.0},
        {"params": context_head_params, "lr": args.context_head_lr, "weight_decay": args.weight_decay},  # NEW
        {"params": [p for p in resampler.parameters() if p.requires_grad],
         "lr": args.resampler_lr, "weight_decay": args.weight_decay},
    ], betas=(0.9, 0.95), eps=1e-8)
    
    
    from torch.optim.lr_scheduler import LambdaLR
    def _resampler_warmup_then_cosine(step):
        warmup = args.resampler_warmup_steps
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, args.max_steps - warmup)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159265))).item()
    
    base_scheduler = CosineAnnealingLR(optimizer, T_max=args.max_steps, eta_min=args.lr * 0.1)
    scheduler = base_scheduler

    # ── Dataset ───────────────────────────────────────────────────────────
    interleaver = Interleaver(
        tokenizer=tokenizer, audio_frame_rate=mimi.frame_rate,
        text_padding=lm_base.text_padding_token_id,
        end_of_text_padding=lm_base.end_of_text_padding_id,
        zero_padding=lm_base.zero_token_id,
        keep_main_only=True,
        use_bos_eos=True,
        main_speaker_label="ASSISTANT",   # matches InterleavedTokenizer.agent_speaker_label
        device=str(device),
    )
    instruct_tokenizer = InterleavedTokenizer(
        mimi=mimi, interleaver=interleaver, duration_sec=args.duration_sec,
        n_ref_slots=args.n_ref_slots,
        max_ref_source_tokens=args.max_ref_source_tokens,
        target_n_audio_codebooks = 16,
    )

    train_loader = build_data_loader(
        instruct_tokenizer=instruct_tokenizer,
        args=args,
        batch_size=args.batch_size,
        seed=42,
        rank=0,
        world_size=1,
        is_eval=False,
    )
    eval_loader = build_data_loader(
        instruct_tokenizer=instruct_tokenizer,
        args=args,
        batch_size=args.batch_size,
        seed=0,
        rank=0,
        world_size=1,
        is_eval=True,
    ) if args.eval_data else None

    # ── Resume ────────────────────────────────────────────────────────────
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(Path(args.resume), lm, resampler, optimizer, scheduler)

    # ── Training loop ─────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(42)
    step = start_step
    log_interval  = args.log_every
    eval_interval = args.eval_every
    save_interval = args.save_every

    running = {k: 0.0 for k in ("loss", "text_loss", "audio_loss", "contrastive", "rag_trigger")}
    running_n = 0
    t0 = time.time()

    logger.info(f"starting training from step {step}")

    optimizer.zero_grad()
    accum_steps = 0

    ref_buffer = ReferenceBuffer()  

    for batch in train_loader:
        if step >= args.max_steps:
            break

        scalars = training_step(
            batch=batch, lm=lm, resampler=resampler,
            ref_buffer=ref_buffer, args=args, rng=rng, device=device,
            tokenizer=tokenizer, ref_tokenizer=ref_tokenizer, step=step,   # <-- add
        )
        accum_steps += 1
        for k, v in scalars.items():
            running[k] += v
        running_n += 1

        if accum_steps >= args.grad_accum:
            torch.nn.utils.clip_grad_norm_(
                list(lm.parameters()) + list(resampler.parameters()), max_norm=1.0,
            )
            optimizer.step()
            scheduler.step()
            if step < args.resampler_warmup_steps:
                warmup_scale = (step + 1) / args.resampler_warmup_steps
                optimizer.param_groups[3]['lr'] = args.resampler_lr * warmup_scale
            optimizer.zero_grad()
            step += 1
            accum_steps = 0

            # ── Logging ───────────────────────────────────────────────────
            if step % log_interval == 0:
                elapsed = time.time() - t0
                avg     = {k: running[k] / max(running_n, 1) for k in running}
                lr      = scheduler.get_last_lr()[0]
                logger.info(
                    f"step {step:5d}/{args.max_steps} "
                    f"loss={avg['loss']:.4f} "
                    f"text={avg['text_loss']:.4f} "
                    f"audio={avg['audio_loss']:.4f} "
                    f"contrastive={avg['contrastive']:.4f} "
                    f"lookup={avg['rag_trigger']:.4f} "
                    f"lr={lr:.2e} "
                    f"elapsed={elapsed:.0f}s"
                )
                running   = {k: 0.0 for k in running}
                running_n = 0

            # ── Eval ──────────────────────────────────────────────────────
            if eval_loader is not None and step % eval_interval == 0:
                lm.eval()
                resampler.eval()
                eval_losses = {k: 0.0 for k in ("loss", "text_loss", "audio_loss", "contrastive")}
                eval_n = 0
                with torch.no_grad():
                    for eval_batch in eval_loader:
                        s = training_step(
                            batch=eval_batch, lm=lm, resampler=resampler,
                            ref_buffer=ref_buffer, args=args, rng=rng, device=device,
                            tokenizer=tokenizer, ref_tokenizer=ref_tokenizer, step=step,   # <-- add
                        )
                        for k in eval_losses:
                            eval_losses[k] += s.get(k, 0.0)
                        eval_n += 1
                        if eval_n >= 50:
                            break
                avg_eval = {k: v / max(eval_n, 1) for k, v in eval_losses.items()}
                logger.info(f"[eval] step {step} loss={avg_eval['loss']:.4f} "
                            f"text={avg_eval['text_loss']:.4f} audio={avg_eval['audio_loss']:.4f} "
                            f"contrastive={avg_eval['contrastive']:.4f}",f"lookup={avg_eval['lookup']:.4f}",)
                lm.train()
                resampler.train()

            if step % save_interval == 0:
                save_checkpoint(out_dir, step, lm, resampler, optimizer, scheduler, stats={"step": step})

    save_checkpoint(out_dir, step, lm, resampler, optimizer, scheduler, stats={"step": step, "final": True})
    
    logger.info(f"training complete at step {step}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data
    ap.add_argument("--train-data",  required=True, help="train.jsonl manifest path")
    ap.add_argument("--eval-data",   default="",    help="eval.jsonl manifest path (optional)")
    ap.add_argument("--duration-sec", type=float, default=100.0)
    ap.add_argument("--batch-size",   type=int,   default=16)

    # Model
    ap.add_argument("--hf-repo",      default=loaders.DEFAULT_REPO)
    ap.add_argument("--moshi-weight", default=None)
    ap.add_argument("--mimi-weight",  default=None)
    ap.add_argument("--tokenizer",    default=None)
    ap.add_argument("--device",       default="cuda")

    # LoRA
    ap.add_argument("--lora-rank",    type=int,   default=128)
    ap.add_argument("--lora-scaling", type=float, default=2.0)
    ap.add_argument("--ft-embed",     action="store_true",
                    help="Also fine-tune text_emb (default: frozen)")
    ap.add_argument("--gradient-checkpointing", action="store_true", default=True)

    ap.add_argument("--n-ref-slots", type=int, default=16,
                     help="Fixed number of resampler output vectors injected per reference (replaces --max-ref-tokens)")
    ap.add_argument("--max-ref-source-tokens", type=int, default=300,
                     help="Cap on raw reference tokens the resampler reads; does NOT affect main sequence length")
    ap.add_argument("--resampler-lr", type=float, default=1e-4,
                     help="Consider a separate, higher LR for the resampler vs LoRA — it's trained from scratch")

    ap.add_argument("--ref-encoder-name", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                     help="Encodes the reference")

    ap.add_argument("--rag-trigger-weight", type=float, default=1.0)
    ap.add_argument("--rag-trigger-pos-weight", type=float, default=20.0,
    help="BCE pos_weight compensating for lookup frames being rare within a sequence")

    # Optimiser
    ap.add_argument("--lr",           type=float, default=2e-6)
    ap.add_argument("--context-head-lr",           type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--max-steps",    type=int,   default=12000)
    ap.add_argument("--grad-accum",   type=int,   default=1,
                    help="Gradient accumulation steps (default: 1)")

    # Loss weights
    ap.add_argument("--first-codebook-weight", dest="first_codebook_weight_multiplier",
                    type=float, default=100.0)
    ap.add_argument("--num-agent-codebooks", type=int, default=8,
                 help="Only the first N audio codebooks (agent/assistant channel) "
                      "count toward audio loss — user channel is input, never a "
                      "generation target")
    ap.add_argument("--text-padding-weight",   type=float, default=0.5)
    ap.add_argument("--ref-token-weight",      type=float, default=5.0)
    ap.add_argument("--lead-weight",           type=float, default=1.0)
    ap.add_argument("--filler-weight",         type=float, default=0.5)
    ap.add_argument("--body-weight",           type=float, default=2.0)
    ap.add_argument("--no-rag-weight",         type=float, default=1.0)
    ap.add_argument("--lookup-loss-weight", type=float, default=1.0,
                help="Separate, undiluted CE term for the <lookup> decision frame — "
                     "NOT part of the weighted-mean text loss, which dilutes rare "
                     "single-frame signal to near-zero regardless of per-frame weight")
    ap.add_argument("--contrastive-weight",    type=float, default=0.3)
    ap.add_argument("--contrastive-margin",    type=float, default=0.3)

    # Checkpointing / logging
    ap.add_argument("--out-dir",    default="checkpoints/rag_lora")
    ap.add_argument("--resume",     default=None,
                    help="Path to checkpoint directory to resume from")
    ap.add_argument("--save-every", type=int, default=4000)
    ap.add_argument("--log-every",  type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=10000)
    ap.add_argument("--contrastive-every", type=int, default=4)

    ap.add_argument("--resampler-layers", type=int, default=4,
    help="Depth of cross+self-attn reasoning blocks in RefResampler")
    ap.add_argument("--resampler-warmup-steps", type=int, default=200,
        help="Linear LR warmup steps for the resampler param group only — "
             "it's a fresh, deep, randomly-initialized adapter feeding "
             "directly into the LM's embedding stream")

    # DataArgs compatibility (required by build_data_loader)
    ap.add_argument("--shuffle", action="store_true", default=True)

    args = ap.parse_args()

    # Expose as DataArgs-compatible attributes
    args.train_data = args.train_data
    args.eval_data  = args.eval_data

    with torch.no_grad():
        pass   # not wrapping train() — we need gradients

    train(args)


if __name__ == "__main__":
    main()