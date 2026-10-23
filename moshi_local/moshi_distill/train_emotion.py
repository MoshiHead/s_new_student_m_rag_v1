"""
train_emotion.py — emotion-conditioned DepFormer fine-tuning
============================================================

Frozen : temporal transformer, audio/text embeddings, out_norm, text head, Mimi.
Trained: the **DepFormer** (depformer_in / depformer_emb / depformer_text_emb /
         depformer stack / linears) + the **EmotionAdapter**.

Objective (audio-first, per "focus on audio based losses in terms of the
DepFormer so we can get the intensity/prosody of the emotions"):

    L = lambda_audio   * CE_audio(DepFormer logits, target codes)   # main
      + lambda_contrast* relu(margin + CE_true - CE_perm)           # make the
                                                                    # emotion matter
      + lambda_anchor  * mean||theta - theta_0||^2                  # optional drift
                                                                    # guard

CE_audio is a per-codebook weighted cross-entropy over the 16 target codebooks.
Codebook 0 (the coarse "semantic + prosody" stream) is up-weighted: it carries
most of the pitch/energy envelope, i.e. the *intensity* of the emotion; the
remaining codebooks refine timbre.  There is no text CE term — the temporal
transformer that produces the text stream is frozen, so it carries no gradient.

Frozen-transformer cache
------------------------
`transformer_out` depends on the codes alone (the temporal transformer is
frozen), so the 20-layer backbone forward can be run ONCE and cached.  Training
then only runs the 6-layer DepFormer — several times faster per step, and the
same data can be revisited for many epochs.  Enable with `--cache-dir`.

Data: the sidecars written by emotion_preprocess.sh, where
`text_conditions.emotion_id` is the discrete label and `<emotion>` is already
prepended to the system prompt.

Usage
-----
  cd /workspace/personaplex
  PY=/workspace/menv/bin/python
  SNAP=/workspace/.cache/huggingface/hub/models--nvidia--personaplex-7b-v1/snapshots/fdaf4090a61cb315c138a1faee287ffd6c716309

  # 1) build the cache once (one frozen-backbone pass), then train
  $PY -m moshi_local.moshi_distill.train_emotion \
      --student-checkpoint ckpt/backbone_S20_lora/step_1800 \
      --mimi-weight $SNAP/tokenizer-e351c8d8-checkpoint125.safetensors \
      --tokenizer   $SNAP/tokenizer_spm_32k_3.model \
      --train-data /workspace/data/emotion_v1/audio/train.jsonl \
      --eval-data  /workspace/data/emotion_v1/audio/train.jsonl \
      --cache-dir  /workspace/data/emotion_v1/cache \
      --out-dir    ckpt/emotion_v1 \
      --batch-size 8 --max-steps 6000
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path

import sentencepiece
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from .backbone_distill import plain_state_dict, _resolve_ckpt_path
from .dataset import build_data_loader
from .distill.data import DistillInterleavedTokenizer
from .emotion_module import EmotionAdapter, depformer_logits
from .emotion_probe import emotion_discrimination, run_generation_probe
from .emotion_recon import recon_loss
from .interleaver import Interleaver
from .models import loaders
from .models.lm import LMModel, _delay_sequence, _undelay_sequence

logger = logging.getLogger(__name__)

EMOTIONS = ["neutral", "happy", "sad", "angry", "fearful",
            "disgust", "surprised", "calm"]


# ---------------------------------------------------------------------------
# Model construction / freezing
# ---------------------------------------------------------------------------

def build_student_from_ckpt(checkpoint: str | Path, device, dtype=torch.float32):
    """Rebuild the pruned backbone straight from its checkpoint — no 7B teacher.

    The checkpoint is self-contained (`kept_layers` + `base_kwargs`), so the
    architecture is reconstructed exactly and the weights loaded verbatim.
    """
    pkg = torch.load(_resolve_ckpt_path(checkpoint), map_location="cpu",
                     weights_only=False)
    if not isinstance(pkg, dict) or "kept_layers" not in pkg:
        raise ValueError(f"{checkpoint} is not a pruned-backbone checkpoint")
    kept = list(pkg["kept_layers"])
    bk = dict(pkg.get("base_kwargs") or {})
    bk["num_layers"] = len(kept)
    # Build on CPU first: LMModel constructs text_linear / out_norm /
    # depformer_in / linears without device/dtype, so one explicit .to() after
    # loading is what actually moves the whole model.
    model = LMModel(**bk)
    sd = pkg["state_dict"]
    sd = {k: (v.float() if v.is_floating_point() else v) for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model = model.to(device=device, dtype=dtype)
    return model, kept, bk


def freeze_backbone_keep_depformer(model: LMModel) -> tuple[int, int]:
    """Trainable = everything whose name starts with 'depformer' or 'linears'.

    That is exactly the DepFormer (its input projections, token embeddings and
    output linears); the temporal transformer / embeddings / out_norm / text
    head stay frozen.  Returns (trainable, total).
    """
    trainable = ("depformer", "linears")
    n_train = n_total = 0
    for name, p in model.named_parameters():
        r = name.startswith(trainable)
        p.requires_grad_(r)
        n_total += p.numel()
        n_train += p.numel() if r else 0
    return n_train, n_total


def cast_frozen_backbone(model: LMModel, dtype) -> None:
    """Move the (non-trainable) backbone to `dtype` to save GPU memory.

    In cached mode the frozen backbone is never even executed during training —
    only the DepFormer runs — so keeping it in bf16 halves its footprint at no
    cost. `depformer_inputs` casts the state back to the projection dtype.

    Keyed off `requires_grad`, not a name list: the trainable set is not fixed
    (the text head can be unfrozen too).
    """
    if dtype is None:
        return
    for _, p in model.named_parameters():
        if not p.requires_grad:
            p.data = p.data.to(dtype)


def offload_frozen_to_cpu(model: LMModel) -> int:
    """Park the frozen backbone on CPU (cached mode never executes it).

    Saves ~9 GB of GPU while training and is free: `plain_state_dict` reads the
    weights wherever they live, and the cached eval never runs the backbone.
    """
    n = 0
    for _, p in model.named_parameters():
        if not p.requires_grad and p.device.type == "cuda":
            p.data = p.data.cpu()
            n += p.numel()
    return n


def onload_frozen_to_device(model: LMModel, device) -> None:
    """Bring the offloaded frozen backbone back for a generation probe.

    Generation runs the temporal transformer (`forward_codes`), which training
    in cached mode never touches — so the probe has to undo the offload.
    """
    device = torch.device(device)
    for _, p in model.named_parameters():
        if not p.requires_grad and p.device != device:
            p.data = p.data.to(device)


def depformer_dim(model: LMModel) -> int:
    return int(model.depformer_in[0].out_features)


# ---------------------------------------------------------------------------
# Forward / losses
# ---------------------------------------------------------------------------

def temporal_forward(model: LMModel, codes: torch.Tensor):
    """Frozen temporal transformer. Returns (seq, transformer_out).

    `seq` is the [B, K, T] DepFormer input token stream; `transformer_out` is
    the post-out_norm [B, T, dim] state the DepFormer consumes.
    """
    B, K, T = codes.shape
    initial = model._get_initial_token().expand(B, -1, -1)
    delayed = _delay_sequence(model.delays, codes, initial)
    delayed = torch.cat([initial, delayed], dim=2)          # [B, K, T+1]
    emb = model.embed_codes(delayed[:, :, :-1])
    transformer_out, _ = model.forward_embeddings(emb)
    return delayed[:, :, 1:], transformer_out


def depformer_input_seq(model: LMModel, codes: torch.Tensor) -> torch.Tensor:
    """`seq` alone (cheap): the delayed code stream the DepFormer consumes."""
    B = codes.shape[0]
    initial = model._get_initial_token().expand(B, -1, -1)
    return _delay_sequence(model.delays, codes, initial)


def undelay_audio(model: LMModel, logits: torch.Tensor, codes: torch.Tensor):
    Ka = model.dep_q
    delays = model.delays[model.audio_offset:model.audio_offset + Ka]
    logits, mask = _undelay_sequence(delays, logits, fill_value=float("nan"))
    mask = mask & (codes[:, model.audio_offset:model.audio_offset + Ka]
                   != model.zero_token_id)
    return logits, mask


def audio_ce(logits: torch.Tensor, codes: torch.Tensor, mask: torch.Tensor,
             codebook_weights: torch.Tensor | None = None,
             first_codebook_mult: float = 1.0) -> torch.Tensor:
    """Weighted cross-entropy over the DepFormer audio codebooks."""
    target = torch.where(mask, codes, torch.zeros_like(codes))
    w = mask.float()
    if codebook_weights is not None:
        w = w * codebook_weights.to(w.device).view(1, -1, 1)
    if first_codebook_mult != 1.0:
        w = w.clone()
        w[:, 0] *= first_codebook_mult
    lf = torch.nan_to_num(logits.reshape(-1, logits.size(-1)).float(),
                          nan=0.0, posinf=0.0, neginf=0.0)
    tf = target.reshape(-1)
    wf = w.reshape(-1)
    per = F.cross_entropy(lf, tf, reduction="none")
    return (per * wf).sum() / wf.sum().clamp(min=1e-8)


def batch_emotion_ids(batch, device, n_emotions: int) -> torch.Tensor:
    ca = batch.condition_attributes
    B = batch.codes.shape[0]
    if ca is None:
        return torch.zeros(B, dtype=torch.long, device=device)
    ids = []
    for c in ca:
        eid = None
        if isinstance(c, dict):
            eid = c.get("emotion_id")
            if eid is None and c.get("emotion") in EMOTIONS:
                eid = EMOTIONS.index(c["emotion"])
        ids.append(0 if eid is None else int(eid) % n_emotions)
    return torch.tensor(ids, dtype=torch.long, device=device)


def codebook_weight_vector(model: LMModel, args) -> torch.Tensor:
    """Per-codebook audio-CE weights.

    `--codebook-scope agent` (default) zeroes the USER stream's codebooks.  The
    user stream carries no emotion — it is *provided* at inference — so keeping
    it in the loss only dilutes the emotion gradient by ~2x (half of `dep_q`).
    """
    Ka = model.dep_q
    if args.codebook_decay and args.codebook_decay > 0:
        w = torch.tensor([args.codebook_decay ** i for i in range(Ka)],
                         dtype=torch.float32)
    else:
        w = torch.ones(Ka, dtype=torch.float32)
    if getattr(args, "codebook_scope", "agent") == "agent":
        w[Ka // 2:] = 0.0
    return w


# ---------------------------------------------------------------------------
# The objective (shared by the on-the-fly and cached paths)
# ---------------------------------------------------------------------------

def _loss_from(codes, t_out, emo_ids, model, adapter, args, cb_weights,
               depformer_init=None, mimi=None, do_recon=True):
    """codes [B,K,T] (padded), t_out [B,T,D] frozen transformer state,
    emo_ids [B]. Returns (loss, stats)."""
    Ka = model.dep_q
    audio_codes = codes[:, model.audio_offset:model.audio_offset + Ka]
    B = codes.shape[0]

    # the cache stores bf16; cast to whatever the trainable DepFormer uses
    t_out = t_out.to(model.depformer_in[0].weight.dtype)

    seq = depformer_input_seq(model, codes)
    emo_vec = adapter(emo_ids)
    logits = depformer_logits(model, seq, t_out, emo_vec,
                              emo_slots=(getattr(args, "emo_slots", 0) or None))
    logits, mask = undelay_audio(model, logits, codes)
    ce = audio_ce(logits, audio_codes, mask, cb_weights, args.first_codebook_mult)

    total = args.lambda_audio * ce
    stats = {"audio_ce": ce.item()}

    if args.lambda_contrast > 0.0 and B > 1:
        perm = torch.randperm(B, device=codes.device)
        logits_p = depformer_logits(model, seq, t_out, adapter(emo_ids[perm]),
                                    emo_slots=(getattr(args, "emo_slots", 0) or None))
        logits_p, mask_p = undelay_audio(model, logits_p, codes)
        ce_perm = audio_ce(logits_p, audio_codes, mask_p, cb_weights,
                           args.first_codebook_mult)
        contrast = F.relu(args.contrast_margin + ce - ce_perm)
        total = total + args.lambda_contrast * contrast
        stats["contrast"] = contrast.item()
        stats["ce_perm"] = ce_perm.item()

    # TEXT cross-entropy.  `t_out` is the frozen transformer's post-out_norm
    # state, and `text_linear` maps it to the text logits — so the text loss can
    # be computed straight from the CACHE (no extra transformer forward), and it
    # is the only thing that teaches the model to SAY the emotional sentence.
    if args.lambda_text > 0.0 and model.text_linear.weight.requires_grad:
        t_logits = model.text_linear(t_out)[:, None]           # [B,1,T,text_card]
        tl, tm = _undelay_sequence(model.delays[:1], t_logits,
                                   fill_value=float("nan"))
        tgt = codes[:, 0]
        card = model.text_card
        m = (tm[:, 0] & (tgt != model.zero_token_id) & (tgt >= 0)
             & (tgt < card)).float()
        lf = torch.nan_to_num(tl[:, 0].float(), nan=0.0)
        safe = torch.where(m > 0, tgt, torch.zeros_like(tgt))
        per = F.cross_entropy(lf.reshape(-1, card), safe.reshape(-1),
                              reduction="none").view_as(m)
        text_ce = (per * m).sum() / m.sum().clamp_min(1e-8)
        total = total + args.lambda_text * text_ce
        stats["text_ce"] = text_ce.item()

    # Waveform-domain reconstruction: decode the predicted (soft) codes with
    # Mimi and compare log-mels against the decoded ground truth.  This is the
    # only term that sees the audio, so it is what penalises incoherent output
    # that per-token CE happily accepts.  Only the AGENT's codebooks are
    # decoded — that is the half Mimi renders for this stream.
    if mimi is not None and args.lambda_recon > 0.0 and do_recon:
        K = Ka // 2
        lrec = recon_loss(mimi, logits[:, :K], audio_codes[:, :K], mask[:, :K])
        total = total + args.lambda_recon * lrec
        stats["recon"] = lrec.item()

    if args.lambda_anchor > 0.0 and depformer_init is not None:
        anchor, n = None, 0
        for p, p0 in zip(depformer_init["params"], depformer_init["values"]):
            term = ((p.float() - p0) ** 2).sum()
            anchor = term if anchor is None else anchor + term
            n += p.numel()
        anchor = anchor / max(1, n)
        total = total + args.lambda_anchor * anchor
        stats["anchor"] = anchor.item()

    stats["loss"] = total.item()
    return total, stats


def training_step(batch, model, adapter, args, device, cb_weights,
                  depformer_init=None, autocast_dtype=None, mimi=None,
                  do_recon=True):
    codes = batch.codes.to(device)
    emo_ids = batch_emotion_ids(batch, device, adapter.n_emotions)
    if autocast_dtype is not None:
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            _, t_out = temporal_forward(model, codes)
    else:
        _, t_out = temporal_forward(model, codes)
    return _loss_from(codes, t_out, emo_ids, model, adapter, args, cb_weights,
                      depformer_init, mimi=mimi, do_recon=do_recon)


# ---------------------------------------------------------------------------
# Frozen-transformer cache
# ---------------------------------------------------------------------------

def expected_samples(manifest: str, duration_sec: float) -> int:
    n = 0
    for line in open(manifest, errors="replace"):
        line = line.strip()
        if line:
            n += max(1, math.ceil(json.loads(line)["duration"] / duration_sec))
    return n


def first_system_prompt(manifest: str) -> str:
    """Persona prompt of the first manifest entry (used for the gen probe)."""
    with open(manifest, errors="replace") as f:
        p = json.loads(f.readline())["path"]
    side = os.path.splitext(p)[0] + ".sidecar.json"
    try:
        d = json.load(open(side))
        return (d.get("metadata", {}).get("system_prompt")
                or (d.get("text_conditions") or {}).get("system_prompt") or "")
    except Exception:
        return ""


@torch.no_grad()
def build_transformer_cache(model, loader, device, n_emotions, n_samples,
                            autocast_dtype=None):
    model.eval()
    items = []
    for batch in loader:
        if len(items) >= n_samples:
            break
        codes = batch.codes.to(device)
        if autocast_dtype is not None:
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                _, t_out = temporal_forward(model, codes)
        else:
            _, t_out = temporal_forward(model, codes)
        emo_ids = batch_emotion_ids(batch, device, n_emotions)
        for i in range(codes.shape[0]):
            if len(items) >= n_samples:
                break
            items.append({
                "codes": codes[i].to(torch.int16).cpu(),
                "t_out": t_out[i].to(torch.bfloat16).cpu(),
                "emotion_id": int(emo_ids[i]),
            })
        if len(items) and len(items) % 500 < codes.shape[0]:
            logger.info(f"[cache] {len(items)}/{n_samples}")
    model.train()
    return items


def collate_cache(items, device, pad_token_id):
    T = max(it["codes"].shape[-1] for it in items)
    T += T % 2
    K = items[0]["codes"].shape[0]
    D = items[0]["t_out"].shape[-1]
    B = len(items)
    codes = torch.full((B, K, T), pad_token_id, dtype=torch.long)
    t_out = torch.zeros((B, T, D), dtype=torch.bfloat16)
    emo = torch.zeros(B, dtype=torch.long)
    for i, it in enumerate(items):
        t = it["codes"].shape[-1]
        codes[i, :, :t] = it["codes"].long()
        t_out[i, :t] = it["t_out"]
        emo[i] = it["emotion_id"]
    return codes.to(device), t_out.to(device), emo.to(device)


def training_step_cached(items, model, adapter, args, device, cb_weights,
                         depformer_init=None, mimi=None, do_recon=True):
    codes, t_out, emo_ids = collate_cache(items, device, model.zero_token_id)
    return _loss_from(codes, t_out, emo_ids, model, adapter, args, cb_weights,
                      depformer_init, mimi=mimi, do_recon=do_recon)


@torch.no_grad()
def _eval_batch(codes, t_out, emo_ids, model, adapter, args, cb_weights):
    Ka = model.dep_q
    audio_codes = codes[:, model.audio_offset:model.audio_offset + Ka]
    seq = depformer_input_seq(model, codes)
    logits = depformer_logits(model, seq, t_out, adapter(emo_ids))
    logits, mask = undelay_audio(model, logits, codes)
    perm = torch.randperm(codes.shape[0], device=codes.device)
    logits_p = depformer_logits(model, seq, t_out, adapter(emo_ids[perm]))
    logits_p, mask_p = undelay_audio(model, logits_p, codes)

    out = {
        "audio_ce": audio_ce(logits, audio_codes, mask, cb_weights,
                             args.first_codebook_mult).item(),
        "audio_ce_perm": audio_ce(logits_p, audio_codes, mask_p, cb_weights,
                                  args.first_codebook_mult).item(),
    }
    # Codebook 0 is the agent's coarse "semantic + prosody" stream — the emotion
    # lives almost entirely there.  Its reliance is the sharpest read: the
    # all-codebook number is diluted by the 7 refinement books and by the user
    # stream, which carries no emotion at all.
    out["ce_cb0"] = audio_ce(logits[:, :1], audio_codes[:, :1],
                             mask[:, :1], None, 1.0).item()
    out["ce_cb0_perm"] = audio_ce(logits_p[:, :1], audio_codes[:, :1],
                                  mask_p[:, :1], None, 1.0).item()
    return out


def _finish_eval(agg, n, adapter, device, model, disc=None):
    model.train()
    if n == 0:
        return {}
    out = {k: v / n for k, v in agg.items()}
    out["emotion_reliance"] = out.get("audio_ce_perm", 0.0) - out.get("audio_ce", 0.0)
    out["reliance_cb0"] = out.get("ce_cb0_perm", 0.0) - out.get("ce_cb0", 0.0)
    if disc and disc.get("n"):
        out["emotion_acc"] = disc["correct"] / max(1, disc["total"])
        out["ce_true"] = disc["ce_true"] / disc["n"]
        out["ce_wrong"] = disc["ce_wrong"] / disc["n"]
        out["ce_gap"] = out["ce_wrong"] - out["ce_true"]
    with torch.no_grad():
        v = adapter(torch.arange(adapter.n_emotions, device=device))
        out["emo_spread"] = v.std(dim=0).mean().item()
    return out


def _disc_acc(disc, codes, t_out, emo_ids, model, adapter, args, cb_weights):
    c, tot, cet, cew, _ = emotion_discrimination(
        codes, t_out, emo_ids, model, adapter, args, cb_weights,
        adapter.n_emotions)
    disc["correct"] += c
    disc["total"] += tot
    disc["ce_true"] += cet
    disc["ce_wrong"] += cew
    disc["n"] += 1


@torch.no_grad()
def evaluate(model, adapter, loader, args, device, cb_weights, max_batches=8):
    model.eval()
    agg, n = {}, 0
    disc = {"correct": 0, "total": 0, "ce_true": 0.0, "ce_wrong": 0.0, "n": 0}
    for batch in loader:
        if n >= max_batches:
            break
        codes = batch.codes.to(device)
        _, t_out = temporal_forward(model, codes)
        emo_ids = batch_emotion_ids(batch, device, adapter.n_emotions)
        d = _eval_batch(codes, t_out, emo_ids, model, adapter, args, cb_weights)
        for k, v in d.items():
            agg[k] = agg.get(k, 0.0) + v
        _disc_acc(disc, codes, t_out, emo_ids, model, adapter, args, cb_weights)
        n += 1
    return _finish_eval(agg, n, adapter, device, model, disc)


@torch.no_grad()
def evaluate_cached(items, model, adapter, args, device, cb_weights, max_batches=8):
    """Evaluate straight from cached transformer states (no frozen forward)."""
    model.eval()
    agg, n, bs = {}, 0, args.batch_size
    disc = {"correct": 0, "total": 0, "ce_true": 0.0, "ce_wrong": 0.0, "n": 0}
    for bi in range(0, max(1, len(items) - bs + 1), bs):
        if n >= max_batches:
            break
        codes, t_out, emo_ids = collate_cache(items[bi:bi + bs], device,
                                              model.zero_token_id)
        d = _eval_batch(codes, t_out, emo_ids, model, adapter, args, cb_weights)
        for k, v in d.items():
            agg[k] = agg.get(k, 0.0) + v
        _disc_acc(disc, codes, t_out, emo_ids, model, adapter, args, cb_weights)
        n += 1
    return _finish_eval(agg, n, adapter, device, model, disc)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(out_dir: Path, step: int, model: LMModel, adapter: EmotionAdapter,
                    kept: list[int], base_kwargs: dict, stats: dict,
                    save_dtype=torch.bfloat16) -> None:
    d = out_dir / f"step_{step}"
    d.mkdir(parents=True, exist_ok=True)
    sd = plain_state_dict(model)
    if save_dtype is not None:
        sd = {k: (v.to(save_dtype) if v.is_floating_point() else v)
              for k, v in sd.items()}
    torch.save({
        "state_dict": sd,
        "kept_layers": kept,
        "save_dtype": str(save_dtype),
        "stats": {k: float(v) for k, v in stats.items()
                  if isinstance(v, (int, float))},
        "emotion_order": EMOTIONS,
        "base_kwargs": {k: v for k, v in base_kwargs.items()
                        if not isinstance(v, torch.dtype)},
    }, d / "student.pt")
    adapter.save(d / "emotion.pt", EMOTIONS)
    (d / "meta.json").write_text(json.dumps(
        {"step": step,
         "stats": {k: float(v) for k, v in stats.items()
                   if isinstance(v, (int, float))}}, indent=2))
    logger.info(f"[ckpt] saved step {step} -> {d} (student.pt + emotion.pt)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--student-checkpoint",
                    default="/workspace/personaplex/ckpt/backbone_S20_lora/step_1800")
    ap.add_argument("--mimi-weight", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--frozen-dtype", default="bfloat16",
                    choices=["bfloat16", "float32"],
                    help="precision kept for the frozen backbone (bf16 halves "
                         "its GPU footprint; it is not executed in cached mode)")
    ap.add_argument("--autocast", default="", choices=["", "bfloat16", "float16"])
    ap.add_argument("--train-data", required=True)
    ap.add_argument("--eval-data", default="")
    ap.add_argument("--duration-sec", type=float, default=8.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--shuffle", action="store_true", default=True)
    ap.add_argument("--cache-dir", default="",
                    help="cache the frozen transformer_out here (big speedup); "
                         "built on first run, reused after")
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--cache-samples", type=int, default=0,
                    help="0 = all samples in the train manifest")
    ap.add_argument("--offload-frozen", action="store_true", default=True,
                    help="park the frozen backbone on CPU in cached mode "
                         "(saves ~9 GB GPU; it is never executed while training)")
    ap.add_argument("--no-offload-frozen", dest="offload_frozen",
                    action="store_false")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--depformer-lr", type=float, default=None)
    ap.add_argument("--adapter-lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=6000)
    ap.add_argument("--lambda-audio", type=float, default=1.0)
    ap.add_argument("--first-codebook-mult", type=float, default=2.0,
                    help="extra weight on codebook 0 (coarse prosody/intensity)")
    ap.add_argument("--codebook-decay", type=float, default=0.0,
                    help=">0: exponential per-codebook weight decay from cb0")
    ap.add_argument("--codebook-scope", default="agent", choices=["agent", "all"],
                    help="'agent' (default) trains only the agent's codebooks; "
                         "'all' also trains the user stream (which carries no "
                         "emotion and is provided at inference)")
    ap.add_argument("--lambda-contrast", type=float, default=0.5)
    ap.add_argument("--contrast-margin", type=float, default=0.1)
    ap.add_argument("--lambda-anchor", type=float, default=0.0,
                    help="optional DepFormer drift guard (mean squared delta)")
    ap.add_argument("--lambda-recon", type=float, default=0.5,
                    help="weight for the WAVEFORM reconstruction loss: Mimi-decode "
                         "the predicted soft codes and match the decoded ground "
                         "truth in log-mel. This is the term that penalises "
                         "incoherent audio that per-token CE accepts. 0 = off.")
    ap.add_argument("--recon-every", type=int, default=1,
                    help="compute the reconstruction loss every N steps "
                         "(it costs 2 Mimi decoder forwards per step)")
    ap.add_argument("--lambda-text", type=float, default=1.0,
                    help="weight for the TEXT cross-entropy. Without it the "
                         "emotion fine-tune never touches the text head, so the "
                         "model keeps SAYING whatever the frozen head produces "
                         "(a greeting) no matter how well the audio is coloured.")
    ap.add_argument("--no-train-text-head", dest="train_text_head",
                    action="store_false", default=True,
                    help="keep the text head frozen (default: train it)")
    ap.add_argument("--adapter-rms", type=float, default=0.25,
                    help="renormalise each emotion's per-codebook vector to this "
                         "RMS (the DepFormer input RMS is ~0.41). Measured, an "
                         "un-normalised adapter was only ~10%% of the input scale "
                         "and the audio CE shrank it further, so the emotion was "
                         "inaudible. 0 = off (learnable out_scale).")
    ap.add_argument("--emo-slots", type=int, default=0,
                    help="optional HARD mask: inject the emotion into only the "
                         "first N codebooks. 0 (default) = no mask — the "
                         "adapter's own per-codebook gate decides, starting from "
                         "codebook 0 only and opening the rest only if the "
                         "reconstruction loss rewards it.")
    ap.add_argument("--adapter-init", default="small", choices=["small", "zero"],
                    help="'small' gives each emotion a distinct vector at step 0 "
                         "(required for the contrastive term to have a gradient); "
                         "'zero' starts bit-identical to the base model")
    ap.add_argument("--gen-every", type=int, default=800,
                    help="generate a per-emotion audio grid every N steps "
                         "(0 = off). This is the only probe that proves the "
                         "emotion is audible.")
    ap.add_argument("--gen-frames", type=int, default=100,
                    help="frames of ANSWER to generate per emotion "
                         "(100 = 8 s at 12.5 Hz)")
    ap.add_argument("--gen-greet", type=int, default=40,
                    help="frames the agent gets to GREET before the user asks "
                         "(40 = 3.2 s). The greeting is the agent's first turn, "
                         "as in the training data; without it the query would "
                         "land on the greeting turn.")
    ap.add_argument("--no-gen-asr", dest="gen_asr", action="store_false",
                    default=True,
                    help="skip round-trip ASR of the generated rows (default: on, "
                         "so you see TARGET vs generated text in the log)")
    ap.add_argument("--gen-voice", default="",
                    help="optional voice prompt (.pt) for the generation probe. "
                         "Empty (default) = no voice prompt: the probe is ~2x "
                         "faster and the emotion comparison is unaffected, since "
                         "every emotion sees the same context either way.")
    ap.add_argument("--out-dir", default="/workspace/personaplex/ckpt/emotion_v1")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--save-dtype", default="bfloat16",
                    choices=["bfloat16", "float32"])
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def train(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    autocast_dtype = ({"bfloat16": torch.bfloat16, "float16": torch.float16}
                      .get(args.autocast))

    logger.info("loading mimi (frozen)")
    mimi = loaders.get_mimi(args.mimi_weight, device)
    mimi.eval()
    for p in mimi.parameters():
        p.requires_grad_(False)

    logger.info(f"loading student backbone from {args.student_checkpoint}")
    model, kept, base_kwargs = build_student_from_ckpt(
        args.student_checkpoint, device, dtype=dtype)
    n_train, n_total = freeze_backbone_keep_depformer(model)
    if args.train_text_head:
        # The WORDS the model says come from this head, and it is frozen by
        # default — measured, its free-running opener is a greeting, which is
        # exactly why every generation greets.  Train it.
        model.text_linear.requires_grad_(True)
        n_train += sum(p.numel() for p in model.text_linear.parameters())
    frozen_dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.frozen_dtype]
    cast_frozen_backbone(model, frozen_dtype)
    logger.info(f"backbone {n_total/1e9:.2f}B params | trainable "
                f"{n_train/1e6:.1f}M (DepFormer + "
                f"{'text head' if args.train_text_head else 'no text head'}) | "
                f"{len(kept)} layers | frozen dtype {args.frozen_dtype}")

    adapter = EmotionAdapter(len(EMOTIONS), depformer_dim(model),
                             n_codebooks=model.dep_q,
                             init=args.adapter_init,
                             target_rms=args.adapter_rms).to(device)
    if args.adapter_rms > 0.0:
        adapter.out_scale.requires_grad_(False)   # magnitude is fixed
    logger.info(f"emotion adapter: "
                f"{sum(p.numel() for p in adapter.parameters())/1e6:.2f}M params "
                f"for {len(EMOTIONS)} emotions {EMOTIONS}")

    tokenizer = sentencepiece.SentencePieceProcessor(args.tokenizer)
    interleaver = Interleaver(
        tokenizer=tokenizer, audio_frame_rate=mimi.frame_rate,
        text_padding=model.text_padding_token_id,
        end_of_text_padding=model.end_of_text_padding_id,
        zero_padding=model.zero_token_id,
        keep_main_only=True, use_bos_eos=True,
        main_speaker_label="ASSISTANT", device=str(device),
    )
    dit = DistillInterleavedTokenizer(
        mimi=mimi, interleaver=interleaver,
        duration_sec=args.duration_sec,
        target_n_audio_codebooks=model.num_audio_codebooks,
        use_sidecar=True, require_sidecar=True, use_rag=False,
    )
    train_loader = build_data_loader(
        instruct_tokenizer=dit, args=args, batch_size=args.batch_size,
        seed=args.seed, rank=0, world_size=1, is_eval=False)

    def make_eval_loader():
        return build_data_loader(
            instruct_tokenizer=dit, args=args, batch_size=args.batch_size,
            seed=0, rank=0, world_size=1, is_eval=True)

    # ── frozen-transformer cache ─────────────────────────────────────────
    cache = None
    cache_path = Path(args.cache_dir) / "transformer_cache.pt" if args.cache_dir else None
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists() and not args.rebuild_cache:
            logger.info(f"loading transformer cache {cache_path}")
            cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        else:
            n_samples = args.cache_samples or expected_samples(
                args.train_data, args.duration_sec)
            logger.info(f"building transformer cache ({n_samples} samples) ...")
            cache = build_transformer_cache(model, train_loader, device,
                                            len(EMOTIONS), n_samples,
                                            autocast_dtype)
            torch.save(cache, cache_path)
            logger.info(f"cache -> {cache_path} ({len(cache)} samples)")
        train_loader = None     # free the generator

    cache_eval: list = []
    if cache is not None and len(cache) > 2 * args.batch_size:
        n_eval = min(max(8, len(cache) // 50), len(cache) - args.batch_size)
        cache_eval = cache[:n_eval]
        cache = cache[n_eval:]
    if cache is not None and args.offload_frozen and cache_eval:
        freed = offload_frozen_to_cpu(model)
        logger.info(f"offloaded {freed/1e9:.2f}B frozen params to CPU "
                    f"(train {len(cache)} / eval {len(cache_eval)})")

    # ── generation-probe inputs: one fixed prompt + one fixed persona ─────
    gen_user_codes = gen_target_codes = gen_system_prompt = None
    if args.gen_every > 0:
        src = cache_eval or cache
        if src:
            c0 = src[0]["codes"].long()
            off = model.audio_offset + 8              # user stream = codebooks 9..16
            gen_user_codes = c0[off:off + 8].unsqueeze(0)     # [1, 8, T]
            gen_target_codes = c0[1:9].unsqueeze(0)           # ground-truth agent
            gen_system_prompt = first_system_prompt(args.train_data)
            logger.info(f"gen probe: {gen_user_codes.shape[-1]} prompt frames, "
                        f"persona={gen_system_prompt[:45]!r}...")
        else:
            logger.warning("gen probe disabled: no cache (pass --cache-dir)")

    # ── optimiser ────────────────────────────────────────────────────────
    df_lr = args.depformer_lr if args.depformer_lr is not None else args.lr
    param_groups = [
        {"params": [p for p in model.parameters() if p.requires_grad],
         "lr": df_lr, "weight_decay": args.weight_decay},
        {"params": list(adapter.parameters()), "lr": args.adapter_lr,
         "weight_decay": 0.0},
    ]
    optimizer = AdamW(param_groups, betas=(0.9, 0.95), eps=1e-8)
    if args.warmup_steps > 0:
        scheduler = SequentialLR(optimizer, [
            LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup_steps),
            CosineAnnealingLR(optimizer,
                              T_max=max(1, args.max_steps - args.warmup_steps),
                              eta_min=df_lr * 0.1),
        ], milestones=[args.warmup_steps])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.max_steps,
                                      eta_min=df_lr * 0.1)

    cb_weights = codebook_weight_vector(model, args).to(device)
    depformer_init = None
    if args.lambda_anchor > 0.0:
        ps = [p for p in model.parameters() if p.requires_grad]
        depformer_init = {"params": ps,
                          "values": [p.detach().clone() for p in ps]}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.save_dtype]

    model.train()
    step, micro, t0 = 0, 0, time.time()
    running, running_n = {}, 0
    optimizer.zero_grad()

    def consume(loss, stats):
        nonlocal step, micro, running, running_n
        (loss / args.grad_accum).backward()
        micro += 1
        running_n += 1
        for k, v in stats.items():
            running[k] = running.get(k, 0.0) + v
        if micro < args.grad_accum:
            return
        micro = 0
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad]
            + list(adapter.parameters()), args.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        step += 1
        if step % args.log_every == 0:
            avg = {k: v / max(running_n, 1) for k, v in running.items()}
            logger.info(f"step {step:5d}/{args.max_steps} "
                        + " ".join(f"{k}={v:.4f}" for k, v in avg.items())
                        + f" lr={scheduler.get_last_lr()[0]:.2e} "
                        + f"scale={adapter.out_scale.item():.3f} "
                        + f"cb_open={int((adapter.cb_gate[:model.dep_q // 2].abs() > 0.1).sum())}"
                        + f"/{model.dep_q // 2} "
                        + f"elapsed={time.time()-t0:.0f}s")
            running, running_n = {}, 0
        if args.eval_data and step % args.eval_every == 0:
            if cache_eval:
                ev = evaluate_cached(cache_eval, model, adapter, args, device,
                                     cb_weights)
            else:
                ev = evaluate(model, adapter, make_eval_loader(), args, device,
                              cb_weights)
            logger.info(f"[eval] step {step} "
                        + " ".join(f"{k}={v:.4f}" for k, v in ev.items()))

        if (gen_user_codes is not None and args.gen_every > 0
                and step % args.gen_every == 0):
            if args.offload_frozen:
                onload_frozen_to_device(model, device)
            _t0 = time.time()
            try:
                summ = run_generation_probe(
                    model, adapter, mimi, tokenizer, gen_user_codes,
                    gen_system_prompt, EMOTIONS, out_dir / "gen", step,
                    n_greet=args.gen_greet, n_ans=args.gen_frames,
                    voice_path=args.gen_voice,
                    device=device, emo_slots=(args.emo_slots or None),
                    target_codes=gen_target_codes, do_asr=args.gen_asr)
                logger.info(f"[gen] step {step} -> {summ['dir']} "
                            f"({time.time()-_t0:.0f}s, wav_diff={summ['max_wav_diff']})")
                txt = Path(summ["dir"]) / "transcripts.txt"
                if txt.exists():
                    for line in txt.read_text().splitlines():
                        logger.info(f"        {line}")
                if summ.get("emotions_identical"):
                    logger.warning(
                        "[gen] every emotion produced the SAME waveform — the "
                        "conditioning is NOT reaching generation.")
            except Exception:
                logger.exception("[gen] generation probe failed (non-fatal)")
            finally:
                if args.offload_frozen:
                    offload_frozen_to_cpu(model)
        if step % args.save_every == 0:
            save_checkpoint(out_dir, step, model, adapter, kept, base_kwargs,
                            {"step": step}, save_dtype)

    if cache is not None:
        logger.info(f"training on the cached transformer states "
                    f"({len(cache)} samples, batch {args.batch_size})")
        epoch = 0
        while step < args.max_steps:
            order = torch.randperm(len(cache)).tolist()
            for bi in range(0, len(order) - args.batch_size + 1, args.batch_size):
                if step >= args.max_steps:
                    break
                items = [cache[i] for i in order[bi:bi + args.batch_size]]
                loss, stats = training_step_cached(
                    items, model, adapter, args, device, cb_weights,
                    depformer_init, mimi=mimi,
                    do_recon=(step % max(1, args.recon_every) == 0))
                consume(loss, stats)
            epoch += 1
            logger.info(f"epoch {epoch} done (step {step})")
    else:
        for batch in train_loader:
            if step >= args.max_steps:
                break
            loss, stats = training_step(batch, model, adapter, args, device,
                                        cb_weights, depformer_init,
                                        autocast_dtype, mimi=mimi,
                                        do_recon=(step % max(1, args.recon_every) == 0))
            consume(loss, stats)

    save_checkpoint(out_dir, step, model, adapter, kept, base_kwargs,
                    {"step": step}, save_dtype)
    logger.info(f"done: {step} steps -> {out_dir}")


def main():
    train(build_args())


if __name__ == "__main__":
    main()
