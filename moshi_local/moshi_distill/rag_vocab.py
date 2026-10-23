"""
rag_vocab.py — grow the 32 000-token text vocab to 32 003 for the RAG control
tokens, on an already-distilled 12.5 Hz student.

Copied from `train_latent.py::add_special_tokens_and_resize` (the only part of
that file that is correct for this model) so the Stage-A trainer does not have
to import `transformers` / `RefResampler`.  Behaviour is unchanged:

  * new ids must be contiguous and appended last: `<ref_start>` 32000,
    `<ref_end>` 32001, `<lookup>` 32002;
  * the old padding/initial row (32000) moves to 32003, so
    `text_initial_token_id` becomes 32003 while `text_padding_token_id` stays 3;
  * three tables are resized: `text_emb` (input), `text_linear` (output) and
    `depformer_text_emb` (the DepFormer's text input — this one is easy to
    miss and produces an OOB lookup at the first `<lookup>` otherwise);
  * gradient hooks zero every gradient except the three new rows. **The hooks do
    not enable `requires_grad`** — call `enable_new_token_rows()` afterwards.
"""

from __future__ import annotations

import logging

import sentencepiece
import torch

from .models.lm import LMModel, ScaledEmbedding

logger = logging.getLogger(__name__)


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
            raise ValueError(f"{tok!r} not in tokenizer vocab — wrong tokenizer.model?")
        expected = old_text_card + i
        if tid != expected:
            raise ValueError(f"{tok!r} has id {tid}, expected {expected} "
                             f"(must be contiguous, appended last).")
        new_ids.append(tid)

    device = lm_base.text_emb.weight.device
    dtype = lm_base.text_emb.weight.dtype
    dim = lm_base.text_emb.embedding_dim
    new_text_card = old_text_card + n_new

    # --- text_emb (main transformer input) ---
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

    # --- text_linear (output projection) ---
    old_linear = lm_base.text_linear
    extra_text = 1 if lm_base.existing_text_padding_id is None else 0
    old_pad_row_out = old_linear.weight.data[old_text_card].clone() if extra_text else None
    old_pad_bias = (
        old_linear.bias.data[old_text_card].clone()
        if (extra_text and old_linear.bias is not None) else None
    )
    new_out_dim = new_text_card + extra_text
    new_linear = torch.nn.Linear(dim, new_out_dim,
                                 bias=old_linear.bias is not None,
                                 device=device, dtype=dtype)
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

    # --- depformer_text_emb (sibling table read by forward_depformer_training) ---
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
    lm_base.depformer_text_emb = new_dep_emb
    lm_base.text_card = new_text_card

    # --- gradient masks: only the new rows may move ---
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

    dep_emb_mask = torch.zeros(new_text_card + 1, 1,
                               device=old_dep_emb.weight.device,
                               dtype=old_dep_emb.weight.dtype)
    dep_emb_mask[new_ids] = 1.0
    new_dep_emb.weight.register_hook(lambda g, m=dep_emb_mask: g * m)

    logger.info("resized text vocab %d -> %d (+%d: %s); pad/initial row moved "
                "%d -> %d; depformer_text_emb resized too",
                old_text_card, new_text_card, n_new,
                dict(zip(new_tokens, new_ids)), old_text_card, new_text_card)
    return dict(zip(new_tokens, new_ids))


def enable_new_token_rows(lm: LMModel) -> None:
    """Turn on gradients for the three resized tables.

    Safe: the hooks installed by `add_special_tokens_and_resize` still zero every
    row except the new control tokens, so this does not let the base vocab drift.
    """
    for mod, name in ((lm.text_emb, "text_emb"),
                      (lm.text_linear, "text_linear"),
                      (lm.depformer_text_emb, "depformer_text_emb")):
        for p in mod.parameters():
            p.requires_grad_(True)
    logger.info("new token rows enabled for %s",
                "text_emb/text_linear/depformer_text_emb")
