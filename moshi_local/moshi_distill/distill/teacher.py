"""
distill/teacher.py
==================
Frozen-teacher forward for distillation (plan §4): PersonaPlex @ 12.5 Hz
under no_grad, returning everything the student needs for

    L_hidden  → 'hidden'      [B, T, D_teacher]  post teacher out_norm
    L_KD text → 'text_logits' [B, 1, T, V_text]  undelayed (NaN at invalid)
    L_KD audio→ 'audio_logits'[B, Ka, T, card]  undelayed (NaN at invalid)

and the matching validity masks. The teacher stays completely frozen —
tensors come back detached by construction.

This is deliberately a mirror of `LMModel.forward_train` that additionally
keeps `transformer_out`; never modify the teacher path itself.

Streaming hygiene: in training we never enter a `streaming(batch_size)`
context, so `StreamingTransformer.forward` recomputes the rope offset from
zero on every batch — no cross-batch offset accumulation occurs, and
`reset_streaming()` is not needed (and would raise when stateless).
"""

from __future__ import annotations

import logging

import torch

from ..models.lm import _delay_sequence, _undelay_sequence

logger = logging.getLogger(__name__)


@torch.no_grad()
def forward_teacher_distill(teacher_lm, codes: torch.Tensor) -> dict[str, torch.Tensor]:
    B, K, T = codes.shape
    assert K == teacher_lm.num_codebooks, (
        f"codes K={K} != teacher num_codebooks={teacher_lm.num_codebooks}"
    )

    initial = teacher_lm._get_initial_token().expand(B, -1, -1)
    delayed = _delay_sequence(teacher_lm.delays, codes, initial)
    delayed = torch.cat([initial, delayed], dim=2)          # [B, K, T+1]

    # main forward: embeddings -> temporal transformer -> out_norm
    emb = teacher_lm.embed_codes(delayed[:, :, :-1])
    transformer_out, _ = teacher_lm.forward_embeddings(emb)  # applies out_norm

    # The delayer output that the DepFormer actually consumes, *before* the
    # undelay: slot k reads `seq[:, audio_offset + k - 1]` and slot 0 reads
    # `seq[:, 0]`. Anything conditioning on previous codebooks (grouped /
    # parallel students) MUST use these delayed inputs, not the frame-aligned
    # targets, or the conditioning is off by the delay pattern.
    seq = delayed[:, :, 1:]

    # frozen heads
    text_logits = teacher_lm.text_linear(transformer_out)[:, None]
    audio_logits = teacher_lm.forward_depformer_training(seq, transformer_out)

    # undelay back to per-code targets, NaN-fill invalid positions
    audio_logits, audio_mask = _undelay_sequence(
        teacher_lm.delays[teacher_lm.audio_offset:
                          teacher_lm.audio_offset + teacher_lm.dep_q],
        audio_logits, fill_value=float("NaN"))
    audio_mask &= (codes[:, teacher_lm.audio_offset:
                         teacher_lm.audio_offset + teacher_lm.dep_q]
                   != teacher_lm.zero_token_id)
    text_logits, text_mask = _undelay_sequence(
        teacher_lm.delays[:1], text_logits, fill_value=float("NaN"))
    text_mask &= (codes[:, :1] != teacher_lm.zero_token_id)

    return {
        "hidden": transformer_out,          # [B, T, D_teacher]
        "text_logits": text_logits,         # [B, 1, T, V_text]
        "text_mask": text_mask,             # [B, 1, T]
        "audio_logits": audio_logits,       # [B, Ka, T, card]
        "audio_mask": audio_mask,           # [B, Ka, T]
        # delayed DepFormer inputs (sequence-aligned, NOT frame-aligned)
        "depformer_inputs": seq[:, teacher_lm.audio_offset:
                                teacher_lm.audio_offset + teacher_lm.dep_q],
        "text_inputs": seq[:, 0],           # [B, T]
    }
