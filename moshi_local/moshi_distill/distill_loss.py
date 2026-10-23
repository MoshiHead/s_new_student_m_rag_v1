"""
distill_loss.py
===============
Distillation objectives (distill_plan.md §7–§10, §15, §18) on top of the
existing RAG loss primitives:

    L_total = L_task + lambda_hidden * L_hidden + lambda_state * L_state
              + lambda_kd * L_KD (+ L_audio)
    (L_audio is included inside L_task via compute_audio_loss)

- L_state: DENSE per-frame matching (plan §7 generalized). Both sides are post
  teacher out_norm, i.e. exactly the vectors the frozen DepFormer/text head
  consume. This is the term that keeps the frozen autoregressive DepFormer on
  the distribution it was trained on; without it the per-frame conditioning is
  only indirectly driven by KD/CE and audio drifts/gets choppy at inference.

- L_hidden: plan §7 pooling. The teacher state at 12.5 Hz, post
  (frozen) teacher out_norm, is pooled pairwise
      H_t = (h_2t + h_(2t+1)) / 2
  then projected to the student dim, and matched with MSE against the
  student's post-norm state s_t. Pair boundaries are STRICTLY the same
  2-frame windows the compressor consumes, so causality is preserved.

- L_KD: temperature softmax KL(teacher || student) on undelayed logits.
  Text head ALWAYS (student must learn teacher text output behaviour);
  audio codebooks so the STATE upstream of the frozen DepFormer matches
  what the teacher DepFormer expects. All gradients flow through the
  student path only — the teacher side is detached by its no_grad forward.

- L_task: reuse compute_rag_losses with an EMPTY seg_metas list — every
  sample here is plain audio (segment_meta=None by construction in
  distill/data.py), so this degenerates to the standard text + audio CE
  with the project's NaN-safe weighted-CE behaviour intact.

NaN convention: undelayed logits contain real NaN at invalid trailing
positions (fill_value=NaN). Everything is sanitized with nan_to_num BEFORE
(weighted) reduction — multiplying NaN by a 0 weight after the fact would
poison gradients otherwise (same hazard documented in rag_loss_new.py).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .models.lm import LMOutput
from .rag_loss_new import compute_rag_losses


# ---------------------------------------------------------------------------
# Validity masks
# ---------------------------------------------------------------------------

def audio_frame_valid(codes: torch.Tensor, teacher_lm) -> torch.Tensor:
    """[B, T] bool — a frame is valid when its first audio codebook holds a
    real token (padding frames use zero_padding = -1 everywhere)."""
    return codes[:, teacher_lm.audio_offset] != teacher_lm.zero_token_id


def pair_valid_mask(frame_valid: torch.Tensor) -> torch.Tensor:
    """[B, T] bool -> [B, T//2] bool. A pair is valid only if BOTH frames are
    (conservative at the trailing padding boundary)."""
    return frame_valid[:, 0::2] & frame_valid[:, 1::2]


# ---------------------------------------------------------------------------
# L_hidden — hidden-state distillation
# ---------------------------------------------------------------------------

def compute_hidden_distill_loss(
    student_hidden: torch.Tensor,   # [B, T2, Ds]  post student out_norm
    teacher_out: torch.Tensor,      # [B, T, Dt]   post teacher out_norm
    hidden_proj,
    pair_valid: torch.Tensor | None = None,  # [B, T2] bool, None -> all valid
) -> torch.Tensor:
    B, T, Dt = teacher_out.shape
    T2, Ds = student_hidden.shape[1], student_hidden.shape[2]
    assert 2 * T2 == T, (
        f"teacher T={T} must be exactly 2x student steps T2={T2}")
    assert Dt == hidden_proj.proj.in_features

    H = teacher_out.view(B, T2, 2, Dt).mean(dim=2)     # (h_2t + h_(2t+1)) / 2
    target = hidden_proj(H)
    diff = (student_hidden.float() - target.float()).square().mean(-1)  # [B, T2]
    if pair_valid is None:
        return diff.mean()
    w = pair_valid.float()
    return (diff * w).sum() / w.sum().clamp(min=1e-8)


# ---------------------------------------------------------------------------
# L_state — dense per-frame state distillation (frozen-head input space)
# ---------------------------------------------------------------------------

def compute_state_distill_loss(
    student_states: torch.Tensor,   # [B, T, Dt] post teacher out_norm
    teacher_hidden: torch.Tensor,   # [B, T, Dt] post teacher out_norm
    frame_valid: torch.Tensor | None = None,  # [B, T] bool, None -> all valid
    mode: str = "mse",
) -> tuple[torch.Tensor, float]:
    """Dense 12.5 Hz state matching, in the space the FROZEN DepFormer consumes.

    `mode` controls the distance:
      - "mse"        : plain MSE. With untrained adapters the states are
                       uncorrelated, so the loss starts around 2*sigma^2 (e.g.
                       ~4.2 measured) -> with `lambda_state=60` that is a >250
                       total loss and gradient clipping smothers the updates.
      - "cosine"     : 1 - cos, bounded, directly optimises the gated metric.
      - "cosine_mse" : cosine + a small MSE anchor (recommended).

    Returns (loss, mean_cosine_similarity) — the cosine is a pure diagnostic
    of how aligned the conditioning is (1.0 == perfect).
    """
    assert student_states.shape == teacher_hidden.shape, (
        f"{student_states.shape} vs {teacher_hidden.shape}")
    s = student_states.float()
    h = teacher_hidden.float()
    cos = F.cosine_similarity(s, h, dim=-1)                     # [B, T]
    if mode == "mse":
        per_frame = (s - h).square().mean(-1)
    elif mode == "cosine":
        per_frame = 1.0 - cos
    elif mode == "cosine_mse":
        per_frame = (1.0 - cos) + 0.1 * (s - h).square().mean(-1)
    else:
        raise ValueError(f"unknown state-loss mode {mode!r}")
    if frame_valid is None:
        loss = per_frame.mean()
        cos_mean = cos.mean()
    else:
        w = frame_valid.float()
        loss = (per_frame * w).sum() / w.sum().clamp(min=1e-8)
        cos_mean = (cos * w).sum() / w.sum().clamp(min=1e-8)
    return loss, cos_mean.item()


# ---------------------------------------------------------------------------
# L_KD — output distillation
# ---------------------------------------------------------------------------

def compute_text_kd_loss(
    student_text_logits: torch.Tensor,  # [B, 1, T, V] (NaN at invalid pos)
    teacher_text_logits: torch.Tensor,  # [B, 1, T, V] (NaN at invalid pos)
    text_mask: torch.Tensor,            # [B, 1, T] bool
    temperature: float = 2.0,
) -> torch.Tensor:
    t = temperature
    t_logits = torch.nan_to_num(teacher_text_logits[:, 0], nan=0.0).float()
    s_logits = torch.nan_to_num(student_text_logits[:, 0], nan=0.0).float()
    p_teacher = torch.softmax(t_logits / t, dim=-1)
    logp_student = torch.log_softmax(s_logits / t, dim=-1)
    kl = (p_teacher * (p_teacher.clamp_min(1e-12).log() - logp_student)).sum(-1)  # [B, T]
    mask = text_mask[:, 0].float()
    return (kl * mask).sum() / mask.sum().clamp(min=1e-8)


def compute_audio_kd_loss(
    student_logits: torch.Tensor,   # [B, Ka, T, card] (NaN at invalid pos)
    teacher_logits: torch.Tensor,   # [B, Ka, T, card] (NaN at invalid pos)
    audio_mask: torch.Tensor,       # [B, Ka, T] bool
    temperature: float = 2.0,
) -> torch.Tensor:
    t = temperature
    t_logits = torch.nan_to_num(teacher_logits, nan=0.0).float()
    s_logits = torch.nan_to_num(student_logits, nan=0.0).float()
    p_teacher = torch.softmax(t_logits / t, dim=-1)
    logp_student = torch.log_softmax(s_logits / t, dim=-1)
    kl = (p_teacher * (p_teacher.clamp_min(1e-12).log() - logp_student)).sum(-1)  # [B, Ka, T]
    w = audio_mask.float()
    return (kl * w).sum() / w.sum().clamp(min=1e-8)


# ---------------------------------------------------------------------------
# L_task — reuse the RAG loss machinery with zero RAG structure
# ---------------------------------------------------------------------------

def compute_task_losses(teacher_lm, student_output: LMOutput, codes: torch.Tensor, args):
    """Plain text + audio CE against ground-truth codes. `seg_metas=[]`
    makes the per-segment weights degenerate to a constant (`no_rag_weight`);
    any audio RAG masking is likewise disabled (no ref spans exist)."""
    return compute_rag_losses(
        model=teacher_lm,
        output=student_output,
        codes=codes,
        seg_metas=[],
        args=args,
    )


# ---------------------------------------------------------------------------
# Combined
# ---------------------------------------------------------------------------

def compute_distill_losses(
    student_lm,
    teacher_lm,
    student_hidden: torch.Tensor,       # [B, T2, Ds]
    student_output: LMOutput,
    teacher_distill_output: dict[str, torch.Tensor],
    codes: torch.Tensor,
    args,
    student_states: torch.Tensor | None = None,   # [B, T, Dt] post teacher out_norm
    phase: str = "joint",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Builds every term, applies loss weights, returns (total, stats dict).

    teacher_distill_output keys: 'hidden' [B, T, Dt],
        'text_logits' [B, 1, T, V], 'audio_logits' [B, Ka, T, card] — all
        from the frozen-teacher no_grad forward (see train_distill.py).

    phase:
      - "joint": full objective (task + hidden + state + KD).
      - "state": only state-alignment terms (hidden + dense state). Use it as a
        warmup phase: get the frozen heads' conditioning right BEFORE asking
        the frozen DepFormer's outputs to match the teacher's distribution.
        Otherwise the KD/CE gradients through the frozen heads fight the state
        objective while the conditioning is still meaningless.
    """
    frame_valid = audio_frame_valid(codes, teacher_lm)
    pair_valid = pair_valid_mask(frame_valid)

    # L_task (text CE + audio CE; degenerates to unweighted for distill data)
    task_text, task_audio = compute_task_losses(teacher_lm, student_output, codes, args)

    # L_hidden (pooled 6.25 Hz state)
    hidden_loss = compute_hidden_distill_loss(
        student_hidden, teacher_distill_output["hidden"], student_lm.hidden_proj,
        pair_valid,
    )

    # L_state (dense 12.5 Hz state, frozen-head input space)
    lambda_state = float(getattr(args, "lambda_state", 0.0) or 0.0)
    state_loss = None
    state_cos = 0.0
    if lambda_state > 0.0 and student_states is not None:
        state_loss, state_cos = compute_state_distill_loss(
            student_states, teacher_distill_output["hidden"], frame_valid,
            mode=getattr(args, "state_loss", "mse"),
        )

    # L_KD text
    kd_text = compute_text_kd_loss(
        student_output.text_logits,
        teacher_distill_output["text_logits"],
        student_output.text_mask,
        temperature=args.kd_temperature,
    )

    stats = {
        "task_text": task_text.item(),
        "task_audio": task_audio.item(),
        "hidden": hidden_loss.item(),
        "kd_text": kd_text.item(),
        "state_cos": state_cos,
    }

    if phase == "state":
        # Alignment warmup: only teach the student the teacher's internal
        # states; the output-distillation/task terms join afterwards.
        total = (
            args.lambda_hidden * hidden_loss
            + (lambda_state * state_loss if state_loss is not None else 0.0)
        )
        if state_loss is not None:
            stats["state"] = state_loss.item()
        stats["loss"] = total.item()
        return total, stats

    total = (
        args.lambda_task_text * task_text
        + args.lambda_task_audio * task_audio
        + args.lambda_hidden * hidden_loss
        + args.lambda_kd_text * kd_text
    )
    if state_loss is not None:
        total = total + lambda_state * state_loss
        stats["state"] = state_loss.item()

    # L_KD audio — keeps the student's upsampled states inside the frozen
    # DepFormer's expected conditioning manifold
    if getattr(args, "lambda_kd_audio", 0.0) > 0.0:
        kd_audio = compute_audio_kd_loss(
            student_output.logits,
            teacher_distill_output["audio_logits"],
            student_output.mask,
            temperature=args.kd_temperature,
        )
        total = total + args.lambda_kd_audio * kd_audio
        stats["kd_audio"] = kd_audio.item()

    stats["loss"] = total.item()
    return total, stats
