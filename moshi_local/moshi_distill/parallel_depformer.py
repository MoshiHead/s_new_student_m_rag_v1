"""
parallel_depformer.py — retrainable grouped DepFormer student
=============================================================

Goal (depformer_retrain_plan.md)
--------------------------------
The frozen 7B backbone runs once per 12.5 Hz frame; the DepFormer then runs a
SERIAL chain of ``dep_q`` small forwards per frame, one per audio codebook,
because codebook ``k`` is conditioned on the sampled codebook ``k-1``.  That
serial chain is the latency bottleneck on Jetson/Orin.  This module retrains a
grouped, partly-parallel DepFormer **student** that predicts every audio
codebook of a frame in a small number of dependent passes, keeping the frozen
backbone bit-identical.

Designs (plan §4/§9, "which path first")
----------------------------------------
* ``schedule="B"`` (default, first path): pass 1 predicts slot 0 (agent cb0,
  whose input is the text token, so it is known exactly); pass 2 predicts all
  remaining slots in parallel from the real cb0 and learned MASK embeddings for
  the rest.  2 dependent passes.
* ``schedule="C"``: B plus one refinement pass in which every slot sees its own
  parallel draft of the other slots.  3 dependent passes.
* ``schedule="A"``: grouped AR with commit passes (smaller groups).  More passes,
  better correlation.

``schedule="E"`` (exact speculative decoding) is intentionally NOT implemented
yet; it is the lossless landing pad if neither B nor C clears the quality gates.

The objective: teacher-only
----------------------------
The loss compares the student ONLY against the frozen teacher model's outputs
(full-distribution KL + top-k/temperature-matched KL + hard CE to the teacher's
greedy tokens).  Ground-truth codes are never used as targets.  Every ``dep_q``
codebook (agent 0..7 and user 8..15) is predicted and accounted for with a
strictly-positive weight; the user stream is never masked.

Data: the USER stream is the query
----------------------------------
Two collection modes:
* ``collect`` (default): one teacher-forced pass over the dual-channel codes
  (kept for reference / ablations).
* ``collect --rollout``: the **primary** mode.  The sample's USER stream is the
  only query fed to the frozen teacher; the teacher *generates* the agent stream
  + text autoregressively, and the student distills from the teacher's outputs
  over that generation.  No ground-truth agent code enters the targets.

Inputs / outputs are the teacher's layout: frame state ``h`` (post ``out_norm``,
4096-d), text token, audio codes ``[B, K, T]`` (K = num_codebooks = 17), and
logits ``[B, dep_q, T, card]``.

The student is **initialised from the frozen teacher's DepFormer weights**
(same layers, same ``weights_per_step`` slices) and only the input layout /
attention masking differs, exactly as the plan requires.

Usage
-----
    # 0. CPU self-test, no downloads
    python -m moshi_distill.parallel_depformer --self-test

    # 1. PRIMARY: teacher rollouts with the USER stream as the query
    python -m moshi_distill.parallel_depformer collect --rollout \
        --train-data data/train.jsonl --out targets.pt --max-batches 500

    # 2. fit the grouped student (teacher logits recomputed on the fly)
    python -m moshi_distill.parallel_depformer fit \
        --targets targets.pt --schedule B --out-dir checkpoints/grouped_depformer

    # 3. serve: ParallelDepFormerGen(student, teacher, ...)
"""

from __future__ import annotations

import argparse
import copy
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models.lm import (
    LMGen,
    AUDIO_TOKENS_PER_STREAM,
    SILENCE_TOKENS,
    SINE_TOKENS,
    _delay_sequence,
    _undelay_sequence,
)
from .utils.sampling import sample_token

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schedule + codebook weighting
# ---------------------------------------------------------------------------

# Relative weights WITHIN one 8-codebook stream (plan §5): higher for cb0
# (semantic / content), lower for the fine acoustic detail.
AGENT_CODEBOOK_WEIGHTS = (3.0, 2.0, 1.5, 1.0, 1.0, 0.7, 0.7, 0.5)


def default_codebook_weights(n: int, user_weight: float = 1.0) -> torch.Tensor:
    """Weights for all ``n`` predicted codebooks (agent then user stream).

    ``user_weight`` scales the user-stream codebooks.  It defaults to 1.0 so
    every codebook is *accounted for*; lower it only if the user stream should
    contribute less, never to zero (that would be the forbidden masking).
    """
    w = []
    for k in range(n):
        base = AGENT_CODEBOOK_WEIGHTS[k % AUDIO_TOKENS_PER_STREAM]
        w.append(base * (user_weight if k >= AUDIO_TOKENS_PER_STREAM else 1.0))
    return torch.tensor(w, dtype=torch.float32)


def build_groups(n: int, schedule: str) -> list[list[int]]:
    """Slot groups processed serially; slots inside a group are parallel."""
    s = schedule.upper()
    if n <= 1:
        return [list(range(n))]
    if s in ("B", "C"):
        return [[0], list(range(1, n))]
    if s in ("AR", "FULL"):
        # Fully autoregressive reference: one slot per group. Reproduces the
        # teacher DepFormer exactly (same pass structure, no masks), so it is
        # the correct overfit sanity check; no speedup.
        return [[k] for k in range(n)]
    if s == "A":
        groups: list[list[int]] = [[0]]
        # Grouped AR per stream: {0}, {1,2}, {3,4}, {5,6,7} (then the same for
        # the next 8-codebook stream).  Small groups -> 4-8 dependent passes.
        starts = [1]
        while starts:
            start = starts.pop(0)
            if start >= n:
                continue
            end = min(start + AUDIO_TOKENS_PER_STREAM, n)
            k = start
            while k < end:
                size = 3 if (end - k) == 3 else 2
                groups.append(list(range(k, min(k + size, end))))
                k += size
            if end < n:
                starts.append(end)
        covered = {i for g in groups for i in g}
        rest = [i for i in range(n) if i not in covered]
        if rest:
            groups.append(rest)
        return groups
    raise ValueError(f"unknown schedule {schedule!r} (use B, C or A)")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class GroupedDepFormerConfig:
    schedule: str = "B"                 # B (first), C, A, AR (overfit reference)
    user_codebook_weight: float = 1.0   # relative weight for codebooks >= 8
    mask_init_scale: float = 0.0        # learned MASK_k init (std)
    # loss (plan §5)
    lam_samp: float = 1.0               # weight of the temp/top-k-matched KL
    samp_top_k: int = 250               # inference top_k (plan §1.10)
    samp_temp: float = 0.8              # inference temp (plan §1.10)
    ce_teacher_weight: float = 1.0      # hard CE to the TEACHER's greedy tokens
    kd_temperature: float = 1.0         # full-distribution KL temperature
    # training-time draft used by the refine (C) pass — matches inference
    draft_use_sampling: bool = True
    draft_temp: float = 0.8
    draft_top_k: int = 250
    # serving: codebooks to actually decode. The user stream (>= 8) is always
    # provided by the server loop, so we only need to decode the agent stream;
    # the MASK context for the skipped slots is identical to training, so the
    # agent logits are bit-identical to a full decode.
    serve_max_slot: int = AUDIO_TOKENS_PER_STREAM


# ---------------------------------------------------------------------------
# The student
# ---------------------------------------------------------------------------

class GroupedDepFormerStudent(nn.Module):
    """Teacher-weight-initialised DepFormer with grouped/masked slot inputs.

    Owns a trainable **copy** of the teacher's DepFormer stack, per-slot input
    projections, token embeddings and output heads (``linears``).  The teacher
    itself is never referenced here, so it stays frozen for label generation.
    """

    def __init__(self, teacher_lm, schedule: str = "B",
                 mask_init_scale: float = 0.0,
                 serve_max_slot: int = AUDIO_TOKENS_PER_STREAM,
                 device=None, dtype=None):
        super().__init__()
        self.num_audio_codebooks = int(teacher_lm.dep_q)
        self.n = int(teacher_lm.dep_q)
        self.card = int(teacher_lm.card)
        self.audio_offset = 1
        self.depformer_multi_linear = teacher_lm.depformer_multi_linear
        self.weights_per_step_schedule = teacher_lm.depformer_weights_per_step_schedule

        dep = teacher_lm.depformer
        self.depformer = copy.deepcopy(dep)
        self.depformer_in = copy.deepcopy(teacher_lm.depformer_in)
        self.depformer_emb = copy.deepcopy(teacher_lm.depformer_emb)
        self.depformer_text_emb = copy.deepcopy(teacher_lm.depformer_text_emb)
        self.linears = copy.deepcopy(teacher_lm.linears)

        self.depformer_dim = int(self.linears[0].in_features)
        # Match the copied teacher weights' dtype (bf16 in the real run) so the
        # MASK parameters and the whole student never get silently cast.
        param_dtype = self.linears[0].weight.dtype
        # Learned MASK_k embeddings for slots whose causal predecessor is not
        # decided yet in the current group (plan §4, designs A/B/C).
        self.mask = nn.Parameter(torch.zeros(self.n, self.depformer_dim,
                                             dtype=param_dtype))
        if mask_init_scale > 0:
            nn.init.normal_(self.mask, std=mask_init_scale)

        self.groups = build_groups(self.n, schedule)
        self.schedule_name = schedule.upper()
        self.refine_passes = 1 if self.schedule_name == "C" else 0
        self.serve_max_slot = int(serve_max_slot)

        # The copy is fresh weights for the student -> trainable. (The source
        # teacher is frozen, so the deepcopy inherits requires_grad=False.)
        for p in self.parameters():
            p.requires_grad_(True)

        # Move device/dtype explicitly: never default dtype to a parameter's
        # dtype (that would up-cast an fp32 mask over a bf16 teacher copy).
        if device is not None:
            self.to(device=device)
        if dtype is not None:
            self.to(dtype=dtype)

    # -- helpers -----------------------------------------------------------

    def _depformer_in_for(self, k: int) -> nn.Module:
        if self.depformer_multi_linear:
            idx = k
            if self.weights_per_step_schedule is not None:
                idx = int(self.weights_per_step_schedule[k])
            return self.depformer_in[idx]
        return self.depformer_in[0]

    def _build_inputs(self, transformer_out: torch.Tensor,
                      tok: torch.Tensor, decided_in: torch.Tensor) -> torch.Tensor:
        """Run the (non-streaming) DepFormer over all slots in one shot.

        Args:
            transformer_out: [B, T, Dt] frozen-backbone state (post out_norm).
            tok: [B, T, n] long. ``tok[:, :, 0]`` is the text token; for k >= 1,
                ``tok[:, :, k]`` is the audio token of codebook ``k-1``.
            decided_in: [B, T, n] bool — is ``tok[:, :, k]`` a real token (vs the
                learned MASK_k for that slot)?
        Returns:
            [B*T, n, depformer_dim] hidden for every slot.
        """
        B, T, _ = transformer_out.shape
        ins = []
        for k in range(self.n):
            tr = self._depformer_in_for(k)(transformer_out)
            if k == 0:
                emb = self.depformer_text_emb(tok[..., 0])
            else:
                real = self.depformer_emb[k - 1](tok[..., k])
                mask_k = self.mask[k].to(real.dtype).view(1, 1, -1)
                emb = torch.where(decided_in[..., k].unsqueeze(-1), real,
                                  mask_k.expand_as(real))
            ins.append((tr + emb).unsqueeze(2))
        x = torch.cat(ins, dim=2).reshape(B * T, self.n, self.depformer_dim)
        return self.depformer(x)

    def _read_heads(self, y: torch.Tensor, slots, B: int, T: int):
        out = []
        for k in slots:
            out.append(self.linears[k](y[:, k]).view(B, T, self.card))
        return out

    # -- teacher-forced training forward -----------------------------------

    def forward_groups(self, transformer_out: torch.Tensor,
                       text_tokens: torch.Tensor,
                       prefix_codes: torch.Tensor | None = None,
                       max_slot: int | None = None) -> torch.Tensor:
        """Grouped-AR student logits, teacher-forced on ``prefix_codes``.

        Args:
            transformer_out: [B, T, Dt].
            text_tokens: [B, T] long.
            prefix_codes: [B, n, T] teacher/GT audio codebooks used to fill the
                already-decided slots (teacher forcing).  ``None`` -> all MASK.
            max_slot: only decode slots ``< max_slot`` (serving fast path).
        Returns:
            logits [B, n, T, card] (teacher layout).
        """
        B, T, _ = transformer_out.shape
        n = self.n
        max_slot = n if max_slot is None else max_slot
        tok = torch.zeros((B, T, n), dtype=torch.long, device=transformer_out.device)
        tok[..., 0] = text_tokens
        if prefix_codes is not None:
            # tok[..., k] = audio codebook k-1 -> prefix_codes[:, k-1]
            tok[..., 1:] = prefix_codes[:, :n - 1].permute(0, 2, 1)
        # NOTE: build a fresh `decided_in` tensor for each group. Mutating a
        # tensor in place after it fed a `torch.where` breaks autograd; a Python
        # list tracks state instead.
        decided_slots = [False] * n
        decided_slots[0] = True

        out: list[torch.Tensor | None] = [None] * n
        for group in self.groups:
            active = [k for k in group if k < max_slot]
            if active:
                decided_in = torch.zeros((B, T, n), dtype=torch.bool,
                                         device=transformer_out.device)
                decided_in[..., 0] = True
                for k in range(1, n):
                    if decided_slots[k]:
                        decided_in[..., k] = True
                y = self._build_inputs(transformer_out, tok, decided_in)
                for k, lg in zip(active, self._read_heads(y, active, B, T)):
                    out[k] = lg
            # deciding output slot k makes codebook k (the input of slot k+1)
            # available for all later groups.
            for k in group:
                if k + 1 < n:
                    decided_slots[k + 1] = True

        for k in range(max_slot):
            if out[k] is None:
                raise RuntimeError(f"schedule {self.schedule_name} never covers slot {k}")
        logits = torch.stack(
            [out[k] if out[k] is not None else
             torch.zeros(B, T, self.card, device=transformer_out.device,
                         dtype=transformer_out.dtype)
             for k in range(n)], dim=1)
        return logits

    def refine(self, transformer_out: torch.Tensor, text_tokens: torch.Tensor,
               draft_codes: torch.Tensor) -> torch.Tensor:
        """Design C refinement: every slot sees the draft of every other slot."""
        B, T, _ = transformer_out.shape
        n = self.n
        tok = torch.zeros((B, T, n), dtype=torch.long, device=transformer_out.device)
        tok[..., 0] = text_tokens
        tok[..., 1:] = draft_codes[:, :n - 1].permute(0, 2, 1)
        decided_in = torch.ones((B, T, n), dtype=torch.bool,
                                device=transformer_out.device)
        y = self._build_inputs(transformer_out, tok, decided_in)
        out = []
        for k in range(n):
            out.append(self.linears[k](y[:, k]).view(B, T, self.card))
        return torch.stack(out, dim=1)

    # -- inference ---------------------------------------------------------

    @torch.no_grad()
    def decode(self, transformer_out: torch.Tensor, text_tokens: torch.Tensor,
               temp: float = 0.8, top_k: int = 250, use_sampling: bool = True,
               max_slot: int | None = None) -> torch.Tensor:
        """Sample slots group by group; returns [B, n, T] (teacher layout).

        ``max_slot`` limits decoding to the first slots (the user stream is
        provided at serving, see ``serve_max_slot``).  The MASK context of the
        skipped slots is unchanged, so decoded agent logits are identical to a
        full decode.
        """
        B, T, _ = transformer_out.shape
        n = self.n
        max_slot = n if max_slot is None else max_slot
        tok = torch.zeros((B, T, n), dtype=torch.long, device=transformer_out.device)
        tok[..., 0] = text_tokens
        decided_in = torch.zeros((B, T, n), dtype=torch.bool,
                                 device=transformer_out.device)
        decided_in[..., 0] = True
        sampled = torch.zeros((B, n, T), dtype=torch.long, device=transformer_out.device)

        for group in self.groups:
            active = [k for k in group if k < max_slot]
            if active:
                y = self._build_inputs(transformer_out, tok, decided_in)
                for k in active:
                    lg = self.linears[k](y[:, k]).float()
                    s = sample_token(lg, use_sampling, temp, top_k).view(B, T)
                    sampled[:, k] = s
                    if k + 1 < n:
                        tok[..., k + 1] = s
            for k in group:
                if k + 1 < n:
                    decided_in[..., k + 1] = True

        if self.refine_passes:
            draft = sampled
            tok2 = torch.zeros_like(tok)
            tok2[..., 0] = text_tokens
            tok2[..., 1:] = draft[:, :n - 1].permute(0, 2, 1)
            decided2 = torch.ones_like(decided_in)
            y = self._build_inputs(transformer_out, tok2, decided2)
            for k in range(max_slot):
                lg = self.linears[k](y[:, k]).float()
                sampled[:, k] = sample_token(lg, use_sampling, temp, top_k).view(B, T)
        return sampled


# ---------------------------------------------------------------------------
# Loss — every codebook, no user-stream masking
# ---------------------------------------------------------------------------

def _flat_kl(p_logits: torch.Tensor, q_logits: torch.Tensor,
             temperature: float) -> torch.Tensor:
    p = torch.softmax(p_logits.float() / temperature, dim=-1)
    logq = torch.log_softmax(q_logits.float() / temperature, dim=-1)
    return (p * (p.clamp_min(1e-12).log() - logq)).sum(-1)          # [B, n, T]


def _topk_kl(p_logits: torch.Tensor, q_logits: torch.Tensor,
             k: int, temperature: float) -> torch.Tensor:
    """KL between the top-k/temp sampling distributions (plan §5 L_samp).

    Uses the teacher's top-k support for both sides so the term stays finite
    where the student puts mass on tokens outside the teacher's top-k.
    """
    v = p_logits.size(-1)
    kk = min(int(k), v)
    idx = torch.topk(p_logits.float(), kk, dim=-1).indices
    p_at = p_logits.float().gather(-1, idx)
    q_at = q_logits.float().gather(-1, idx)
    p = torch.softmax(p_at / temperature, dim=-1)
    logq = torch.log_softmax(q_at / temperature, dim=-1)
    return (p * (p.clamp_min(1e-12).log() - logq)).sum(-1)          # [B, n, T]


def _per_codebook_mean(per_cb: torch.Tensor, mask: torch.Tensor,
                       weights: torch.Tensor) -> torch.Tensor:
    """Weighted mean over frames, kept per codebook. per_cb [B,n,T]."""
    w = mask.float() * weights.view(1, -1, 1)
    num = (per_cb * w).sum(dim=(0, 2))
    den = w.sum(dim=(0, 2)).clamp(min=1e-8)
    return num / den


def parallel_depformer_loss(
    student_logits: torch.Tensor,       # [B, n, T, card]  (all dep_q codebooks)
    teacher_logits: torch.Tensor,       # [B, n, T, card]
    teacher_tokens: torch.Tensor,       # [B, n, T] (argmax of teacher logits)
    mask: torch.Tensor,                 # [B, n, T] valid frames (never means "user")
    codebook_weights: torch.Tensor | None = None,
    lam_samp: float = 1.0,
    samp_top_k: int = 250,
    samp_temp: float = 0.8,
    ce_teacher_weight: float = 1.0,
    kd_temperature: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Teacher-only distillation over EVERY predicted codebook.

        loss = sum_k w_k * (L_kl[k] + lam*L_samp[k] + ce_w*L_ce_teacher[k]) / sum_k w_k

    **Every target comes from the teacher model.** ``teacher_logits`` is the
    frozen-teacher distribution and ``teacher_tokens`` is its argmax; the
    ground-truth data codes are never referenced. There is deliberately no
    ``num_agent_codebooks`` masking: codebooks 8..15 (user stream) carry the
    same loss weight as 0..7 and are never silently dropped.

    ``mask`` only excludes padding / invalid frames.
    """
    B, n, T, card = student_logits.shape
    assert teacher_logits.shape == student_logits.shape, (
        f"teacher {tuple(teacher_logits.shape)} vs student "
        f"{tuple(student_logits.shape)}")
    assert teacher_tokens.shape == (B, n, T), teacher_tokens.shape
    assert mask.shape == (B, n, T), mask.shape

    p = torch.nan_to_num(teacher_logits, nan=0.0, posinf=0.0, neginf=0.0)
    q = torch.nan_to_num(student_logits, nan=0.0, posinf=0.0, neginf=0.0)

    if codebook_weights is None:
        codebook_weights = default_codebook_weights(n)
    w = codebook_weights.to(student_logits.device).float()
    assert w.numel() == n
    assert bool((w > 0).all()), (
        "codebook weights must be strictly positive for every codebook; "
        "a zero weight would silently mask a codebook (e.g. the user stream). "
        f"got {w.tolist()}")

    # Guard: every codebook must have valid frames. A fully-zero column would
    # mean a codebook was masked out — exactly what this loss must never do.
    cover = mask.float().sum(dim=(0, 2))
    assert bool((cover > 0).all()), (
        "codebook(s) fully masked: "
        f"{(cover <= 0).nonzero().flatten().tolist()}")

    kl_cb = _per_codebook_mean(_flat_kl(p, q, kd_temperature), mask, w)
    samp_cb = _per_codebook_mean(_topk_kl(p, q, samp_top_k, samp_temp), mask, w)

    # Hard teacher-token CE: target = argmax of the TEACHER logits (not data).
    tt = torch.where(mask, teacher_tokens, torch.zeros_like(teacher_tokens))
    ce_t = F.cross_entropy(q.reshape(-1, card), tt.reshape(-1),
                           reduction="none").view(B, n, T)
    ce_teacher_cb = _per_codebook_mean(ce_t, mask, w)

    per_cb = kl_cb + lam_samp * samp_cb + ce_teacher_weight * ce_teacher_cb
    total = (per_cb * w).sum() / w.sum().clamp(min=1e-8)

    stats: dict[str, float] = {}

    with torch.no_grad():
        agree = ((q.argmax(-1) == teacher_tokens) & mask).float()
        agree_cb = agree.sum(dim=(0, 2)) / mask.float().sum(dim=(0, 2)).clamp(min=1)
    n_agent = min(AUDIO_TOKENS_PER_STREAM, n)

    stats.update({
        "kl": float(kl_cb.mean()),
        "samp": float(samp_cb.mean()),
        "ce_teacher": float(ce_teacher_cb.mean()),
        "loss_agent": float(per_cb[:n_agent].mean()),
        "loss_user": float(per_cb[n_agent:].mean()) if n > n_agent else 0.0,
        "agree_agent": float(agree_cb[:n_agent].mean()),
        "agree_user": float(agree_cb[n_agent:].mean()) if n > n_agent else 0.0,
        # proof / regression guard that all codebooks are accounted for:
        "cb_covered": int((cover > 0).sum()),
        "cb_weight_mass": float((w * cover).sum() / w.sum()),
    })
    return total, stats


# ---------------------------------------------------------------------------
# Targets (cache hidden states + greedy tokens; logits recomputed on the fly)
# ---------------------------------------------------------------------------

def undelay_depformer_logits(logits_seq: torch.Tensor, teacher_lm,
                             n_audio: int) -> torch.Tensor:
    """Map sequence-aligned student logits to frame-aligned logits.

    The student consumes the delayed DepFormer inputs and emits logits at the
    same sequence positions as ``forward_depformer_training``; the loss needs
    them frame-aligned, exactly like the teacher's ``audio_logits``.
    """
    out, _ = _undelay_sequence(
        teacher_lm.delays[teacher_lm.audio_offset:
                          teacher_lm.audio_offset + n_audio],
        logits_seq, fill_value=float("nan"))
    return out


@torch.no_grad()
def build_targets_from_teacher(teacher_lm, codes: torch.Tensor,
                               n_audio: int | None = None,
                               include_teacher_logits: bool = True
                               ) -> dict[str, torch.Tensor]:
    """Teacher-forced targets for ALL ``dep_q`` audio codebooks.

    ``codes`` is the full duplex layout [B, K=1+n_q, T].  Codebooks 8..15 (the
    user stream) are kept, not sliced away.  ``depformer_inputs`` / ``text_inputs``
    are the *delayed* inputs the DepFormer consumes (sequence-aligned); the
    grouped student conditions on those, then un-delays its logits.
    """
    from .distill.teacher import forward_teacher_distill

    with torch.no_grad():
        t = forward_teacher_distill(teacher_lm, codes)

    n = int(n_audio) if n_audio else int(teacher_lm.dep_q)
    assert 1 <= n <= teacher_lm.dep_q, (n, teacher_lm.dep_q)

    logits = t["audio_logits"][:, :n]
    mask = t["audio_mask"][:, :n]
    teacher_tokens = torch.nan_to_num(logits, nan=-1e4).argmax(-1)

    out = {
        "hidden": t["hidden"],                                   # [B, T, Dt]
        "codes": codes,                                          # [B, K, T]
        "text": t["text_inputs"],                                # [B, T] delayed
        "depformer_inputs": t["depformer_inputs"][:, :n],        # [B, n, T] delayed
        "teacher_tokens": teacher_tokens,                        # [B, n, T] targets
        "mask": mask,                                            # [B, n, T] targets
        "n_audio": torch.tensor(n),
    }
    if include_teacher_logits:
        out["teacher_logits"] = logits                          # [B, n, T, card]
    return out


@torch.no_grad()
def recompute_teacher_logits(teacher_lm, codes: torch.Tensor,
                             transformer_out: torch.Tensor,
                             n_audio: int | None = None
                             ) -> tuple[torch.Tensor, torch.Tensor]:
    """Frozen-teacher DepFormer logits from cached backbone states (no teacher
    backbone re-run, no logits stored on disk — plan §3.2)."""
    B, K, T = codes.shape
    n = int(n_audio) if n_audio else int(teacher_lm.dep_q)
    transformer_out = transformer_out.to(next(teacher_lm.parameters()).dtype)
    initial = teacher_lm._get_initial_token().expand(B, -1, -1)
    delayed = _delay_sequence(teacher_lm.delays, codes, initial)
    delayed = torch.cat([initial, delayed], dim=2)
    audio_logits = teacher_lm.forward_depformer_training(delayed[:, :, 1:],
                                                         transformer_out)
    audio_logits, audio_mask = _undelay_sequence(
        teacher_lm.delays[teacher_lm.audio_offset:
                          teacher_lm.audio_offset + teacher_lm.dep_q],
        audio_logits, fill_value=float("nan"))
    audio_mask &= (codes[:, teacher_lm.audio_offset:
                         teacher_lm.audio_offset + teacher_lm.dep_q]
                   != teacher_lm.zero_token_id)
    return audio_logits[:, :n], audio_mask[:, :n]


# ---------------------------------------------------------------------------
# Collection + fitting
# ---------------------------------------------------------------------------

def _pad_cat(tensors: list[torch.Tensor], dim: int, value) -> torch.Tensor:
    """Concatenate along dim=0 after padding ``dim`` to the max length.

    Distill samples have variable frame counts (system-prompt length varies), so
    batches cannot be concatenated directly. Padding uses the tokenizer's padding
    convention (``zero_token_id`` for codes) so the validity mask ignores it.
    """
    m = max(t.shape[dim] for t in tensors)
    parts = []
    for t in tensors:
        shape = list(t.shape)
        shape[dim] = m
        buf = torch.full(shape, value, dtype=t.dtype)
        idx = [slice(None)] * t.dim()
        idx[dim] = slice(0, t.shape[dim])
        buf[tuple(idx)] = t
        parts.append(buf)
    return torch.cat(parts, dim=0)


@torch.no_grad()
def collect_targets(teacher_lm, loader, device, max_batches: int,
                    max_frames: int | None = None,
                    n_audio: int | None = None) -> dict[str, torch.Tensor]:
    """Cache what the grouped student needs: frame states, greedy agent+user
    teacher targets, delayed DepFormer inputs.  Teacher logits are NOT cached."""
    n = int(n_audio) if n_audio else int(teacher_lm.dep_q)
    acc: dict[str, list[torch.Tensor]] = {
        k: [] for k in ("hidden", "codes", "text", "depformer_inputs",
                        "teacher_tokens", "mask")}
    n_frames = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        codes = batch.codes.to(device)
        t = build_targets_from_teacher(teacher_lm, codes, n_audio=n,
                                       include_teacher_logits=False)
        acc["hidden"].append(t["hidden"].to(next(teacher_lm.parameters()).dtype).cpu())
        acc["codes"].append(codes.cpu())
        acc["text"].append(t["text"].cpu())
        acc["depformer_inputs"].append(t["depformer_inputs"].cpu())
        acc["teacher_tokens"].append(t["teacher_tokens"].cpu())
        acc["mask"].append(t["mask"].cpu())
        n_frames += codes.shape[0] * codes.shape[-1]
        if max_frames is not None and n_frames >= max_frames:
            break
        logger.info("collect: batch %d (%d frames total)", i + 1, n_frames)

    pad = {
        "hidden": (1, 0.0),
        "codes": (2, teacher_lm.zero_token_id),
        "text": (1, teacher_lm.zero_token_id),
        "depformer_inputs": (2, teacher_lm.zero_token_id),
        "teacher_tokens": (2, teacher_lm.zero_token_id),
        "mask": (2, False),
    }
    out = {k: _pad_cat(v, pad[k][0], pad[k][1]) for k, v in acc.items()}
    out["n_audio"] = torch.tensor(n)
    logger.info("collect: %d samples x %d frames x %d codebooks",
                out["hidden"].shape[0], out["hidden"].shape[1], n)
    return out


@torch.no_grad()
def _forced_prompt_len(codes: torch.Tensor, zero_token_id: int) -> int:
    """Length of the leading forced `<system>` persona block, if present.

    The interleaver prepends the system prompt as forced frames whose agent
    codebooks are exactly ``SILENCE_TOKENS`` and user codebooks exactly
    ``SINE_TOKENS``, with real (non-padding) text tokens.  Detecting that
    leading run lets the rollout replay the persona before generating.
    Returns 0 when the chunk has no prompt (e.g. start_sec > 0).
    """
    if codes.shape[1] < 1 + 2 * AUDIO_TOKENS_PER_STREAM:
        return 0
    aps = AUDIO_TOKENS_PER_STREAM
    agent = codes[0, 1:1 + aps]                          # [8, T]
    user = codes[0, 1 + aps:1 + 2 * aps]                 # [8, T]
    silence = torch.as_tensor(SILENCE_TOKENS, dtype=agent.dtype,
                              device=agent.device).unsqueeze(1)
    sine = torch.as_tensor(SINE_TOKENS, dtype=user.dtype,
                           device=user.device).unsqueeze(1)
    match = ((agent == silence).all(0) & (user == sine).all(0)
             & (codes[0, 0] != zero_token_id))
    n, T = 0, codes.shape[-1]
    while n < T and bool(match[n]):
        n += 1
    return n


def collect_rollouts(teacher_lm, loader, device, max_batches: int,
                     max_frames: int | None = None, n_audio: int | None = None,
                     temp: float = 0.8, top_k: int = 250,
                     max_samples: int | None = None) -> dict[str, torch.Tensor]:
    """Teacher rollouts as distillation targets (plan §3.1 primary source).

    Input = the **system prompt (persona) + the sample's USER stream**.  The
    leading forced ``<system>`` block carried in the codes is replayed exactly
    like the serving prompt path (text + agent silence + user sine forced), then
    the USER codebooks are provided and the teacher *generates* the agent stream
    + text autoregressively.  ``h`` and the DepFormer logits are computed
    teacher-forced over the teacher's OWN generation, so **no ground-truth agent
    code ever enters the targets**.

    One sample at a time (streaming via ``LMGen``); the output is cached in the
    same layout as :func:`collect_targets`, so ``fit`` is unchanged.
    """
    from .models.lm import LMGen

    n = int(n_audio) if n_audio else int(teacher_lm.dep_q)
    gen = LMGen(teacher_lm, device=device, use_sampling=True, temp=temp,
                top_k=top_k, top_k_text=25, return_logits=False,
                fast_provided_tail=False)
    gen.eval()

    acc: dict[str, list[torch.Tensor]] = {
        k: [] for k in ("hidden", "codes", "text", "depformer_inputs",
                        "teacher_tokens", "mask")}
    n_frames, n_samples, n_prompt_frames = 0, 0, 0
    stop = False
    for i, batch in enumerate(loader):
        if i >= max_batches or stop:
            break
        codes_all = batch.codes
        for b in range(codes_all.shape[0]):
            if max_samples is not None and n_samples >= max_samples:
                stop = True
                break
            codes = codes_all[b:b + 1].to(device)
            T = codes.shape[-1]
            aps = AUDIO_TOKENS_PER_STREAM
            lo = aps + 1
            agent = codes[:, 1:1 + aps]                  # forced during the prompt
            user = codes[:, lo: lo + aps]                # the query stream
            text = codes[:, 0]                           # forced during the prompt
            prompt_len = _forced_prompt_len(codes, teacher_lm.zero_token_id)
            n_prompt_frames += prompt_len
            with gen.streaming(1):
                gen.reset_streaming()
                frames = []
                for f in range(T):
                    if f < prompt_len:
                        # replay the persona block exactly as serving does
                        out = gen.step(input_tokens=user[:, :, f:f + 1],
                                       moshi_tokens=agent[:, :, f:f + 1],
                                       text_token=text[:, f])
                    else:
                        # USER stream is the query; agent + text are generated
                        out = gen.step(input_tokens=user[:, :, f:f + 1])
                    if out is None:
                        continue
                    if isinstance(out, tuple):
                        out = out[0]
                    if out is None:
                        continue
                    frames.append(out[:, :, 0])            # [1, 17]
            if not frames:
                logger.warning("rollout: sample produced no frames; skipping")
                continue
            codes_gen = torch.stack(frames, dim=2)          # [1, 17, F]
            t = build_targets_from_teacher(teacher_lm, codes_gen, n_audio=n,
                                           include_teacher_logits=False)
            acc["hidden"].append(t["hidden"].to(next(teacher_lm.parameters()).dtype).cpu())
            acc["codes"].append(codes_gen.cpu())
            acc["text"].append(t["text"].cpu())
            acc["depformer_inputs"].append(t["depformer_inputs"].cpu())
            acc["teacher_tokens"].append(t["teacher_tokens"].cpu())
            acc["mask"].append(t["mask"].cpu())
            n_frames += codes_gen.shape[0] * codes_gen.shape[-1]
            n_samples += 1
            logger.info("rollout: sample %d (prompt %d frames, %d frames total)",
                        n_samples, prompt_len, n_frames)
            if max_frames is not None and n_frames >= max_frames:
                stop = True
                break

    pad = {
        "hidden": (1, 0.0),
        "codes": (2, teacher_lm.zero_token_id),
        "text": (1, teacher_lm.zero_token_id),
        "depformer_inputs": (2, teacher_lm.zero_token_id),
        "teacher_tokens": (2, teacher_lm.zero_token_id),
        "mask": (2, False),
    }
    out = {k: _pad_cat(v, pad[k][0], pad[k][1]) for k, v in acc.items()}
    out["n_audio"] = torch.tensor(n)
    logger.info("rollout: %d samples x %d frames x %d codebooks "
                "(%d forced prompt frames replayed)",
                out["hidden"].shape[0], out["hidden"].shape[1], n,
                n_prompt_frames)
    return out


def _sample_codes(logits: torch.Tensor, use_sampling: bool, temp: float,
                  top_k: int) -> torch.Tensor:
    """Sample every codebook independently from [B, n, T, card]; returns [B,n,T]."""
    B, n, T, card = logits.shape
    flat = logits.permute(0, 2, 1, 3).reshape(-1, card)
    s = sample_token(flat.float(), use_sampling, temp, top_k)
    return s.view(B, T, n).permute(0, 2, 1)


def fit(cfg: GroupedDepFormerConfig, targets: dict[str, torch.Tensor],
        teacher_lm, device, epochs: int = 4, batch_size: int = 2,
        lr: float = 3e-4, weight_decay: float = 0.01, warmup_steps: int = 200,
        log_every: int = 20,
        out_dir: str | Path = "checkpoints/grouped_depformer") -> GroupedDepFormerStudent:
    """Fit the grouped student on cached hidden states / codes.

    Teacher logits are recomputed from the cached ``hidden`` with the frozen
    teacher DepFormer, so no logits are stored on disk. The objective is
    teacher-only (no ground-truth codes).
    """
    device = torch.device(device)
    teacher_lm.eval()
    for p in teacher_lm.parameters():
        p.requires_grad_(False)
    n = int(targets["n_audio"].item()) if "n_audio" in targets \
        else int(teacher_lm.dep_q)

    student = GroupedDepFormerStudent(
        teacher_lm, schedule=cfg.schedule, mask_init_scale=cfg.mask_init_scale,
        serve_max_slot=cfg.serve_max_slot, device=device)
    student.train()
    weights = default_codebook_weights(n, cfg.user_codebook_weight)

    opt = torch.optim.AdamW(student.parameters(), lr=lr,
                            weight_decay=weight_decay, betas=(0.9, 0.95))

    hidden = targets["hidden"]
    codes_all = targets["codes"]
    text = targets["text"]
    dep_inputs = targets["depformer_inputs"]
    ttok = targets["teacher_tokens"]
    mask = targets["mask"]
    N = hidden.shape[0]

    step, t0 = 0, time.time()
    running: dict[str, float] = {}
    run_n = 0
    for epoch in range(epochs):
        perm = torch.randperm(N)
        for s in range(0, N, batch_size):
            sel = perm[s:s + batch_size]
            h = hidden[sel].to(device, dtype=next(student.parameters()).dtype)
            codes_b = codes_all[sel].to(device)
            tx = text[sel].to(device)
            pin = dep_inputs[sel].to(device)
            tb = ttok[sel].to(device)
            mb = mask[sel].to(device)

            p_logits, p_mask = recompute_teacher_logits(teacher_lm, codes_b, h,
                                                        n_audio=n)
            # The validity mask must match; trust the cached one but keep the
            # recomputed one as a consistency check.
            assert p_logits.shape[1] == n and p_mask.shape[1] == n

            # Student consumes the DELAYED DepFormer inputs and emits
            # sequence-aligned logits, which are then un-delayed to match the
            # teacher's frame-aligned targets.
            logits_seq = student.forward_groups(h, tx, prefix_codes=pin)
            logits = undelay_depformer_logits(logits_seq, teacher_lm, n)
            loss, stats = parallel_depformer_loss(
                logits, p_logits, tb, mb, codebook_weights=weights,
                lam_samp=cfg.lam_samp, samp_top_k=cfg.samp_top_k,
                samp_temp=cfg.samp_temp, ce_teacher_weight=cfg.ce_teacher_weight,
                kd_temperature=cfg.kd_temperature)

            if student.refine_passes:
                draft = _sample_codes(logits_seq.detach(), cfg.draft_use_sampling,
                                      cfg.draft_temp, cfg.draft_top_k)
                logits2_seq = student.refine(h, tx, draft)
                logits2 = undelay_depformer_logits(logits2_seq, teacher_lm, n)
                loss2, stats2 = parallel_depformer_loss(
                    logits2, p_logits, tb, mb, codebook_weights=weights,
                    lam_samp=cfg.lam_samp, samp_top_k=cfg.samp_top_k,
                    samp_temp=cfg.samp_temp, ce_teacher_weight=cfg.ce_teacher_weight,
                    kd_temperature=cfg.kd_temperature)
                loss = 0.5 * (loss + loss2)
                stats = {k: 0.5 * (stats.get(k, 0.0) + stats2.get(k, 0.0))
                         for k in set(stats) | set(stats2)}

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            step += 1
            if step <= warmup_steps:
                for g in opt.param_groups:
                    g["lr"] = lr * step / max(1, warmup_steps)

            stats_run = {**stats, "loss": loss.item()}
            for k, v in stats_run.items():
                running[k] = running.get(k, 0.0) + v
            run_n += 1
            if step % log_every == 0:
                avg = {k: v / max(run_n, 1) for k, v in running.items()}
                logger.info("epoch %d step %d %s (%.0fs)", epoch, step,
                            " ".join(f"{k}={v:.4f}" for k, v in avg.items()),
                            time.time() - t0)
                running, run_n = {}, 0

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": student.state_dict(), "cfg": asdict(cfg),
                "n_audio": n},
               out_dir / "grouped_depformer.pt")
    logger.info("saved %s", out_dir / "grouped_depformer.pt")
    return student


# ---------------------------------------------------------------------------
# Serving adapter
# ---------------------------------------------------------------------------

class ParallelDepFormerGen(LMGen):
    """``LMGen`` whose DepFormer step runs the grouped student.

    The student predicts ALL ``dep_q`` codebooks, but the serving loop provides
    the user stream every frame (``fast_provided_tail=True``), so only the agent
    stream is decoded (``cfg.serve_max_slot``) and the provided tail is passed
    through bit-identically.  The MASK context of the skipped slots is identical
    to training, so the agent logits do not change.
    """

    def __init__(self, depformer_student: GroupedDepFormerStudent, *args, **kwargs):
        kwargs.setdefault("fast_provided_tail", True)
        super().__init__(*args, **kwargs)
        assert self.fast_provided_tail, (
            "ParallelDepFormerGen requires fast_provided_tail=True (the "
            "serving loop provides the user stream every frame)")
        assert not self.return_logits and not self.report_loss, (
            "the grouped head does not reproduce the sequential DepFormer "
            "logits; disable return_logits/report_loss")
        self.depformer_student = depformer_student
        self._n = depformer_student.n
        assert self.lm_model.dep_q == self._n, (self.lm_model.dep_q, self._n)

    @torch.no_grad()
    def depformer_step(self, text_token, transformer_out, audio_tokens,
                       audio_provided) -> torch.Tensor:
        B = text_token.shape[0]
        max_slot = self.depformer_student.serve_max_slot
        sampled = self.depformer_student.decode(
            transformer_out, text_token[:, None], temp=self.temp,
            top_k=self.top_k, use_sampling=self.use_sampling,
            max_slot=max_slot)                                     # [B, n, T=1]
        sampled = sampled[:, :, 0]                                  # [B, n]
        agent = torch.where(audio_provided[:, :max_slot],
                            audio_tokens[:, :max_slot], sampled[:, :max_slot])
        if max_slot < self._n:
            tokens = torch.cat([agent, audio_tokens[:, max_slot:]], dim=1)
        else:
            tokens = agent
        assert tokens.shape == (B, self.lm_model.dep_q), tokens.shape
        return tokens


# ---------------------------------------------------------------------------
# Self test + CLI
# ---------------------------------------------------------------------------

def _self_test() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from .models.lm import LMModel

    delays = [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
    torch.manual_seed(0)
    teacher = LMModel(
        delays=delays, n_q=16, dep_q=16, card=64, text_card=32,
        dim=64, num_heads=4, num_layers=2, hidden_scale=2,
        existing_text_padding_id=3, context=64, causal=True,
        depformer_dim=32, depformer_dim_feedforward=64,
        depformer_num_heads=4, depformer_num_layers=2,
        depformer_multi_linear=True, depformer_weights_per_step=True,
        depformer_weights_per_step_schedule=list(range(16)),
        depformer_causal=True, depformer_layer_scale=None, depformer_context=8,
        depformer_max_period=10000, depformer_gating="silu", depformer_pos_emb="none")
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    g = torch.Generator().manual_seed(7)
    codes = torch.randint(0, 64, (2, 17, 16), generator=g)
    codes[:, 0] = torch.randint(1, 31, (2, 16), generator=g)
    codes[:, :, -1:] = -1

    t = build_targets_from_teacher(teacher, codes, include_teacher_logits=True)
    assert t["teacher_tokens"].shape == (2, 16, 16), t["teacher_tokens"].shape
    assert t["depformer_inputs"].shape == (2, 16, 16)
    print("targets: hidden", tuple(t["hidden"].shape),
          "teacher_tokens", tuple(t["teacher_tokens"].shape))

    # S0 sanity (plan §6): a fully-autoregressive student (groups = one slot
    # each) must reproduce the teacher DepFormer logits EXACTLY, which validates
    # both the teacher-weight copy and the delayed-input alignment.
    full = GroupedDepFormerStudent(teacher, schedule="B")
    full.groups = [[k] for k in range(16)]
    full_seq = full.forward_groups(t["hidden"], t["text"],
                                   prefix_codes=t["depformer_inputs"])
    full_logits = undelay_depformer_logits(full_seq, teacher, 16)
    diff = torch.nan_to_num((full_logits - t["teacher_logits"]).abs(), nan=0.0)
    max_diff = float((diff * t["mask"].unsqueeze(-1)).max())
    print(f"S0 full-AR max|student - teacher| = {max_diff:.3e}")
    assert max_diff < 1e-3, max_diff

    student = GroupedDepFormerStudent(teacher, schedule="B")
    logits_seq = student.forward_groups(t["hidden"], t["text"],
                                        prefix_codes=t["depformer_inputs"])
    logits = undelay_depformer_logits(logits_seq, teacher, 16)
    assert logits.shape == (2, 16, 16, 64), logits.shape
    print("forward_groups:", tuple(logits.shape))

    # --- delay audit (plan §1.5, §3.2) ------------------------------------
    # (a) fit recomputes teacher targets from cached h; it must equal the
    #     collect-time teacher logits (both mirror LMModel.forward_train).
    p_logits, _ = recompute_teacher_logits(teacher, codes, t["hidden"], 16)
    d = torch.nan_to_num((p_logits - t["teacher_logits"]).abs(), nan=0.0)
    md = float((d * t["mask"].unsqueeze(-1)).max())
    print(f"delay audit: recompute_teacher_logits max|diff| = {md:.3e}")
    assert md < 1e-3, md

    # (b) design-C refine fed the teacher-forced delayed draft must also
    #     reproduce the teacher: proves the draft<->slot delay indexing is
    #     consistent (draft of codebook k-1 at sequence pos t is the token slot
    #     k consumes at pos t).
    ref_seq = student.refine(t["hidden"], t["text"], t["depformer_inputs"])
    ref_logits = undelay_depformer_logits(ref_seq, teacher, 16)
    d = torch.nan_to_num((ref_logits - t["teacher_logits"]).abs(), nan=0.0)
    md = float((d * t["mask"].unsqueeze(-1)).max())
    print(f"delay audit: refine(teacher-forced delayed draft) max|diff| = {md:.3e}")
    assert md < 1e-3, md

    weights = default_codebook_weights(16)
    # teacher-only objective: targets are teacher_logits / teacher_tokens only.
    loss, stats = parallel_depformer_loss(
        logits, t["teacher_logits"], t["teacher_tokens"],
        t["mask"], codebook_weights=weights)
    assert torch.isfinite(loss)
    assert stats["cb_covered"] == 16, stats
    print("loss:", {k: round(v, 4) for k, v in stats.items()})

    # all-codebooks accounting: every head / input proj must receive gradient,
    # including the user stream (8..15).
    loss.backward()
    for k in range(16):
        assert student.linears[k].weight.grad is not None, f"head {k} no grad"
        assert student.depformer_in[k].weight.grad is not None, f"in {k} no grad"
    assert student.mask.grad is not None and torch.isfinite(student.mask.grad).all()
    print("all 16 codebook heads + input projections received gradients")

    # refine (design C) smoke
    student_c = GroupedDepFormerStudent(teacher, schedule="C")
    logits_c_seq = student_c.forward_groups(t["hidden"], t["text"],
                                            prefix_codes=t["depformer_inputs"])
    draft = _sample_codes(logits_c_seq.detach(), True, 0.8, 64)
    logits_c2_seq = student_c.refine(t["hidden"], t["text"], draft)
    assert logits_c2_seq.shape == logits_c_seq.shape
    logits_c2 = undelay_depformer_logits(logits_c2_seq, teacher, 16)
    loss_c, _ = parallel_depformer_loss(
        logits_c2, t["teacher_logits"], t["teacher_tokens"],
        t["mask"], codebook_weights=weights)
    assert torch.isfinite(loss_c)
    print("schedule C refine ok")

    # serving adapter: decode the agent stream, pass the user stream through
    gen = ParallelDepFormerGen(student, teacher, device="cpu", sample_rate=24000,
                               frame_rate=12.5, use_sampling=False, top_k=1,
                               top_k_text=1)
    gen.streaming_forever(1)
    tt = torch.full((2,), 5, dtype=torch.long)
    st = t["hidden"][:, 0:1]
    audio_tok = torch.full((2, 16), 9, dtype=torch.long)
    prov = torch.ones((2, 16), dtype=torch.bool)
    out = gen.depformer_step(tt, st, audio_tok, prov)
    assert out.shape == (2, 16), out.shape
    assert (out[:, 8:] == 9).all(), "trailing (provided) codebooks must pass through"
    print("serving adapter ok:", tuple(out.shape))
    print("GROUPED DEPFORMER SELF-TEST PASSED")


def _cmd_collect(args) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    from .dataset import build_data_loader
    from .distill.data import DistillInterleavedTokenizer
    from .interleaver import Interleaver
    from .models import loaders
    import sentencepiece
    from huggingface_hub import hf_hub_download

    device = torch.device(args.device)
    tokenizer = sentencepiece.SentencePieceProcessor(
        args.tokenizer or hf_hub_download(args.hf_repo, loaders.TEXT_TOKENIZER_NAME))
    mimi = loaders.get_mimi(
        args.mimi_weight or hf_hub_download(args.hf_repo, loaders.MIMI_NAME), device)
    mimi.eval()
    teacher = loaders.get_moshi_lm(
        args.moshi_weight or hf_hub_download(args.hf_repo, loaders.MOSHI_NAME),
        device=device)
    teacher.eval()

    interleaver = Interleaver(
        tokenizer=tokenizer, audio_frame_rate=mimi.frame_rate,
        text_padding=teacher.text_padding_token_id,
        end_of_text_padding=teacher.end_of_text_padding_id,
        zero_padding=teacher.zero_token_id,
        keep_main_only=True, use_bos_eos=True,
        main_speaker_label="ASSISTANT", device=str(device))
    dtok = DistillInterleavedTokenizer(
        mimi=mimi, interleaver=interleaver, duration_sec=args.duration_sec,
        target_n_audio_codebooks=teacher.num_audio_codebooks,
        use_sidecar=args.use_sidecar,
        require_sidecar=not args.allow_missing_sidecar,
        system_prompt_default=args.default_system_prompt)
    loader = build_data_loader(instruct_tokenizer=dtok, args=args,
                               batch_size=args.batch_size, seed=0, rank=0,
                               world_size=1, is_eval=False)
    if args.rollout:
        logger.info("collect: TEACHER ROLLOUT mode (user stream = query, "
                    "teacher generates agent; no ground-truth agent codes)")
        targets = collect_rollouts(teacher, loader, device, args.max_batches,
                                   args.max_frames, n_audio=args.n_audio,
                                   temp=args.rollout_temp, top_k=args.rollout_top_k,
                                   max_samples=args.rollout_samples)
    else:
        targets = collect_targets(teacher, loader, device, args.max_batches,
                                  args.max_frames, n_audio=args.n_audio)
    torch.save(targets, args.out)
    logger.info("wrote %s", args.out)


def _cmd_fit(args) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    from .models import loaders
    from huggingface_hub import hf_hub_download

    device = torch.device(args.device)
    targets = torch.load(args.targets, map_location="cpu", weights_only=False)
    teacher = loaders.get_moshi_lm(
        args.moshi_weight or hf_hub_download(args.hf_repo, loaders.MOSHI_NAME),
        device=device)
    teacher.eval()
    cfg = GroupedDepFormerConfig(
        schedule=args.schedule,
        user_codebook_weight=args.user_codebook_weight,
        serve_max_slot=args.serve_max_slot,
        ce_teacher_weight=args.ce_teacher_weight,
    )
    fit(cfg, targets, teacher, device, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, log_every=args.log_every,
        out_dir=args.out_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    sub = ap.add_subparsers(dest="cmd")

    c = sub.add_parser("collect")
    c.add_argument("--train-data", required=True)
    c.add_argument("--eval-data", default="")
    c.add_argument("--duration-sec", type=float, default=40.0)
    c.add_argument("--batch-size", type=int, default=8)
    c.add_argument("--shuffle", dest="shuffle", action="store_true", default=True,
                   help="shuffle the data sources (build_data_loader arg)")
    c.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    c.add_argument("--use-sidecar", dest="use_sidecar", action="store_true",
                   default=True)
    c.add_argument("--no-sidecar", dest="use_sidecar", action="store_false")
    c.add_argument("--hf-repo", default="nvidia/personaplex-7b-v1")
    c.add_argument("--moshi-weight", default=None)
    c.add_argument("--mimi-weight", default=None)
    c.add_argument("--tokenizer", default=None)
    c.add_argument("--device", default="cuda")
    c.add_argument("--max-batches", type=int, default=500)
    c.add_argument("--max-frames", type=int, default=400_000)
    c.add_argument("--n-audio", type=int, default=None,
                   help="codebooks to predict; default = teacher dep_q (all 16)")
    c.add_argument("--allow-missing-sidecar", action="store_true")
    c.add_argument("--default-system-prompt", default=None)
    c.add_argument("--rollout", action="store_true",
                   help="collect from TEACHER ROLLOUTS: use the USER stream as "
                        "the only query, let the teacher generate the agent "
                        "stream, and distill from the teacher's outputs only "
                        "(no ground-truth codes).")
    c.add_argument("--rollout-temp", type=float, default=0.8)
    c.add_argument("--rollout-top-k", type=int, default=250)
    c.add_argument("--rollout-samples", type=int, default=None,
                   help="cap the number of rollout samples (useful for probes)")
    c.add_argument("--out", default="parallel_targets.pt")
    c.set_defaults(func=_cmd_collect)

    f = sub.add_parser("fit")
    f.add_argument("--targets", required=True)
    f.add_argument("--out-dir", default="checkpoints/grouped_depformer")
    f.add_argument("--device", default="cuda")
    f.add_argument("--epochs", type=int, default=4)
    f.add_argument("--batch-size", type=int, default=2)
    f.add_argument("--lr", type=float, default=3e-4)
    f.add_argument("--log-every", type=int, default=20)
    f.add_argument("--ce-teacher-weight", type=float, default=1.0,
                   help="weight of the hard CE to the teacher's greedy tokens")
    f.add_argument("--schedule", default="B", choices=["B", "C", "A", "AR"])
    f.add_argument("--user-codebook-weight", type=float, default=1.0)
    f.add_argument("--serve-max-slot", type=int, default=AUDIO_TOKENS_PER_STREAM)
    f.add_argument("--hf-repo", default="nvidia/personaplex-7b-v1")
    f.add_argument("--moshi-weight", default=None)
    f.set_defaults(func=_cmd_fit)

    args = ap.parse_args()
    if args.self_test or args.cmd is None:
        _self_test()
        return
    args.func(args)


if __name__ == "__main__":
    main()
