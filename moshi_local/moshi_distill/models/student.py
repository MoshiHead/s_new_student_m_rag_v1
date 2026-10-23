"""
models/student.py
=================
Student-side modules for the PersonaPlex 6.25 Hz temporal distillation
(distill_plan.md §5–§7, §12, §20).

Pieces (all trainable unless marked otherwise):

- `FramePairCompressor`   12.5 -> 6.25 Hz: z_t = MLP([m_2t ; m_(2t+1)]).
                           Causal by construction: z_t only ever sees frames
                           <= 2t+1 (full-sequence reshape, never look-ahead).
- `AudioStateUpsampler`   6.25 -> 12.5 Hz: [a_2t, a_(2t+1)] = MLP(s_t), with
                           output_dim defaulting to the TEACHER dim (4096) so
                           the frozen teacher DepFormer stack can consume the
                           states directly (plan §14 Option A).
- `HiddenProjection`      Linear teacher_dim -> student_dim for the pooled
                           teacher target of L_hidden (plan §7).
- `StudentLM`             Embedding tables (same layout as the teacher: 1 text
                           codebook + n_q audio codebooks) + small streaming
                           Temporal Transformer + the three modules above.
                           Audio heads (text_linear / depformer stack) live
                           ON THE FROZEN TEACHER and are reused through
                           `forward_train(codes, teacher_lm)`.

Rate note (plan §13): everything here is training-time teacher-forced. The
user-visible audio rate stays 12.5 Hz; only the STUDENT'S Temporal Transformer
runs at 6.25 Hz (T/2 steps), and `AudioStateUpsampler` reconstructs a 12.5 Hz
state per frame for DepFormer.

Streaming caveat: the compressor reshapes [m_2t, m_2t+1] pairs, which is exact
for offline/full-sequence teacher forcing. Streaming inference will need a
1-frame buffer (store m at odd t, emit z when the pair completes) — implement
when wiring the student into LMGen / server, not before training.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.nn as nn

from ..modules.transformer import StreamingTransformer, create_norm_fn
from .lm import (
    LMOutput,
    ScaledEmbedding,
    _delay_sequence,
    _undelay_sequence,
)

logger = logging.getLogger(__name__)


@dataclass
class StudentConfig:
    """Defaults target a ~2.5B student inside an A100-80GB budget with the
    ~14GB bf16 teacher resident. Reduce `dim`/`num_layers` if memory/tflops
    demand it; reduce codebooks only together with a trainable DepFormer
    (plan §14), NOT while reusing the frozen teacher DepFormer."""

    # temporal transformer — defaults land at ~2.3B params
    dim: int = 2560
    num_layers: int = 24
    num_heads: int = 20      # 2560 / 20 = 128 head dim
    hidden_scale: float = 4.0
    context: int = 1500     # student steps; 1500 @ 6.25 Hz == 3000 frames @ 12.5 Hz
    norm: str = "rms_norm_f32"
    gating: str = "silu"
    positional_embedding: str = "rope"   # teacher main transformer uses rope
    max_period: float = 10_000.0
    layer_scale: float | None = None

    # audio/table layout — must match the teacher's
    card: int = 2048
    text_card: int = 32_000
    n_q: int = 16
    dep_q: int = 16

    # adapter modules
    compressor_hidden: int | None = None   # None -> 1x dim bottleneck MLP
    upsampler_hidden: int | None = None    # None -> 1x dim bottleneck MLP

    # --- 6.25 Hz serving-schedule fix -------------------------------------
    # The server consumes, per 12.5 Hz frame p, the upsampled state a_p for odd
    # p and a_(p-1) for even p (the pair completes one frame late). Without a
    # fix the even slots are conditioned on the previous frame's state, which
    # measures only ~0.65 cosine-similar to what the frozen heads need (teacher
    # adjacent-frame hidden cosine) -> repeated/doubled tokens.
    #   "off"     : train as-is (old behaviour, kept for compatibility)
    #   "augment" : train with the SAME schedule the server uses AND inject the
    #               current frame's embedding into the head input
    #               (zero-initialised residual, so training starts from the
    #               un-augmented behaviour).
    streaming_fix: str = "off"
    phase_hidden: int | None = None        # None -> dim
    # With the layer-init recipe the student state already lives in the
    # teacher's dimension, so the learned hidden projection must be the
    # identity (a trainable projection could collapse the L_hidden target).
    identity_hidden_proj: bool = False


class FramePairCompressor(nn.Module):
    """z_t = MLP([m_2t ; m_(2t+1)]) — plan §5.

    Concatenation of two frames makes the pair boundary explicit; the causal
    requirement (z_t must not see m_(2t+2)) is satisfied because the reshape
    `[B, T2, 2*size]` never mixes across pairs.
    """

    def __init__(self, student_dim: int, hidden: int | None = None,
                 device=None, dtype=None):
        super().__init__()
        hidden = hidden or student_dim
        self.in_norm = nn.LayerNorm(2 * student_dim, device=device, dtype=dtype)
        self.proj = nn.Linear(2 * student_dim, hidden, device=device, dtype=dtype)
        self.act = nn.GELU()
        self.out = nn.Linear(hidden, student_dim, device=device, dtype=dtype)
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.normal_(self.out.weight, std=0.02)

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        B, T, D = m.shape
        assert T % 2 == 0, (
            f"FramePairCompressor needs an even number of frames, got {T} "
            f"(pad the sample to duration_sec*12.5 frames, even)."
        )
        pairs = m.view(B, T // 2, 2 * D)          # never looks ahead
        z = self.out(self.act(self.proj(self.in_norm(pairs))))
        return z


class AudioStateUpsampler(nn.Module):
    """s_t -> two 12.5 Hz states [a_(2t), a_(2t+1)] — plan §12/§13.

    `out_dim` = teacher dim so `a` can be handed straight to the frozen
    teacher DepFormer / text head without another projection.
    """

    def __init__(self, student_dim: int, out_dim: int,
                 hidden: int | None = None, device=None, dtype=None):
        super().__init__()
        hidden = hidden or student_dim
        self.in_norm = nn.LayerNorm(student_dim, device=device, dtype=dtype)
        self.proj = nn.Linear(student_dim, hidden, device=device, dtype=dtype)
        self.act = nn.GELU()
        self.out = nn.Linear(hidden, 2 * out_dim, device=device, dtype=dtype)
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.normal_(self.out.weight, std=0.02)
        self.out_dim = out_dim

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        B, T2, _ = s.shape
        up = self.out(self.act(self.proj(self.in_norm(s))))   # [B, T2, 2*out]
        # [:, t, 0:out] = a_(2t), [:, t, out:] = a_(2t+1); flattening the
        # trailing 2*out axis yields index n = 2t + r — interleaved in time.
        return up.view(B, 2 * T2, self.out_dim)


class HiddenProjection(nn.Module):
    """Teacher hidden dim -> student hidden dim (plan §7). Trained together
    with the student so the pooled target H_t = (h_2t + h_(2t+1))/2 lands in
    the student's representation space."""

    def __init__(self, teacher_dim: int, student_dim: int, device=None, dtype=None):
        super().__init__()
        self.proj = nn.Linear(teacher_dim, student_dim, device=device, dtype=dtype)
        nn.init.normal_(self.proj.weight, std=teacher_dim ** -0.5)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        return self.proj(H)


class _IdentityProj(nn.Module):
    """Parameter-free stand-in for `HiddenProjection` (interface-compatible).

    Used when the student state already lives in the teacher's dimension
    (layer-init recipe): no projection is needed and none should exist, since a
    trainable projection could collapse the L_hidden target.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.proj = SimpleNamespace(in_features=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class StreamingStateFix(nn.Module):
    """Training-time fix for the 6.25 Hz serving schedule (see StudentConfig).

    At serving, frame p's head input is:
        head_state_p = used_p + phase_proj[p % 2](m_p)
    with
        used_p = a_p        for odd p   (the pair completes in time)
        used_p = a_(p-1)    for even p  (the pair is one frame late)
    and `m_p` the embedding of frame p (available at that step, because the
    model has already embedded the current frame into its pair buffer).

    The projections are zero-initialised, so an untrained checkpoint behaves
    exactly like the un-augmented model and training learns only the correction
    needed for the late slots.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden: int | None = None,
                 device=None, dtype=None):
        super().__init__()
        hidden = hidden or out_dim
        self.phase_proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, hidden, bias=False, device=device, dtype=dtype),
                nn.GELU(),
                nn.Linear(hidden, out_dim, bias=False, device=device, dtype=dtype),
            )
            for _ in range(2)
        ])
        for seq in self.phase_proj:
            nn.init.zeros_(seq[-1].weight)     # zero-init residual: no-op at start

    def phase_add(self, phase: int, m: torch.Tensor) -> torch.Tensor:
        return self.phase_proj[phase](m)

    def forward(self, states: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        T = states.shape[1]
        phase = torch.arange(T, device=states.device) % 2
        add0 = self.phase_proj[0](m)
        add1 = self.phase_proj[1](m)
        add = torch.where(phase.view(1, -1, 1) == 0, add0, add1)
        return states + add


def serving_schedule_states(a: torch.Tensor) -> torch.Tensor:
    """Rebuild the per-frame states the SERVER consumes from the training-path
    states: odd slots keep a_p, even slots get a_(p-1) (one pair late)."""
    if a.shape[1] <= 2:
        return a
    used = a.clone()
    used[:, 2::2] = a[:, 1:-1:2]
    return used


def served_head_states(student: "StudentLM", a: torch.Tensor,
                       m: torch.Tensor) -> torch.Tensor:
    """The head-conditioning states for the CURRENT cfg: unchanged when the fix
    is off, otherwise the serving schedule + the per-phase frame injection.
    Used by forward_train, the server, and the offline checks so that all three
    agree by construction."""
    if getattr(student.cfg, "streaming_fix", "off") != "augment":
        return a
    assert student.streaming_fix is not None
    return student.streaming_fix(serving_schedule_states(a), m)


class StudentLM(nn.Module):
    """Student: compressor + small Temporal Transformer + upsampler.

    HEADS ARE NOT OWNED here. The frozen teacher's `text_linear` and
    DepFormer stack are used inside `forward_train(codes, teacher_lm)`, so
    student logits and teacher logits always come out of the SAME weights —
    L_KD then directly measures the upstream (student) states' behaviour.

    Config must match the teacher's audio layout (`card`, `text_card`, `n_q`,
    `dep_q`, and the delay pattern) or the frozen DepFormer breaks.
    """

    def __init__(self, cfg: StudentConfig, teacher_lm=None,
                 delays: list[int] | None = None,
                 teacher_dim: int | None = None,
                 device=None, dtype=None):
        super().__init__()
        self.cfg = cfg
        if teacher_lm is not None:
            delays = list(teacher_lm.delays)
            cfg.card = teacher_lm.card
            cfg.text_card = teacher_lm.text_card
            cfg.n_q = teacher_lm.num_audio_codebooks
            cfg.dep_q = teacher_lm.dep_q
            teacher_dim = teacher_lm.dim
        assert delays is not None, "pass teacher_lm or delays"
        assert teacher_dim is not None, "pass teacher_lm or teacher_dim"
        self.delays = delays

        self.num_audio_codebooks = cfg.n_q
        self.audio_offset = 1
        self.num_codebooks = 1 + cfg.n_q

        zero_idx = -1  # LMModel.zero_token_id convention
        emb_kwargs = dict(device=device, dtype=dtype, zero_idx=zero_idx)
        self.emb = nn.ModuleList(
            ScaledEmbedding(cfg.card + 1, cfg.dim, **emb_kwargs)
            for _ in range(cfg.n_q)
        )
        self.text_emb = ScaledEmbedding(cfg.text_card + 1, cfg.dim, **emb_kwargs)

        self.compressor = FramePairCompressor(
            cfg.dim, hidden=cfg.compressor_hidden, device=device, dtype=dtype)
        self.transformer = StreamingTransformer(
            d_model=cfg.dim,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
            dim_feedforward=int(cfg.hidden_scale * cfg.dim),
            norm=cfg.norm,
            causal=True,
            layer_scale=cfg.layer_scale,
            context=cfg.context,
            max_period=cfg.max_period,
            gating=cfg.gating,
            positional_embedding=cfg.positional_embedding,
            device=device,
            dtype=dtype,
        )
        self.out_norm = create_norm_fn(cfg.norm, cfg.dim)
        if cfg.identity_hidden_proj:
            assert cfg.dim == teacher_dim, (
                "identity_hidden_proj requires the student dim to equal the "
                f"teacher dim (got {cfg.dim} vs {teacher_dim})")
            self.hidden_proj: nn.Module = _IdentityProj(cfg.dim)
        else:
            self.hidden_proj = HiddenProjection(
                teacher_dim, cfg.dim, device=device, dtype=dtype)
        self.upsampler = AudioStateUpsampler(
            cfg.dim, out_dim=teacher_dim, hidden=cfg.upsampler_hidden,
            device=device, dtype=dtype)
        # 6.25 Hz serving-schedule fix (no-op at init, see StudentConfig)
        self.streaming_fix: StreamingStateFix | None = None
        if cfg.streaming_fix == "augment":
            self.streaming_fix = StreamingStateFix(
                cfg.dim, teacher_dim, hidden=cfg.phase_hidden,
                device=device, dtype=dtype)
        else:
            assert cfg.streaming_fix == "off", cfg.streaming_fix

    # -- embeddings ---------------------------------------------------------

    def embed(self, sequence: torch.Tensor) -> torch.Tensor:
        """Same codebook-summing convention as LMModel.embed_codes."""
        B, K, S = sequence.shape
        assert K == self.num_codebooks, (
            f"unexpected codebook count {K} != {self.num_codebooks}")
        input_ = None
        for cb_index in range(self.num_audio_codebooks):
            audio_emb = self.emb[cb_index](sequence[:, cb_index + self.audio_offset])
            input_ = audio_emb if input_ is None else input_ + audio_emb
        text_emb = self.text_emb(sequence[:, 0])
        input_ = text_emb if input_ is None else input_ + text_emb
        return input_

    # -- main student path --------------------------------------------------

    def forward_main(self, m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """m: per-frame student embeddings [B, T, Ds] (post `embed`).
        Returns (s, a):
            s [B, T2, Ds]  post-norm student states (L_hidden target side)
            a [B, T, Dout] upsampled 12.5 Hz states for the frozen heads
        """
        z = self.compressor(m)
        u = self.transformer(z)
        s = self.out_norm(u)
        a = self.upsampler(s)
        return s, a

    # -- full teacher-forced training forward -------------------------------

    def forward_train(
        self,
        codes: torch.Tensor,
        teacher_lm,
        gradient_checkpointing: bool = False,
        return_states: bool = False,
    ):
        """Teacher-forced student forward, mirroring LMModel.forward_train.

        Args:
            codes: [B, K=17, T] interleaved codes (same layout the teacher
                consumes — the student sees the SAME delayed stream).
            teacher_lm: frozen PersonaPlex; provides delays, the initial
                token, text_linear and the DepFormer stack (all reused frozen).
            gradient_checkpointing: checkpoint the student's main path.
            return_states: also return the per-frame head-conditioning states
                `a` (post teacher out_norm). These are what the FROZEN DepFormer
                consumes, so dense state distillation (L_state) needs them.

        Returns:
            student_hidden [B, T2, Ds] and LMOutput with UNDELAYED logits
            (real NaN at the invalid trailing positions, same convention as
            the teacher — loss functions must sanitize BEFORE CE/KL).
            If `return_states`, also returns `a` [B, T, Dt].
        """
        assert list(teacher_lm.delays) == self.delays
        B, K, T = codes.shape
        assert T % 2 == 0, (
            "Student forward needs an even frame count T for 2-frame "
            "compression; pick duration_sec so duration_sec*12.5 is even.")

        initial = teacher_lm._get_initial_token().expand(B, -1, -1)
        delayed = _delay_sequence(self.delays, codes, initial)
        delayed = torch.cat([initial, delayed], dim=2)  # [B, K, T+1]

        m = self.embed(delayed[:, :, :-1])  # [B, T, Ds]

        if gradient_checkpointing:
            from torch.utils.checkpoint import checkpoint
            s, a = checkpoint(lambda mm: self.forward_main(mm), m,
                              use_reentrant=False)
        else:
            s, a = self.forward_main(m)

        # teacher post-out_norm normalization, exactly where the teacher
        # applies it, before its frozen heads
        norm = teacher_lm.out_norm
        if norm is not None:
            a = norm(a)

        # 6.25 Hz serving-schedule fix: hand the heads exactly the states the
        # server will hand them (odd slots exact, even slots one pair late with
        # the current frame injected), and supervise those same states, so the
        # training objective and the serving behaviour cannot diverge.
        if self.streaming_fix is not None:
            a = served_head_states(self, a, m)

        text_logits = teacher_lm.text_linear(a)[:, None]        # [B, 1, T, text_card]
        audio_logits = teacher_lm.forward_depformer_training(    # [B, Ka, T, card]
            delayed[:, :, 1:], a)

        logits, logits_mask = _undelay_sequence(
            self.delays[self.audio_offset:self.audio_offset + self.cfg.dep_q],
            audio_logits, fill_value=float("NaN"))
        logits_mask &= (codes[:, self.audio_offset:self.audio_offset + self.cfg.dep_q]
                        != teacher_lm.zero_token_id)
        text_logits, text_logits_mask = _undelay_sequence(
            self.delays[:1], text_logits, fill_value=float("NaN"))
        text_logits_mask &= (codes[:, :1] != teacher_lm.zero_token_id)

        out = LMOutput(logits, logits_mask, text_logits, text_logits_mask)
        if return_states:
            return s, out, a
        return s, out
