"""
halfrate_teacher.py — A3: 6.25 Hz adapters around the teacher's OWN layers
==========================================================================

Idea
----
The 6.25 Hz student (models/student.py) buys half the temporal steps by
training a NEW 2.5B transformer — a new brain, which is why answers/audio got
weaker. A3 keeps the teacher's weights and only adds two small adapters:

    teacher embeddings (frozen)
        -> FramePairCompressor    (12.5 -> 6.25 Hz, trainable, ~20M)
        -> TEACHER TRANSFORMER    (frozen! half the calls, half the KV)
        -> AudioStateUpsampler    (6.25 -> 12.5 Hz, trainable, ~60M)
        -> frozen teacher out_norm / text_linear / DepFormer   (unchanged)

Only the two adapters train; the "intelligence" (all 7B layers, embeddings,
audio head) is byte-identical to the teacher. Training reuses the existing
trainer + loss stack unchanged (dense `L_state`, `L_hidden`, `L_KD`, `L_task`):

    # in train_distill.train():
    #   from .halfrate_teacher import HalfRateTeacher, HalfRateTeacherConfig
    #   backbone = HalfRateTeacher(cfg, teacher_lm=teacher, device=device,
    #                              dtype=args.dtype)
    # everything else (training_step / compute_distill_losses / save_checkpoint)
    # works as-is because this class keeps StudentLM's forward_train contract.

The streaming caveat is identical to the student's: pairs complete on odd
delayed positions, so alternating frames reuse the freshest state (1-frame
older context). The dense `L_state` objective keeps that perturbation small
because the adapters are supervised to reproduce the teacher's per-frame
hidden states exactly.

Usage
-----
    python -m moshi_distill.halfrate_teacher --self-test      # CPU check
    python -m moshi_distill.halfrate_teacher --print-wiring   # how to plug in

Training: use the normal trainer with the two-line swap shown above.
Serving:  use HalfRateTeacherGen (same adapter pattern as server.StudentLMGen).
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from .models.lm import LMGen, LMOutput, _LMGenState, _delay_sequence, _undelay_sequence
from .models.student import FramePairCompressor, AudioStateUpsampler
from .utils.compile import CUDAGraphed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class HalfRateTeacherConfig:
    """Adapter sizes; the model dims are filled in from the loaded teacher so
    a checkpoint can always be rebuilt (`asdict(cfg)` is saved with the
    weights, like StudentLM)."""

    # adapters
    compressor_hidden: int | None = None      # None -> 1x dim bottleneck MLP
    upsampler_hidden: int | None = None       # None -> 1x dim bottleneck MLP
    # derived from the teacher (do not set by hand)
    dim: int = 0
    card: int = 0
    text_card: int = 0
    n_q: int = 0
    dep_q: int = 0
    delays: list[int] = field(default_factory=list)


class _ProjShim(nn.Module):
    """Interface-compatible stand-in for StudentLM.hidden_proj.

    The A3 state already lives in the TEACHER's hidden dimension, so no learned
    projection is needed (and none should exist: a learnable projection could
    collapse the L_hidden target). `compute_hidden_distill_loss` asserts
    `hidden_proj.proj.in_features`, hence the shim.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.proj = SimpleNamespace(in_features=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

class HalfRateTeacher(nn.Module):
    """Trainable adapters around the frozen teacher transformer.

    Drop-in for `StudentLM` in train_distill.train(): same `forward_train`
    signature/return contract, `cfg` dataclass, `hidden_proj` attribute.
    """

    def __init__(self, cfg: HalfRateTeacherConfig, teacher_lm=None,
                 device=None, dtype=None):
        super().__init__()
        assert teacher_lm is not None, "HalfRateTeacher needs the loaded teacher"
        # derive dims / token layout from the frozen teacher
        cfg.dim = teacher_lm.dim
        cfg.card = teacher_lm.card
        cfg.text_card = teacher_lm.text_card
        cfg.n_q = teacher_lm.num_audio_codebooks
        cfg.dep_q = teacher_lm.dep_q
        cfg.delays = list(teacher_lm.delays)
        self.cfg = cfg

        self.delays = list(cfg.delays)
        self.num_audio_codebooks = cfg.n_q
        self.num_codebooks = 1 + cfg.n_q
        self.audio_offset = 1

        self.compressor = FramePairCompressor(
            cfg.dim, hidden=cfg.compressor_hidden, device=device, dtype=dtype)
        self.upsampler = AudioStateUpsampler(
            cfg.dim, out_dim=cfg.dim, hidden=cfg.upsampler_hidden,
            device=device, dtype=dtype)
        self.hidden_proj = _ProjShim(cfg.dim)      # no parameters

    # -- trainable surface --------------------------------------------------

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def param_groups(self, lr: float, weight_decay: float):
        return [{"params": self.trainable_parameters(),
                 "lr": lr, "weight_decay": weight_decay}]

    def extra_repr(self) -> str:
        n = sum(p.numel() for p in self.trainable_parameters())
        return f"dim={self.cfg.dim} trainable={n:,}"

    @classmethod
    def load(cls, path, teacher_lm, device=None, dtype=None) -> "HalfRateTeacher":
        pkg = torch.load(path, map_location="cpu")
        cfg = HalfRateTeacherConfig(**pkg["cfg"])
        backbone = cls(cfg, teacher_lm=teacher_lm, device=device, dtype=dtype)
        backbone.load_state_dict(pkg["state_dict"], strict=True)
        backbone.eval()
        return backbone

    # -- main path ----------------------------------------------------------

    def forward_main(self, m: torch.Tensor, teacher_lm):
        """m: teacher embeddings of the delayed frames [B, T, Dt] (the teacher's
        own `embed_codes` output). Returns (s, a):
            s [B, T/2, Dt]  post out_norm 6.25 Hz state (hidden-loss side)
            a [B, T,   Dt]  upsampled + normed 12.5 Hz head conditioning
        """
        z = self.compressor(m)                        # [B, T/2, Dt]
        u = teacher_lm.transformer(z)                 # FROZEN 7B layers, half rate
        s = teacher_lm.out_norm(u) if teacher_lm.out_norm is not None else u
        a = self.upsampler(s)                         # [B, T, Dt]
        if teacher_lm.out_norm is not None:
            a = teacher_lm.out_norm(a)
        return s, a

    # -- teacher-forced training forward (StudentLM contract) ---------------

    def forward_train(self, codes: torch.Tensor, teacher_lm,
                      gradient_checkpointing: bool = False,
                      return_states: bool = False):
        assert list(teacher_lm.delays) == self.delays
        B, K, T = codes.shape
        assert K == self.num_codebooks, (K, self.num_codebooks)
        assert T % 2 == 0, (
            "HalfRateTeacher needs an even frame count T for 2-frame compression")

        initial = teacher_lm._get_initial_token().expand(B, -1, -1)
        delayed = _delay_sequence(self.delays, codes, initial)
        delayed = torch.cat([initial, delayed], dim=2)          # [B, K, T+1]

        # FROZEN teacher embeddings: no new embedding tables to learn, and the
        # input stream matches exactly what the teacher itself would consume.
        m = teacher_lm.embed_codes(delayed[:, :, :-1])          # [B, T, Dt]

        if gradient_checkpointing:
            from torch.utils.checkpoint import checkpoint
            s, a = checkpoint(lambda mm: self.forward_main(mm, teacher_lm), m,
                              use_reentrant=False)
        else:
            s, a = self.forward_main(m, teacher_lm)

        text_logits = teacher_lm.text_linear(a)[:, None]         # [B, 1, T, V]
        audio_logits = teacher_lm.forward_depformer_training(    # [B, Ka, T, C]
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


# ---------------------------------------------------------------------------
# Serving adapter (same pattern as server.StudentLMGen)
# ---------------------------------------------------------------------------

@dataclass
class _HalfRateGenState(_LMGenState):
    hr_pending: torch.Tensor | None = None
    hr_last_state: torch.Tensor | None = None

    def reset(self):
        super().reset()
        self.hr_pending = None
        if self.hr_last_state is not None:
            self.hr_last_state.zero_()


class HalfRateTeacherGen(LMGen):
    """LMGen whose temporal backbone is the frozen teacher transformer fed at
    6.25 Hz through the trained adapters.

    Wiring (server.py ServerState) — note the teacher is still `lm`:

        self.lm_gen = HalfRateTeacherGen(backbone, lm, **lm_gen_kwargs)
        # backbone = HalfRateTeacher.load(path, teacher_lm=lm, device=...)
        # (keep the fast_provided_tail / voice-prompt handling as for the teacher)
    """

    def __init__(self, backbone: HalfRateTeacher, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backbone = backbone
        # the frozen teacher transformer/embeddings are reused, so the teacher
        # must have been passed as `lm_model` to LMGen.__init__
        self._teacher = self.lm_model

    def _init_streaming_state(self, batch_size: int) -> _LMGenState:
        state = super()._init_streaming_state(batch_size)
        state.graphed_main = CUDAGraphed(self._halfrate_forward_codes, disable=True)
        state.__class__ = _HalfRateGenState
        state.hr_pending = None
        state.hr_last_state = torch.zeros(
            batch_size, 1, self._teacher.dim,
            device=self._teacher.device,
            dtype=next(iter(self._teacher.parameters())).dtype)
        return state

    @torch.no_grad()
    def _halfrate_forward_codes(self, input_tokens: torch.Tensor):
        state = self._streaming_state
        m = self._teacher.embed_codes(input_tokens)              # [B, 1, Dt]
        if state.hr_pending is None:
            state.hr_pending = m
        else:
            pair = torch.cat([state.hr_pending, m], dim=1)       # [B, 2, Dt]
            z = self.backbone.compressor(pair)                   # [B, 1, Dt]
            u = self._teacher.transformer(z)                     # frozen, 6.25 Hz
            s = self._teacher.out_norm(u) if self._teacher.out_norm is not None else u
            a = self.backbone.upsampler(s)                       # [B, 2, Dt]
            if self._teacher.out_norm is not None:
                a = self._teacher.out_norm(a)
            state.hr_pending = None
            state.hr_last_state = a[:, 1:2]
        transformer_out = state.hr_last_state
        text_logits = self._teacher.text_linear(transformer_out)[:, None]
        return transformer_out, text_logits

    @staticmethod
    def load(path: str | Path, teacher_lm, device=None, dtype=None) -> "HalfRateTeacher":
        """Alias of `HalfRateTeacher.load` (kept for symmetry with
        server.StudentLMGen-style wiring)."""
        return HalfRateTeacher.load(path, teacher_lm=teacher_lm,
                                    device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Self test / CLI
# ---------------------------------------------------------------------------

def _tiny_teacher(dim=64, card=64, text_card=32):
    from .models.lm import LMModel
    t = LMModel(
        delays=[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
        n_q=16, dep_q=16, card=card, text_card=text_card,
        dim=dim, num_heads=4, num_layers=2, hidden_scale=2,
        existing_text_padding_id=3, context=64, causal=True,
        depformer_dim=32, depformer_dim_feedforward=64,
        depformer_num_heads=4, depformer_num_layers=2,
        depformer_multi_linear=True, depformer_weights_per_step=True,
        depformer_weights_per_step_schedule=list(range(16)),
        depformer_causal=True, depformer_layer_scale=None, depformer_context=8,
        depformer_max_period=10000, depformer_gating="silu", depformer_pos_emb="none")
    t.eval()
    for p in t.parameters():
        p.requires_grad_(False)
    return t


def _self_test() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    import argparse as _ap
    import tempfile

    torch.manual_seed(0)
    teacher = _tiny_teacher()
    cfg = HalfRateTeacherConfig(compressor_hidden=None, upsampler_hidden=None)
    model = HalfRateTeacher(cfg, teacher_lm=teacher, dtype=torch.float32)
    n_train = sum(p.numel() for p in model.trainable_parameters())
    n_teacher = sum(p.numel() for p in teacher.parameters())
    print(f"trainable adapters: {n_train:,}  (teacher frozen: {n_teacher:,})")

    g = torch.Generator().manual_seed(7)
    codes = torch.randint(0, 64, (2, 17, 16), generator=g)
    codes[:, 0] = torch.randint(1, 31, (2, 16), generator=g)
    codes[:, :, -1:] = -1

    s, out, a = model.forward_train(codes, teacher_lm=teacher, return_states=True)
    assert s.shape == (2, 8, 64), s.shape          # 6.25 Hz state
    assert a.shape == (2, 16, 64), a.shape         # 12.5 Hz conditioning
    assert out.logits.shape == (2, 16, 16, 64), out.logits.shape
    assert out.text_logits.shape == (2, 1, 16, 32), out.text_logits.shape
    print("forward_train ok:", tuple(s.shape), tuple(a.shape))

    # contract with the EXISTING loss stack (incl. dense L_state)
    from .distill_loss import compute_distill_losses
    from .distill.teacher import forward_teacher_distill
    t_out = forward_teacher_distill(teacher, codes)
    args = _ap.Namespace(
        lambda_task_text=1.0, lambda_task_audio=1.0, lambda_hidden=1.0,
        lambda_state=10.0, lambda_kd_text=1.0, lambda_kd_audio=1.0,
        kd_temperature=2.0, first_codebook_weight_multiplier=100.0,
        num_agent_codebooks=None, text_padding_weight=0.5, no_rag_weight=1.0)
    loss, stats = compute_distill_losses(
        student_lm=model, teacher_lm=teacher, student_hidden=s,
        student_output=out, teacher_distill_output=t_out, codes=codes,
        args=args, student_states=a)
    assert torch.isfinite(loss), loss
    loss.backward()
    assert model.compressor.proj.weight.grad is not None
    assert model.upsampler.out.weight.grad is not None
    assert all(p.grad is None for p in teacher.parameters()), "teacher must stay frozen"
    print(f"loss stack ok: total={loss.item():.3f} state_cos={stats['state_cos']:.3f}")

    # save / load round trip
    tmp = Path(tempfile.mkdtemp(prefix="hr_self_test_"))
    torch.save({"state_dict": model.state_dict(), "cfg": asdict(model.cfg)},
               tmp / "halfrate.pt")
    fresh = HalfRateTeacherGen.load(tmp / "halfrate.pt", teacher_lm=teacher,
                                    dtype=torch.float32)
    for k, v in model.state_dict().items():
        assert torch.equal(v, fresh.state_dict()[k]), k
    print("checkpoint round trip ok")

    # serving adapter: drive the LMGen path with provided frames
    gen = HalfRateTeacherGen(fresh, teacher, device="cpu", sample_rate=24000,
                             frame_rate=12.5, use_sampling=False, top_k=1,
                             top_k_text=1, fast_provided_tail=True)
    gen.streaming_forever(1)
    one = codes[0:1]
    outs = 0
    for f in range(6):
        r = gen.step(moshi_tokens=one[:, 1:9, f: f + 1],
                     input_tokens=one[:, 9:17, f: f + 1],
                     text_token=one[:, 0, f])
        if r is not None:
            assert r.shape[1] == gen.lm_model.dep_q + 1, r.shape
            outs += 1
    assert outs > 0, "no outputs produced"
    print(f"serving adapter ok ({outs} frames emitted)")
    print("HALF-RATE TEACHER SELF-TEST PASSED")


WIRING = """\
# --- train_distill.py: swap the backbone (nothing else changes) -------------
from .halfrate_teacher import HalfRateTeacher, HalfRateTeacherConfig

cfg = HalfRateTeacherConfig(compressor_hidden=args.compressor_hidden,
                            upsampler_hidden=args.upsampler_hidden)
backbone = HalfRateTeacher(cfg, teacher_lm=teacher, device=device,
                           dtype=args.dtype)
# then use `backbone` wherever `student` was created; the optimizer should use
# backbone.trainable_parameters(). training_step / compute_distill_losses /
# save_checkpoint work unchanged (they only need forward_train + cfg +
# hidden_proj). Recommended: --lambda-state 10 --state-warmup-steps 1000.

# --- server.py: serve it ---------------------------------------------------
from .halfrate_teacher import HalfRateTeacher, HalfRateTeacherGen

backbone = HalfRateTeacher.load(args.halfrate_checkpoint, teacher_lm=lm,
                                device=args.device, dtype=torch.bfloat16)
self.lm_gen = HalfRateTeacherGen(backbone, lm, **lm_gen_kwargs)
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--print-wiring", action="store_true")
    args = ap.parse_args()
    if args.print_wiring:
        print(WIRING)
        return
    _self_test()


if __name__ == "__main__":
    main()
