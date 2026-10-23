"""
distill/smoke_test.py
=====================
CPU-only end-to-end check of the distillation plumbing with a TINY teacher,
no downloads, no CUDA:

    tiny PersonaPlex-shaped teacher (LMModel, random init, frozen)
        → forward_teacher_distill (hidden states + undelayed logits)
    tiny StudentLM (compressor → student transformer → upsampler)
        → forward_train(codes, teacher)  [frozen teacher heads]
    → all four loss terms → backward → weights sanity → AdamW step

Run:
    python -m moshi.distill.smoke_test        (from the repo root, or set PYTHONPATH)
    python distill/smoke_test.py              (from inside the moshi package dir)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import torch

from .teacher import forward_teacher_distill
from ..models.lm import LMModel
from ..models.student import StudentConfig, StudentLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smoke")

TEACHER_DELAYS = [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]


def build_tiny_teacher() -> LMModel:
    teacher = LMModel(
        delays=TEACHER_DELAYS,
        n_q=16, dep_q=16,
        card=64, text_card=32,
        dim=64, num_heads=4, num_layers=2, hidden_scale=2,
        existing_text_padding_id=3,
        context=64,
        depformer_dim=32, depformer_dim_feedforward=64,
        depformer_num_heads=4, depformer_num_layers=2,
        depformer_multi_linear=True,
        depformer_weights_per_step=True,
        depformer_weights_per_step_schedule=list(range(16)),
        depformer_causal=True,
        depformer_layer_scale=None,
        depformer_context=8,
        depformer_max_period=10000,
        depformer_gating="silu",
        depformer_pos_emb="none",
    )
    teacher.eval()
    return teacher


def build_fake_codes(B=2, card=64) -> torch.Tensor:
    gen = torch.Generator().manual_seed(7)
    K, T = 17, 16
    codes = torch.randint(0, card, (B, K, T), generator=gen)
    # trailing right-padding (zero_padding = -1) on the last frame
    codes[:, :, -1:] = -1
    # text channel mostly inert like a zero-transcript stream
    codes[:, 0, :8] = torch.randint(0, 32, (8,), generator=gen)
    codes[:, 0, 8:] = -1
    return codes


def build_fake_rag_args():
    import argparse
    return argparse.Namespace(
        lambda_task_text=1.0, lambda_task_audio=1.0,
        lambda_hidden=1.0,
        lambda_kd_text=1.0, lambda_kd_audio=1.0,
        kd_temperature=2.0,
        first_codebook_weight_multiplier=100.0,
        num_agent_codebooks=None,
        text_padding_weight=0.5,
        no_rag_weight=1.0,
    )


def main():
    t0 = time.time()
    torch.manual_seed(0)

    teacher = build_tiny_teacher()
    for p in teacher.parameters():
        p.requires_grad_(False)

    cfg = StudentConfig(dim=32, num_layers=2, num_heads=4, hidden_scale=2,
                        context=32)
    student = StudentLM(cfg, teacher_lm=teacher, dtype=torch.float32)
    student.train()

    B = 2
    codes = build_fake_codes(B=B, card=teacher.card)

    # ---- teacher forward ----
    T = codes.shape[-1]
    t_out = forward_teacher_distill(teacher, codes)
    assert t_out["hidden"].shape == (B, T, 64), t_out["hidden"].shape
    assert t_out["text_logits"].shape == (B, 1, T, teacher.text_card), t_out["text_logits"].shape
    assert t_out["audio_logits"].shape == (B, teacher.dep_q, T, teacher.card), t_out["audio_logits"].shape
    logger.info(f"teacher fwd ok: hidden {t_out['hidden'].shape}, "
                f"text {t_out['text_logits'].shape}, audio {t_out['audio_logits'].shape}")

    # ---- student forward ----
    student_hidden, student_out = student.forward_train(codes, teacher_lm=teacher)
    T2 = T // 2
    assert student_hidden.shape == (B, T2, cfg.dim), student_hidden.shape
    assert student_out.text_logits.shape == (B, 1, T, teacher.text_card), student_out.text_logits.shape
    assert student_out.logits.shape == (B, teacher.dep_q, T, teacher.card), student_out.logits.shape
    logger.info(f"student fwd ok: hidden {student_hidden.shape}")

    # NOTE: text table row text_card exists (+1 padding row convention) while
    # the teacher's own table is built as text_card (+extra_text). Both are
    # valid target spaces for CE here since compute_text_loss_rag weights
    # invalid positions to zero.

    # ---- losses + backward ----
    from ..distill_loss import compute_distill_losses
    args = build_fake_rag_args()
    loss, stats = compute_distill_losses(
        student_lm=student, teacher_lm=teacher,
        student_hidden=student_hidden, student_output=student_out,
        teacher_distill_output=t_out, codes=codes, args=args,
    )
    logger.info(f"loss={loss.item():.4f} stats={stats}")
    assert torch.isfinite(loss), "loss must be finite"

    loss.backward()

    checks = {
        "text_emb": student.text_emb.weight.grad is not None,
        "compressor": student.compressor.proj.weight.grad is not None,
        "transformer": next(student.transformer.parameters()).grad is not None,
        "upsampler": student.upsampler.out.weight.grad is not None,
        "hidden_proj": student.hidden_proj.proj.weight.grad is not None,
    }
    assert all(checks.values()), f"missing grads: {checks}"
    logger.info(f"grads ok: {checks}")

    # frozen-teacher grad leak check
    assert all(p.grad is None for p in teacher.parameters()), "teacher must get no grads"

    # ---- optimizer step ----
    trainable = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=1e-3)
    torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
    n_params = sum(p.numel() for p in trainable)
    opt.step()

    logger.info(f"opt step ok, trainable params: {n_params:,}")

    # ---- trainer-level step + checkpoint roundtrip ----
    from ..train_distill import training_step, save_checkpoint, load_checkpoint
    args = build_fake_rag_args()
    args.gradient_checkpointing = False
    args.grad_clip = 1.0

    batch = argparse.Namespace(codes=codes)
    stats = training_step(batch, student, teacher, args, "cpu")
    logger.info(f"trainer step ok: {stats}")

    import tempfile
    from pathlib import Path
    tmp = Path(tempfile.mkdtemp(prefix="distill_smoke_"))
    scheduler = None
    save_checkpoint(tmp, 10, student, opt, scheduler, stats={"step": 10})
    fresh_cfg = StudentConfig(dim=32, num_layers=2, num_heads=4, hidden_scale=2,
                              context=32)
    fresh = StudentLM(fresh_cfg, teacher_lm=teacher, dtype=torch.float32)
    fresh.train()
    resumed = load_checkpoint(tmp / "step_10", fresh, opt, scheduler)
    assert resumed == 10, resumed
    for k, v in student.state_dict().items():
        assert torch.equal(v, fresh.state_dict()[k]), f"mismatched key {k}"
    logger.info("checkpoint roundtrip ok (resume -> step 10)")

    logger.info(f"SMOKE TEST PASSED in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
