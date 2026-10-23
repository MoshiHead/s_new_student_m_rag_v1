"""
train_distill.py
================
PersonaPlex 6.25 Hz temporal distillation (distill_plan.md §19 Stage 4–7).

    Audio → Mimi (frozen) → codes [B, 17, T]
        ├── TEACHER (frozen, on-the-fly @ 12.5 Hz, no stored logits):
        │       personaPlex-7B forward → hidden states + text/audio logits
        └── STUDENT (trainable):
                2-frame compressor → 6.25 Hz Temporal Transformer
                → 6.25→12.5 Hz upsampler → FROZEN teacher heads (DepFormer)

    L = L_task(text CE + audio CE)
        + lambda_hidden * L_hidden(pooled 6.25 Hz state ↔ teacher hidden)
        + lambda_state  * L_state(per-frame 12.5 Hz state ↔ teacher hidden)
        + lambda_kd_text  * KL(teacher || student, T)
        + lambda_kd_audio * KL(teacher || student, T)

Design constraints honored:
- teacher COMPLETELY frozen; student's heads ARE the teacher's heads (frozen)
- Mimi untouched; 16 codebooks and the delay convention identical to teacher
- no KV compression, no quantization, no streaming inference yet (plan §21)

Training recipe notes (audio quality):
- L_state is the dense per-frame term and is what keeps the FROZEN DepFormer
  on the state manifold it was trained on. L_hidden alone only supervises the
  pooled 6.25 Hz state, which is not enough for coherent generation.
- Prefer `--state-warmup-steps` (e.g. a few hundred..thousand) so the student
  first learns the teacher's internal states, then the task/KD objectives join.
- Keep `--lambda-state` at least ~O(1); the task CE (first-codebook weight 100)
  is large and will otherwise drag the conditioning off the teacher manifold.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import sentencepiece
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from huggingface_hub import hf_hub_download

from .dataset import build_data_loader
from .distill.data import DistillInterleavedTokenizer
from .distill.teacher import forward_teacher_distill
from .distill_loss import compute_distill_losses
from .interleaver import Interleaver
from .models import loaders
from .models.lm import LMModel
from .models.student import StudentConfig, StudentLM

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(out_dir: Path, step: int, student: StudentLM,
                    optimizer=None, scheduler=None, stats: dict | None = None,
                    save_optimizer: bool = False) -> None:
    """Save the student weights (and a tiny step marker).

    The optimizer state is NOT saved by default: for the layer-init student it
    is tens of GB per checkpoint (AdamW keeps 2 fp32 moments per trainable
    parameter; 5.4B params -> ~43 GB) while the model itself is ~11 GB in bf16.
    Pass `save_optimizer=True` only if you actually need exact resume.
    """
    ckpt_dir = out_dir / f"step_{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": student.state_dict(),
                "cfg": asdict(student.cfg)}, ckpt_dir / "student.pt")
    # tiny marker so --resume still knows the step (and can keep logging)
    try:
        (ckpt_dir / "meta.json").write_text(
            json.dumps({"step": step,
                        "stats": {k: float(v) for k, v in (stats or {}).items()
                                  if isinstance(v, (int, float))}}),
            encoding="utf-8")
    except Exception as e:  # never fail a checkpoint over metadata
        logger.warning(f"[ckpt] could not write meta.json: {e}")

    if save_optimizer and optimizer is not None:
        torch.save({
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "step": step, "stats": stats,
        }, ckpt_dir / "optim.pt")
        logger.info(f"[ckpt] saved step {step} -> {ckpt_dir} (with optimizer state)")
    else:
        logger.info(f"[ckpt] saved step {step} -> {ckpt_dir} "
                    f"(model only; no optimizer state)")


def load_checkpoint(ckpt_dir: Path, student: StudentLM,
                    optimizer=None, scheduler=None) -> int:
    student_path = ckpt_dir / "student.pt"
    pkg = torch.load(student_path, map_location="cpu")
    missing, unexpected = student.load_state_dict(pkg["state_dict"], strict=True)
    if missing or unexpected:
        logger.warning(f"[ckpt] student state: {len(missing)} missing, "
                       f"{len(unexpected)} unexpected keys")

    # step marker (written even when the optimizer is not saved)
    step = 0
    meta_path = ckpt_dir / "meta.json"
    if meta_path.exists():
        try:
            step = int(json.loads(meta_path.read_text(encoding="utf-8")).get("step", 0))
        except Exception:
            step = 0

    optim_path = ckpt_dir / "optim.pt"
    if optim_path.exists() and optimizer is not None:
        state = torch.load(optim_path, map_location="cpu")
        optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None and state.get("scheduler"):
            scheduler.load_state_dict(state["scheduler"])
        step = state.get("step", step)
        logger.info(f"[ckpt] resumed from step {step} (optimizer state restored)")
    else:
        logger.warning(
            f"[ckpt] no optimizer state in {ckpt_dir}: resuming the WEIGHTS only "
            f"(step={step}); optimizer moments and the LR schedule start fresh")
    return step


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def training_step(batch, student: StudentLM, teacher: LMModel, args, device,
                  phase: str = "joint") -> dict:
    codes = batch.codes.to(device)
    # No streaming state is ever started during training, so
    # StreamingTransformer.forward recomputes position offsets from 0 for
    # every batch — nothing to reset between steps.

    # teacher forward — frozen, no gradients stored
    teacher_out = forward_teacher_distill(teacher, codes)
    teacher_out = {k: v.to(device) for k, v in teacher_out.items()}

    # student forward — frozen teacher heads are used internally
    student_hidden, student_output, student_states = student.forward_train(
        codes, teacher_lm=teacher,
        gradient_checkpointing=args.gradient_checkpointing,
        return_states=True,
    )

    loss, stats = compute_distill_losses(
        student_lm=student,
        teacher_lm=teacher,
        student_hidden=student_hidden,
        student_output=student_output,
        teacher_distill_output=teacher_out,
        codes=codes,
        args=args,
        student_states=student_states,
        phase=phase,
    )
    if loss.requires_grad:
        # no-op under torch.no_grad() (eval) — avoids "does not require grad"
        loss.backward()
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    device = torch.device(args.device)

    # ── Teacher stack ─────────────────────────────────────────────────────
    logger.info("loading tokenizer")
    if args.tokenizer is None:
        args.tokenizer = hf_hub_download(args.hf_repo, loaders.TEXT_TOKENIZER_NAME)
    tokenizer = sentencepiece.SentencePieceProcessor(args.tokenizer)

    logger.info("loading mimi (frozen, shared by data pipeline + teacher)")
    if args.mimi_weight is None:
        args.mimi_weight = hf_hub_download(args.hf_repo, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(args.mimi_weight, device)
    mimi.eval()
    for p in mimi.parameters():
        p.requires_grad_(False)

    logger.info("loading personaPlex teacher (frozen)")
    if args.moshi_weight is None:
        args.moshi_weight = hf_hub_download(args.hf_repo, loaders.MOSHI_NAME)
    teacher: LMModel = loaders.get_moshi_lm(args.moshi_weight, device=device)
    teacher.eval()
    teacher_params = sum(p.numel() for p in teacher.parameters())
    for p in teacher.parameters():
        p.requires_grad_(False)

    # ── Student stack ─────────────────────────────────────────────────────
    # `--resume` also implies the checkpoint's architecture, so resuming a
    # layer-init output needs no duplicated size flags.
    init_arg = args.student_init or args.resume
    if init_arg:
        # Architecture comes from the checkpoint (e.g. a layer_init.py output):
        # avoids having to mirror every size flag on the command line.
        init_path = Path(init_arg)
        if init_path.is_dir():
            init_path = init_path / "student.pt"
        pkg = torch.load(init_path, map_location="cpu")
        cfg = StudentConfig(**pkg["cfg"])
        logger.info(f"student cfg from {init_path}: dim={cfg.dim} "
                    f"layers={cfg.num_layers} heads={cfg.num_heads} "
                    f"context={cfg.context} streaming_fix={cfg.streaming_fix} "
                    f"identity_hidden_proj={cfg.identity_hidden_proj}")
    else:
        cfg = StudentConfig(
            dim=args.student_dim,
            num_layers=args.student_layers,
            num_heads=args.student_heads,
            hidden_scale=args.student_hidden_scale,
            context=args.student_context,
            upsampler_hidden=args.upsampler_hidden,
            compressor_hidden=args.compressor_hidden,
            streaming_fix=args.streaming_fix,
            phase_hidden=args.phase_hidden,
            identity_hidden_proj=args.identity_hidden_proj,
        )
    student = StudentLM(cfg, teacher_lm=teacher, device=device,
                        dtype=args.dtype)
    if getattr(args, "student_freeze_layers", False):
        for layer in student.transformer.layers:
            for p in layer.parameters():
                p.requires_grad_(False)
        logger.warning("student transformer layers frozen: training only the "
                       "adapters (compressor/upsampler/%s/out_norm)",
                       "streaming_fix" if cfg.streaming_fix != "off" else "hidden_proj")
    tl = getattr(args, "trainable_layers", None)
    if tl is not None:
        n = len(student.transformer.layers)
        tl = min(int(tl), n)
        for i, layer in enumerate(student.transformer.layers):
            keep = i >= (n - tl)
            for p in layer.parameters():
                p.requires_grad_(keep)
        logger.warning("keeping only the TOP %d/%d transformer layers trainable "
                       "(+ adapters) to fit memory", tl, n)
    student_params = sum(p.numel() for p in student.parameters())
    trainable_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    logger.info(
        f"teacher={teacher_params:,} params | student={student_params:,} params "
        f"({trainable_params:,} trainable) | dim={cfg.dim} layers={cfg.num_layers}"
    )

    student.train()

    # ── Optimizer ─────────────────────────────────────────────────────────
    # Fresh adapters (compressor/upsampler/streaming-fix/out_norm) learn a new
    # mapping and want a higher LR than the (pre-trained) transformer layers.
    named = [(n, p) for n, p in student.named_parameters() if p.requires_grad]
    if args.adapter_lr is not None:
        layer_ps = [p for n, p in named if n.startswith("transformer.") or n.startswith("emb")]
        adapter_ps = [p for n, p in named if not (n.startswith("transformer.") or n.startswith("emb"))]
        logger.info(f"param groups: layers/embeddings {sum(p.numel() for p in layer_ps):,} @ lr={args.lr:g} "
                    f"| adapters {sum(p.numel() for p in adapter_ps):,} @ lr={args.adapter_lr:g}")
        param_groups = [
            {"params": layer_ps, "lr": args.lr, "weight_decay": args.weight_decay},
            {"params": adapter_ps, "lr": args.adapter_lr, "weight_decay": args.weight_decay},
        ]
    else:
        param_groups = [{"params": [p for _, p in named],
                         "lr": args.lr, "weight_decay": args.weight_decay}]
    optimizer = AdamW(param_groups, betas=(0.9, 0.95), eps=1e-8) if args.optim == "adamw" else None
    if args.optim == "adafactor":
        # Adafactor keeps only a scaled row/column second moment (~P instead of
        # 2*P fp32), which is what can make a 24-layer (5.4B trainable) student
        # fit in 80GB next to the frozen 7B teacher. Its kwargs changed across
        # torch versions, so try the known signatures in order.
        from torch.optim import Adafactor
        last_err = None
        for kwargs in (
            dict(lr=args.lr, beta2_decay=-0.8, eps=(None, 1e-3),
                 weight_decay=args.weight_decay),                       # torch >= 2.7
            dict(lr=args.lr, eps=(1e-30, 1e-3), clip_threshold=1.0,
                 decay_rate=-0.8, scale_parameter=False, relative_step=False,
                 warmup_init=False, weight_decay=args.weight_decay),   # 1.12 .. 2.6
            dict(lr=args.lr, weight_decay=args.weight_decay),          # minimal
        ):
            try:
                optimizer = Adafactor(param_groups, **kwargs)
                logger.info(f"adafactor kwargs used: {sorted(kwargs)}")
                break
            except TypeError as e:
                last_err, optimizer = e, None
        if optimizer is None:
            raise last_err
    logger.info(f"optimizer: {args.optim}")

    if args.warmup_steps > 0:
        if args.warmup_steps >= args.max_steps:
            logger.warning(
                f"--warmup-steps {args.warmup_steps} >= --max-steps "
                f"{args.max_steps}: the LR never leaves the warmup "
                f"(start_factor 0.1) and the cosine T_max collapses to 1. "
                f"Lower --warmup-steps (e.g. 5% of --max-steps).")
        scheduler = SequentialLR(optimizer, [
            LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup_steps),
            CosineAnnealingLR(optimizer,
                              T_max=max(1, args.max_steps - args.warmup_steps),
                              eta_min=args.lr * 0.1),
        ], milestones=[args.warmup_steps])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.max_steps,
                                      eta_min=args.lr * 0.1)

    # ── Data ──────────────────────────────────────────────────────────────
    interleaver = Interleaver(
        tokenizer=tokenizer, audio_frame_rate=mimi.frame_rate,
        text_padding=teacher.text_padding_token_id,
        end_of_text_padding=teacher.end_of_text_padding_id,
        zero_padding=teacher.zero_token_id,
        keep_main_only=True,
        use_bos_eos=True,
        main_speaker_label="ASSISTANT",
        device=str(device),
    )
    distill_tokenizer = DistillInterleavedTokenizer(
        mimi=mimi,
        interleaver=interleaver,
        duration_sec=args.duration_sec,
        target_n_audio_codebooks=teacher.num_audio_codebooks,
        use_sidecar=args.use_sidecar,
        require_sidecar=not args.allow_missing_sidecar,
        system_prompt_default=args.default_system_prompt,
        use_rag=args.use_rag,
        ref_mode=args.ref_mode,
        n_ref_slots=args.n_ref_slots,
        max_ref_source_tokens=args.max_ref_source_tokens,
        ref_tokens_per_slot=args.ref_tokens_per_slot,
        min_ref_slots=args.min_ref_slots,
        ref_slots_after_query=args.ref_slots_after_query,
    )

    train_loader = build_data_loader(
        instruct_tokenizer=distill_tokenizer,
        args=args,
        batch_size=args.batch_size,
        seed=42,
        rank=0,
        world_size=1,
        is_eval=False,
    )
    # Eval loader: build a FRESH generator for every evaluation.
    # build_data_loader() returns a generator, so a single instance is
    # exhausted after one pass — reusing it silently yields 0 batches and
    # logs all-zero metrics for every eval after the first. Rebuilding is
    # cheap (the chunk manifest is cached) and deterministic (seed fixed,
    # eval shuffling off), so each eval scores the same held-out batches.
    def make_eval_loader():
        return build_data_loader(
            instruct_tokenizer=distill_tokenizer,
            args=args,
            batch_size=args.batch_size,
            seed=0,
            rank=0,
            world_size=1,
            is_eval=True,
        )

    has_eval = bool(args.eval_data)
    if not has_eval:
        logger.warning("no --eval-data given: validation is DISABLED "
                       "(no [eval] lines will be logged)")

    # ── Resume ────────────────────────────────────────────────────────────
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(Path(args.resume), student, optimizer, scheduler)

    # ── Loop ──────────────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stat_keys = ("task_text", "task_audio", "hidden", "state", "kd_text", "kd_audio")
    running = {k: 0.0 for k in stat_keys}
    running_n = 0
    accum = 0
    step = start_step
    t0 = time.time()
    logger.info(f"starting distillation from step {step}")

    optimizer.zero_grad()

    # autotune-collection of "kd_audio" only exists when lambda>0
    stat_keys_live = list(stat_keys)

    for batch in train_loader:
        if step >= args.max_steps:
            break

        # Phase: alignment warmup teaches the student the teacher's internal
        # states first (no task/KD), then the full objective joins. This is
        # what keeps the frozen DepFormer's conditioning meaningful.
        phase = "state" if step < args.state_warmup_steps else "joint"
        stats = training_step(batch, student, teacher, args, device, phase=phase)
        accum += 1
        running_n += 1
        for k, v in stats.items():
            running[k] = running.get(k, 0.0) + v

        if accum >= args.grad_accum:
            torch.nn.utils.clip_grad_norm_(
                [p for p in student.parameters() if p.requires_grad],
                max_norm=args.grad_clip,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            accum = 0

            # ── Logging ───────────────────────────────────────────────────
            if step % args.log_every == 0:
                elapsed = time.time() - t0
                avg = {k: running[k] / max(running_n, 1) for k in running}
                parts = " ".join(
                    f"{k}={avg[k]:.4f}" for k in stat_keys_live if k in avg
                )
                logger.info(
                    f"step {step:5d}/{args.max_steps} loss={avg.get('loss', 0.0):.4f} "
                    f"{parts} state_cos={avg.get('state_cos', 0.0):.3f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e} "
                    f"elapsed={elapsed:.0f}s"
                )
                if step in (args.state_warmup_steps,) and args.state_warmup_steps > 0:
                    logger.info(
                        "alignment warmup finished (state_cos=%.3f): switching to "
                        "the full joint objective", avg.get("state_cos", 0.0))
                running = {k: 0.0 for k in stat_keys}
                running_n = 0

            # ── Eval ──────────────────────────────────────────────────────
            if has_eval and step % args.eval_every == 0:
                student.eval()
                eval_stats: dict[str, float] = {}
                eval_n = 0
                with torch.no_grad():
                    for eval_batch in make_eval_loader():
                        s = training_step(eval_batch, student, teacher, args,
                                          device)
                        for k, v in s.items():
                            eval_stats[k] = eval_stats.get(k, 0.0) + v
                        eval_n += 1
                        if eval_n >= args.eval_batches:
                            break
                student.train()
                if eval_n == 0:
                    logger.warning(
                        "[eval] step %d: eval loader produced NO batches — "
                        "check --eval-data (path/manifest/sidecars)", step)
                else:
                    avg = {k: v / eval_n for k, v in eval_stats.items()}
                    logger.info(
                        f"[eval] step {step} n={eval_n} " + " ".join(
                            f"{k}={avg.get(k, 0.0):.4f}" for k in stat_keys)
                        + f" state_cos={avg.get('state_cos', 0.0):.3f}"
                    )

            if step % args.save_every == 0:
                save_checkpoint(out_dir, step, student, optimizer, scheduler,
                                stats={"step": step},
                                save_optimizer=args.save_optimizer)

    save_checkpoint(out_dir, step, student, optimizer, scheduler,
                    stats={"step": step, "final": True},
                    save_optimizer=args.save_optimizer)
    logger.info(f"distillation complete at step {step}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data (same jsonl manifests as the RAG finetune, minus sidecar needs)
    ap.add_argument("--train-data", required=True, help="train.jsonl manifest path")
    ap.add_argument("--eval-data", default="", help="eval.jsonl manifest path (optional)")
    ap.add_argument("--duration-sec", type=float, default=100.0,
                    help="chunk length; keep duration_sec*12.5 EVEN (100s -> 1250)")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--use-sidecar", action="store_true", default=True,
                    help="use sidecar transcript/speaker masks when present "
                         "(falls back to mono + zero-text automatically)")
    ap.add_argument("--no-sidecar", dest="use_sidecar", action="store_false",
                    help="never read sidecar files")
    ap.add_argument("--allow-missing-sidecar", action="store_true", default=False,
                    help="permit samples without a .sidecar.json (degraded mono "
                         "fallback: agent stream mixes both speakers and the "
                         "text stream is inert — trains the garbled-audio / "
                         "no-text failure mode; NOT recommended)")
    ap.add_argument("--default-system-prompt", type=str, default=None,
                    help="persona text injected as a <system>...</system> block "
                         "at the start of every recording when the sidecar has "
                         "no system_prompt. The inference server always feeds "
                         "the persona this way, so training must contain the "
                         "convention or the model cannot say the greeting.")

    # RAG / context injection (prefill-only latent ref slots).
    ap.add_argument("--use-rag", action="store_true", default=False,
                    help="splice <lookup> + <ref_start> [K slots] <ref_end> and "
                         "return segment_meta/reference_text so the RAG losses "
                         "and RefAdapter injection can be applied. Off by "
                         "default: the plain distillation path is unchanged.")
    ap.add_argument("--ref-mode", choices=["slots", "off"], default="slots")
    ap.add_argument("--n-ref-slots", type=int, default=16)
    ap.add_argument("--max-ref-source-tokens", type=int, default=400)
    ap.add_argument("--ref-tokens-per-slot", type=int, default=4)
    ap.add_argument("--min-ref-slots", type=int, default=1)
    ap.add_argument("--ref-slots-after-query", action="store_true", default=True,
                    help="place the ref block right after <lookup> (default)")
    ap.add_argument("--ref-slots-after-filler", dest="ref_slots_after_query",
                    action="store_false")

    # Teacher / models
    ap.add_argument("--hf-repo", default=loaders.DEFAULT_REPO)
    ap.add_argument("--moshi-weight", default=None)
    ap.add_argument("--mimi-weight", default=None)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--device", default="cuda")

    # Student architecture (2-3B default per plan §6)
    ap.add_argument("--student-dim", type=int, default=2560)
    ap.add_argument("--student-layers", type=int, default=24)
    ap.add_argument("--student-heads", type=int, default=20)
    ap.add_argument("--student-hidden-scale", type=float, default=4.0)
    ap.add_argument("--student-context", type=int, default=1500,
                    help="student steps of context (1500 @ 6.25 Hz = 240s)")
    ap.add_argument("--upsampler-hidden", type=int, default=None)
    ap.add_argument("--compressor-hidden", type=int, default=None)
    ap.add_argument("--streaming-fix", choices=["off", "augment"], default="augment",
                    help="6.25 Hz serving-schedule fix (see models/student.py). "
                         "'augment' trains with the exact schedule the server "
                         "uses and injects the current frame embedding into the "
                         "late slots; required for coherent generation at "
                         "6.25 Hz. 'off' reproduces the old behaviour.")
    ap.add_argument("--phase-hidden", type=int, default=None)
    ap.add_argument("--identity-hidden-proj", action="store_true",
                    help="use a parameter-free hidden projection (required by "
                         "the layer-init recipe, where the student state is "
                         "already in the teacher's dimension)")
    ap.add_argument("--student-init", type=str, default=None,
                    help="path to a student.pt (or its directory) whose cfg "
                         "defines the student architecture, e.g. produced by "
                         "layer_init.py. Overrides the --student-* size flags.")
    ap.add_argument("--student-freeze-layers", action="store_true",
                    help="freeze the temporal transformer layers and train only "
                         "the adapters (memory saver for the layer-init student)")
    ap.add_argument("--trainable-layers", type=int, default=None,
                    help="keep only the top N transformer layers trainable "
                         "(plus the adapters). Use for the 24-layer student, "
                         "which does not fit AdamW at full fine-tuning in 80GB.")
    ap.add_argument("--optim", choices=["adamw", "adafactor"], default="adamw",
                    help="adafactor needs ~1/2 the optimizer memory of AdamW "
                         "(one scaled row/col moment instead of two fp32 "
                         "moments) and is recommended for the 24-layer student")

    # Losses (plan §18)
    ap.add_argument("--lambda-task-text", type=float, default=1.0)
    ap.add_argument("--lambda-task-audio", type=float, default=1.0)
    ap.add_argument("--lambda-hidden", type=float, default=1.0,
                    help="weight for pooled-teacher hidden-state MSE")
    ap.add_argument("--lambda-state", type=float, default=10.0,
                    help="weight for DENSE per-frame state distillation in the "
                         "frozen heads' input space. This is what keeps the "
                         "frozen DepFormer on-distribution. Measured: 1.0 lets "
                         "the task CE drag the conditioning off the teacher "
                         "manifold (state_cos 0.99 -> 0.52, i.e. choppy audio); "
                         "~10-20 keeps it at 0.99 while still training. Set 0 "
                         "only to reproduce the old (broken) behaviour.")
    ap.add_argument("--state-warmup-steps", type=int, default=1000,
                    help="train ONLY the state-alignment terms (hidden + state) "
                         "for this many steps before enabling task/KD losses. "
                         "Cheap and makes the initial alignment reliable; the "
                         "frozen heads' input must be right before their output "
                         "distribution can be matched. 0 disables the phase.")
    ap.add_argument("--state-loss", choices=["mse", "cosine", "cosine_mse"],
                    default="mse",
                    help="distance for the dense state term. 'cosine_mse' is "
                         "recommended for the layer-init run: plain MSE starts "
                         "at ~2*sigma^2 (~4.2) with untrained adapters, which "
                         "with lambda-state 60 gives a >250 loss whose gradients "
                         "are smothered by --grad-clip.")
    ap.add_argument("--lambda-kd-text", type=float, default=1.0,
                    help="weight for text-logit KL distillation")
    ap.add_argument("--lambda-kd-audio", type=float, default=1.0,
                    help="weight for audio-logit KL distillation "
                         "(set 0 to save memory/compute)")
    ap.add_argument("--kd-temperature", type=float, default=2.0)

    # Task-loss internals (passthrough to rag_loss_new; keep near defaults)
    ap.add_argument("--first-codebook-weight", dest="first_codebook_weight_multiplier",
                    type=float, default=100.0)
    ap.add_argument("--num-agent-codebooks", type=int, default=None,
                    help="restrict audio CE to the first N codebooks "
                         "(default: all 16, standard duplex)")
    ap.add_argument("--text-padding-weight", type=float, default=0.3)
    ap.add_argument("--no-rag-weight", type=float, default=1.0)

    # Optimizer
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="fresh student modules -> larger LR than LoRA finetunes")
    ap.add_argument("--adapter-lr", type=float, default=None,
                    help="separate LR for the fresh adapters (compressor / "
                         "upsampler / streaming-fix / out_norm), leaving --lr for "
                         "the pre-trained transformer layers and embeddings. "
                         "Recommended for the layer-init recipe, e.g. "
                         "--lr 2e-5 --adapter-lr 3e-4.")
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--grad-clip", type=float, default=1.0,
                    help="gradient-norm clip; raise it (e.g. 10) when the state "
                         "term is large so early alignment is not frozen")
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--grad-checkpointing", dest="gradient_checkpointing",
                    action="store_true", default=True)
    ap.add_argument("--no-grad-checkpointing", dest="gradient_checkpointing",
                    action="store_false",
                    help="disable activation checkpointing (only if you have "
                         "plenty of VRAM; it increases memory usage a lot)")
    ap.add_argument("--dtype", default="bfloat16",

                    choices=["bfloat16", "float32"],
                    help="student parameter dtype (teacher always bf16)")

    # Checkpointing / logging
    ap.add_argument("--out-dir", default="checkpoints/distill_v1")
    ap.add_argument("--resume", default=None,
                    help="checkpoint dir to resume from, e.g. .../step_1000")
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--save-optimizer", action="store_true", default=False,
                    help="ALSO save optimizer+scheduler state (tens of GB per "
                         "checkpoint for the layer-init student: AdamW keeps 2 "
                         "fp32 moments per trainable parameter, ~43GB at 5.4B). "
                         "Off by default; without it --resume restores only the "
                         "weights and the step marker in meta.json.")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--eval-batches", type=int, default=50)

    # DataArgs compat for build_data_loader
    ap.add_argument("--shuffle", action="store_true", default=True)

    args = ap.parse_args()
    args.train_data = args.train_data
    args.eval_data = args.eval_data
    args.dtype = getattr(torch, args.dtype)

    train(args)


if __name__ == "__main__":
    main()
