"""
eval_student.py — state-fidelity gate for distilled 6.25 Hz students
====================================================================

Measures the ONE number that predicts whether a student checkpoint can drive
the frozen DepFormer coherently:

    cos( student's per-frame 12.5 Hz state, teacher's per-frame hidden )

measured on REAL conversation audio (the data pipeline's own codes). Random
codes are out-of-distribution and give meaningless low cosines (~0.2), so this
script runs the actual `DistillInterleavedTokenizer` on training manifest
samples.

Interpretation (measured on the teacher itself):
    >= 0.95   good enough for the frozen heads
    ~0.85     what a short/under-trained run looks like -> salad text
    <= 0.70   conditioning is unrelated -> unintelligible

It also reports the teacher's own adjacent-frame hidden cosine, which bounds
how much the 6.25 Hz schedule's 1-frame reuse can cost (measured ~0.65 on
personaplex, i.e. the even slots see a state ~0.65-similar to what they need).

Usage
-----
    python -m moshi_distill.eval_student \
        --student-checkpoint ckpt/distill_v1/step_2000 \
        --moshi-weight .../model.safetensors \
        --mimi-weight  .../tokenizer-e351c8d8-checkpoint125.safetensors \
        --tokenizer    .../tokenizer_spm_32k_3.model \
        --train-data data/train_split.jsonl --samples 3
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import sphn
import sentencepiece
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from .interleaver import Interleaver
from .distill.data import DistillInterleavedTokenizer
from .models import loaders
from .models.lm import _delay_sequence
from .models.student import StudentConfig, StudentLM
from .server import torch_auto_device

logger = logging.getLogger(__name__)


@torch.no_grad()
def student_vs_teacher_cos(student: StudentLM, teacher, codes: torch.Tensor) -> tuple[float, float]:
    """(served-state cosine, teacher adjacent-frame cosine) on these codes.

    Uses `forward_train(return_states=True)`, i.e. exactly the head-conditioning
    states the server will consume (including the streaming fix when enabled),
    so this number is comparable to what generation sees.
    """
    if codes.shape[-1] % 2:
        codes = codes[:, :, :-1]
    _, _, a = student.forward_train(codes, teacher_lm=teacher, return_states=True)
    B = codes.shape[0]
    initial = teacher._get_initial_token().expand(B, -1, -1)
    delayed = torch.cat([initial, _delay_sequence(teacher.delays, codes, initial)], dim=2)
    t_hidden = teacher.forward_embeddings(teacher.embed_codes(delayed[:, :, :-1]))[0]
    valid = codes[:, teacher.audio_offset] != teacher.zero_token_id
    cos = F.cosine_similarity(a.float(), t_hidden.float(), dim=-1)       # [B, T]
    w = valid.float()
    frame_cos = ((cos * w).sum() / w.sum().clamp(min=1)).item()
    adj = F.cosine_similarity(t_hidden[:, :-1].float(),
                              t_hidden[:, 1:].float(), dim=-1).mean().item()
    return frame_cos, adj


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-checkpoint", required=True)
    ap.add_argument("--moshi-weight", default=None)
    ap.add_argument("--mimi-weight", default=None)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--hf-repo", default=loaders.DEFAULT_REPO)
    ap.add_argument("--train-data", required=True, help="jsonl manifest")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--duration-sec", type=float, default=40.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--allow-missing-sidecar", action="store_true")
    args = ap.parse_args()

    device = torch_auto_device(args.device)
    if args.mimi_weight is None:
        args.mimi_weight = hf_hub_download(args.hf_repo, loaders.MIMI_NAME)
    if args.tokenizer is None:
        args.tokenizer = hf_hub_download(args.hf_repo, loaders.TEXT_TOKENIZER_NAME)
    if args.moshi_weight is None:
        args.moshi_weight = hf_hub_download(args.hf_repo, loaders.MOSHI_NAME)

    mimi = loaders.get_mimi(args.mimi_weight, device)
    teacher = loaders.get_moshi_lm(args.moshi_weight, device=device)
    teacher.eval()

    ckpt = args.student_checkpoint
    if os.path.isdir(ckpt):
        ckpt = os.path.join(ckpt, "student.pt")
    pkg = torch.load(ckpt, map_location="cpu")
    cfg = StudentConfig(**pkg["cfg"])
    student = StudentLM(cfg, teacher_lm=teacher, device=device, dtype=torch.bfloat16)
    student.load_state_dict(pkg["state_dict"], strict=True)
    student.eval()
    logger.info("student: dim=%d layers=%d ctx=%d", cfg.dim, cfg.num_layers, cfg.context)

    tok = sentencepiece.SentencePieceProcessor(args.tokenizer)
    inter = Interleaver(tokenizer=tok, audio_frame_rate=mimi.frame_rate,
                        text_padding=3, end_of_text_padding=0, zero_padding=-1,
                        keep_main_only=True, use_bos_eos=True,
                        main_speaker_label="ASSISTANT", device=str(device))
    dtok = DistillInterleavedTokenizer(
        mimi=mimi, interleaver=inter, duration_sec=args.duration_sec,
        target_n_audio_codebooks=teacher.num_audio_codebooks,
        require_sidecar=not args.allow_missing_sidecar)

    cosines = []
    n = 0
    for line in open(args.train_data):
        if n >= args.samples:
            break
        d = json.loads(line)
        p = d["path"]
        n += 1
        try:
            pcm, sr = sphn.read(p)
            pcm = sphn.resample(pcm, src_sample_rate=sr, dst_sample_rate=mimi.sample_rate)
            sample = dtok(pcm, 0.0, p)
            cos, adj = student_vs_teacher_cos(student, teacher, sample.codes.to(device))
            cosines.append(cos)
            logger.info("[%d] %s frames=%d state_cos=%.4f (teacher adjacent-frame "
                        "cos=%.4f)", n, os.path.basename(p), sample.codes.shape[-1],
                        cos, adj)
        except Exception as e:  # keep going: one bad file shouldn't kill the gate
            logger.error("[%d] %s failed: %s: %s", n, os.path.basename(p),
                         type(e).__name__, e)

    if cosines:
        mean = sum(cosines) / len(cosines)
        logger.info("MEAN state_cos = %.4f over %d samples (%s)", mean, len(cosines),
                    "OK (>=0.95)" if mean >= 0.95 else
                    "TOO LOW: retrain with stronger state dist" if mean < 0.95 else "")
    else:
        logger.error("no samples evaluated")


if __name__ == "__main__":
    main()
