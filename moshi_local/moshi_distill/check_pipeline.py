"""
check_pipeline.py
=================
Offline end-to-end sanity check of the distilled-student serving path,
WITHOUT the websocket / opus plumbing (that part is then not a variable).

It does three things:

  1. STREAMING-vs-TRAINING parity on the real weights: steps the
     StudentLMGen main-backbone one frame at a time (exactly what LMGen.step
     feeds it) and asserts the consumed 12.5 Hz states match the ones
     `StudentLM.forward_train` produces (exact on even delayed positions,
     1-frame reuse on odd ones - inherent to the causal 2-frame compressor).
  2. GENERATION: prompts the model via the real LMGen prompt path
     (text prompt + 0.5 s silence, voice prompt optional) then streams
     SILENT user-audio frames and decodes what the agent generates.
     Writes `out.wav` plus a text dump, so you can *listen* to whether the
     greeting/speech is intact without the browser in the loop.
  3. TEACHER A/B: run the same generation with --student-checkpoint omitted
     to compare against the full teacher. If teacher is clean and student is
     choppy, the student checkpoint/pipeline is the suspect, not the server.

Usage (on the serve box, same args as server.py):
    python -m moshi_distill.check_pipeline \
        --moshi-weight /path/model.safetensors \
        --mimi-weight  /path/mimi.safetensors \
        --tokenizer    /path/tokenizer.model \
        --student-checkpoint /path/to/step_5000 \
        --text-prompt  "Your name is Jane. You are a friendly agent." \
        --seconds 4 \
        [--voice-prompt /path/to/s0.wav]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import sentencepiece
import sphn
import torch
from huggingface_hub import hf_hub_download

from .models import loaders
from .models.lm import _delay_sequence
from .models.student import StudentConfig, StudentLM
from .server import ServerState, seed_all, torch_auto_device, wrap_with_system_tags

logger = logging.getLogger(__name__)


def _write_wav(path: str | Path, sample_rate: int, mono_float32: np.ndarray) -> None:
    """Write mono float audio as a 16-bit WAV.

    `sphn` changes its writer name/signature between versions
    (write_wav(filename, data, sample_rate) vs write_pcm_wav_s16(...)), so try
    it first and fall back to a stdlib writer that always works."""
    try:
        if hasattr(sphn, "write_wav"):
            sphn.write_wav(str(path), np.asarray(mono_float32, dtype=np.float32),
                           int(sample_rate))
            return
        if hasattr(sphn, "write_pcm_wav_s16"):
            sphn.write_pcm_wav_s16(str(path), int(sample_rate), mono_float32)
            return
    except Exception as e:  # noqa: BLE001
        logger.warning("sphn wav writer failed (%s); using stdlib fallback", e)
    import wave
    clipped = np.clip(np.asarray(mono_float32, dtype=np.float32), -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(pcm16.tobytes())


def load_stack(args, device):
    if args.mimi_weight is None:
        args.mimi_weight = hf_hub_download(args.hf_repo, loaders.MIMI_NAME)
    if args.tokenizer is None:
        args.tokenizer = hf_hub_download(args.hf_repo, loaders.TEXT_TOKENIZER_NAME)
    if args.moshi_weight is None:
        args.moshi_weight = hf_hub_download(args.hf_repo, loaders.MOSHI_NAME)

    tokenizer = sentencepiece.SentencePieceProcessor(args.tokenizer)
    mimi = loaders.get_mimi(args.mimi_weight, device)
    other_mimi = loaders.get_mimi(args.mimi_weight, device)
    lm = loaders.get_moshi_lm(args.moshi_weight, device=device)
    lm.eval()

    student = None
    if args.student_checkpoint:
        ckpt = Path(args.student_checkpoint)
        if ckpt.is_dir():
            ckpt = ckpt / "student.pt"
        pkg = torch.load(ckpt, map_location="cpu")
        student = StudentLM(StudentConfig(**pkg["cfg"]), teacher_lm=lm,
                            device=device, dtype=torch.bfloat16)
        student.load_state_dict(pkg["state_dict"], strict=True)
        student.eval()
        logger.info("student loaded from %s", ckpt)
    return tokenizer, mimi, other_mimi, lm, student


def parity_check(state: ServerState, lm, student: StudentLM, device):
    """Streaming main-backbone vs training forward (real weights)."""
    lm_gen = state.lm_gen
    teacher_delays = list(lm.delays)
    # IMPORTANT: clear any streaming state left over from warmup() BEFORE
    # computing the reference, otherwise the reference forward runs at a
    # nonzero position offset and can never match the fresh streaming run.
    lm_gen.reset_streaming()
    B, T = 1, 16
    gen = torch.Generator().manual_seed(7)
    codes = torch.randint(0, lm.card, (B, lm.num_codebooks, T), generator=gen)
    codes[:, :, -1:] = -1
    codes[:, 0, :8] = torch.randint(0, lm.text_card, (8,), generator=gen)
    codes[:, 0, 8:] = -1
    codes = codes.to(device)          # keep the whole check on one device

    # ---- reference: training-time embedding -> compressor -> transformer ... heads
    initial = lm._get_initial_token().expand(B, -1, -1)
    delayed = _delay_sequence(teacher_delays, codes, initial)
    delayed = torch.cat([initial, delayed], dim=2)
    # Reference = the head-conditioning states the TRAINING path produces,
    # including the 6.25 Hz serving fix when the checkpoint enables it, so this
    # stays an apples-to-apples comparison with the streaming path below.
    s_ref, _, a_ref = student.forward_train(codes, teacher_lm=lm, return_states=True)

    # ---- CONDITIONING QUALITY: student's per-frame states vs the teacher's.
    # This is the single best "will the audio be coherent?" number: the frozen
    # DepFormer only behaves if these match. A well-distilled checkpoint should
    # be well above ~0.9; near 0 means the conditioning is off (choppy audio).
    with torch.no_grad():
        teacher_hidden = lm.forward_embeddings(lm.embed_codes(delayed[:, :, :-1]))[0]
        valid = codes[:, lm.audio_offset] != lm.zero_token_id
        cos = torch.nn.functional.cosine_similarity(
            a_ref.float(), teacher_hidden.float(), dim=-1)
        w = valid.float()
        state_cos = ((cos * w).sum() / w.sum().clamp(min=1e-8)).item()
    logger.info("STUDENT per-frame state cosine vs teacher: %.4f "
                "(>=0.9 good, <0.7 conditioning is off -> choppy audio)", state_cos)

    # ---- streaming: exactly as LMGen.step drives it (input = cache[o-1]) ---
    # NOTE: must start at o=1: the compressor pairs the ARRIVAL sequence, so
    # skipping the first frame shifts the pairing phase by one and every
    # comparison mismatches (that would be a bug in this check, not the model).
    lm_gen.reset_streaming()
    mismatch = 0
    cos_even, cos_odd = [], []
    odd_stale = []
    fix_on = getattr(student.cfg, "streaming_fix", "off") == "augment"
    with torch.no_grad():
        for o in range(1, 17):
            frame = delayed[:, :, o - 1: o]
            t_out, _ = lm_gen._student_forward_codes(frame)
            if o == 1:
                continue          # bootstrap: pair buffered, no state consumed yet
            # With the serving fix the reference already IS the per-slot served
            # state; without it we compare against the state the server reuses.
            want = a_ref[:, o - 1] if fix_on else a_ref[:, (o - 1 if o % 2 == 0 else o - 2)]
            # bf16 rounding makes max|d| ~0.06 on ~1.2-RMS vectors, so compare
            # with a relative/cosine criterion instead of max-abs tolerance.
            got = t_out.squeeze().float()
            ref = want.squeeze().float()
            cos = torch.nn.functional.cosine_similarity(got, ref, dim=-1).mean().item()
            rel = ((got - ref).abs().mean() / ref.abs().mean().clamp(min=1e-6)).item()
            ok = (cos > 0.999) and (rel < 0.03)
            if not ok:
                mismatch += 1
                idx = o - 1 if o % 2 == 0 else o - 2
                logger.warning("parity MISMATCH at step %d (vs a_ref[:, %d], "
                               "cos=%.4f rel=%.4f)", o, idx, cos, rel)
            # how far is the SERVED state from the teacher's state at the
            # position this step is actually predicting? (even steps == exact
            # slot, odd steps == the 1-pair-stale slot by construction)
            served = t_out.squeeze().float()
            exact = teacher_hidden[:, o - 1].squeeze().float()
            stale = teacher_hidden[:, max(0, o - 2)].squeeze().float()
            c_exact = torch.nn.functional.cosine_similarity(
                served, exact, dim=-1).mean().item()
            c_stale = torch.nn.functional.cosine_similarity(
                served, stale, dim=-1).mean().item()
            (cos_even if o % 2 == 0 else cos_odd).append(c_exact)
            if o % 2 == 1:
                odd_stale.append(c_stale)
    if mismatch == 0:
        logger.info("PARITY OK: streaming states == training forward on real weights "
                    "(even steps exact, odd steps 1-frame reuse)")
    else:
        logger.error("PARITY FAILED at %d steps - the serving backbone does not "
                     "match the training path; investigate before anything else",
                     mismatch)
    if cos_even:
        logger.info("streaming-vs-teacher cosine: even steps %.4f | odd steps "
                    "vs exact slot %.4f (staleness cost), vs their own stale slot "
                    "%.4f (checkpoint fidelity)",
                    sum(cos_even) / len(cos_even),
                    (sum(cos_odd) / len(cos_odd)) if cos_odd else float("nan"),
                    (sum(odd_stale) / len(odd_stale)) if odd_stale else float("nan"))
    return mismatch == 0


def generate(state: ServerState, tokenizer, mimi, device, args):
    """Drive LMGen exactly like the server's opus_loop, minus websocket/opus."""
    lm_gen = state.lm_gen
    frame_size = state.frame_size

    lm_gen.reset_streaming()
    lm_gen.text_prompt_tokens = tokenizer.encode(
        wrap_with_system_tags(args.text_prompt)) if args.text_prompt else None
    lm_gen.step_system_prompts(mimi)      # voice prompt only if set on lm_gen

    silent = torch.zeros(1, 1, frame_size, dtype=torch.float32, device=device)
    codes = mimi.encode(silent)           # SILENT user stream frames
    n_frames = int(args.seconds * mimi.frame_rate)

    pcm_out, transcript = [], []
    for i in range(n_frames):
        for c in range(codes.shape[-1]):
            tokens = lm_gen.step(codes[:, :, c: c + 1])
            if tokens is None:
                continue
            out = mimi.decode(tokens[:, 1:9]).cpu()
            pcm_out.append(out[0, 0].numpy())
            t = tokens[0, 0, 0].item()
            if t not in (0, 3):
                transcript.append(tokenizer.id_to_piece(t).replace("▁", " "))

    wav = np.concatenate(pcm_out) if pcm_out else np.zeros(0, dtype=np.float32)
    out_path = Path(args.out or ("out_student" if state.lm_gen.__class__.__name__ == "StudentLMGen" else "out_teacher") + ".wav")
    _write_wav(out_path, mimi.sample_rate, wav)
    text = "".join(transcript)
    Path(out_path).with_suffix(".txt").write_text(text, encoding="utf8")
    logger.info("generated %s (%d samples) | transcript: %s",
                out_path, len(wav), repr(text[:400]))


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", type=str)
    ap.add_argument("--moshi-weight", type=str)
    ap.add_argument("--mimi-weight", type=str)
    ap.add_argument("--hf-repo", type=str, default=loaders.DEFAULT_REPO)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--student-checkpoint", type=str)
    ap.add_argument("--text-prompt", type=str,
                    default="Your name is Jane. You enjoy good conversation.")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--out", type=str)
    args = ap.parse_args()

    device = torch_auto_device(args.device)
    seed_all(42424242)
    tokenizer, mimi, other_mimi, lm, student = load_stack(args, device)
    mode = "student" if student is not None else "teacher"
    logger.info("running in %s mode", mode)

    state = ServerState(mimi=mimi, other_mimi=other_mimi, text_tokenizer=tokenizer,
                        lm=lm, device=device, student=student)
    state.warmup()

    if student is not None:
        parity_check(state, lm, student, device)

    generate(state, tokenizer, mimi, device, args)
    logger.info("done (%s mode). Listen to %s and re-run WITHOUT "
                "--student-checkpoint for the teacher A/B.", mode, args.out or "out_*.wav")


if __name__ == "__main__":
    with torch.no_grad():
        main()
