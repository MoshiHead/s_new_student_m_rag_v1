"""
rag_cache.py — pre-tokenize the RAG v3 corpus into a training cache.
====================================================================

Why a cache: the Stage-A trainer needs (a) Mimi encoding, (b) speaker-masked
stereo channels, (c) RAG splicing, (d) turn-end truncation.  All four are
deterministic and CPU/GPU-bound, so doing them once turns a ~1.5 h epoch into a
~15 min epoch and makes ablation runs cheap.

Output (one ``.pt``):

    {"entries": [ {codes: int16 [17,T], ref_ids: int64 [n], turns: [[kind,s,e]],
                   plen: int, kind: str, conv_id: str, has_ref: bool}, ... ],
     "meta": {"tokenizer": ..., "duration_sec": 48, "n_slots": 16,
              "text_card": 32003, "tail_frames": 10, "n": ...}}

``codes`` are truncated 10 frames (0.8 s) after the last turn end, so the
student never sees the teacher/TTS free-run tail.  Truncation is exact: the
transformer is causal and the depformer has a fixed 1-frame delay, so frames
past the cut cannot influence frames before it.

Usage::

    python -m moshi_local.moshi_distill.rag_cache \
        --audio-dir /workspace/data/rag_v3/audio \
        --snap /workspace/.cache/huggingface/hub/models--nvidia--personaplex-7b-v1/snapshots/fdaf4090a61cb315c138a1faee287ffd6c716309 \
        --out /workspace/personaplex/rag_v3_tokens.pt \
        --duration-sec 48 --tail-sec 0.8
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import logging
import os
import time

import numpy as np
import torch

from .distill.data import DistillInterleavedTokenizer
from .interleaver import Interleaver

logger = logging.getLogger(__name__)

# Grounded / abstain classes get a spliced reference block; the two RAG-free
# classes never do (their sidecars carry no reference).
ALL_KINDS = ("grounded_real", "grounded_cf", "grounded_distract",
             "abstain", "no_rag_qa", "regular_chat")


def conv_id_of(name: str) -> str:
    """Strip the class suffix so train/eval splits cannot leak a conversation."""
    base = os.path.basename(name)
    for suf in (".sidecar.json", ".new.combined.json.aligned",
                ".new.combined.json", ".wav", ".json.aligned"):
        if base.endswith(suf):
            base = base[: -len(suf)]
    for suf in ("_cf", "_dx", "_ab", "_chat", "_qa"):
        if base.endswith(suf):
            base = base[: -len(suf)]
    return base


def split_of(conv_id: str, eval_frac: float = 0.06) -> str:
    h = hashlib.sha1(conv_id.encode("utf-8")).hexdigest()
    return "eval" if (int(h[:8], 16) / 0xFFFFFFFF) < eval_frac else "train"


def _read_wav(path: str, target_sr: int) -> np.ndarray:
    import sphn
    # NB: sphn.read returns (data[C, T], sample_rate) — the opposite of the
    # librosa/soundfile convention.
    wav, sr = sphn.read(path)
    if sr != target_sr:
        wav = sphn.resample(wav, src_sample_rate=sr, dst_sample_rate=target_sr)
    if wav.ndim == 1:
        wav = wav[None, :]
    return np.ascontiguousarray(wav.astype(np.float32))


def _turn_end_frame(turns) -> int:
    return max((int(t.end_frame) for t in turns), default=-1)


def build(args) -> dict:
    from .models import loaders

    device = torch.device(args.device)
    mimi = loaders.get_mimi(args.mimi_weight, device)
    mimi.eval()
    import sentencepiece
    tokenizer = sentencepiece.SentencePieceProcessor(model_file=args.tokenizer)

    interleaver = Interleaver(
        tokenizer=tokenizer, audio_frame_rate=mimi.frame_rate,
        text_padding=3, end_of_text_padding=0, zero_padding=-1,
        keep_main_only=True, use_bos_eos=True,
        main_speaker_label="ASSISTANT", device=str(device))

    dtok = DistillInterleavedTokenizer(
        mimi=mimi, interleaver=interleaver, duration_sec=args.duration_sec,
        target_n_audio_codebooks=16, require_sidecar=True,
        use_rag=True, n_ref_slots=args.n_slots,
        max_ref_source_tokens=args.max_ref_tokens,
        ref_tokens_per_slot=args.ref_tokens_per_slot)

    sidecars = sorted(glob.glob(os.path.join(args.audio_dir, "*.sidecar.json")))
    if args.limit:
        sidecars = sidecars[: args.limit]
    logger.info("found %d sidecars", len(sidecars))

    tail = int(round(args.tail_sec * mimi.frame_rate))
    entries, skipped, seen_kind = [], {}, {}
    t0 = time.time()

    for i, sc in enumerate(sidecars):
        wav_path = sc[: -len(".sidecar.json")] + ".wav"
        if not os.path.exists(wav_path):
            skipped["no_wav"] = skipped.get("no_wav", 0) + 1
            continue
        try:
            with open(sc) as f:
                side = json.load(f)
            kind = (side.get("metadata") or {}).get("rag_kind")
            if args.kinds and kind not in args.kinds:
                skipped["kind_filter"] = skipped.get("kind_filter", 0) + 1
                continue
            wav = _read_wav(wav_path, mimi.sample_rate)
            sample = dtok(wav, 0.0, wav_path)
        except Exception as e:  # noqa: BLE001 - one bad file must not kill 14k
            skipped["error"] = skipped.get("error", 0) + 1
            if skipped["error"] <= 5:
                logger.warning("skip %s: %s", os.path.basename(wav_path), e)
            continue

        meta = sample.segment_meta
        if meta is None or not meta.turns:
            skipped["no_meta"] = skipped.get("no_meta", 0) + 1
            continue
        end = _turn_end_frame(meta.turns)
        if end <= 0:
            skipped["no_turn_end"] = skipped.get("no_turn_end", 0) + 1
            continue

        T = sample.codes.shape[-1]
        keep = min(T, end + tail)
        # [1,17,T] -> [17,T]: the batch dim is re-added at collate time
        codes = sample.codes[0, :, :keep].to(torch.int16).contiguous()

        ref = sample.reference_text if sample.reference_text else ""
        ref_ids = (torch.tensor(tokenizer.encode(ref)[: args.max_ref_tokens],
                                dtype=torch.long)
                   if ref else torch.zeros(0, dtype=torch.long))

        plen = 0
        try:
            from .backbone_distill import forced_prompt_len
            plen = int(forced_prompt_len(codes.long()[None], -1))
        except Exception:  # noqa: BLE001
            plen = 0
        if plen <= 0:
            # fallback: the persona block is the only leading run with real text
            # and silence audio; if detection failed, assume the standard block.
            skipped["plen_zero"] = skipped.get("plen_zero", 0) + 1

        turns = [(t.kind, int(t.start_frame), int(min(t.end_frame, keep)))
                 for t in meta.turns if t.start_frame < keep]
        entries.append({
            "codes": codes,
            "ref_ids": ref_ids,
            "turns": turns,
            "plen": plen,
            "kind": kind,
            "conv_id": conv_id_of(os.path.basename(sc)),
            "has_ref": bool(ref_ids.numel()),
        })
        seen_kind[kind] = seen_kind.get(kind, 0) + 1

        if (i + 1) % 200 == 0:
            el = time.time() - t0
            logger.info("%d/%d  kept=%d  %.1f files/s  eta=%.0f min",
                        i + 1, len(sidecars), len(entries),
                        (i + 1) / max(el, 1e-9), (len(sidecars) - i - 1) / max((i + 1) / max(el, 1e-9), 1e-9) / 60)

    frame_counts = [int(e["codes"].shape[-1]) for e in entries]
    plens = [int(e["plen"]) for e in entries]
    meta = {
        "tokenizer": args.tokenizer,
        "duration_sec": args.duration_sec,
        "n_slots": args.n_slots,
        "tail_frames": tail,
        "kinds": seen_kind,
        "skipped": skipped,
        "n": len(entries),
        "frames_min": min(frame_counts) if frame_counts else 0,
        "frames_p50": int(np.median(frame_counts)) if frame_counts else 0,
        "frames_max": max(frame_counts) if frame_counts else 0,
        "plen_min": min(plens) if plens else 0,
        "plen_max": max(plens) if plens else 0,
    }
    logger.info("meta: %s", json.dumps(meta))
    return {"entries": entries, "meta": meta}


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--snap", required=True,
                    help="HF snapshot dir with mimi + 32k tokenizer assets")
    ap.add_argument("--tokenizer", default=None,
                    help="32 003-piece tokenizer.model (default: <snap>/tokenizer.model)")
    ap.add_argument("--mimi-weight", default=None,
                    help="default: <snap>/tokenizer-e351c8d8-checkpoint125.safetensors")
    ap.add_argument("--out", required=True)
    ap.add_argument("--duration-sec", type=float, default=48.0)
    ap.add_argument("--tail-sec", type=float, default=0.8)
    ap.add_argument("--n-slots", type=int, default=16)
    ap.add_argument("--max-ref-tokens", type=int, default=256)
    ap.add_argument("--ref-tokens-per-slot", type=int, default=4)
    ap.add_argument("--kinds", nargs="*", default=list(ALL_KINDS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.tokenizer is None:
        # The 32 003-piece RAG tokenizer lives with the source tree, not the
        # upstream snapshot (which only has the 32k one).
        args.tokenizer = "/workspace/personaplex/moshi_local/tokenizer.model"
    if args.mimi_weight is None:
        args.mimi_weight = os.path.join(
            args.snap, "tokenizer-e351c8d8-checkpoint125.safetensors")

    pkg = build(args)
    torch.save(pkg, args.out)
    logger.info("wrote %s (%.1f MB)", args.out,
                os.path.getsize(args.out) / 1e6)


if __name__ == "__main__":
    main()
