"""
build_sidecars.py
=================
Converts your existing per-audio JSON files into the extended sidecar format
that InterleavedTokenizer expects, and writes the JSONL manifest.

Input (what you already have — one JSON per WAV):
-------------------------------------------------
    {
      "alignments": [
        ["Hello.", [0.0, 1.7], "MODEL"],
        ["I",      [2.54, 2.8], "USER"],
        ...
      ],
      "metadata": {                        # may also be flattened at top level
        "id":        "-415623028232955",
        "system_prompt": "...",
        "Conversation": [...],              # NOTE: key may be "Conversation" or "conversation"
        "finance_turn_index": 3,            # or "qa_turn_index" — 0 is a valid index!
        "reference": "The seventh season of Once Upon a Time ...",
        "raw_response": "[LEAD] I actually know a bit about this one. [FILLER] Hold on, let me find that information for you. [BODY] The seventh season ..."
      }
    }

Output (written to a NEW file — the input JSON is never modified):
--------------------------------------------------------------------
    <stem>.sidecar.json   (default; override with --output-suffix)
    {
      "alignments": [...],                  # unchanged, serialised as-is
      "duration": 18.4,
      "segment_meta": {
        "turns": [
          {"kind": "regular", "start_sec": ..., "end_sec": ...},
          {"kind": "lead",    "start_sec": ..., "end_sec": ...},
          {"kind": "lookup",  "start_sec": ..., "end_sec": ...},
          {"kind": "filler",  "start_sec": ..., "end_sec": ...},
          {"kind": "ref",     "start_sec": ..., "end_sec": ...},
          {"kind": "body",    "start_sec": ..., "end_sec": ...},
          ...
        ]
      },
      "text_conditions": {
        "streaming_sum_path": "-415623028232955_ref.pt",
        "reference": "The seventh season of Once Upon a Time ..."
      },
      "metadata": { ... }   ← preserved unchanged
    }

    data/train.jsonl:
    {"path": "/abs/path/file.wav", "duration": 18.4}

Speaker labels
--------------
    "MODEL" / "ASSISTANT" → Moshi  (main speaker)
    "USER"                → user

Usage
-----
    python build_sidecars.py \
        --audio-dir /data/audio \
        --manifest  data/train.jsonl

    # Dry-run: print what would change, write nothing
    python build_sidecars.py --audio-dir /data/audio --dry-run

    # Overwrite the original JSON in place (old behaviour, NOT recommended)
    python build_sidecars.py --audio-dir /data/audio --in-place
"""

from __future__ import annotations

import argparse
import json
import re
import wave
from pathlib import Path
from typing import Optional

from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Speaker label constants — match your JSON. Some pipeline versions emit
# "MODEL", others "ASSISTANT"; both are treated as the main speaker.
SPEAKER_MODEL_LABELS = {"MODEL", "ASSISTANT", "MOSHI", "SPEAKER_MAIN"}
SPEAKER_USER = "USER"

Alignment = tuple[str, tuple[float, float], str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_wav_duration(wav_path: Path) -> float:
    with wave.open(str(wav_path), "r") as w:
        return w.getnframes() / w.getframerate()


def normalise(text: str) -> list[str]:
    """Lowercase, strip punctuation → list of words for fuzzy matching."""
    return re.sub(r"[^a-z0-9\s]", "", text.lower()).split()


def is_model_speaker(spk: str) -> bool:
    return str(spk).strip().upper() in SPEAKER_MODEL_LABELS


def find_boundary(
    alignments: list[Alignment],
    target_text: str,
    search_from: int = 0,
    search_until: int | None = None,
) -> tuple[int, int]:
    """
    Find (start_idx, end_idx) into alignments that best match target_text
    among MODEL-speaker words, starting search at search_from.

    Uses a sliding window with word-overlap scoring so minor transcription
    differences (punctuation, capitalisation) don't break the match.

    Raises ValueError if no match found.
    """
    target_words = normalise(target_text)
    if not target_words:
        raise ValueError(f"empty target_text: {target_text!r}")

    if search_until is None:
        search_until = len(alignments)

    # Flatten words while retaining their source alignment index. This handles
    # alignments containing more than one word instead of silently keeping only
    # the first word.
    filtered = []
    for i, (w, _, spk) in enumerate(alignments):
        if i < search_from or i >= search_until or not is_model_speaker(spk):
            continue
        nw = normalise(w)
        filtered.extend((i, word) for word in nw)

    window = len(target_words)
    if window == 0 or len(filtered) < window:
        raise ValueError(f"No MODEL alignments after idx {search_from}")

    best_score = -1
    best_start = search_from
    best_end = search_from

    for i in range(len(filtered) - window + 1):
        chunk = [w for _, w in filtered[i: i + window]]
        score = sum(1 for a, b in zip(chunk, target_words) if a == b)
        if score > best_score:
            best_score = score
            best_start = filtered[i][0]
            best_end = filtered[i + window - 1][0]

    if best_score <= 0 or best_score / window < 0.4:
        raise ValueError(f"No alignment match for text={target_text[:60]!r}")

    return best_start, best_end


def _safe_find(alignments, text, start, end):
    """find_boundary that returns None instead of raising (lead/filler are
    optional and the body is derived structurally, so a missed match must not
    abort the record)."""
    if not text:
        return None
    try:
        return find_boundary(alignments, text, start, end)
    except ValueError:
        return None


def parse_raw_response(raw: str) -> tuple[str, str, str]:
    """
    Extract LEAD, FILLER, BODY segments from raw_response.
    Expects tags like [LEAD] ... [FILLER] ... [BODY] ...
    Returns (lead, filler, body) stripped.
    """
    lead_match = re.search(r'\[LEAD\](.*?)(?=\[FILLER\]|$)', raw, re.DOTALL | re.IGNORECASE)
    filler_match = re.search(r'\[FILLER\](.*?)(?=\[BODY\]|$)', raw, re.DOTALL | re.IGNORECASE)
    body_match = re.search(r'\[BODY\](.*?)$', raw, re.DOTALL | re.IGNORECASE)

    lead = lead_match.group(1).strip() if lead_match else ""
    filler = filler_match.group(1).strip() if filler_match else ""
    body = body_match.group(1).strip() if body_match else ""

    return lead, filler, body


def _get_case_insensitive(d: dict, *keys, default=None):
    """Try each key in order (as given, and as-is only — no fuzzy matching).
    First present (non-None) value wins. Fixes the 'Conversation' vs
    'conversation' key-casing mismatch that was silently breaking turn
    resolution."""
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _get_turn_index(meta: dict) -> Optional[int]:
    """finance_turn_index / qa_turn_index — 0 is a VALID index, so this must
    not use `a or b` (which treats 0 as falsy and silently falls through)."""
    fti = meta.get("finance_turn_index")
    if fti is not None:
        return fti
    return meta.get("qa_turn_index")


def resolve_rag_model_turn(meta: dict, model_turns: list[tuple[int, int]]) -> Optional[int]:
    """finance_turn_index/qa_turn_index indexes the full conversation[] list
    (both speakers) and points at the USER turn whose FOLLOWING assistant
    reply is the RAG-grounded turn — not at the assistant turn itself,
    and not at a MODEL-only turn count. Translate accordingly.

    Returns the index into model_turns (MODEL-only turn count) that
    corresponds to the RAG turn, or None if it can't be resolved.

    Every failure path logs a warning — resolution should never fail
    silently, since a silent failure here just quietly drops the ref
    segment downstream.
    """
    conv = _get_case_insensitive(meta, "conversation", "Conversation")
    idx = _get_turn_index(meta)

    if conv is None:
        logger.warning("resolve_rag_model_turn: no 'conversation'/'Conversation' key in metadata")
        return None
    if idx is None:
        logger.warning("resolve_rag_model_turn: no 'finance_turn_index'/'qa_turn_index' in metadata")
        return None
    if not (0 <= idx < len(conv)):
        logger.warning(f"turn_index {idx} out of range for conversation len {len(conv)}")
        return None
    if str(conv[idx].get("role", "")).lower() != "user":
        logger.warning(f"turn_index {idx} points at role={conv[idx].get('role')!r}, expected 'user'")
        return None

    rag_conv_idx = idx + 1
    if rag_conv_idx >= len(conv) or str(conv[rag_conv_idx].get("role", "")).lower() != "assistant":
        logger.warning(f"conversation[{rag_conv_idx}] is not an assistant turn following the question")
        return None

    # Count assistant-role entries up to and including rag_conv_idx —
    # that count (0-indexed) is the corresponding entry in model_turns,
    # since enumerate_model_turns walks only MODEL-speaker runs in the
    # same left-to-right order as assistant entries appear in conversation.
    model_turn_num = sum(
        1 for c in conv[: rag_conv_idx + 1]
        if str(c.get("role", "")).lower() == "assistant"
    ) - 1

    if not (0 <= model_turn_num < len(model_turns)):
        logger.warning(
            f"resolved model_turn_num={model_turn_num} out of range for "
            f"{len(model_turns)} detected MODEL turns — alignment/conversation "
            f"turn-count mismatch (e.g. a MODEL turn got merged/split differently "
            f"by enumerate_model_turns than assistant entries suggest)"
        )
        return None

    return model_turn_num


# ---------------------------------------------------------------------------
# Core: extend one JSON sidecar
# ---------------------------------------------------------------------------

def enumerate_model_turns(alignments: list[Alignment]) -> list[tuple[int, int]]:
    """Contiguous MODEL-speaker runs, bounded by any non-MODEL speech.
    Returns list of (start_idx, end_idx) into alignments, one per turn,
    in conversation order."""
    turns = []
    run_start = None
    for i, (_, _, spk) in enumerate(alignments):
        if is_model_speaker(spk) and run_start is None:
            run_start = i
        elif not is_model_speaker(spk) and run_start is not None:
            turns.append((run_start, i - 1))
            run_start = None
    if run_start is not None:
        turns.append((run_start, len(alignments) - 1))
    return turns


def build_sidecar(wav_path, data, ref_duration_sec=0.16, lookup_duration_sec=0.16):
    # Merge nested and flattened metadata. The sidecar contract needs the
    # resolved reference in text_conditions, regardless of where the source
    # JSON stored it.
    SIDECAR_KEYS = {"alignments", "duration", "segment_meta", "text_conditions", "metadata"}
    meta = {k: v for k, v in data.items() if k not in SIDECAR_KEYS}
    nested_meta = data.get("metadata") or {}
    if isinstance(nested_meta, dict):
        meta.update(nested_meta)

    source_conditions = data.get("text_conditions") or {}
    text_conditions = dict(source_conditions) if isinstance(source_conditions, dict) else {}
    # The current trainer computes reference vectors from text with its local
    # resampler; stale precomputed streaming-sum paths belong to the removed
    # adapter pipeline and must not be advertised in new sidecars.
    text_conditions.pop("streaming_sum_path", None)
    reference = (
        text_conditions.get("reference")
        or meta.get("reference")
        or data.get("reference")
        or ""
    )
    if reference:
        text_conditions["reference"] = reference

    rag_idx_raw = _get_turn_index(meta)

    # Only grounded / abstain turns carry a lookup+ref+body RAG structure.
    # `no_rag_qa` (answered from parametric memory) and `regular_chat`
    # (pure small-talk) must be emitted as plain `regular` turns so the
    # trigger head never learns to fire on them. `rag_kind` is written by
    # build_rag_v2.py; legacy records without it keep the old behaviour.
    rag_kind = str(meta.get("rag_kind") or "").strip().lower()
    is_grounded_turn = bool(reference) and rag_kind not in {
        "no_rag", "no_rag_qa", "regular", "regular_chat",
    }

    raw = data["alignments"]
    alignments = [(e[0], (float(e[1][0]), float(e[1][1])), e[2]) for e in raw]
    model_turns = enumerate_model_turns(alignments)
    rag_turn_num = (
        resolve_rag_model_turn(meta, model_turns) if is_grounded_turn else None
    )
    if is_grounded_turn and rag_turn_num is None and model_turns:
        # Fallback: the grounded QA reply is the LAST assistant turn in these
        # datasets. A turn-count mismatch (an extra/missing ASR segment) would
        # otherwise drop the whole record.
        rag_turn_num = len(model_turns) - 1

    turn_segments = []
    matched_rag_turn = False

    for turn_num, (t_start, t_end) in enumerate(model_turns):
        if turn_num == rag_turn_num:
            matched_rag_turn = True
            raw_response = meta.get("raw_response", "")
            if not raw_response:
                raise ValueError(f"turn {turn_num} marked as RAG turn but no raw_response")
            lead_text, filler_text, body_text = parse_raw_response(raw_response)
            if not body_text:
                raise ValueError(f"could not parse body from raw_response for turn {turn_num}")

            # Bound every match to this assistant turn. Lead/filler are
            # optional; the body is the REMAINDER of the turn after them, so a
            # mis-transcribed number/name in the body (Whisper often renders
            # "Jacob W." as "J.K.W.", dates as words, etc.) can never drop the
            # record. This is exact because the turn is lead + filler + body in
            # order.
            cursor = t_start
            lead_match = _safe_find(alignments, lead_text, cursor, t_end + 1)
            if lead_match is not None:
                cursor = lead_match[1] + 1
            filler_match = _safe_find(alignments, filler_text, cursor, t_end + 1)
            if filler_match is not None:
                cursor = filler_match[1] + 1
            body_s = min(cursor, t_end)
            body_e = t_end

            body_start_sec = alignments[body_s][1][0]
            turn_start_sec = alignments[t_start][1][0]   # start of the whole model turn
            turn_segments.append({
                "kind": "lookup",
                "start_sec": turn_start_sec,
                "end_sec": turn_start_sec + lookup_duration_sec,
            })
            if lead_match is not None:
                lead_s, lead_e = lead_match
                turn_segments.append({
                    "kind": "lead",
                    "start_sec": max(alignments[lead_s][1][0], turn_start_sec + lookup_duration_sec),
                    "end_sec": alignments[lead_e][1][1],
                })
            if filler_match is not None:
                fill_s, fill_e = filler_match
                turn_segments.append({
                    "kind": "filler",
                    "start_sec": alignments[fill_s][1][0],
                    "end_sec": alignments[fill_e][1][1],
                })
            # The actual ref block is synthesized by InterleavedTokenizer
            # immediately after <lookup>; do not emit a second placeholder ref
            # span here.
            turn_segments.append({
                "kind": "body",
                "start_sec": body_start_sec,
                "end_sec": alignments[body_e][1][1],
            })
        else:
            turn_segments.append({
                "kind": "regular",
                "start_sec": alignments[t_start][1][0],
                "end_sec": alignments[t_end][1][1],
            })

    # Best-effort: if no MODEL turn could be matched, emit the recording as
    # plain `regular` instead of dropping it (the audio/transcript are still
    # valid distillation data; it just contributes no grounding signal).
    if reference and rag_idx_raw is not None and not matched_rag_turn:
        logger.warning(
            "reference set and turn_index=%r present but no MODEL turn matched "
            "(rag_turn_num=%r) — emitting as regular", rag_idx_raw, rag_turn_num)

    # `alignments` is serialised UNCHANGED — no <ref>/<lookup> words added.
    serialised = [[w, [s, e], spk] for w, (s, e), spk in alignments]
    duration = get_wav_duration(wav_path)

    return {
        "alignments": serialised,
        "duration": duration,
        "segment_meta": {"turns": turn_segments},
        "text_conditions": text_conditions,
        "metadata": meta,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--audio-dir", required=True,
                    help="Directory containing .wav + .json pairs")
    ap.add_argument("--manifest", default="data/train.jsonl",
                    help="Output JSONL manifest (default: data/train.jsonl)")
    ap.add_argument("--output-suffix", default=".sidecar.json",
                    help="Suffix for the new output file, written alongside the "
                         "input WAV (default: .sidecar.json). Ignored if --in-place.")
    ap.add_argument("--in-place", action="store_true",
                    help="DANGEROUS: overwrite the original *.new.combined.json "
                         "instead of writing a new file. Off by default so you can "
                         "always recover the source data.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would change, write nothing")
    args = ap.parse_args()

    audio_dir = Path(args.audio_dir)
    assert audio_dir.exists(), f"Not found: {audio_dir}"

    wav_files = sorted(audio_dir.glob("*.wav"))
    print(f"[sidecars] {len(wav_files)} WAV files in {audio_dir}")

    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)

    stats = {"written": 0, "skipped": 0, "errors": {}}
    manifest_lines: list[str] = []

    for wav_path in tqdm(wav_files, desc="sidecars", unit="file"):
        json_path = wav_path.with_suffix(".new.combined.json")

        if not json_path.exists():
            tqdm.write(f"  [SKIP] {wav_path.stem}: no JSON")
            stats["skipped"] += 1
            continue

        try:
            with open(json_path) as f:
                data = json.load(f)

            if "alignments" not in data:
                raise KeyError("JSON has no 'alignments' key")

            sidecar = build_sidecar(wav_path, data)

            out_path = json_path if args.in_place else wav_path.with_suffix(args.output_suffix)

            if not args.dry_run:
                with open(out_path, "w") as f:
                    json.dump(sidecar, f, indent=2)

            manifest_lines.append(json.dumps({
                "path": str(wav_path.absolute()),
                "duration": sidecar["duration"],
            }))
            stats["written"] += 1

            if args.dry_run:
                turns = sidecar["segment_meta"]["turns"]
                kinds = [t["kind"] for t in turns]
                ref_turn = next((t for t in turns if t["kind"] == "ref"), None)
                if ref_turn:
                    tqdm.write(
                        f"  [DRY] {wav_path.stem}: "
                        f"ref=[{ref_turn['start_sec']:.2f},{ref_turn['end_sec']:.2f}]s "
                        f"-> would write {out_path.name}"
                    )
                else:
                    tqdm.write(f"  [DRY] {wav_path.stem}: turns={kinds} (no ref segment)")

        except Exception as e:
            reason = f"{type(e).__name__}: {str(e)[:100]}"
            stats["errors"][reason] = stats["errors"].get(reason, 0) + 1
            tqdm.write(f"  [ERR] {wav_path.stem}: {reason}")
            stats["skipped"] += 1

    if not args.dry_run:
        with open(args.manifest, "w") as f:
            f.write("\n".join(manifest_lines) + "\n")

    print(f"\n── complete ────────────────────────────────────────────")
    print(f"  written  : {stats['written']}")
    print(f"  skipped  : {stats['skipped']}")
    if stats["errors"]:
        print("  errors:")
        for r, c in sorted(stats["errors"].items(), key=lambda x: -x[1]):
            print(f"    {c:4d}  {r}")
    if not args.dry_run:
        print(f"  manifest → {args.manifest}")
        if not args.in_place:
            print(f"  sidecars written as *{args.output_suffix} next to each WAV "
                  f"(originals untouched — pass --in-place to overwrite instead)")


if __name__ == "__main__":
    main()
