"""Query-conditioned extractive compression of the retrieved passage.

The RAG student reads the reference out of a fixed prefill block of ``K``
frames (``K = clamp(ceil(n_tokens / 4), 1, 16)``).  ``K`` tokens are only
~20-25% of a typical passage, so *which* ``K`` matters: the block has to carry
the sentence that answers the question, and the first ``K`` tokens often do
not (SQuAD answers sit anywhere in the paragraph).

This script scores every ``K``-token window of the passage against the user's
question with the same MiniLM encoder the retriever uses, and stores the best
window per conversation::

    {conv_id: [token ids]}    ->  rag_v3_refsel.pt

The serving path applies the identical function (query transcript + retrieved
passage -> best window), so train and serve see the same compression.

Usage (CPU is fine, ~5 min for the 9.5 k ref-bearing entries)::

    python -m moshi_local.moshi_distill.make_refsel \
        --tokens /workspace/personaplex/rag_v3_tokens.pt \
        --sidecars /workspace/data/rag_v3/audio \
        --out /workspace/personaplex/rag_v3_refsel.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger("refsel")

ST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# window scoring (shared with the server)
# ---------------------------------------------------------------------------

def window_starts(n: int, k: int, stride: int | None = None) -> list[int]:
    if n <= k:
        return [0]
    stride = stride or max(1, k // 4)
    starts = list(range(0, n - k + 1, stride))
    if starts[-1] != n - k:                 # always include the tail
        starts.append(n - k)
    return starts


def pick_window(scores: np.ndarray) -> int:
    """Index into :func:`window_starts` of the best-scoring window."""
    return int(np.argmax(scores))


def compress_passage(ids: list[int], k: int, query_vec: np.ndarray,
                     embed, sp) -> list[int]:
    """Best ``k``-token window of ``ids`` for ``query_vec`` (serving side)."""
    ids = [int(i) for i in ids]
    n = len(ids)
    if n <= k:
        return ids
    starts = window_starts(n, k)
    texts = [sp.decode(ids[s:s + k]) for s in starts]
    mat = embed(texts)
    return ids[starts[pick_window(mat @ query_vec)]:][:k]


# ---------------------------------------------------------------------------
# building the table
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--sidecars", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer",
                    default="/workspace/personaplex/moshi_local/tokenizer.model")
    ap.add_argument("--model", default=ST_MODEL)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stats", default="")
    args = ap.parse_args()

    torch.set_num_threads(int(os.environ.get("RAG_NUM_THREADS", "8")))
    import sentencepiece as spm
    from sentence_transformers import SentenceTransformer

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    st = SentenceTransformer(args.model, device=args.device)
    st.eval()

    def embed(texts: list[str]) -> np.ndarray:
        v = st.encode(texts, batch_size=64, convert_to_numpy=True,
                      normalize_embeddings=True, show_progress_bar=False)
        return v.astype(np.float32)

    pkg = torch.load(args.tokens, map_location="cpu", weights_only=False)
    entries = pkg["entries"]
    logger.info("cache: %d entries", len(entries))

    jobs = []                                  # (conv_id, ids, k, query)
    for e in entries:
        if not e.get("has_ref"):
            continue
        sl = [t for t in e["turns"] if t[0] == "ref_slot"]
        if not sl:
            continue
        k = int(sl[0][2] - sl[0][1])
        ids = [int(i) for i in e["ref_ids"].tolist() if int(i) > 0]
        if k <= 0 or not ids:
            continue
        jobs.append((int(e.get("conv_id", 0)), ids, k))
        if args.limit and len(jobs) >= args.limit:
            break
    logger.info("ref-bearing entries: %d", len(jobs))

    # one pass over the sidecars for the question text (synthesised classes get
    # a suffix: `<conv>_cf.sidecar.json`, `<conv>_dx.sidecar.json`, ...)
    queries, missing = [], 0
    for conv_id, _ids, _k in jobs:
        q = ""
        for suf in ("", "_cf", "_dx", "_abstain", "_ab"):
            p = Path(args.sidecars) / f"{conv_id}{suf}.sidecar.json"
            if not p.exists():
                continue
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                q = ((d.get("metadata") or {}).get("user") or "").strip()
            except Exception as exc:            # unreadable sidecar
                logger.debug("sidecar %s: %s", p, exc)
            if q:
                break
        if not q:
            missing += 1
        queries.append(q)
    logger.info("queries found: %d / %d (missing %d -> first-k fallback)",
                sum(bool(q) for q in queries), len(queries), missing)

    t0 = time.time()
    uniq = sorted({q for q in queries if q})
    qmat = {q: i for i, q in enumerate(uniq)}
    qvec = embed(uniq) if uniq else np.zeros((0, 384), dtype=np.float32)
    logger.info("encoded %d unique questions in %.0fs", len(uniq), time.time() - t0)

    out: dict[int, list[int]] = {}
    n_fallback = n_windows = 0
    for i, (conv_id, ids, k) in enumerate(jobs):
        q = queries[i]
        if not q or len(ids) <= k:
            out[conv_id] = ids[:k]
            n_fallback += 1
            continue
        starts = window_starts(len(ids), k)
        vecs = embed([sp.decode(ids[s:s + k]) for s in starts])
        best = starts[pick_window(vecs @ qvec[qmat[q]])]
        out[conv_id] = ids[best:best + k]
        n_windows += len(starts)
        if (i + 1) % 500 == 0:
            logger.info("  %d/%d (%.0fs, %.1f windows/entry)",
                        i + 1, len(jobs), time.time() - t0,
                        n_windows / (i + 1))
        if args.limit and i + 1 >= args.limit:
            break

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    logger.info("wrote %s | %d passages (fallback %d) in %.0fs",
                args.out, len(out), n_fallback, time.time() - t0)

    if args.stats:
        import statistics
        lens = [len(v) for v in out.values()]
        Path(args.stats).write_text(json.dumps({
            "n": len(out), "fallback": n_fallback,
            "len_mean": float(statistics.mean(lens)),
            "len_min": min(lens), "len_max": max(lens),
        }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
