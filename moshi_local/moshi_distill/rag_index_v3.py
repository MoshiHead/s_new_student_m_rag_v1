"""
rag_index_v3.py — retrieval index over the RAG v3 references (numpy only).
=========================================================================

The existing `/workspace/rag_index` is a 30-chunk smoke test.  This builds a
real index from the corpus itself:

    passage = sidecar `text_conditions.reference`   (what goes into the slots)
    query   = the USER words of the same conversation (what the server hears)

Both are hashed into signed char/word n-gram features, IDF-weighted and
L2-normalised.  A bucket inverted index gives sub-millisecond top-k retrieval
for 14 k passages without faiss / scipy / sentence-transformers (none of which
are installed in `/workspace/menv`).

The held-out split is by `conv_id`, identical to `rag_cache.split_of`, so the
hit@1 report is on unseen conversations.  Gate: hit@1 >= 0.85.  If it is lower,
retrieval is the bottleneck and no amount of Stage-A training can ground the
answers — install `sentence-transformers` (the MiniLM weights are already in
the HF cache) and re-run with `--backend` left as the upgrade path.

Usage::

    python -m moshi_local.moshi_distill.rag_index_v3 \
        --audio-dir /workspace/data/rag_v3/audio \
        --out /workspace/personaplex/rag_index_v3
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import logging
import os
import re
import time

import numpy as np

logger = logging.getLogger(__name__)

D = 1 << 18          # 262 144 buckets; ~1 k non-zeros/passage -> few collisions
WORD_RE = re.compile(r"[a-z0-9]+")


def _h(key: str) -> int:
    return int.from_bytes(hashlib.blake2b(key.encode("utf-8"),
                                          digest_size=8).digest(), "big")


def features(text: str) -> dict[int, float]:
    toks = WORD_RE.findall(text.lower())
    out: dict[int, float] = {}

    def add(key: str, w: float) -> None:
        h = _h(key)
        b = h & (D - 1)
        sign = 1.0 if (h >> 63) & 1 else -1.0
        out[b] = out.get(b, 0.0) + sign * w

    for t in toks:
        add("w:" + t, 1.0)
    for a, b in zip(toks, toks[1:]):
        add("b:" + a + "_" + b, 0.6)
    s = " " + " ".join(toks) + " "
    for n in (4, 5):
        for i in range(len(s) - n + 1):
            add(f"c{n}:" + s[i:i + n], 0.35)
    return out


def conv_id_of(name: str) -> str:
    base = os.path.basename(name)
    for suf in (".sidecar.json", ".wav"):
        if base.endswith(suf):
            base = base[: -len(suf)]
    for suf in ("_cf", "_dx", "_ab"):
        if base.endswith(suf):
            base = base[: -len(suf)]
    return base


def split_of(conv_id: str, eval_frac: float = 0.06) -> str:
    h = hashlib.sha1(conv_id.encode("utf-8")).hexdigest()
    return "eval" if (int(h[:8], 16) / 0xFFFFFFFF) < eval_frac else "train"


def user_text(side: dict, max_words: int = 60) -> str:
    words = [a[0] for a in (side.get("alignments") or [])
             if len(a) >= 3 and a[2] == "USER"]
    return " ".join(words[-max_words:])


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kinds", nargs="*",
                    default=["grounded_real", "grounded_cf",
                             "grounded_distract", "abstain"])
    ap.add_argument("--eval-frac", type=float, default=0.06)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.audio_dir, "*.sidecar.json")))
    if args.limit:
        files = files[: args.limit]
    logger.info("scanning %d sidecars", len(files))

    docs, seen = [], set()
    for p in files:
        try:
            with open(p) as f:
                side = json.load(f)
        except Exception:
            continue
        if (side.get("metadata") or {}).get("rag_kind") not in args.kinds:
            continue
        ref = (side.get("text_conditions") or {}).get("reference") or ""
        q = user_text(side)
        if not ref or not q:
            continue
        key = hashlib.sha1(ref.strip().lower().encode()).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        cid = conv_id_of(os.path.basename(p))
        docs.append({"id": len(docs), "text": ref, "query": q, "conv_id": cid,
                     "kind": (side.get("metadata") or {}).get("rag_kind"),
                     "split": split_of(cid, args.eval_frac)})
    logger.info("kept %d unique passages", len(docs))

    t0 = time.time()
    feats = [features(d["text"]) for d in docs]
    qfeats = [features(d["query"]) for d in docs]
    logger.info("hashed %d docs in %.0fs", len(docs), time.time() - t0)

    df = np.zeros(D, dtype=np.float32)
    for f in feats:
        df[list(f)] += 1.0
    idf = np.log((len(docs) + 1.0) / (df + 1.0)).astype(np.float32) + 1.0

    def to_vec(f: dict[int, float]) -> tuple[np.ndarray, np.ndarray]:
        idx = np.fromiter(f.keys(), dtype=np.int64, count=len(f))
        val = np.fromiter((f[i] for i in f), dtype=np.float32, count=len(f))
        val = val * idf[idx]
        n = float(np.linalg.norm(val))
        if n > 0:
            val = val / n
        return idx, val

    vecs = [to_vec(f) for f in feats]
    qvecs = [to_vec(f) for f in qfeats]

    # inverted index: bucket -> (doc ids, weights)
    posting: dict[int, list[tuple[int, float]]] = {}
    for di, (idx, val) in enumerate(vecs):
        for b, w in zip(idx.tolist(), val.tolist()):
            posting.setdefault(b, []).append((di, w))
    p_ids = {b: np.fromiter((d for d, _ in v), dtype=np.int64, count=len(v))
             for b, v in posting.items()}
    p_w = {b: np.fromiter((w for _, w in v), dtype=np.float32, count=len(v))
           for b, v in posting.items()}
    logger.info("posting lists: %d buckets", len(p_ids))

    def top_k(qv, k: int = 5) -> list[int]:
        idx, val = qv
        scores = np.zeros(len(docs), dtype=np.float32)
        for b, w in zip(idx.tolist(), val.tolist()):
            ids = p_ids.get(b)
            if ids is not None:
                scores[ids] += w * p_w[b]
        return np.argsort(-scores)[:k].tolist()

    ev = [i for i, d in enumerate(docs) if d["split"] == "eval"]
    hit1 = hit5 = 0
    for i in ev:
        top = top_k(qvecs[i], 5)
        hit1 += int(i in top[:1])
        hit5 += int(i in top)
    n_ev = max(len(ev), 1)
    logger.info("held-out conversations: %d | hit@1=%.3f hit@5=%.3f",
                len(ev), hit1 / n_ev, hit5 / n_ev)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "passages.jsonl"), "w") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
    np.savez_compressed(
        os.path.join(args.out, "index.npz"),
        doc_ids=np.concatenate([vecs[i][0] for i in range(len(docs))]) if docs else np.zeros(0, np.int64),
        doc_ptr=np.cumsum([0] + [len(vecs[i][0]) for i in range(len(docs))]).astype(np.int64),
        doc_val=np.concatenate([vecs[i][1] for i in range(len(docs))]) if docs else np.zeros(0, np.float32),
        idf=idf,
        bucket_ids=np.fromiter(p_ids.keys(), dtype=np.int64, count=len(p_ids)) if p_ids else np.zeros(0, np.int64),
        bucket_ptr=np.cumsum([0] + [len(p_ids[b]) for b in p_ids]).astype(np.int64) if p_ids else np.zeros(1, np.int64),
        bucket_docs=np.concatenate([p_ids[b] for b in p_ids]) if p_ids else np.zeros(0, np.int64),
        bucket_w=np.concatenate([p_w[b] for b in p_ids]) if p_ids else np.zeros(0, np.float32),
    )
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump({"n_docs": len(docs), "D": D, "eval_frac": args.eval_frac,
                   "hit@1": hit1 / n_ev, "hit@5": hit5 / n_ev,
                   "n_eval": len(ev)}, f, indent=2)
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
