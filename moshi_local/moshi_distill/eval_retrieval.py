"""
eval_retrieval.py — honest retrieval eval for the RAG v3 corpus.
==============================================================

`rag_index_v3.py` mixed every class into one index and measured hit@1 over
all of them.  That is the wrong question: only ``grounded_real`` pairs a real
user question with the passage that actually answers it.  ``grounded_cf``
synthesises a counterfactual passage, ``grounded_distract`` swaps in another
document and ``abstain`` injects an unrelated one — none of those are
retrievable by design, so counting them as misses only measures the corpus
recipe.

This script asks the serving question instead:

    KB   = deduped `grounded_real` references (what the server can retrieve)
    q    = the LAST user turn of a conversation (what STT hands the server)
    hit@k = the query's own passage is in the top-k

It also reports the top-1 cosine score per class, which is what a serving
score threshold would be calibrated on (regular_chat / abstain should score
low; grounded_real should score high).

Usage::

    python -m moshi_local.moshi_distill.eval_retrieval \
        --audio-dir /workspace/data/rag_v3/audio --out /workspace/personaplex/rag_index_v3
"""

from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import logging
import os

import numpy as np

from .rag_index_v3 import D, conv_id_of, features, split_of, user_text

logger = logging.getLogger(__name__)


def last_user_turn(side: dict) -> str:
    """The transcript the server would have: the final contiguous USER run."""
    al = side.get("alignments") or []
    turns, cur = [], []
    for a in al:
        if len(a) < 3:
            continue
        if a[2] == "USER":
            cur.append(a[0])
        elif cur:
            turns.append(" ".join(cur))
            cur = []
    if cur:
        turns.append(" ".join(cur))
    return turns[-1] if turns else ""


def load(audio_dir: str):
    files = sorted(glob.glob(os.path.join(audio_dir, "*.sidecar.json")))
    out = []
    for p in files:
        try:
            with open(p) as f:
                s = json.load(f)
        except Exception:
            continue
        kind = (s.get("metadata") or {}).get("rag_kind")
        ref = (s.get("text_conditions") or {}).get("reference") or ""
        if not ref:
            continue
        out.append({
            "path": p, "kind": kind, "ref": ref,
            "q_last": last_user_turn(s), "q_60": user_text(s),
            "conv_id": conv_id_of(os.path.basename(p)),
        })
    return out


def build(docs):
    feats = [features(d["ref"]) for d in docs]
    df = np.zeros(D, dtype=np.float32)
    for f in feats:
        df[list(f)] += 1.0
    idf = (np.log((len(docs) + 1.0) / (df + 1.0)) + 1.0).astype(np.float32)

    def vec(f):
        idx = np.fromiter(f.keys(), dtype=np.int64, count=len(f))
        val = np.fromiter((f[i] for i in f), dtype=np.float32, count=len(f))
        val = val * idf[idx]
        n = float(np.linalg.norm(val))
        return idx, (val / n if n > 0 else val)

    mat = [vec(f) for f in feats]
    # dense-ish scoring is fine at 5k docs: one sparse dot per query.
    post = collections.defaultdict(list)
    for di, (idx, val) in enumerate(mat):
        for b, w in zip(idx.tolist(), val.tolist()):
            post[b].append((di, w))
    post = {b: (np.array([d for d, _ in v], np.int64),
                np.array([w for _, w in v], np.float32))
            for b, v in post.items()}
    return idf, post, len(docs)


def score(idf, post, ndocs, qfeat):
    idx = np.fromiter(qfeat.keys(), dtype=np.int64, count=len(qfeat))
    val = np.fromiter((qfeat[i] for i in qfeat), dtype=np.float32,
                      count=len(qfeat))
    val = val * idf[idx]
    n = float(np.linalg.norm(val))
    if n > 0:
        val = val / n
    scores = np.zeros(ndocs, dtype=np.float32)
    for b, w in zip(idx.tolist(), val.tolist()):
        p = post.get(b)
        if p is not None:
            scores[p[0]] += w * p[1]
    return scores


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--eval-frac", type=float, default=0.06)
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    rows = load(args.audio_dir)
    logger.info("loaded %d sidecars with a reference", len(rows))

    # ---- KB: dedupe grounded_real passages --------------------------------
    kb, seen = [], set()
    for r in rows:
        if r["kind"] != "grounded_real":
            continue
        key = hashlib.sha1(r["ref"].strip().lower().encode()).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        kb.append(r["ref"])
    logger.info("KB: %d unique grounded_real passages", len(kb))

    idf, post, ndocs = build([{"ref": t} for t in kb])

    # map passage -> doc id (for hit@1); first occurrence wins, as in the KB
    doc_of = {}
    for i, t in enumerate(kb):
        doc_of.setdefault(hashlib.sha1(t.strip().lower().encode()).hexdigest(), i)

    for mode in ("q_last", "q_60"):
        stats = collections.defaultdict(lambda: {"n": 0, "hit1": 0, "hit5": 0,
                                                 "top1": []})
        for r in rows:
            key = hashlib.sha1(r["ref"].strip().lower().encode()).hexdigest()
            di = doc_of.get(key)
            if di is None:
                continue          # its passage is not in the serving KB
            q = r[mode].strip()
            if not q:
                continue
            s = score(idf, post, ndocs, features(q))
            top = np.argsort(-s)[: args.top_k]
            st = stats[r["kind"]]
            st["n"] += 1
            st["hit1"] += int(top[0] == di)
            st["hit5"] += int(di in top)
            st["top1"].append(float(s[di]) if di < len(s) else 0.0)
        logger.info("--- query mode: %s ---", mode)
        for kind, st in sorted(stats.items()):
            n = max(st["n"], 1)
            logger.info("%-18s n=%5d  hit@1=%.3f  hit@5=%.3f",
                        kind, st["n"], st["hit1"] / n, st["hit5"] / n)

    # ---- score separation: real queries vs. non-groundable classes --------
    # top-1 score for the *first* hit, not the true passage, approximates what
    # the server would inject for that transcript.
    def top1_score(q):
        s = score(idf, post, ndocs, features(q))
        return float(s.max()) if len(s) else 0.0

    sep = collections.defaultdict(list)
    for r in rows:
        if r["kind"] in ("regular_chat", "no_rag_qa", "abstain",
                         "grounded_distract", "grounded_real"):
            sep[r["kind"]].append(top1_score(r["q_last"]))
    logger.info("--- top-1 score distribution (serving threshold calibration) ---")
    for kind, v in sorted(sep.items()):
        a = np.array(v)
        logger.info("%-18s n=%5d  p10=%.3f  p50=%.3f  p90=%.3f",
                    kind, len(a), np.percentile(a, 10), np.percentile(a, 50),
                    np.percentile(a, 90))


if __name__ == "__main__":
    main()
