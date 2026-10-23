"""
eval_retrieval_dense.py — MiniLM retrieval over the grounded_real KB.
===================================================================

The hashed lexical index tops out at hit@1 ~0.59 on grounded_real.  The
MiniLM weights are already in the HF cache, so this builds the dense index the
server will actually use:

    KB  = unique `grounded_real` references (the retrievable corpus)
    q   = last USER turn (what STT gives the server)

Reports hit@1 / hit@5 for dense retrieval, and again after a cross-encoder
rerank of the dense top-N (the cross-encoder is cached too).  Also writes the
serving artefacts:

    <out>/passages.jsonl   {"id", "text", "kind", "conv_id"}
    <out>/emb.npy          float32 [N, 384], L2-normalised
    <out>/meta.json

Usage::

    python -m moshi_local.moshi_distill.eval_retrieval_dense \
        --audio-dir /workspace/data/rag_v3/audio \
        --out /workspace/personaplex/rag_index_dense
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

from .eval_retrieval import last_user_turn

logger = logging.getLogger(__name__)
ST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load(audio_dir: str):
    rows, seen = [], {}
    for p in sorted(glob.glob(os.path.join(audio_dir, "*.sidecar.json"))):
        try:
            with open(p) as f:
                s = json.load(f)
        except Exception:
            continue
        kind = (s.get("metadata") or {}).get("rag_kind")
        ref = (s.get("text_conditions") or {}).get("reference") or ""
        if not ref:
            continue
        key = hashlib.sha1(ref.strip().lower().encode()).hexdigest()
        rows.append({"kind": kind, "ref": ref, "key": key,
                     "q": last_user_turn(s),
                     "path": os.path.basename(p)})
        if kind == "grounded_real":
            seen.setdefault(key, ref)
    return rows, seen


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--out", default="/workspace/personaplex/rag_index_dense")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--top-k", type=int, default=20,
                    help="dense candidates passed to the reranker")
    ap.add_argument("--rerank", type=int, default=1)
    ap.add_argument("--hybrid", type=int, default=1,
                    help="also add lexical top-20 candidates before reranking")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from sentence_transformers import CrossEncoder, SentenceTransformer

    rows, kb = load(args.audio_dir)
    docs = list(kb.values())
    logger.info("loaded %d sidecars | KB %d unique grounded_real passages",
                len(rows), len(docs))

    model = SentenceTransformer(ST_MODEL, device=args.device)
    emb = model.encode(docs, batch_size=args.batch_size, convert_to_numpy=True,
                       normalize_embeddings=True, show_progress_bar=False)
    emb = emb.astype(np.float32)
    logger.info("encoded KB -> %s", emb.shape)

    doc_of, for_id = {}, []
    for i, r in enumerate(rows):
        if r["kind"] == "grounded_real" and r["key"] not in doc_of:
            doc_of[r["key"]] = len(for_id)
            for_id.append(r)

    qs = [r["q"] for r in for_id]
    qe = model.encode(qs, batch_size=args.batch_size, convert_to_numpy=True,
                      normalize_embeddings=True, show_progress_bar=False)
    qe = qe.astype(np.float32)
    sims = qe @ emb.T
    top = np.argsort(-sims, axis=1)[:, : args.top_k]
    hit1 = float(np.mean(top[:, 0] == np.arange(len(for_id))))
    hit5 = float(np.mean([i in top[j, :5] for j, i in enumerate(range(len(for_id)))]))
    logger.info("DENSE hit@1=%.3f hit@5=%.3f (n=%d)", hit1, hit5, len(for_id))

    if args.rerank:
        ce = CrossEncoder(RERANK_MODEL, device=args.device)
        cand = [list(top[j]) for j in range(len(for_id))]

        if args.hybrid:
            # Lexical retrieval is weak on its own (hit@1 ~0.59) but catches
            # exact entity/number matches that MiniLM sometimes misses.
            from .eval_retrieval import build as lex_build, score as lex_score
            from .rag_index_v3 import features as lex_features
            lidf, lpost, lnd = lex_build([{"ref": t} for t in docs])
            for j, q in enumerate(qs):
                ls = lex_score(lidf, lpost, lnd, lex_features(q))
                extra = np.argsort(-ls)[:20].tolist()
                seen = set(cand[j])
                cand[j].extend(i for i in extra if i not in seen)
            logger.info("hybrid candidates/query: median %d",
                        int(np.median([len(c) for c in cand])))

        pairs = [(for_id[j]["q"], docs[i]) for j in range(len(for_id))
                 for i in cand[j]]
        scores = ce.predict(pairs, batch_size=256, show_progress_bar=False)
        scores = np.asarray(scores)
        off = 0
        rr = []
        for j in range(len(for_id)):
            n = len(cand[j])
            s = scores[off:off + n]
            off += n
            order = np.argsort(-s)
            rr.append([cand[j][i] for i in order])
        rh1 = float(np.mean([rr[j][0] == j for j in range(len(for_id))]))
        rh5 = float(np.mean([j in rr[j][:5] for j in range(len(for_id))]))
        logger.info("RERANKED(cand=%d) hit@1=%.3f hit@5=%.3f",
                    int(np.median([len(c) for c in cand])), rh1, rh5)
        hit1, hit5 = rh1, rh5

    # ---- top-1 score per class (threshold calibration) --------------------
    sel = [r for r in rows if r["q"].strip()]
    qe2 = model.encode([r["q"] for r in sel], batch_size=args.batch_size,
                       convert_to_numpy=True, normalize_embeddings=True,
                       show_progress_bar=False).astype(np.float32)
    mx = (qe2 @ emb.T).max(axis=1)
    by = collections.defaultdict(list)
    for r, v in zip(sel, mx):
        by[r["kind"]].append(float(v))
    for kind, v in sorted(by.items()):
        a = np.array(v)
        logger.info("score %-18s n=%5d p10=%.3f p50=%.3f p90=%.3f",
                    kind, len(a), np.percentile(a, 10), np.percentile(a, 50),
                    np.percentile(a, 90))

    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "emb.npy"), emb)
    with open(os.path.join(args.out, "passages.jsonl"), "w") as f:
        for i, r in enumerate(docs):
            f.write(json.dumps({"id": i, "text": r, "key": docs_key(r)}) + "\n")
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump({"model": ST_MODEL, "rerank": RERANK_MODEL,
                   "n_docs": len(docs), "dim": int(emb.shape[1]),
                   "hit@1": hit1, "hit@5": hit5,
                   "rerank_top": args.top_k if args.rerank else 0}, f, indent=2)
    logger.info("wrote %s", args.out)


def docs_key(text: str) -> str:
    return hashlib.sha1(text.strip().lower().encode()).hexdigest()


if __name__ == "__main__":
    main()
