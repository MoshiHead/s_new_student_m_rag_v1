"""rag_retriever_check.py — does the serving retriever reproduce eval numbers?
============================================================================

`eval_retrieval_dense.py` reports hit@1 for the offline pipeline, but the
server has its own `DenseRetriever` (numpy KB + lexical union + cross-encoder).
A drift between the two would silently degrade the demo, so this loads the
*serving* class and measures hit@1 on a sample of grounded_real queries.

    HF_HOME=/workspace/.cache/huggingface HF_HUB_OFFLINE=1 \
    python -m moshi_local.moshi_distill.rag_retriever_check --n 150 --device cpu
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

from .eval_retrieval_dense import load
from .server_student_rag import DenseRetriever

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audio-dir", default="/workspace/data/rag_v3/audio")
    ap.add_argument("--index", default="/workspace/personaplex/rag_index_dense")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows, _ = load(args.audio_dir)
    real = [r for r in rows if r["kind"] == "grounded_real" and r["q"].strip()]
    logger.info("%d rows | %d grounded_real with a query", len(rows), len(real))
    rng = np.random.default_rng(args.seed)
    pick = rng.permutation(len(real))[: min(args.n, len(real))]

    retr = DenseRetriever(args.index, device=args.device)
    hits, missing = 0, 0
    for i in pick:
        r = real[int(i)]
        got = retr.retrieve(r["q"], top_k=1)
        if not got:
            missing += 1
            continue
        if got[0]["text"].strip().lower() == r["ref"].strip().lower():
            hits += 1
    n = len(pick)
    logger.info("serving-path hit@1 = %.3f (n=%d, empty=%d)",
                hits / max(n, 1), n, missing)


if __name__ == "__main__":
    main()
