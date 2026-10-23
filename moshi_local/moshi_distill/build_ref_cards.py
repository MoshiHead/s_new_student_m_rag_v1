"""build_ref_cards.py — precompute the fact card for every cached reference.

Writes ``{conv_id: [token ids]}`` (at most K ids per entry, K = the entry's
ref_slot count), so the trainer can use the same block content as the server
without loading the tokenizer in the data path.

    python -m moshi_local.moshi_distill.build_ref_cards \
        --tokens rag_v3_tokens.pt --out rag_v3_cards.pt
"""

from __future__ import annotations

import argparse
import logging

import torch

from .rag_ref_select import select_ids

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer",
                    default="/workspace/personaplex/moshi_local/tokenizer.model")
    args = ap.parse_args()

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    pkg = torch.load(args.tokens, map_location="cpu", weights_only=False)

    cards: dict[int, list[int]] = {}
    n = 0
    for e in pkg["entries"]:
        if not e.get("has_ref"):
            continue
        turns = [(str(a), int(b), int(c)) for a, b, c in e["turns"]]
        slot = [t for t in turns if t[0] == "ref_slot"]
        if not slot:
            continue
        k = max(slot[0][2] - slot[0][1], 1)
        ids = [int(t) for t in e["ref_ids"].tolist() if int(t) > 0]
        cards[int(e["conv_id"])] = select_ids(sp, ids, k)
        n += 1
    torch.save({"cards": cards}, args.out)
    lens = [len(v) for v in cards.values()]
    logger.info("wrote %s: %d cards, mean len %.1f", args.out, len(cards),
                sum(lens) / max(len(lens), 1))


if __name__ == "__main__":
    main()
