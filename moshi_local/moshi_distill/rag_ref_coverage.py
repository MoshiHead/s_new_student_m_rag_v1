"""rag_ref_coverage.py — does the block actually contain what the body says?

Two very different failures show up in the eval transcripts:

  gerbera    body needs 1710/1743, block stops at "...botanist and medical"
             -> the fact was never in the block (truncation)
  Love...Air body needs 1960/1961, block contains "is a 1960 disco song sung
             by John Paul Young" -> the fact *is* there but the prior (1977)
             wins (conflict)

This measures, over the whole token cache:

  cov@head   fraction of the first N body token *instances* whose id occurs in
             the block (K = len(ref_slot), i.e. exactly what the model sees)
  cov@all    same over the whole body
  digit      fraction of body digit-token instances present in the block
  full@head  fraction of entries with cov@head == 1.0

CPU only, ~seconds for 14k entries.

    python -m moshi_local.moshi_distill.rag_ref_coverage --tokens rag_v3_tokens.pt
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict

import torch

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--head", type=int, default=20)
    ap.add_argument("--tokenizer",
                    default="/workspace/personaplex/moshi_local/tokenizer.model")
    args = ap.parse_args()

    pkg = torch.load(args.tokens, map_location="cpu", weights_only=False)
    entries = pkg["entries"]

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    cache: dict[int, str] = {}

    def piece(t: int) -> str:
        s = cache.get(t)
        if s is None:
            s = cache[t] = sp.decode([int(t)])
        return s

    agg = defaultdict(lambda: {"n": 0, "head_hit": 0, "head_tot": 0,
                               "full": 0, "body_hit": 0, "body_tot": 0,
                               "dig_hit": 0, "dig_tot": 0,
                               "miss_head": 0, "miss_beyond": 0})
    for e in entries:
        if not e.get("has_ref"):
            continue
        turns = [(str(k), int(s), int(x)) for k, s, x in e["turns"]]
        slot = [t for t in turns if t[0] == "ref_slot"]
        body = [t for t in turns if t[0] == "body"]
        if not slot or not body:
            continue
        codes = e["codes"].long()
        if codes.dim() == 3:
            codes = codes[0]
        n = codes.shape[-1]
        s, x = slot[0][1], min(slot[0][2], n)
        k = max(x - s, 0)
        # the cache keeps -1 in the slots; the block content is ref_ids[:k]
        block = set(int(t) for t in e["ref_ids"].tolist()[:k] if t > 0)
        bs, be = max(body[0][1], 0), min(body[0][2], n)
        gt = [int(t) for t in codes[0, bs:be].tolist() if t > 0]
        if not gt or not block:
            continue
        head = gt[:args.head]
        a = agg[e["kind"]]
        a["n"] += 1
        a["head_hit"] += sum(t in block for t in head)
        a["head_tot"] += len(head)
        a["body_hit"] += sum(t in block for t in gt)
        a["body_tot"] += len(gt)
        a["full"] += int(all(t in block for t in head))
        digs = [t for t in gt if any(ch.isdigit() for ch in piece(t))]
        if digs:
            a["dig_hit"] += sum(t in block for t in digs)
            a["dig_tot"] += len(digs)

    print(f"{'kind':16s} {'n':>6s} {'cov@head':>9s} {'full@head':>10s} "
          f"{'cov@body':>9s} {'digit':>7s}")
    tot = defaultdict(int)
    for kind, a in sorted(agg.items()):
        cov = a["head_hit"] / max(a["head_tot"], 1)
        full = a["full"] / max(a["n"], 1)
        covb = a["body_hit"] / max(a["body_tot"], 1)
        dig = a["dig_hit"] / max(a["dig_tot"], 1) if a["dig_tot"] else float("nan")
        print(f"{kind:16s} {a['n']:6d} {cov:9.3f} {full:10.3f} {covb:9.3f} "
              f"{dig:7.3f}")
        for k in a:
            tot[k] += a[k]
    a = tot
    cov = a["head_hit"] / max(a["head_tot"], 1)
    covb = a["body_hit"] / max(a["body_tot"], 1)
    dig = a["dig_hit"] / max(a["dig_tot"], 1) if a["dig_tot"] else float("nan")
    print(f"{'ALL':16s} {a['n']:6d} {cov:9.3f} "
          f"{a['full'] / max(a['n'], 1):10.3f} {covb:9.3f} {dig:7.3f}")
    print(f"\ndigit tokens seen in bodies: {a['dig_tot']}")


if __name__ == "__main__":
    main()
