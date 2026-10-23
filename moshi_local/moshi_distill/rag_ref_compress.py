"""rag_ref_compress.py — how much of the answer actually fits in the block?

The cache sized the ref block as a *latent* block: K = ceil(n_tokens / 4)
(4-token chunks).  In text mode the trainer writes `ref_ids[:K]`, i.e. the
first K tokens verbatim — so a 50-token passage is truncated to 13 tokens
instead of being covered by 13 slots.

Measured over the whole cache, only ~31% of the first-20 body tokens and ~36%
of body digit tokens are inside the block.  That, not model capacity, is the
main reason "the model answers from its prior": for most samples the fact was
never in the block.

This compares ways to fill the same K frames:

  verbatim   ref_ids[:K]                     (current)
  content    stopword-dropped passage, [:K]
  hybrid     first 4 tokens verbatim, then stopword-dropped, [:K]
  content+4  stopword-dropped passage, [:K+4]  (4 extra frames ~ 0.3 s)

CPU only.

    python -m moshi_local.moshi_distill.rag_ref_compress --tokens rag_v3_tokens.pt
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict

import torch

logger = logging.getLogger(__name__)

STOP = set("""
a an the this that these those it its he she they them his her their him
is are was were be been being am do does did have has had having
of to in on at by for from with without into onto over under about after
before during between among through across against along around behind
and or but nor so than then as if because while although though since
not no nor only just also very more most much many few some any all both
each other another such own same can could may might must shall should will
would i you we me my our your us one two three first second new
there here where when who whom whose which what how why
named called known used made
""".split())


def piece(p, i):
    s = p.id_to_piece(int(i))
    return s.replace("\u2581", " ").strip().lower()


def content_ids(p, ids, stop):
    out = []
    for t in ids:
        w = piece(p, t)
        if w in stop:
            continue
        out.append(int(t))
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--tokenizer",
                    default="/workspace/personaplex/moshi_local/tokenizer.model")
    ap.add_argument("--head", type=int, default=20)
    args = ap.parse_args()

    import sentencepiece as spm
    p = spm.SentencePieceProcessor(model_file=args.tokenizer)
    pkg = torch.load(args.tokens, map_location="cpu", weights_only=False)

    def variants(ids, k):
        con = content_ids(p, ids, STOP)
        return {
            "verbatim": list(ids[:k]),
            "content": con[:k],
            "hybrid": list(ids[:4]) + con[:max(k - 4, 0)],
            "content+4": con[:k + 4],
        }

    agg = defaultdict(lambda: defaultdict(lambda: {"n": 0, "dig_hit": 0,
                                                   "dig_tot": 0, "con_hit": 0,
                                                   "con_tot": 0, "len": 0}))
    for e in pkg["entries"]:
        if not e.get("has_ref"):
            continue
        turns = [(str(a), int(b), int(c)) for a, b, c in e["turns"]]
        slot = [t for t in turns if t[0] == "ref_slot"]
        body = [t for t in turns if t[0] == "body"]
        if not slot or not body:
            continue
        codes = e["codes"].long()
        n = codes.shape[-1]
        k = max(min(slot[0][2], n) - slot[0][1], 1)
        ids = [int(t) for t in e["ref_ids"].tolist() if int(t) > 0]
        gt = [int(t) for t in codes[0, max(body[0][1], 0):min(body[0][2], n)].tolist()
              if t > 0]
        if not ids or not gt:
            continue
        digs = [t for t in gt if any(ch.isdigit() for ch in piece(p, t))]
        cons = [t for t in gt[:args.head] if piece(p, t) not in STOP]
        for name, blk in variants(ids, k).items():
            bs = set(blk)
            a = agg[e["kind"]][name]
            a["n"] += 1
            a["len"] += len(blk)
            a["dig_hit"] += sum(t in bs for t in digs)
            a["dig_tot"] += len(digs)
            a["con_hit"] += sum(t in bs for t in cons)
            a["con_tot"] += len(cons)

    names = ["verbatim", "content", "hybrid", "content+4"]
    print(f"{'kind':16s} {'variant':11s} {'n':>5s} {'len':>6s} "
          f"{'digit_cov':>9s} {'head_content_cov':>16s}")
    for kind in sorted(agg):
        for name in names:
            a = agg[kind][name]
            d = a["dig_hit"] / max(a["dig_tot"], 1)
            c = a["con_hit"] / max(a["con_tot"], 1)
            print(f"{kind:16s} {name:11s} {a['n']:5d} "
                  f"{a['len'] / max(a['n'], 1):6.1f} {d:9.3f} {c:16.3f}")
    print()
    for name in names:
        d = sum(agg[k][name]["dig_hit"] for k in agg) / max(
            sum(agg[k][name]["dig_tot"] for k in agg), 1)
        c = sum(agg[k][name]["con_hit"] for k in agg) / max(
            sum(agg[k][name]["con_tot"] for k in agg), 1)
        print(f"ALL {name:11s} digit_cov={d:.3f} head_content_cov={c:.3f}")


if __name__ == "__main__":
    main()
