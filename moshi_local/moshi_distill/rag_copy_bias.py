"""rag_copy_bias.py — can a decode-time pointer bias beat the prior, with no
training at all?

The step-1500 eval has both failure modes side by side:

  gerbera     block = "...botanist and medical"   -> 1710/1743 never in block
  Love...Air  block = "is a 1960 disco song ..."  -> 1960 IS in block, but the
                                                     model says 1977 (memory)

The second one is not a capacity problem, it is a preference problem: the
model's logit for "1977" already beats "1960" at that frame.  A pointer/copy
bias — add beta to the logits of exactly the tokens that sit in the block —
is the cheapest way to flip that preference, and it costs one vector add per
frame at serving (no retraining, no extra frames).

This measures, on real cached samples, what the bias does to

  * body CE (does it hurt the fluent part?),
  * body accuracy,
  * digit-token accuracy (the year/number conflicts),
  * head-12 accuracy (the tokens the passage can actually move).

    python -m moshi_local.moshi_distill.rag_copy_bias \
        --ckpt ckpt/rag_S23/step_1500/student.pt --tokens rag_v3_tokens.pt \
        --kinds grounded_cf --per-kind 20
"""

from __future__ import annotations

import argparse
import logging

import torch
import torch.nn.functional as F

from .rag_probe import load_rag_checkpoint
from .rag_text_probe import clean, make_decoder
from .rag_train import collate

logger = logging.getLogger(__name__)


@torch.no_grad()
def arm_logits(model, codes):
    out = model.forward_train_with_context(codes, None, None)
    return torch.nan_to_num(out.text_logits[:, 0].float(), nan=0.0)


def score(logits, gt, body, block_ids, beta, digit_ids, beta_digit):
    lg = logits.clone()
    if block_ids and beta:
        lg[body, block_ids] += beta
    if digit_ids and beta_digit:
        lg[body, digit_ids] += beta_digit
    pred = lg.argmax(-1)
    g = gt[body]
    ce = F.cross_entropy(lg[body], g.clamp_min(0), reduction="none").mean()
    acc = (pred[body] == g).float().mean()
    d = torch.tensor([int(t) in digit_ids for t in g.tolist()],
                     dtype=torch.bool, device=g.device)
    dacc = (pred[body][d] == g[d]).float().mean() if d.any() else torch.tensor(float("nan"))
    h = min(12, body.numel())
    hacc = (pred[body][:h] == g[:h]).float().mean()
    return float(ce), float(acc), float(dacc), float(hacc), pred


def run(args) -> None:
    device = torch.device(args.device)
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    model, _adapter, pkg = load_rag_checkpoint(args.ckpt, device, dtype=dtype)
    model.eval()
    ref_pos = pkg.get("ref_pos", "pre_body")

    pkg_tok = torch.load(args.tokens, map_location="cpu", weights_only=False)
    entries = pkg_tok["entries"]

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    decode = make_decoder(sp)

    want = [int(c) for c in args.include_conv.split(",") if c.strip()]
    kinds = [k for k in args.kinds.split(",") if k]
    picked, seen = [], {}
    for i, e in enumerate(entries):
        if not e.get("has_ref") or e["kind"] not in kinds:
            continue
        if int(e.get("conv_id", 0)) in want:
            picked.append(i)
            continue
        if seen.get(e["kind"], 0) >= args.per_kind:
            continue
        seen[e["kind"]] = seen.get(e["kind"], 0) + 1
        picked.append(i)
    logger.info("samples: %d (%s)", len(picked),
                ", ".join(f"{entries[i]['kind']}" for i in picked[:6]))

    grid = args.grid.split(",")
    acc = {g: {"n": 0, "ce": 0.0, "acc": 0.0, "dacc": 0.0, "hacc": 0.0,
               "dcnt": 0} for g in grid}
    shower = []
    for i in picked:
        e = entries[i]
        b = collate(entries, [i], device, "text", None, ref_pos)
        codes = b["codes"]
        meta = b["metas"][0]
        bodys = meta.spans_of("body")
        if not bodys:
            continue
        body = torch.arange(bodys[0].start_frame, bodys[0].end_frame,
                            device=device)
        slot = meta.spans_of("ref_slot")
        block_ids = []
        if slot:
            block_ids = [int(t) for t in
                         codes[0, 0, slot[0].start_frame:slot[0].end_frame].tolist()
                         if t > 0]
            block_ids = sorted(set(block_ids) - {0, 1, 2, 3})
        gt = codes[0, 0]
        digit_ids = [t for t in block_ids
                     if any(ch.isdigit() for ch in decode([t]))]

        lg = arm_logits(model, codes)
        for gs in grid:
            beta, beta_d = (float(x) for x in gs.split(":"))
            ce, a, da, ha, _ = score(lg, gt, body, block_ids, beta,
                                     digit_ids, beta_d)
            s = acc[gs]
            s["n"] += 1
            s["ce"] += ce
            s["acc"] += a
            s["hacc"] += ha
            if da == da:
                s["dacc"] += da
                s["dcnt"] += 1
        if len(shower) < args.show:
            strip = "".join(ch for ch in str(e["kind"]))
            ce0, a0, _, ha0, p0 = score(lg, gt, body, block_ids, 0.0,
                                        digit_ids, 0.0)
            ce1, a1, _, ha1, p1 = score(lg, gt, body, block_ids, args.show_beta,
                                        digit_ids, args.show_beta_digit)
            shower.append((strip, e, gt, body, p0, p1, ce0, ce1, ha0, ha1))

    print("\n=== bias grid (beta_all:beta_digit) ===")
    print(f"{'grid':>10s} {'n':>4s} {'body_ce':>8s} {'acc':>6s} "
          f"{'head12':>7s} {'digit_acc':>9s}")
    for gs in grid:
        s = acc[gs]
        n = max(s["n"], 1)
        print(f"{gs:>10s} {s['n']:4d} {s['ce'] / n:8.4f} {s['acc'] / n:6.3f} "
              f"{s['hacc'] / n:7.3f} "
              f"{(s['dacc'] / max(s['dcnt'], 1)):9.3f}")

    for kind, e, gt, body, p0, p1, ce0, ce1, ha0, ha1 in shower:
        g = gt[body]
        print(f"\n--- {kind} conv={e.get('conv_id')} "
              f"body_ce {ce0:.4f} -> {ce1:.4f}  head12 {ha0:.3f} -> {ha1:.3f}")
        print(f"  gt  | {clean(decode(g[:18].tolist()))}")
        print(f"  b0  | {clean(decode(p0[body][:18].tolist()))}")
        print(f"  b+  | {clean(decode(p1[body][:18].tolist()))}")


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--tokenizer",
                    default="/workspace/personaplex/moshi_local/tokenizer.model")
    ap.add_argument("--kinds", default="grounded_cf")
    ap.add_argument("--per-kind", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--grid", default="0:0,1:0,2:0,3:0,0:2,0:4,2:4")
    ap.add_argument("--show", type=int, default=2)
    ap.add_argument("--show-beta", type=float, default=2.0)
    ap.add_argument("--show-beta-digit", type=float, default=4.0)
    ap.add_argument("--include-conv",
                    default="5188255051165725078",   # Love Is in the Air
                    help="conv ids always included (the conflict sample)")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
