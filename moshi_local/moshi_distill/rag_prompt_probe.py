"""rag_prompt_probe.py — where should the passage go: mid-stream block or prompt?

Compares, on the same cached samples and with **no training**:

  block    passage tokens written into the <ref_start> .. <ref_end> block
           (the current text-mode conditioning)
  noref    same layout, block left pristine (the ablation arm)
  prompt   passage prepended at the end of the persona prompt, block pristine
  promptc  control: the same number of extra prompt frames, text padding only

The prompt arms are the in-distribution option: text in the prompt is exactly
what the persona block already is, so the frozen student does not have to learn
a new mid-stream format.

Also prints the per-position body CE curve, because the mean over a 100-frame
body hides the first few tokens — the only ones the passage can actually help.

    python -m moshi_local.moshi_distill.rag_prompt_probe \
        --ckpt ckpt/rag_S23/step_1250/student.pt --tokens rag_v3_tokens.pt \
        --kinds grounded_cf --per-kind 4 --k 16
"""

from __future__ import annotations

import argparse
import logging
import statistics

import torch
import torch.nn.functional as F

from .interleaver import SegmentMeta, TurnSpan
from .rag_probe import load_rag_checkpoint
from .rag_text_probe import clean, make_decoder
from .rag_train import collate

logger = logging.getLogger(__name__)


@torch.no_grad()
def ce_curve(model, codes, turns, body_range, device):
    """CE per frame, plus the argmax tokens, for one [17, T] sample."""
    c = codes.to(device)[None]
    meta = SegmentMeta(turns=[TurnSpan(kind=k, start_frame=s, end_frame=e)
                              for k, s, e in turns if s < c.shape[-1]],
                       batch_idx=0)
    out = model.forward_train_with_context(c, None, None)
    logits = torch.nan_to_num(out.text_logits[:, 0].float(), nan=0.0)
    gt = c[0, 0].clamp_min(0)
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                         gt.reshape(-1), reduction="none")
    pred = out.text_logits[:, 0].argmax(-1)[0]
    s, e = body_range
    s, e = max(s, 0), min(e, int(c.shape[-1]))
    return ce[s:e].cpu(), pred[s:e].cpu(), gt[s:e].cpu()


def body_range_of(turns):
    for k, s, e in turns:
        if k == "body":
            return s, e
    return 0, 0


def prompt_variant(entry, k, ids, fill: str):
    """Insert k frames at the end of the persona prompt."""
    c = entry["codes"].long()
    n = int(c.shape[-1])
    pl = max(1, min(int(entry["plen"]), n))
    frame = c[:, pl - 1:pl].repeat(1, k).clone()      # prompt-audio pattern
    frame[0, :] = 3                                   # text padding
    if fill == "passage":
        m = min(k, len(ids))
        if m:
            frame[0, :m] = torch.tensor(ids[:m], dtype=torch.long)
    new = torch.cat([c[:, :pl], frame, c[:, pl:]], dim=-1)
    turns = [(kk, s + k, min(x + k, n + k)) for kk, s, x in entry["turns"] if s < n]
    return new, turns, pl


def run(args) -> None:
    device = torch.device(args.device)
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    model, adapter, pkg = load_rag_checkpoint(args.ckpt, device, dtype=dtype)
    model.eval()
    ref_pos = pkg.get("ref_pos", "pre_body")
    logger.info("ckpt step=%s ref_mode=%s ref_pos=%s dtype=%s",
                pkg.get("step"), pkg.get("ref_mode"), ref_pos, dtype)

    pkg_tok = torch.load(args.tokens, map_location="cpu", weights_only=False)
    entries = pkg_tok["entries"]
    kinds = [k for k in args.kinds.split(",") if k] if args.kinds else None
    picked, seen = [], {}
    for i, e in enumerate(entries):
        if not e["has_ref"]:
            continue
        if kinds and e["kind"] not in kinds:
            continue
        if seen.get(e["kind"], 0) >= args.per_kind:
            continue
        seen[e["kind"]] = seen.get(e["kind"], 0) + 1
        picked.append(i)
    logger.info("samples: %s", [(i, entries[i]["kind"]) for i in picked])

    import sentencepiece as spm
    decode = make_decoder(spm.SentencePieceProcessor(model_file=args.tokenizer))

    agg: dict[str, list] = {}
    for i in picked:
        e = entries[i]
        ref_ids = [int(t) for t in e["ref_ids"].tolist() if int(t) > 0]
        k = min(args.k, max(1, len(ref_ids)))

        arms = {}
        b = collate(entries, [i], device, "text", None, ref_pos)
        sl = b["metas"][0].spans_of("ref_slot")[0]
        body = (b["metas"][0].spans_of("body")[0].start_frame,
                b["metas"][0].spans_of("body")[0].end_frame)
        arms["block"] = (b["codes"][0], [(t.kind, t.start_frame, t.end_frame)
                                         for t in b["metas"][0].turns], body)
        arms["noref"] = (b["codes_noref"][0],
                         [(t.kind, t.start_frame, t.end_frame)
                          for t in b["metas"][0].turns], body)

        pr_t, pr_turns, pl = prompt_variant(e, k, ref_ids, "passage")
        ct_t, ct_turns, _ = prompt_variant(e, k, ref_ids, "pad")
        arms["prompt"] = (pr_t, pr_turns, body_range_of(pr_turns))
        arms["promptc"] = (ct_t, ct_turns, body_range_of(ct_turns))

        print(f"\n### {e['kind']} conv={e.get('conv_id')} T={int(e['codes'].shape[-1])} "
              f"plen={pl} k={k} ref_tokens={len(ref_ids)}")
        print(f"  REF: {clean(decode(ref_ids[:k]))}")
        ce_by_arm = {}
        for name, (codes, turns, br) in arms.items():
            ce, pred, gt = ce_curve(model, codes, turns, br, device)
            ce_by_arm[name] = {
                "mean": float(ce.mean()),
                "head4": float(ce[:4].mean()) if ce.numel() >= 4 else float("nan"),
                "head12": float(ce[:12].mean()) if ce.numel() >= 12 else float("nan"),
                "ce": ce.tolist(),
            }
            print(f"  {name:8s} body_ce={ce_by_arm[name]['mean']:.4f} "
                  f"head4={ce_by_arm[name]['head4']:.4f} "
                  f"head12={ce_by_arm[name]['head12']:.4f}")
            print(f"           gt   | {clean(decode(gt[:14].tolist()))}")
            print(f"           pred | {clean(decode(pred[:14].tolist()))}")
        agg.setdefault("block-noref", []).append(
            ce_by_arm["noref"]["mean"] - ce_by_arm["block"]["mean"])
        agg.setdefault("prompt-promptc", []).append(
            ce_by_arm["promptc"]["mean"] - ce_by_arm["prompt"]["mean"])
        agg.setdefault("prompt-noref", []).append(
            ce_by_arm["noref"]["mean"] - ce_by_arm["prompt"]["mean"])
        agg.setdefault("prompt-promptc-head4", []).append(
            ce_by_arm["promptc"]["head4"] - ce_by_arm["prompt"]["head4"])
        agg.setdefault("block-noref-head4", []).append(
            ce_by_arm["noref"]["head4"] - ce_by_arm["block"]["head4"])

    print("\n=== summary (positive = the passage helped) ===")
    for k2, v in agg.items():
        print(f"  {k2:22s} mean {statistics.mean(v):+.4f}  "
              f"n={len(v)}  (per sample: "
              + ", ".join(f"{x:+.3f}" for x in v) + ")")

    if args.json:
        import json
        with open(args.json, "w") as f:
            json.dump(agg, f, indent=2)
        print("wrote", args.json)


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--tokenizer", default="/workspace/personaplex/moshi_local/tokenizer.model")
    ap.add_argument("--kinds", default="grounded_cf")
    ap.add_argument("--per-kind", type=int, default=4)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
