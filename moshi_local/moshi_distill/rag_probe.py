"""
rag_probe.py — reference-ablation probe for a trained RAG student.
=================================================================

Answers the only question that matters after Stage A: **does the model actually
use the injected slots?**  It re-runs the same batch three ways and compares the
CE on the grounded body frames only:

    with_ref   : adapter(reference)              <- real conditioning
    zero_ref   : zeros (no context in the slots)  <- baseline
    shuffled   : another sample's reference       <- is it content, or just noise?

A model that ignores the slots gives `zero_ref == with_ref`.  A model that reads
them gives `with_ref < zero_ref`, and a content-sensitive model gives
`shuffled > with_ref`.

Also reports `<lookup>` accuracy (does the model emit the token at the labelled
frame) and validates the **reload path** (checkpoint -> LMModel -> adapter), which
is exactly what the server does at startup.

Usage::

    python -m moshi_local.moshi_distill.rag_probe \
        --ckpt /workspace/personaplex/ckpt/rag_S21/step_5000/student.pt \
        --tokens /workspace/personaplex/rag_v3_tokens.pt --limit 200
"""

from __future__ import annotations

import argparse
import json
import logging

import torch
import torch.nn.functional as F

from .interleaver import SegmentMeta, TurnSpan
from .models.lm import LMModel
from .ref_adapter import RefAdapter, RefAddAdapter, RefBagAdapter

logger = logging.getLogger(__name__)


def load_rag_checkpoint(path, device, dtype=torch.bfloat16):
    pkg = torch.load(path, map_location="cpu", weights_only=False)
    kw = dict(pkg["base_kwargs"])
    kw["num_layers"] = len(pkg["kept_layers"])
    model = LMModel(device=device, dtype=dtype, **kw)
    model = model.to(device=device, dtype=dtype)
    model.load_state_dict(pkg["state_dict"])
    model.eval()
    adapter = None
    ad = pkg.get("adapter")
    if ad is not None:
        kind = pkg.get("adapter_kind")
        if kind is None:
            kind = "add" if "scale" in ad else "latent"
        if kind == "bag":
            adapter = RefBagAdapter(
                model.dim, model.text_emb, n_slots=int(pkg["n_slots"]),
                decay=float(pkg.get("ref_decay") or 0.85))
        elif kind == "add":
            adapter = RefAddAdapter(model.dim, model.text_emb,
                                    n_slots=int(pkg["n_slots"]))
        else:
            adapter = RefAdapter(model.dim, model.text_emb,
                                 n_slots=int(pkg["n_slots"]))
        adapter = adapter.to(device=device, dtype=dtype)
        adapter.load_state_dict(ad)
        adapter.eval()
    logger.info("loaded %s | text_card=%d | ref_mode=%s | n_slots=%d | step=%s",
                path, model.text_card, pkg.get("ref_mode", "latent"),
                pkg["n_slots"], pkg.get("step"))
    return model, adapter, pkg


def collate(entries, idxs, device):
    sel = [entries[i] for i in idxs]
    B = len(sel)
    T = max(int(e["codes"].shape[-1]) for e in sel)
    codes = torch.full((B, 17, T), -1, dtype=torch.long)
    metas, has_ref = [], []
    for b, e in enumerate(sel):
        c = e["codes"]
        n = int(c.shape[-1])
        codes[b, :, :n] = c.long()
        metas.append(SegmentMeta(
            turns=[TurnSpan(k, s, min(x, n)) for k, s, x in e["turns"] if s < n],
            batch_idx=b))
        has_ref.append(bool(e["has_ref"]))
    L = max(int(e["ref_ids"].numel()) for e in sel)
    ref_ids = torch.zeros(B, max(L, 1), dtype=torch.long)
    ref_mask = torch.zeros(B, max(L, 1), dtype=torch.long)
    for b, e in enumerate(sel):
        n = int(e["ref_ids"].numel())
        if n:
            ref_ids[b, :n] = e["ref_ids"]
            ref_mask[b, :n] = 1
    return codes.to(device), metas, has_ref, ref_ids.to(device), ref_mask.to(device)


@torch.no_grad()
def body_ce(model, codes, metas, spans, vecs):
    out = model.forward_train_with_context(codes, spans, vecs)
    w = torch.zeros(codes.shape[0], codes.shape[-1], device=codes.device)
    for meta in metas:
        for sp in meta.spans_of("body"):
            w[meta.batch_idx, sp.start_frame:sp.end_frame] = 1.0
    logits = torch.nan_to_num(out.text_logits[:, 0].float(), nan=0.0)
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                         codes[:, 0].reshape(-1).clamp_min(0), reduction="none")
    return float((ce * w.reshape(-1)).sum() / w.sum().clamp_min(1e-8))


@torch.no_grad()
def lookup_stats(model, codes, metas, spans, vecs, lookup_id):
    """Hit rate at the labelled frame + false-positive rate everywhere else.

    The FP rate matters: a model that spams `<lookup>` gets a perfect hit rate
    while being useless at inference.
    """
    out = model.forward_train_with_context(codes, spans, vecs)
    pred = out.text_logits[:, 0].argmax(-1)
    hit = tot = fp = neg = 0
    for meta in metas:
        is_lookup = torch.zeros(codes.shape[-1], dtype=torch.bool,
                                device=codes.device)
        for sp in meta.spans_of("lookup"):
            lo, hi = max(sp.start_frame, 0), min(sp.end_frame, codes.shape[-1])
            is_lookup[lo:hi] = True
            for f in range(lo, hi):
                tot += 1
                hit += int(pred[meta.batch_idx, f].item() == lookup_id)
        p = pred[meta.batch_idx] == lookup_id
        fp += int((p & ~is_lookup).sum().item())
        neg += int((~is_lookup).sum().item())
    return hit / max(tot, 1), tot, fp / max(neg, 1)


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lookup-id", type=int, default=32002)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)
    model, adapter, pkg = load_rag_checkpoint(args.ckpt, device)
    entries = torch.load(args.tokens, map_location="cpu",
                         weights_only=False)["entries"]
    idxs = [i for i, e in enumerate(entries) if e["has_ref"]][: args.limit]
    logger.info("probing %d entries with a reference", len(idxs))

    tot = {"with_ref": 0.0, "zero_ref": 0.0, "shuffled": 0.0,
           "hit": 0.0, "fp": 0.0, "n": 0, "n_lookup": 0}
    for j in range(0, len(idxs), args.batch_size):
        batch_idx = idxs[j:j + args.batch_size]
        codes, metas, has_ref, ref_ids, ref_mask = collate(entries, batch_idx, device)
        vecs = adapter(ref_ids, ref_mask)
        spans = RefAdapter.spans_from_metas(metas, has_ref)
        roll = torch.roll(ref_ids, 1, dims=0)
        roll_mask = torch.roll(ref_mask, 1, dims=0)
        vecs_shuf = adapter(roll, roll_mask)
        n = codes.shape[0]
        tot["with_ref"] += body_ce(model, codes, metas, spans, vecs) * n
        tot["zero_ref"] += body_ce(model, codes, metas, spans,
                                   torch.zeros_like(vecs)) * n
        tot["shuffled"] += body_ce(model, codes, metas, spans, vecs_shuf) * n
        hit, nl, fp = lookup_stats(model, codes, metas, spans, vecs,
                                   args.lookup_id)
        tot["hit"] += hit * nl
        tot["n_lookup"] += nl
        tot["fp"] += fp * n
        tot["n"] += n

    n = max(tot["n"], 1)
    res = {
        "n_samples": tot["n"],
        "n_lookup": tot["n_lookup"],
        "body_ce_with_ref": tot["with_ref"] / n,
        "body_ce_zero_ref": tot["zero_ref"] / n,
        "body_ce_shuffled": tot["shuffled"] / n,
        "lookup_hit": tot["hit"] / max(tot["n_lookup"], 1),
        "lookup_fp_rate": tot["fp"] / n,
    }
    res["ref_gain"] = res["body_ce_zero_ref"] - res["body_ce_with_ref"]
    res["content_gain"] = res["body_ce_shuffled"] - res["body_ce_with_ref"]
    print(json.dumps(res, indent=2))
    verdict = ("ADAPTER USED" if res["ref_gain"] > 0.02 else
               "ADAPTER IGNORED (ref gain ~ 0)")
    print("verdict:", verdict)
    if res["lookup_hit"] < 0.5:
        print("warning: <lookup> hit rate < 0.5 — the trigger did not learn")


if __name__ == "__main__":
    main()
