"""
rag_text_probe.py — read the model's answer, not just its loss
==============================================================

`rag_probe.py` gives the numeric verdict (body CE with ref vs with zeroed
slots).  This script prints the *text* behind that number, so a delta of 0.02
can be checked against what the model actually says:

    per sample and per turn:   GT | prediction (ref arm) | prediction (zero arm)
    body span:                 token accuracy of each arm, and which body tokens
                               the reference alone got right ("grounded tokens")

Teacher forcing only: every frame is fed from the corpus, so the arms see the
identical context and differ solely in the K slot vectors.  That is the
cleanest read on "is the reference used, and is the content right?" — sampling
noise plays no part.

    # after ~500 steps
    python -m moshi_local.moshi_distill.rag_text_probe \
        --ckpt ckpt/rag_S21/step_500/student.pt \
        --tokens rag_v3_tokens.pt --per-kind 1

    # += a third arm that gets another sample's reference (content check)
    python -m moshi_local.moshi_distill.rag_text_probe --ckpt ... --tokens ... \
        --per-kind 2 --arms all --json /workspace/personaplex/logs/text_probe_500.json

    # CPU self-test of the whole print path, no checkpoint needed
    python -m moshi_local.moshi_distill.rag_text_probe --tiny
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import torch

from .rag_probe import collate, load_rag_checkpoint
from .ref_adapter import RefAdapter

logger = logging.getLogger(__name__)

SPECIALS = {32000: "<ref_start>", 32001: "<ref_end>", 32002: "<lookup>"}
DEFAULT_KINDS = ["grounded_real", "grounded_cf", "grounded_distract",
                 "abstain", "no_rag_qa", "regular_chat"]


# ---------------------------------------------------------------------------
# text helpers
# ---------------------------------------------------------------------------

def make_decoder(spm):
    """ids -> text, with the three control tokens spelled out."""
    if spm is None:
        return lambda ids: " ".join(str(i) for i in ids)

    def decode(ids):
        return "".join(SPECIALS[i] if i in SPECIALS else spm.decode([i])
                       for i in ids)

    return decode


def seg_text(row: torch.Tensor, s: int, e: int, decode) -> str:
    """Decode frames [s, e) of one text-token row (placeholders dropped)."""
    e = min(e, int(row.shape[0]))
    ids = [int(t) for t in row[s:e].tolist() if t >= 0]
    return clean(decode(ids))


def clean(text: str) -> str:
    return text.replace("\u2581", " ").replace("\n", " ").strip()


# ---------------------------------------------------------------------------
# one sample
# ---------------------------------------------------------------------------

@torch.no_grad()
def probe_sample(model, adapter, entry, device, decode,
                 shuffled_ids=None, oracle: bool = False,
                 text_arm: bool = False, ref_mode: str = "latent",
                 refsel: dict | None = None, ref_pos: str = "pre_body",
                 ref_cards: dict | None = None, n_slots: int = 16,
                 ref_delay: int = 0, anchor: str = "lookup",
                 text_pad: int = 3) -> dict:
    codes, metas, has_ref, ref_ids, ref_mask = collate([entry], [0], device)
    spans = None
    if ref_mode in ("text", "add"):
        from .rag_train import collate as _rc
        d = _rc([entry], [0], device, ref_mode, None, ref_pos,
                ref_cards=ref_cards, n_slots=n_slots, ref_delay=ref_delay,
                anchor=anchor, text_pad=text_pad)
        codes, codes_noref = d["codes"], d["codes_noref"]
        metas, has_ref = d["metas"], d["has_ref"]
        ref_ids, ref_mask = d["ref_ids"], d["ref_mask"]
    else:
        codes_noref = codes.clone()

    if ref_mode == "add":
        from .ref_adapter import spans_and_rows
        spans, rows = spans_and_rows(metas, has_ref, "ref_add")
        vecs = (adapter(ref_ids, ref_mask)[rows]
                if adapter is not None else None)
    else:
        spans = RefAdapter.spans_from_metas(metas, has_ref)
        vecs = adapter(ref_ids, ref_mask) if adapter is not None else None

    if adapter is None:
        arms = {"text": (codes, None), "noref": (codes_noref, None)}
    else:
        arms = {"ref": (codes, vecs), "zero": (codes, torch.zeros_like(vecs))}
        if shuffled_ids is not None:
            s_ids = shuffled_ids.to(device)
            if s_ids.dim() == 1:
                s_ids = s_ids[None]
            arms["shuffled"] = (codes, adapter(s_ids, torch.ones_like(s_ids)))
        if oracle and spans and has_ref[0]:
            # "Can the LM read the slots at all?" — inject the body's own
            # upcoming token embeddings instead of the adapter's.
            body = metas[0].spans_of("body")
            b0 = body[0].start_frame if body else 0
            k = spans[0][2] - spans[0][1]
            ids = codes[0, 0, b0:b0 + k]
            if ids.numel() == k:
                arms["oracle"] = (codes, model.text_emb(ids[None]))
        if text_arm and spans and has_ref[0]:
            # The proposal under test: put the *passage text* into the prefill
            # block instead of latents (extractive compression to the K frames
            # we can afford).  No adapter, no projection -- the LM reads its
            # own token embeddings, so nothing new has to be learned.
            txt = codes.clone()
            rewritten = False
            for b, (m, h) in enumerate(zip(metas, has_ref)):
                if not h:
                    continue
                sl = m.spans_of("ref_slot")
                if not sl:
                    continue
                s, e = sl[0].start_frame, sl[0].end_frame
                k = min(e - s, int(ref_mask[b].sum()))
                if k > 0:
                    ids = (refsel or {}).get(int(entry.get("conv_id", 0)))
                    ids = (list(ids)[:k] if ids
                           else ref_ids[b, :k].tolist())
                    txt[b, 0, s:s + len(ids)] = torch.tensor(
                        [int(i) for i in ids], dtype=torch.long)
                    rewritten = True
            if rewritten:
                arms["text"] = (txt, torch.zeros_like(vecs))

    gt = codes[0, 0].detach()                      # stays on the device
    gt_cpu = gt.cpu()
    T = int(gt.shape[0])
    meta = metas[0]
    rec = {"kind": entry["kind"], "conv_id": int(entry.get("conv_id", 0)),
           "T": T, "has_ref": bool(has_ref[0]),
           "n_ref_tokens": int(ref_mask[0].sum()),
           "ref_text": clean(decode([int(i) for i in ref_ids[0].tolist()
                                     if i > 0])),
           "body_ce": {}, "turns": []}

    body_mask = torch.zeros(T, dtype=torch.bool)
    for sp in meta.spans_of("body"):
        body_mask[max(sp.start_frame, 0):min(sp.end_frame, T)] = True

    preds = {}
    for name, (c, v) in arms.items():
        out = model.forward_train_with_context(c, spans, v)
        preds[name] = out.text_logits[:, 0].argmax(-1)[0].cpu()
        logits = torch.nan_to_num(out.text_logits[:, 0].float(), nan=0.0)
        ce = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), gt.reshape(-1).clamp_min(0),
            reduction="none")
        rec["body_ce"][name] = float(ce[body_mask].mean()) if body_mask.any() else 0.0

    for sp in meta.turns:
        s, e = max(sp.start_frame, 0), min(sp.end_frame, T)
        if e <= s:
            continue
        row = {"kind": sp.kind, "range": [s, e],
               "gt": seg_text(gt_cpu, s, e, decode)}
        for name, p in preds.items():
            row[f"pred_{name}"] = seg_text(p, s, e, decode)
        if row["kind"] == "body":
            row["k"] = e - s
        rec["turns"].append(row)

    stats = {}
    gt_off = gt_cpu                            # predictions are on the cpu
    for name, p in preds.items():
        ok = int(((p == gt_off) & body_mask).sum())
        n = int(body_mask.sum())
        stats[name] = {"right": ok, "n": n, "acc": ok / max(n, 1)}
    if "ref" in preds and "zero" in preds:
        pair = ("ref", "zero")
    elif "text" in preds and "noref" in preds:
        pair = ("text", "noref")
    else:
        pair = None
    if pair is not None:
        pr, pz = preds[pair[0]], preds[pair[1]]
        stats["grounded"] = int(((pr == gt_off) & (pz != gt_off) & body_mask).sum())
        stats["anti_grounded"] = int(((pz == gt_off) & (pr != gt_off) & body_mask).sum())
        stats["arms_differ"] = int(((pr != pz) & body_mask).sum())
        rec["pair"] = list(pair)
    rec["body_stats"] = stats
    return rec


# ---------------------------------------------------------------------------
# printing
# ---------------------------------------------------------------------------

def print_sample(rec) -> None:
    print("=" * 100)
    print(f"[{rec['kind']}] conv={rec['conv_id']} T={rec['T']} "
          f"ref_tokens={rec['n_ref_tokens']}")
    if rec["ref_text"]:
        print(f"  REF (retrieved passage): {rec['ref_text'][:220]}")
    print("  body CE  " + "  ".join(
        f"{k}={v:.4f}" for k, v in rec["body_ce"].items())
        + (f"  delta(with-without)="
           f"{rec['body_ce'][rec['pair'][1]] - rec['body_ce'][rec['pair'][0]]:+.4f}"
           if rec.get("pair") else ""))
    bs = rec["body_stats"]
    print("  body acc " + "  ".join(
        f"{k}={v['acc']:.3f}({v['right']}/{v['n']})"
        for k, v in bs.items() if isinstance(v, dict)))
    if "grounded" in bs:
        print(f"  body tokens right only with ref: {bs['grounded']}   "
              f"only without: {bs['anti_grounded']}   "
              f"arms differ on: {bs['arms_differ']}")
    for t in rec["turns"]:
        s, e = t["range"]
        print(f"  --- {t['kind']} [{s}:{e}]")
        print(f"      GT       | {t['gt'][:320]}")
        for name in ("ref", "zero", "text", "noref", "oracle", "shuffled"):
            key = f"pred_{name}"
            if key in t:
                print(f"      {name:8s} | {t[key][:320]}")


# ---------------------------------------------------------------------------
# modes
# ---------------------------------------------------------------------------

def run(args) -> None:
    device = torch.device(args.device)
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    model, adapter, pkg = load_rag_checkpoint(args.ckpt, device, dtype=dtype)
    if adapter is not None:
        adapter.to(device)
    ref_mode = pkg.get("ref_mode", "latent")
    ref_pos = args.ref_pos or pkg.get("ref_pos", "pre_lead")
    logger.info("dtype=%s ref_mode=%s adapter=%s", dtype, ref_mode,
                "yes" if adapter is not None else "no")
    entries = torch.load(args.tokens, map_location="cpu",
                         weights_only=False)["entries"]

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(pkg.get("tokenizer") or args.tokenizer)
    decode = make_decoder(sp)

    kinds = args.kinds.split(",") if args.kinds else DEFAULT_KINDS
    counts = {k: 0 for k in kinds}
    picked = []
    for i, e in enumerate(entries):
        if e["kind"] in counts and counts[e["kind"]] < args.per_kind:
            counts[e["kind"]] += 1
            picked.append(i)
    logger.info("picked %s", counts)

    refsel = (torch.load(args.refsel, map_location="cpu", weights_only=False)
              if args.refsel else None)
    cards = (torch.load(args.ref_cards, map_location="cpu",
                        weights_only=False)["cards"]
             if args.ref_cards else None)
    if ref_mode == "add" and cards is None:
        logger.warning("add-mode checkpoint without --ref-cards: falling back "
                       "to the cached passage prefix (not what it trained on)")
    recs = []
    for pos, i in enumerate(picked):
        shuffled = None
        if args.arms == "all":
            j = picked[(pos + 1) % len(picked)]
            shuffled = entries[j]["ref_ids"]
        rec = probe_sample(model, adapter, entries[i], device, decode, shuffled,
                           oracle=args.oracle, text_arm=args.text_arm,
                           ref_mode=ref_mode, refsel=refsel, ref_pos=ref_pos,
                           ref_cards=cards, n_slots=int(pkg["n_slots"]),
                           ref_delay=args.ref_delay,
                           anchor=args.ref_anchor
                           or (pkg.get("ref_anchor") or "lookup"))
        print_sample(rec)
        recs.append(rec)

    if args.json:
        Path(args.json).write_text(json.dumps(recs, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")

    if recs and args.arms == "all":
        g = sum(r["body_stats"].get("grounded", 0) for r in recs)
        d = sum(r["body_stats"].get("arms_differ", 0) for r in recs)
        print(f"\ntotal body tokens read only via the correct reference: {g} "
              f"(arms differ on {d})")


def tiny() -> None:
    """CPU self-test: exercises collate/forward/print without a checkpoint."""
    from .rag_eval_check import fake_entry, tiny_model
    from .ref_adapter import RefAdapter as RA

    torch.manual_seed(0)
    model = tiny_model()
    adapter = RA(model.dim, model.text_emb, n_slots=16)
    entries = [fake_entry("grounded_real", True),
               fake_entry("grounded_cf", True),
               fake_entry("abstain", True),
               fake_entry("regular_chat", False)]

    class _SP:                      # ids -> letters, enough to see any diff
        def decode(self, ids):
            return "".join(chr(ord("a") + (i % 26)) for i in ids)

    decode = make_decoder(_SP())
    for e in entries[:2]:
        rec = probe_sample(model, adapter, e, torch.device("cpu"), decode)
        print_sample(rec)
        assert rec["turns"] and rec["body_stats"]["ref"]["n"] > 0
        last = rec["turns"][-1]
        assert "pred_ref" in last and "pred_zero" in last
        assert set(rec["body_ce"]) == {"ref", "zero"}
    # shuffled arm path
    rec = probe_sample(model, adapter, entries[0], torch.device("cpu"),
                       decode, entries[1]["ref_ids"])
    assert "shuffled" in rec["body_ce"], rec["body_ce"]
    print_sample(rec)
    # oracle arm (read-path test)
    rec = probe_sample(model, adapter, entries[0], torch.device("cpu"),
                       decode, oracle=True)
    assert "oracle" in rec["body_ce"], rec["body_ce"]
    print_sample(rec)
    # text-ref arm (the redesign under test)
    rec = probe_sample(model, adapter, entries[0], torch.device("cpu"),
                       decode, text_arm=True)
    assert "text" in rec["body_ce"], rec["body_ce"]
    print_sample(rec)
    # text-mode checkpoint: no adapter at all, ablation = pristine block
    rec = probe_sample(model, None, entries[0], torch.device("cpu"), decode,
                       ref_mode="text")
    assert set(rec["body_ce"]) == {"text", "noref"}, rec["body_ce"]
    assert rec.get("pair") == ["text", "noref"], rec.get("pair")
    print_sample(rec)

    # pause carrier: card in the existing pre-body pad frames, zero new frames
    ep = fake_entry("grounded_cf", True)
    ep["codes"][0, 4:11] = 3
    ep["codes"][0, 15] = 3
    ep["codes"][0, 2] = 5
    ep["plen"] = 2
    ep["turns"] = [("regular", 0, 2), ("lookup", 2, 3), ("ref", 11, 15),
                   ("ref_slot", 12, 14), ("body", 16, 22)]
    rec = probe_sample(model, None, ep, torch.device("cpu"), decode,
                       ref_mode="text", ref_pos="pause")
    assert set(rec["body_ce"]) == {"text", "noref"}, rec["body_ce"]
    assert rec["T"] == 20, rec["T"]           # nothing added to the sequence
    print_sample(rec)
    print("TEXT_PROBE_TINY_OK")


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    torch.set_num_threads(int(os.environ.get("RAG_NUM_THREADS", "8")))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--ckpt")
    ap.add_argument("--tokens")
    ap.add_argument("--tokenizer",
                    default="/workspace/personaplex/moshi_local/tokenizer.model")
    ap.add_argument("--per-kind", type=int, default=1)
    ap.add_argument("--kinds", default="", help="comma list, default all six")
    ap.add_argument("--arms", default="ref_zero", choices=["ref_zero", "all"])
    ap.add_argument("--oracle", action="store_true",
                    help="extra arm: inject the body's own token embeddings "
                         "into the slots (tests the read path itself)")
    ap.add_argument("--text-arm", action="store_true",
                    help="extra arm: write the passage's own token ids into "
                         "the prefill block (no adapter) and read them as text")
    ap.add_argument("--refsel", default="",
                    help="optional {conv_id: ids} query-selected windows for "
                         "the text arm")
    ap.add_argument("--ref-cards", default="",
                    help="fact cards pt (rag_v3_cards.pt); required to probe "
                         "an add-mode checkpoint exactly as it was trained")
    ap.add_argument("--ref-delay", type=int, default=0,
                    help="add mode: frames between <lookup> and vector 0")
    ap.add_argument("--ref-anchor", default="",
                    choices=["", "body", "lookup"],
                    help="add mode: override where the carrier rides; empty = "
                         "use what the checkpoint was trained with")
    ap.add_argument("--ref-pos", default="",
                    choices=["", "pre_body", "pre_lead", "pause"],
                    help="override the checkpoint's block layout; empty = "
                         "use what the checkpoint was trained with")
    ap.add_argument("--json", default="", help="dump the records to this path")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.tiny:
        tiny()
        return
    if not (args.ckpt and args.tokens):
        ap.error("--ckpt and --tokens are required unless --tiny")
    run(args)


if __name__ == "__main__":
    main()
