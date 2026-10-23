"""rag_eval_check.py — smoke-test the eval/metric path without training.

`rag_train.evaluate` is only exercised every 500 steps, so a bug there is
expensive.  This runs it on a handful of cached samples (inference only, no
optimizer, no weight updates) and prints the aggregate + per-kind body CE.

    python -m moshi_local.moshi_distill.rag_eval_check \
        --ckpt ckpt/rag_S21/step_250/student.pt \
        --tokens rag_v3_tokens.pt

    # CPU, no checkpoint: random tiny model + synthetic entries.  Exercises
    # evaluate(), the per-kind ablation and the training-time probe.
    python -m moshi_local.moshi_distill.rag_eval_check --tiny
"""

from __future__ import annotations

import argparse
import collections
import logging
import os
import types

import torch

from .models.lm import LMModel
from .rag_probe import load_rag_checkpoint
from .rag_train import body_ce, evaluate, seg_weights_for
from .ref_adapter import RefAdapter

logger = logging.getLogger(__name__)

TINY_ARGS = dict(
    batch_size=2, ref_weight=0.0, ref_context_weight=0.0,
    ref_slot_weight=0.0, lookup_weight=6.0, lead_weight=1.0,
    filler_weight=1.0, body_weight=2.0, no_rag_weight=1.0,
    text_padding_weight=0.5, audio_weight=0.5, first_cb_weight=2.0,
    anchor_weight=0.0, lookup_id=5, ref_mode="latent", body_head=6)


def tiny_model() -> LMModel:
    """1/64-scale twin of the S20 student (same delays / depformer paths)."""
    torch.manual_seed(0)
    delays = [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
    model = LMModel(device="cpu", dtype=torch.float32,
                    delays=delays, n_q=16, dep_q=16,
                    card=32, text_card=64, dim=64, num_heads=4,
                    hidden_scale=2, num_layers=2, context=64,
                    gating="silu", norm="layer_norm", causal=True,
                    depformer_dim=32, depformer_num_heads=4,
                    depformer_dim_feedforward=64, depformer_num_layers=2,
                    depformer_causal=True, depformer_context=8,
                    depformer_multi_linear=True,
                    depformer_weights_per_step=True,
                    depformer_pos_emb="none")
    model.eval()
    return model


def fake_entry(kind: str, has_ref: bool, T: int = 24) -> dict:
    """One synthetic corpus entry: prompt, lookup, ref block, body."""
    card, text_card, lookup = 32, 64, 5
    codes = torch.randint(0, card, (17, T))
    codes[0] = torch.randint(0, text_card, (T,))
    rs, re = 12, 14                     # ref_slot span, K = 2
    codes[0, 10] = lookup               # <lookup>
    codes[0, rs - 1] = 6                # <ref_start>
    codes[0, rs:re] = -1                # slot placeholders
    codes[0, re] = 7                    # <ref_end>
    turns = [("regular", 6, 10), ("lookup", 10, 11), ("ref", rs - 1, re + 1),
             ("ref_slot", rs, re), ("body", re + 2, T - 2)]
    return {"codes": codes, "turns": turns, "has_ref": has_ref,
            "plen": 6, "kind": kind,
            "ref_ids": torch.randint(1, text_card, (6 if has_ref else 0,))}


def tiny() -> None:
    torch.manual_seed(0)
    model = tiny_model()
    adapter = RefAdapter(model.dim, model.text_emb, n_slots=16)
    entries = [fake_entry("grounded_real", True),
               fake_entry("grounded_cf", True),
               fake_entry("no_rag_qa", False),
               fake_entry("regular_chat", False)]
    args = types.SimpleNamespace(**TINY_ARGS)
    ev = evaluate(model, adapter, entries, [0, 1, 2, 3], args, "cpu")
    print(f"[tiny] text={ev['text']:.4f} audio={ev['audio']:.4f} "
          f"lookup_hit={ev['lookup_hit']:.3f} "
          f"body_ce_ref={ev['body_ce_ref']:.4f} "
          f"body_ce_zero={ev['body_ce_zero']:.4f} "
          f"delta={ev['body_ce_zero'] - ev['body_ce_ref']:+.4f}")
    for kind, s in sorted(ev["by_kind"].items()):
        print(f"[tiny] {kind:16s} n={s['n']:3d} ref={s['ref']:.4f} "
              f"zero={s['zero']:.4f} delta={s['delta']:+.4f}")
    assert set(ev["by_kind"]) == {"grounded_real", "grounded_cf"}, ev["by_kind"]
    # no-ref samples must be excluded from the body ablation
    assert ev["body_ce_ref"] > 0 and ev["body_ce_zero"] > 0

    # the live probe uses the same helper on an uncollated batch
    from .rag_train import collate
    batch = collate(entries, [0, 1], "cpu")
    with torch.no_grad():
        vecs = adapter(batch["ref_ids"], batch["ref_mask"])
        ref_ce, kinds, n_ref, ref_head = body_ce(model, batch, args, vecs)
        zero_ce, _, _, zero_head = body_ce(model, batch, args, torch.zeros_like(vecs))
    print(f"[tiny] probe n_ref={n_ref} ref={ref_ce:.4f} zero={zero_ce:.4f} "
          f"delta={zero_ce - ref_ce:+.4f} head_delta={zero_head - ref_head:+.4f} "
          f"kinds={sorted(kinds)}")
    assert n_ref == 2 and set(kinds) == {"grounded_real", "grounded_cf"}
    assert ref_head == ref_head and zero_head == zero_head

    # ---- text mode: passage tokens in the block, no adapter at all --------
    args_t = types.SimpleNamespace(**{**TINY_ARGS, "ref_mode": "text"})
    bt = collate(entries, [0, 1], "cpu", "text", None)
    # the ref block must actually carry the passage ids now
    slot = bt["metas"][0].spans_of("ref_slot")[0]
    rs, re_ = slot.start_frame, slot.end_frame
    assert not torch.equal(bt["codes"][0, 0, rs:re_], bt["codes_noref"][0, 0, rs:re_])
    assert torch.equal(bt["codes"][0, 0, rs:re_], entries[0]["ref_ids"][:re_ - rs])
    # layout (pre_body): <lookup> abuts the block, block abuts the body
    lk = bt["metas"][0].spans_of("lookup")[0]
    rf = bt["metas"][0].spans_of("ref")[0]
    bd = bt["metas"][0].spans_of("body")[0]
    assert lk.end_frame == rf.start_frame, (lk, rf)
    assert rf.end_frame == bd.start_frame, (rf, bd)
    assert int(bt["codes"][0, 0, lk.start_frame]) == TINY_ARGS["lookup_id"]
    assert int(bt["codes"][0, 0, rf.start_frame]) == 6       # <ref_start>
    assert int(bt["codes"][0, 0, rf.end_frame - 1]) == 7      # <ref_end>
    assert bt["codes"].shape[-1] == 24                        # length preserved
    with torch.no_grad():
        ref_ce, kinds, n_ref, ref_head = body_ce(model, bt, args_t, None)
        noref_ce, _, _, noref_head = body_ce(model, bt, args_t, None, noref=True)
    print(f"[tiny] text-mode n_ref={n_ref} ref={ref_ce:.4f} "
          f"noref={noref_ce:.4f} kinds={sorted(kinds)}")
    assert n_ref == 2 and ref_ce > 0 and noref_ce > 0
    evt = evaluate(model, None, entries, [0, 1, 2, 3], args_t, "cpu")
    print(f"[tiny] text-mode eval body_ce_ref={evt['body_ce_ref']:.4f} "
          f"body_ce_zero={evt['body_ce_zero']:.4f} "
          f"delta={evt['body_ce_zero'] - evt['body_ce_ref']:+.4f}")
    assert set(evt["by_kind"]) == {"grounded_real", "grounded_cf"}

    # ---- pause mode: card in the existing pre-body pad frames --------------
    # Zero added frames: the ids replace padding text tokens that are already
    # there, the audio of those frames is the model's own pause, and <lookup>
    # moves to the frame just before the card.
    ep = fake_entry("grounded_cf", True)
    ep["codes"][0, 4:11] = 3                  # pad run before the block
    ep["codes"][0, 15] = 3                    # ... and after it (pre-body)
    ep["codes"][0, 2] = 5                     # <lookup> early, as in the corpus
    ep["plen"] = 2                            # keep the persona prefix short
    ep["turns"] = [("regular", 0, 2), ("lookup", 2, 3), ("ref", 11, 15),
                   ("ref_slot", 12, 14), ("body", 16, 22)]
    bp = collate([ep], [0], "cpu", "text", None, "pause")
    mp = bp["metas"][0]
    lk, sl, bd = (mp.spans_of("lookup")[0], mp.spans_of("ref_slot")[0],
                  mp.spans_of("body")[0])
    K = int(ep["ref_ids"][:16].numel())
    assert bp["codes"].shape[-1] == 20, bp["codes"].shape       # nothing added
    assert bd.start_frame == 12 and K == 6, (bd, K)
    assert (sl.start_frame, sl.end_frame) == (6, 12), sl        # abuts the body
    assert (lk.start_frame, lk.end_frame) == (5, 6), lk
    assert int(bp["codes"][0, 0, 5]) == 5                       # <lookup> moved
    assert int(bp["codes"][0, 0, 2]) == 3                       # old slot padded
    assert torch.equal(bp["codes"][0, 0, 6:12], ep["ref_ids"][:6])
    assert torch.equal(bp["codes_noref"][0, 0, 6:12],
                       torch.full((6,), 3, dtype=torch.long))   # ablation arm
    wp = seg_weights_for(bp["codes"], bp["metas"], bp["plens"], args_t,
                         head_weight=False)[0]
    assert float(wp[5]) == 6.0, float(wp[5])                    # lookup weight
    assert float(wp[6:12].abs().max()) == 0.0, wp[6:12]         # card not predicted
    # the card frames keep their real pause audio (not PAD, not silence)
    assert int(bp["codes"][0, 1, 6]) == int(ep["codes"][1, 6])
    assert int(bp["codes"][0, 1, 7]) != -1
    args_p = types.SimpleNamespace(**{**TINY_ARGS, "ref_mode": "text",
                                      "ref_pos": "pause"})
    evp = evaluate(model, None, [ep], [0], args_p, "cpu")
    print(f"[tiny] pause-mode n={evp['by_kind']['grounded_cf']['n']} "
          f"ref={evp['body_ce_ref']:.4f} zero={evp['body_ce_zero']:.4f}")
    assert evp["by_kind"]["grounded_cf"]["n"] == 1
    assert evp["body_ce_ref"] > 0 and evp["body_ce_zero"] > 0

    # wrong-card control: swap the two cards, same frames, same budget
    from .rag_train import shuffle_cards
    torch.manual_seed(7)
    eq = fake_entry("grounded_real", True)
    eq["codes"][0, 4:11] = 3
    eq["codes"][0, 15] = 3
    eq["codes"][0, 2] = 5
    eq["plen"] = 2
    eq["turns"] = list(ep["turns"])
    bq = collate([ep, eq], [0, 1], "cpu", "text", None, "pause")
    s0 = bq["metas"][0].spans_of("ref_slot")[0]
    s1 = bq["metas"][1].spans_of("ref_slot")[0]
    sq = shuffle_cards(bq)
    assert sq is not None
    k = min(s0.end_frame - s0.start_frame, s1.end_frame - s1.start_frame)
    assert torch.equal(sq[0, 0, s0.start_frame:s0.start_frame + k],
                       bq["codes"][1, 0, s1.start_frame:s1.start_frame + k])
    assert torch.equal(sq[1, 0, s1.start_frame:s1.start_frame + k],
                       bq["codes"][0, 0, s0.start_frame:s0.start_frame + k])
    assert not torch.equal(sq[0], bq["codes"][0])
    # a single reference row has nothing to swap with
    assert shuffle_cards(bp) is None

    # the sample picker used for the printed text probe
    from .rag_train import pick_text_probe
    assert pick_text_probe(entries, [0, 1, 2, 3], 2) == [0, 1]
    assert pick_text_probe(entries, [], 6) == [0, 1, 2, 3]
    assert pick_text_probe(entries, [], 1) == [0]

    # The write must reach the body positions: a random model's *CE* barely
    # moves (its output at the body frames is dominated by the teacher-forced
    # body tokens), so assert on the logits instead.
    with torch.no_grad():
        spans = RefAdapter.spans_from_metas(batch["metas"], batch["has_ref"])
        v = adapter(batch["ref_ids"], batch["ref_mask"])
        o1 = model.forward_train_with_context(batch["codes"], spans, v)
        o0 = model.forward_train_with_context(batch["codes"], spans,
                                              torch.zeros_like(v))
    d = (o1.text_logits[:, 0] - o0.text_logits[:, 0]).abs()
    body_max = float(d[:, 16:22].max())          # body span of `fake_entry`
    print(f"[tiny] ref->body logit shift: all={float(d.max()):.4f} "
          f"body={body_max:.4f}")
    assert spans == [(0, 13, 15), (1, 13, 15)], spans   # block moved pre-body
    assert float(d.max()) > 1e-2 and body_max > 1e-4, (float(d.max()), body_max)

    # ---- additive mode: no block, vectors on the frames after <lookup> ----
    from .ref_adapter import RefBagAdapter
    from .rag_train import _ref_vectors, ref_add_span, strip_ref_block

    T = 30
    card, text_card = 32, 64
    codes = torch.randint(0, card, (17, T))
    codes[0] = torch.randint(0, text_card, (T,))
    codes[0, 6] = 5                     # <lookup>
    codes[0, 7] = 6                     # <ref_start>
    codes[0, 8:10] = -1                 # K = 2 slots
    codes[0, 10] = 7                    # <ref_end>
    turns = [("regular", 0, 6), ("lookup", 6, 7), ("ref", 7, 11),
             ("ref_slot", 8, 10), ("lead", 11, 20), ("filler", 20, 24),
             ("body", 24, T)]
    entry = {"codes": codes, "turns": turns, "has_ref": True, "plen": 6,
             "kind": "grounded_cf",
             "ref_ids": torch.tensor([11, 12, 13, 14, 15, 16])}
    # stripping removes <ref_start> + slots + <ref_end>, everything slides left
    c2, t2 = strip_ref_block(codes, turns)
    assert c2.shape[-1] == T - 4, c2.shape
    kinds2 = [k for k, _, _ in t2]
    assert "ref" not in kinds2 and "ref_slot" not in kinds2, kinds2
    assert int(c2[0, 6]) == 5                         # <lookup> kept
    tmap = {k: (s, x) for k, s, x in t2}
    assert tmap["body"] == (20, T - 4), tmap["body"]
    assert tmap["lookup"] == (6, 7)
    # span starts on the <lookup> frame: inject() puts vec 0 on emb[7] = the
    # embedding of codes[6] (exactly the call the server makes next)
    assert ref_add_span(t2, 4) == (6, 10)
    assert ref_add_span(t2, 4, body_start=8) == (6, 8)   # truncated at body

    args_a = types.SimpleNamespace(**{**TINY_ARGS, "ref_mode": "add",
                                      "n_slots": 4})
    add_adapter = RefBagAdapter(model.dim, model.text_emb, n_slots=4,
                                init_scale=0.0)
    assert float(add_adapter.scale) == 0.0             # exact no-op at 0
    default_ad = RefBagAdapter(model.dim, model.text_emb, n_slots=4)
    assert abs(float(default_ad.scale) - 1.0) < 1e-6, float(default_ad.scale)
    # the native invariant: a one-token card at alpha=1 *is* that token's text
    # embedding -- the carrier hands the LM a word it already reads instead of
    # a random 4096-d vector in a new modality
    _one = torch.tensor([[2]])
    _v1 = default_ad(_one, torch.ones(1, 1, dtype=torch.long))
    assert torch.allclose(_v1[0, 0], model.text_emb(_one)[0, 0], atol=1e-6)
    # sqrt(K) normalisation keeps the magnitude of ~one word for any card size
    _v4 = default_ad(torch.tensor([[11, 12, 13, 14]]),
                     torch.ones(1, 4, dtype=torch.long))
    _n4 = _v4.norm(dim=-1)
    # magnitude of ~one word, independent of card length (the real model's
    # rows are ~0.88; the tiny random twin's are ~7, hence the ratio)
    _row = model.text_emb(torch.tensor([[11, 12, 13, 14]]))[0].norm(dim=-1).mean()
    assert 0.5 * float(_row) < float(_n4[0, 0]) < 1.6 * float(_row), (_n4, _row)
    # ... and it decays on the spoken frames after <lookup>
    assert abs(float(_n4[0, 1] / _n4[0, 0]) - 0.85) < 1e-4, _n4
    # an empty card must inject exactly zero, even after fc1 learns a bias
    _z = default_ad(torch.tensor([[11, 12]]), torch.zeros(1, 2, dtype=torch.long))
    assert float(_z.abs().max()) == 0.0
    b_add = collate([entry], [0], "cpu", "add", None, ref_delay=0, n_slots=4)
    assert b_add["codes"].shape[-1] == T - 4, b_add["codes"].shape
    assert not b_add["metas"][0].spans_of("ref")
    sp = b_add["metas"][0].spans_of("ref_add")
    assert sp and (sp[0].start_frame, sp[0].end_frame) == (6, 10), sp
    assert b_add["ref_ids"].tolist() == [[11, 12, 13, 14]]
    assert int(b_add["codes"][0, 0, 6]) == 5

    # body anchor: <lookup> swaps into the frame before the answer, the carrier
    # rides the answer's first frames -- still zero inserted frames
    b_body = collate([entry], [0], "cpu", "add", None, ref_delay=0, n_slots=4,
                     anchor="body", text_pad=3)
    spb = b_body["metas"][0].spans_of("ref_add")
    assert spb and (spb[0].start_frame, spb[0].end_frame) == (19, 23), spb
    lkb = b_body["metas"][0].spans_of("lookup")
    assert (lkb[0].start_frame, lkb[0].end_frame) == (19, 20), lkb
    assert int(b_body["codes"][0, 0, 19]) == 5      # <lookup> moved here
    assert int(b_body["codes"][0, 0, 6]) == 3       # old slot -> text pad
    assert b_body["codes"].shape == b_add["codes"].shape
    assert _ref_vectors(add_adapter, b_body)[0] == [(0, 19, 23)]

    # scale = 0 must reproduce the no-reference forward bit-exactly
    spans_a, vecs_a = _ref_vectors(add_adapter, b_add)
    assert spans_a == [(0, 6, 10)], spans_a
    assert vecs_a.shape == (1, 4, model.dim), vecs_a.shape
    with torch.no_grad():
        o_add = model.forward_train_with_context(b_add["codes"], spans_a, vecs_a)
        o_off = model.forward_train_with_context(b_add["codes"], None, None)
    assert torch.equal(o_add.text_logits, o_off.text_logits)

    # ... and a non-zero scale must reach the body positions
    add_adapter.scale.data.fill_(1.0)
    with torch.no_grad():
        v1 = _ref_vectors(add_adapter, b_add)[1]
        o1 = model.forward_train_with_context(b_add["codes"], spans_a, v1)
        o0 = model.forward_train_with_context(b_add["codes"], spans_a,
                                              torch.zeros_like(v1))
    d = (o1.text_logits[:, 0] - o0.text_logits[:, 0]).abs()
    # the write lands on the injection frames and, being causal, reaches the
    # body (the tiny random model damps it hard, hence the loose second bound)
    assert float(d[:, 7:11].max()) > 1e-3, float(d.max())
    assert float(d[:, 20:].max()) > 1e-6, float(d.max())
    print(f"[tiny] add-mode: block stripped ({T}->{T - 4}), span {spans_a}, "
          f"inject shift {float(d[:, 7:11].max()):.4f}, "
          f"body shift {float(d[:, 20:].max()):.2e}")

    # mixed batch: a no-reference row must not shift the live rows
    b_mix = collate([entry, entries[2]], [0, 1], "cpu", "add", None, n_slots=4)
    spans_m, vecs_m = _ref_vectors(add_adapter, b_mix)
    assert spans_m == [(0, 6, 10)], spans_m
    assert vecs_m.shape[0] == 1, vecs_m.shape
    # ref dropout zeroes the vectors without touching the frames
    torch.manual_seed(1)
    b_drop = collate([entry], [0], "cpu", "add", None, ref_dropout=1.0,
                     n_slots=4)
    assert b_drop["ref_keep"].tolist() == [0.0]
    assert float(_ref_vectors(add_adapter, b_drop)[1].abs().max()) == 0.0
    assert b_drop["codes"].shape == b_add["codes"].shape

    # the gradient must reach the gate, otherwise the channel can never open
    fresh = RefBagAdapter(model.dim, model.text_emb, n_slots=4)   # live at 1.0
    sp_f, v_f = _ref_vectors(fresh, b_add)
    out_f = model.forward_train_with_context(b_add["codes"], sp_f, v_f)
    logits = out_f.text_logits[:, 0].float()
    tgt = b_add["codes"][:, 0]
    loss = torch.nn.functional.cross_entropy(
        logits[:, 1:].reshape(-1, logits.shape[-1]), tgt[:, 1:].reshape(-1))
    loss.backward()
    gs = float(fresh.scale.grad)
    print(f"[tiny] add-mode: scale={float(fresh.scale):.3f} "
          f"grad={gs:+.3e} (tiny twin; the real grad scales with model size)")
    assert abs(gs) > 1e-10, "no gradient on scale -> the channel can never open"
    for p in model.parameters():
        p.grad = None
    print("EVAL_CHECK_TINY_OK")


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    torch.set_num_threads(int(os.environ.get("RAG_NUM_THREADS", "8")))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tiny", action="store_true",
                    help="CPU self-test with a random tiny model")
    ap.add_argument("--ckpt")
    ap.add_argument("--tokens")
    ap.add_argument("--per-kind", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.tiny:
        tiny()
        return
    if not (args.ckpt and args.tokens):
        ap.error("--ckpt and --tokens are required unless --tiny")

    device = torch.device(args.device)
    model, adapter, _ = load_rag_checkpoint(args.ckpt, device)
    entries = torch.load(args.tokens, map_location="cpu",
                         weights_only=False)["entries"]

    picked, counts = [], collections.Counter()
    for i, e in enumerate(entries):
        if counts[e["kind"]] < args.per_kind:
            counts[e["kind"]] += 1
            picked.append(i)
    logger.info("picked %s", dict(counts))

    ev_args = types.SimpleNamespace(
        batch_size=2, ref_weight=0.0, ref_context_weight=0.0,
        ref_slot_weight=0.0, lookup_weight=6.0, lead_weight=1.0,
        filler_weight=1.0, body_weight=2.0, no_rag_weight=1.0,
        text_padding_weight=0.5, audio_weight=0.5, first_cb_weight=2.0,
        anchor_weight=0.0, lookup_id=32002)

    ev = evaluate(model, adapter, entries, picked, ev_args, device)
    print(f"[eval-check] text={ev['text']:.4f} audio={ev['audio']:.4f} "
          f"lookup_hit={ev['lookup_hit']:.3f} "
          f"body_ce_ref={ev['body_ce_ref']:.4f} "
          f"body_ce_zero={ev['body_ce_zero']:.4f} "
          f"delta={ev['body_ce_zero'] - ev['body_ce_ref']:+.4f}")
    for kind, s in sorted(ev["by_kind"].items()):
        print(f"[eval-check] {kind:16s} n={s['n']:3d} "
              f"ref={s['ref']:.4f} zero={s['zero']:.4f} delta={s['delta']:+.4f}")


if __name__ == "__main__":
    main()
