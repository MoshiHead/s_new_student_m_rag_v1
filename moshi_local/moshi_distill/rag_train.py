"""
rag_train.py — Stage A: teach the distilled 12.5 Hz student the RAG pathway.
==========================================================================

Trains, on top of the current S20 checkpoint:

  * `<lookup>` as a plain text token (weighted CE, no extra head),
  * a chunk-pool `RefAdapter` over the LM's own `text_emb` that fills the 16
    ref-slot frames with latent context (slot j = the j-th token chunk),
  * LoRA on the backbone, so the student learns to *use* the slots,
  * the three new token rows (`<ref_start>`, `<ref_end>`, `<lookup>`).

No teacher KD: the teacher's `text_card` is 32 000 (it cannot represent
`<lookup>`) and it never saw the reference, so its logits encode exactly the
ungrounded behaviour we are removing.  Supervision is the corpus itself.

Losses
------
    L = L_text(weighted CE) + 0.5 * L_audio(agent codebooks, CE)
        + anchor_weight * L_hidden(to the frozen init)

`ref` / `ref_slot` frames carry weight 0 — the server force-feeds that block,
the model must never learn to emit it.

Usage::

    # smoke: must overfit 4 samples (lookup emit rate -> 1.0)
    python -m moshi_local.moshi_distill.rag_train --tokens ... --overfit 4 --max-steps 200

    # real run
    python -m moshi_local.moshi_distill.rag_train --tokens ... --epochs 3 \
        --batch-size 8 --lora-r 64 --lora-lr 1e-4 --out-dir /workspace/personaplex/ckpt/rag_S21
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .backbone_distill import (apply_lora, freeze_transformer,
                               forced_prompt_len, plain_state_dict,
                               teacher_kwargs)
from .interleaver import SegmentMeta, TurnSpan
from .models.lm import LMModel
from .rag_loss_new import (compute_audio_loss, compute_text_loss_rag,
                           make_audio_synthetic_mask, make_segment_weight_tensor)
from .rag_vocab import add_special_tokens_and_resize, enable_new_token_rows
from .ref_adapter import (RefAdapter, RefAddAdapter, RefBagAdapter,
                          spans_and_rows)

logger = logging.getLogger(__name__)

PAD = -1


# ---------------------------------------------------------------------------
# Model load / save
# ---------------------------------------------------------------------------

def load_student_standalone(checkpoint: str | Path, device, dtype,
                            text_card: int | None = None) -> tuple[LMModel, dict]:
    """Rebuild the pruned student from a `backbone_distill` checkpoint alone.

    No teacher needed: the checkpoint stores `base_kwargs` and a full state
    dict, and the pruned depth is `len(kept_layers)`.
    """
    pkg = torch.load(checkpoint, map_location="cpu", weights_only=False)
    kept = list(pkg["kept_layers"])
    kw = dict(pkg.get("base_kwargs") or teacher_kwargs())
    kw["num_layers"] = len(kept)
    if text_card is not None:
        kw["text_card"] = int(text_card)
    model = LMModel(device=device, dtype=dtype, **kw)
    # LMModel builds text_linear / out_norm / depformer_in / linears without
    # device/dtype, so move the assembled model explicitly (build_student does
    # the same thing).
    model = model.to(device=device, dtype=dtype)
    missing, unexpected = model.load_state_dict(pkg["state_dict"], strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected ckpt keys: {unexpected[:5]}")
    if missing:
        raise RuntimeError(f"missing ckpt keys: {missing[:5]}")
    model._kept_layers = kept
    model._base_kwargs = kw
    model.eval()
    return model, pkg


def save_rag_checkpoint(out_dir: Path, student: LMModel, adapter: RefAdapter | None,
                        kept_layers, base_kwargs, n_slots: int,
                        tokenizer_path: str, step: int, stats: dict,
                        ref_mode: str = "latent",
                        ref_pos: str = "pre_body",
                        adapter_kind: str | None = None,
                        ref_anchor: str = "lookup") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    kw = dict(base_kwargs)
    kw["text_card"] = int(student.text_card)   # 32003 after the resize
    sd = plain_state_dict(student)
    sd = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v)
          for k, v in sd.items()}
    if adapter_kind is None:
        if isinstance(adapter, RefBagAdapter):
            adapter_kind = "bag"
        elif isinstance(adapter, RefAddAdapter):
            adapter_kind = "add"
        else:
            adapter_kind = ("latent" if adapter is not None else None)
    torch.save({
        "state_dict": sd,
        "adapter": (None if adapter is None else
                    {k: v.to(torch.bfloat16) if v.is_floating_point() else v
                     for k, v in adapter.state_dict().items()}),
        "adapter_kind": adapter_kind,
        # the carrier profile is not in the state dict but defines the vector
        "ref_decay": (float(adapter.decay)
                      if adapter is not None and hasattr(adapter, "decay")
                      else None),
        "ref_mode": ref_mode,
        "ref_pos": ref_pos,
        "ref_anchor": ref_anchor,
        "kept_layers": kept_layers,
        "base_kwargs": {k: v for k, v in kw.items()
                        if not isinstance(v, torch.dtype)},
        "n_slots": n_slots,
        "tokenizer": tokenizer_path,
        "step": step,
        "stats": {k: float(v) for k, v in stats.items()},
    }, out_dir / "student.pt")
    logger.info("saved %s", out_dir / "student.pt")


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------

def _rotate_span(s: int, e: int, a: int, b: int, bs: int):
    """Map a frame span through ``[a, b) -> just before bs``."""
    d = b - a
    if e <= a or s >= bs:
        return s, e
    if a <= s and e <= b:                  # the moved unit
        off = bs - d - a
        return s + off, e + off
    if b <= s and e <= bs:                 # the lead/filler sliding left
        return s - d, e - d
    return None                            # straddles a boundary


def move_ref_before_body(codes, turns, gap: int = 0):
    """Rotate ``<lookup> + ref block`` to sit immediately before the body.

    Training layout goes from

        <lookup>, <ref_start>, slots, <ref_end>, ... lead ... filler ... body

    to

        lead ... filler ..., <lookup>, <ref_start>, slots, <ref_end>, body

    so the reference sits 0 frames from the tokens it has to ground.  The body
    keeps its absolute frame index and the total length is unchanged, so
    padding, ``plen`` and the persona prefix are untouched.  Serving needs no
    change: the model emits ``<lookup>`` itself right before the block, so the
    injection trigger stays a single deterministic point.

    ``turns`` is ``[(kind, start, end)]`` with exclusive ends.  Entries without
    a reference (or already adjacent) are returned untouched.
    """
    t = {k: (s, x) for k, s, x in turns}
    if not all(k in t for k in ("lookup", "ref", "body")):
        return codes, turns
    a = t["lookup"][0]
    b = t["ref"][1]
    bs = t["body"][0]
    if gap > 0:                            # leave `gap` frames before the body
        # only land inside the existing silence, never inside the filler
        prev_end = max((x for k, s, x in turns
                        if x <= t["body"][0] and k != "body"), default=b)
        gap = min(int(gap), max(0, t["body"][0] - prev_end))
        bs = t["body"][0] - gap
    if a >= t["ref"][0] or b > bs or b == bs:
        return codes, turns
    new = torch.cat([codes[..., :a], codes[..., b:bs],
                     codes[..., a:b], codes[..., bs:]], dim=-1)
    if new.shape[-1] != codes.shape[-1]:
        return codes, turns
    moved = []
    for k, s, x in turns:
        r = _rotate_span(s, x, a, b, bs)
        if r is None:
            return codes, turns            # unexpected overlap: leave as is
        if r[0] < r[1]:
            moved.append((k, r[0], r[1]))
    moved.sort(key=lambda x: x[1])
    return new, moved


def entry_rotatable(entry: dict) -> bool:
    """Can ``move_ref_before_body`` rotate this entry?

    A handful of cached samples carry a body turn that starts *inside* the
    filler span (bad sidecar), so the block could not be placed before the
    body without splitting the filler.  Those are dropped from training rather
    than trained in the legacy layout.
    """
    if not entry.get("has_ref"):
        return True
    n = int(entry["codes"].shape[-1])
    turns = [(k, s, min(x, n)) for k, s, x in entry["turns"] if s < n]
    t = {k: (s, x) for k, s, x in turns}
    if not all(k in t for k in ("lookup", "ref", "body")):
        return False                       # span truncated out of the window
    a, b, bs = t["lookup"][0], t["ref"][1], t["body"][0]
    if t["ref"][1] <= t["ref"][0] or t["body"][1] <= t["body"][0]:
        return False                       # empty span (truncated sample)
    if a >= t["ref"][0] or b > bs:
        return False
    return all(_rotate_span(s, x, a, b, bs) is not None for _, s, x in turns)


def strip_ref_block(codes, turns):
    """Delete the prefill block ``<ref_start> [K] <ref_end>`` from a sample.

    Additive mode has no reserved frames -- the reference rides on the frames
    the model speaks anyway -- so the block must not exist in the training
    sequence either, or the model would learn to read it there and would have
    nothing to read at serving time.

    Returns ``(codes_without_block, remapped_turns)``; the total length shrinks
    by ``ref_end - ref_start`` (K + 2 frames) and everything after the block
    slides left.  Spans that straddle either edge are refused (the caller then
    keeps the sample as cached).
    """
    t = {k: (s, x) for k, s, x in turns}
    if "ref" not in t:
        return codes, turns
    a, b = t["ref"]
    n = int(codes.shape[-1])
    a, b = max(0, int(a)), min(int(b), n)
    if b <= a:
        return codes, turns
    d = b - a
    new = torch.cat([codes[..., :a], codes[..., b:]], dim=-1)
    out = []
    for k, s, x in turns:
        if x <= a:
            out.append((k, s, x))
        elif s >= b:
            out.append((k, s - d, x - d))
        elif s >= a and x <= b:
            continue                        # the removed block itself
        else:
            return codes, turns             # straddles: leave untouched
    return new, out


def ref_add_span(turns, k: int, delay: int = 0, body_start: int | None = None):
    """Where the K additive vectors go, in **code frame** coordinates.

    The span starts at the ``<lookup>`` frame: ``RefAdapter.inject`` writes
    ``emb[s + 1 + j] += vec[j]``, and ``emb[p]`` is the embedding of
    ``codes[p - 1]``, so ``s = lookup`` lands vec 0 on the lookup frame's own
    embedding -- exactly the call the server makes one frame after it emits
    ``<lookup>``.  The span is truncated at the body so the answer tokens never
    carry a reference vector they were not trained on.
    """
    t = {kk: (s, x) for kk, s, x in turns}
    if "lookup" not in t or k <= 0:
        return None
    s = t["lookup"][0] + int(max(0, delay))
    e = s + int(k)
    if body_start is not None:
        e = min(e, int(body_start))
    if e <= s:
        return None
    return s, e


def entry_addable(entry: dict) -> bool:
    """Can this entry be trained in additive mode (block stripped, room left)?"""
    if not entry.get("has_ref"):
        return True
    n = int(entry["codes"].shape[-1])
    turns = [(k, s, min(x, n)) for k, s, x in entry["turns"] if s < n]
    t = {k: (s, x) for k, s, x in turns}
    if not all(k in t for k in ("lookup", "ref", "body")):
        return False
    a, b = t["ref"]
    if b <= a or b > n or t["body"][0] <= t["lookup"][0]:
        return False
    # the block must come out cleanly (no span straddles either edge)
    c2, _ = strip_ref_block(entry["codes"], turns)
    if int(c2.shape[-1]) >= n:
        return False
    # at least one frame between <lookup> and the body after stripping
    return t["body"][0] - (b - a) > t["lookup"][0]


def collate(entries: list[dict], idxs: list[int], device,
            ref_mode: str = "latent", refsel: dict | None = None,
            ref_pos: str = "pre_body", ref_jitter: int = 0,
            ref_dropout: float = 0.0, ref_cards: dict | None = None,
            ref_delay: int = 0, n_slots: int = 16,
            anchor: str = "lookup", text_pad: int = 3) -> dict:
    sel = [entries[i] for i in idxs]
    B = len(sel)
    add_mode = ref_mode == "add"
    # "pause": the text card rides in the pad-text frames that already sit
    # right before the answer (median 17 of them; the audio there is pause).
    # Same frames, same text channel S23 proved (+0.10 body CE), no adapter and
    # no block of silence.
    pause_mode = ref_mode == "text" and ref_pos == "pause"

    # Layout pass.  `add` deletes the prefill block (the reference is injected
    # into existing frames); every other mode keeps it in place, optionally
    # rotating the unit to sit right before the body.
    prep = []
    for e in sel:
        c = e["codes"]
        n = int(c.shape[-1])
        turns = [(k, s, min(x, n)) for k, s, x in e["turns"] if s < n]
        ok = True
        if add_mode or pause_mode:
            if e["has_ref"]:
                c2, t2 = strip_ref_block(c, turns)
                if int(c2.shape[-1]) == n:
                    ok = False             # block could not be removed cleanly
                c, turns = c2, t2
        elif ref_pos == "pre_body":
            gap = random.randint(0, ref_jitter) if ref_jitter > 0 else 0
            c, turns = move_ref_before_body(c, turns, gap)
        prep.append((c, turns, ok))

    T = max(int(c.shape[-1]) for c, _, _ in prep)
    codes = torch.full((B, 17, T), PAD, dtype=torch.long)
    metas, has_ref, plens, turns_per = [], [], [], []
    ref_keep = [1.0] * B
    add_ids: list[list[int] | None] = [None] * B
    pause_card: list[tuple[int, int] | None] = [None] * B
    for b, (e, (c, turns, ok)) in enumerate(zip(sel, prep)):
        codes[b, :, :c.shape[-1]] = c.long()
        turns_per.append(turns)
        meta = SegmentMeta(
            turns=[TurnSpan(kind=k, start_frame=s, end_frame=x)
                   for k, s, x in turns],
            batch_idx=b)
        h = bool(e["has_ref"]) and ok
        if add_mode and h:
            # What the adapter will actually read: the fact card, one vector
            # per token, spread over the frames right after <lookup>.
            k_max = max(1, int(n_slots))
            ids = (ref_cards or {}).get(int(e.get("conv_id", 0)))
            if ids is None:
                ids = e["ref_ids"][:k_max].tolist()
            ids = [int(i) for i in list(ids) if int(i) > 0][:k_max]
            t = {k: (s, x) for k, s, x in turns}
            sp = None
            if ids and anchor == "body" and "lookup" in t and "body" in t:
                # Body anchor: move the single <lookup> token from wherever the
                # corpus put it (right after the user's turn, ~85 frames before
                # the answer) to the frame immediately before the body, and put
                # the carrier on the frames from there.  Nothing is inserted,
                # nothing is deleted: <lookup> swaps with the text token of one
                # existing frame, and vec 0 lands on that frame (muted) with the
                # rest on the answer's first frames.  The server's queue starts
                # at <lookup>, so training and serving agree with no delay.
                # Why: measured |dCE_body/dv| falls from 0.47 at body-1 to
                # 0.08-0.17 on the old lookup position, and the gate confirmed
                # it (held-out delta +0.013, shuf +0.001 after 300 steps).
                lu_old = int(t["lookup"][0])
                lu_new = max(0, int(t["body"][0]) - 1)
                if lu_new != lu_old and lu_new < int(c.shape[-1]):
                    look = int(codes[b, 0, lu_old])
                    codes[b, 0, lu_old] = int(text_pad)
                    codes[b, 0, lu_new] = look
                    # Re-append the lookup span: weight assignment is
                    # last-wins, and the carrier frame now sits inside the
                    # filler span.  Without this the <lookup> weight is 1.0
                    # instead of `lookup_weight`.
                    turns = [(kk, s, x) for kk, s, x in turns if kk != "lookup"]
                    turns.append(("lookup", lu_new, lu_new + 1))
                    turns.sort(key=lambda z: z[1])
                    meta.turns = [TurnSpan(kind=kk, start_frame=s, end_frame=x)
                                  for kk, s, x in turns]
                sp = (lu_new, min(lu_new + len(ids), int(c.shape[-1])))
                if sp[1] <= sp[0]:
                    sp = None
            elif ids:
                sp = (ref_add_span(turns, len(ids), ref_delay,
                                   body_start=t["body"][0] if "body" in t else None)
                      if ids else None)
            if sp is None:
                h = False
            else:
                meta.turns.append(TurnSpan("ref_add", sp[0], sp[1]))
                add_ids[b] = ids[: sp[1] - sp[0]]
                if ref_dropout > 0 and random.random() < ref_dropout:
                    ref_keep[b] = 0.0        # zero vectors, frames untouched
        elif pause_mode and h:
            # Text card inside the pause the model already has.  The pad-text
            # run [body-n, body-1] holds no words (audio is pause), so the card
            # ids go there verbatim, `<lookup>` moves to the frame before the
            # card, and the body is untouched.  Same text channel as the S23
            # block (the only injection that measured a grounding gain), but on
            # frames that already exist: zero added silence.
            t = {k: (s, x) for k, s, x in turns}
            if "body" not in t or "lookup" not in t:
                h = False
            else:
                b0 = int(t["body"][0])
                n_pad = 0
                j = b0 - 1
                # Pad-text ids: 3 is the text padding token, 0 marks the last
                # padding frame of a word (the corpus puts it right before the
                # answer).  The model attributes are accepted too so the same
                # code works on any tiny/twin model.
                pad_ids = {int(text_pad), 3, 0}
                while j >= 0 and int(codes[b, 0, j]) in pad_ids:
                    n_pad += 1
                    j -= 1
                ids = (ref_cards or {}).get(int(e.get("conv_id", 0)))
                if ids is None:
                    ids = refsel.get(int(e.get("conv_id", 0))) if refsel else None
                if ids is None:
                    ids = e["ref_ids"][: int(n_slots)].tolist()
                ids = [int(i) for i in list(ids) if int(i) > 0]
                k = min(len(ids), int(n_slots), max(0, n_pad - 1))
                if k < 2:
                    h = False
                else:
                    lu_old = int(t["lookup"][0])
                    lu_new = b0 - k - 1
                    if lu_new < 0:
                        h = False
                    else:
                        look = int(codes[b, 0, lu_old])
                        codes[b, 0, lu_old] = int(text_pad)
                        codes[b, 0, lu_new] = look
                        drop = ref_dropout > 0 and random.random() < ref_dropout
                        if not drop:
                            codes[b, 0, lu_new + 1:lu_new + 1 + k] = torch.tensor(
                                ids[:k], dtype=torch.long, device=codes.device)
                            pause_card[b] = (lu_new + 1, lu_new + 1 + k)
                        turns = [(kk, s, x) for kk, s, x in turns if kk != "lookup"]
                        turns.append(("lookup", lu_new, lu_new + 1))
                        turns.sort(key=lambda z: z[1])
                        # Trailing span so it overrides lead/filler/body weights:
                        # the card frames are fed, never predicted.
                        turns.append(("ref_slot", lu_new + 1, lu_new + 1 + k))
                        turns_per[b] = turns
                        meta.turns = [TurnSpan(kind=kk, start_frame=s, end_frame=x)
                                      for kk, s, x in turns]
        metas.append(meta)
        has_ref.append(h)
        plens.append(int(e["plen"]))

    if add_mode:
        L = max((len(x) for x in add_ids if x), default=1) or 1
        ref_ids = torch.zeros(B, L, dtype=torch.long)
        ref_mask = torch.zeros(B, L, dtype=torch.long)
        for b, ids in enumerate(add_ids):
            if ids:
                ref_ids[b, :len(ids)] = torch.tensor(ids, dtype=torch.long)
                ref_mask[b, :len(ids)] = 1
    else:
        L = max((int(e["ref_ids"].numel()) for e in sel), default=1) or 1
        ref_ids = torch.zeros(B, L, dtype=torch.long)
        ref_mask = torch.zeros(B, L, dtype=torch.long)
        for b, e in enumerate(sel):
            ids = e["ref_ids"]
            n = int(ids.numel())
            if n:
                ref_ids[b, :n] = ids
                ref_mask[b, :n] = 1
    # Pristine copy: the ref block as cached (latent slots / no text).  This is
    # the "reference removed" arm for the text mode ablation.
    codes_noref = codes.clone()
    if pause_mode:
        # Card removed, `<lookup>` and every frame index unchanged: the honest
        # "no reference" arm for the pause carrier.
        for b, sp in enumerate(pause_card):
            if sp is not None:
                codes_noref[b, 0, sp[0]:sp[1]] = int(text_pad)

    if ref_mode == "text" and not pause_mode:
        # Extractive compression: the passage's own tokens are written into the
        # prefill block, so the LM reads them with its ordinary text embedding
        # instead of having to learn a new input modality.
        for b, e in enumerate(sel):
            if not e["has_ref"]:
                continue
            if ref_dropout > 0 and random.random() < ref_dropout:
                continue                   # reference dropout (MoshiRAG uses 0.2)
            n = int(e["codes"].shape[-1])
            sl = [t for t in turns_per[b] if t[0] == "ref_slot"]
            if not sl:
                continue
            s, x = sl[0][1], min(sl[0][2], n)
            k = x - s
            if k <= 0:
                continue
            ids = (ref_cards or {}).get(int(e.get("conv_id", 0)))
            if ids is None:
                ids = refsel.get(int(e.get("conv_id", 0))) if refsel else None
            if ids is None:
                ids = e["ref_ids"][:k].tolist()
            ids = [int(i) for i in list(ids)[:k] if int(i) > 0]
            if ids:
                codes[b, 0, s:s + len(ids)] = torch.tensor(ids, dtype=torch.long)
    return {
        "codes": codes.to(device),
        "codes_noref": codes_noref.to(device),
        "ref_mode": ref_mode,
        "metas": metas,
        "has_ref": has_ref,
        "plens": torch.tensor(plens, dtype=torch.long, device=device),
        "ref_ids": ref_ids.to(device),
        "ref_mask": ref_mask.to(device),
        "ref_keep": torch.tensor(ref_keep, dtype=torch.float32, device=device),
        "kinds": [e["kind"] for e in sel],
    }


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def seg_weights_for(codes, metas, plens, args, head_weight: bool = True):
    B, _, T = codes.shape
    w = make_segment_weight_tensor(
        B, T, metas, codes.device,
        ref_weight=args.ref_weight,
        lookup_weight=args.lookup_weight,
        lead_weight=args.lead_weight,
        filler_weight=args.filler_weight,
        ref_context_weight=args.ref_context_weight,
        ref_slot_weight=args.ref_slot_weight,
        body_weight=args.body_weight,
        no_rag_weight=args.no_rag_weight,
    )
    # persona block is fed verbatim by the server -> not a target
    for b in range(B):
        p = int(plens[b].item())
        if p > 0:
            w[b, :p] = 0.0
    # Body *head* emphasis: the passage can only change the first few body
    # tokens (everything after that is teacher-forced continuity).  With a
    # uniform body weight those few tokens are ~10x under-weighted relative to
    # the easy tail, so the "use the reference" gradient is diluted.  Ramp the
    # weight from `body_head_weight` at the body start down to 1.0 over
    # `body_head` frames.
    hw = float(getattr(args, "body_head_weight", 1.0) or 1.0)
    nh = int(getattr(args, "body_head", 0) or 0)
    if head_weight and hw > 1.0 and nh > 0:
        for b in range(B):
            for sp in metas[b].spans_of("body"):
                lo = sp.start_frame
                for t in range(nh):
                    f = lo + t
                    if f >= T:
                        break
                    ramp = 1.0 + (hw - 1.0) * (1.0 - t / nh)
                    w[b, f] = w[b, f] * ramp
    return w


def _ref_vectors(adapter, batch):
    """Aligned ``(spans, vecs)`` for one batch, ``vecs[i]`` <-> ``spans[i]``.

    ``RefAdapter.inject`` indexes the vector tensor by *span* order, so rows
    without a reference have to be gathered out -- a mixed batch otherwise
    injects row b's vector into row b+1's frames.
    """
    if adapter is None:
        return None, None
    vecs = adapter(batch["ref_ids"], batch["ref_mask"])
    kind = "ref_add" if batch.get("ref_mode") == "add" else "ref_slot"
    spans, rows = spans_and_rows(batch["metas"], batch["has_ref"], kind)
    if not spans:
        return None, None
    live = vecs[rows]
    keep = batch.get("ref_keep")
    if keep is not None:
        live = live * keep[rows].view(-1, 1, 1).to(live.dtype)
    return spans, live


def shuffle_cards(batch):
    """Wrong-card control for the *text* carrier (no adapter to roll).

    Every reference row reads the next reference row's card in its **own**
    pause frames, same budget: if the body CE does not move, the model is not
    reading the content, it is reading "a card is present".  Frames the donor
    card cannot fill keep the no-reference padding, so the sequence length and
    every span are unchanged.  Returns ``None`` when there is nothing to swap.
    """
    codes = batch["codes"]
    metas = batch["metas"]
    has = batch["has_ref"]
    spans = {}
    for m, h in zip(metas, has):
        sl = m.spans_of("ref_slot")
        if h and sl:
            spans[m.batch_idx] = (sl[0].start_frame, sl[0].end_frame)
    rows = sorted(spans)
    if len(rows) < 2:
        return None
    out = codes.clone()
    for i, b in enumerate(rows):
        b2 = rows[(i + 1) % len(rows)]
        s, e = spans[b]
        s2, e2 = spans[b2]
        k = min(e - s, e2 - s2)
        if k <= 0:
            continue
        out[b, 0, s:s + k] = codes[b2, 0, s2:s2 + k]
        if s + k < e:
            out[b, 0, s + k:e] = batch["codes_noref"][b, 0, s + k:e]
    return out


def rag_step(student, adapter, anchor, batch, args):
    codes = batch["codes"]
    B, _, T = codes.shape
    metas = batch["metas"]

    spans, vecs = _ref_vectors(adapter, batch)

    out = student.forward_train_with_context(codes, spans, vecs)

    seg_w = seg_weights_for(codes, metas, batch["plens"], args)
    pad_ids = {student.text_padding_token_id, student.end_of_text_padding_id}
    text_loss = compute_text_loss_rag(
        out.text_logits, codes[:, :1], out.text_mask, seg_w,
        text_padding_weight=args.text_padding_weight,
        text_padding_ids=pad_ids)

    synth = make_audio_synthetic_mask(B, T, metas, codes.device)
    if batch.get("ref_mode") == "add":
        # The <lookup> frame's audio never reaches the user (the server returns
        # silence for the control frame), so scoring audio there would only
        # pull the injected carrier back out of the residual.  The spoken
        # frames after it keep their audio weight: that is the speech guard.
        keep = torch.ones(B, T, device=codes.device, dtype=synth.dtype)
        for meta in metas:
            for sp in meta.spans_of("lookup"):
                i = sp.start_frame
                if 0 <= i < T:
                    keep[meta.batch_idx, i] = 0.0
        synth = synth * keep
    audio_loss = compute_audio_loss(
        out.logits, codes[:, student.audio_offset:student.audio_offset + 16],
        out.mask, first_codebook_weight_multiplier=args.first_cb_weight,
        synthetic_keep=synth, num_agent_codebooks=8)

    loss = text_loss + args.audio_weight * audio_loss
    stats = {"text": float(text_loss), "audio": float(audio_loss)}
    if adapter is not None and getattr(adapter, "scale", None) is not None:
        stats["ref_scale"] = float(adapter.scale)
        if getattr(adapter, "alpha", None) is not None:
            stats["ref_alpha"] = float(adapter.alpha)
    if vecs is not None:
        stats["ref_norm"] = float(vecs.detach().float().norm(dim=-1).mean())

    # The anchor KL is a "do not drift from the frozen init" term.  In additive
    # mode the reference *is* a deliberate change of the text logits, so an
    # anchor that cannot read the injected vectors would penalise precisely the
    # grounding signal we are training (and an anchor that does read them is
    # just the base model plus noise).  Audio CE keeps the prior in place.
    if anchor is not None and args.anchor_weight > 0 and batch.get("ref_mode") != "add":
        with torch.no_grad():
            ref_out = anchor.forward_train_with_context(codes, spans, vecs)
        # LMOutput does not expose the backbone hidden state, so anchor in
        # text-logit space: this is exactly the behaviour we do not want to
        # drift (the audio prior is already protected by L_audio).
        p = torch.log_softmax(ref_out.text_logits.float()[:, 0], dim=-1)
        q = torch.log_softmax(out.text_logits.float()[:, 0], dim=-1)
        kl = (p.exp() * (p - q)).sum(-1)
        m = out.text_mask[:, 0].float()
        anc = (kl * m).sum() / m.sum().clamp(min=1e-8)
        loss = loss + args.anchor_weight * anc
        stats["anchor"] = float(anc)

    # diagnostics: predicted <lookup> at its own frame. emb[t] = codes[t-1], so
    # the output at frame t is what emits codes[t].
    with torch.no_grad():
        pred = out.text_logits[:, 0].argmax(-1)
        hit = tot = 0
        for meta in metas:
            for sp in meta.spans_of("lookup"):
                i = sp.start_frame
                if 0 <= i < T:
                    tot += 1
                    hit += int(pred[meta.batch_idx, i].item() == args.lookup_id)
        stats["lookup_hit"] = hit / max(tot, 1)

    return loss, stats


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

@torch.no_grad()
def body_ce(student, batch, args, vecs=None, noref: bool = False) -> tuple[float, dict, int]:
    """Weighted body CE (plus per-kind means) for one batch.

    Only entries that carry a reference contribute: with zeroed slots they are
    identical to the with-reference arm, so including them would just dilute
    the ablation.  Returns ``(ce, {kind: mean_ce}, n_ref)``.

    ``noref=True`` scores the batch with the reference removed (zero latent
    vectors in latent mode, the pristine block in text mode), which is the
    honest ablation for the text path.
    """
    has_ref = batch["has_ref"]
    n_ref = int(sum(has_ref))
    if n_ref == 0:
        return 0.0, {}, 0, float("nan")
    codes = batch["codes_noref"] if noref else batch["codes"]
    B, _, T = codes.shape
    spans = live = None
    if vecs is not None:
        kind = "ref_add" if batch.get("ref_mode") == "add" else "ref_slot"
        spans, rows = spans_and_rows(batch["metas"], has_ref, kind)
        live = vecs[rows] if spans else None
    out = student.forward_train_with_context(codes, spans, live)
    w = seg_weights_for(codes, batch["metas"], batch["plens"], args,
                        head_weight=False)
    body = torch.zeros_like(w)
    for meta, h in zip(batch["metas"], has_ref):
        if not h:
            continue
        for sp in meta.spans_of("body"):
            body[meta.batch_idx, sp.start_frame:sp.end_frame] = 1.0
    w = w * body
    ce = F.cross_entropy(
        torch.nan_to_num(out.text_logits[:, 0].float(), nan=0.0)
        .reshape(-1, out.text_logits.shape[-1]),
        codes[:, 0].reshape(-1).clamp_min(0), reduction="none").reshape(B, T)
    total = float((ce * w).sum() / w.sum().clamp_min(1e-8))
    # Body *head* CE: only the first `body_head` frames of each body span.  The
    # mean over a ~100-frame body is dominated by local continuity (the body is
    # teacher-forced, so every token after the first few is nearly deterministic)
    # and understates the reference effect by ~10x.  This is the number that
    # tracks "did the passage change the answer".
    n_head = int(getattr(args, "body_head", 0) or 0)
    head = float("nan")
    if n_head > 0:
        head_mask = torch.zeros_like(body)
        for meta, h in zip(batch["metas"], has_ref):
            if not h:
                continue
            for sp in meta.spans_of("body"):
                lo = sp.start_frame
                head_mask[meta.batch_idx, lo:lo + n_head] = 1.0
        head = float((ce * head_mask).sum() / head_mask.sum().clamp_min(1e-8))
    per_kind: dict = {}
    for b, (h, kind) in enumerate(zip(has_ref, batch["kinds"])):
        m = body[b].bool()
        if h and m.any():
            d = per_kind.setdefault(kind, {"sum": 0.0, "n": 0})
            d["sum"] += float(ce[b][m].mean())
            d["n"] += 1
    return (total, {k: v["sum"] / v["n"] for k, v in per_kind.items()},
            n_ref, head)


@torch.no_grad()
def evaluate(student, adapter, entries, idxs, args, device) -> dict:
    student.eval()
    agg = {"text": 0.0, "audio": 0.0, "lookup_hit": 0.0, "body_ce_ref": 0.0,
           "body_ce_zero": 0.0, "body_head_ref": 0.0, "body_head_zero": 0.0,
           "n": 0, "n_ref": 0}
    by_kind: dict = {}
    for j in range(0, len(idxs), args.batch_size):
        batch = collate(entries, idxs[j:j + args.batch_size], device,
                        getattr(args, "ref_mode", "latent"),
                        getattr(args, "_refsel", None),
                        getattr(args, "ref_pos", "pre_body"),
                        ref_cards=getattr(args, "_ref_cards", None),
                        ref_delay=getattr(args, "ref_delay", 0),
                        n_slots=getattr(args, "n_slots", 16),
                        anchor=getattr(args, "ref_anchor", "body"),
                        text_pad=getattr(student, "text_padding_token_id", 3))
        B = batch["codes"].shape[0]
        _, st = rag_step(student, adapter, None, batch, args)
        for key in ("text", "audio", "lookup_hit"):
            agg[key] += st[key] * B
        agg["n"] += B

        vecs = (adapter(batch["ref_ids"], batch["ref_mask"])
                if adapter is not None else None)
        ref_ce, kind_ref, n_ref, ref_head = body_ce(student, batch, args, vecs)
        agg["n_ref"] += n_ref
        if n_ref == 0:
            continue                       # nothing to ablate in this batch
        zero_ce, kind_zero, _, zero_head = body_ce(
            student, batch, args,
            torch.zeros_like(vecs) if vecs is not None else None,
            noref=vecs is None)
        agg["body_ce_ref"] += ref_ce * n_ref
        agg["body_ce_zero"] += zero_ce * n_ref
        if ref_head == ref_head:           # not NaN -> head metric enabled
            agg["body_head_ref"] += ref_head * n_ref
            agg["body_head_zero"] += zero_head * n_ref
        for kind, val in kind_ref.items():
            d = by_kind.setdefault(kind, {"ref": 0.0, "zero": 0.0, "n": 0})
            d["ref"] += val
            d["zero"] += kind_zero.get(kind, val)
            d["n"] += 1
    student.train()
    n = max(agg.pop("n"), 1)
    n_ref = max(agg.pop("n_ref"), 1)
    res = {k: v / (n_ref if k.startswith("body_") else n)
           for k, v in agg.items()}
    res["head_delta"] = res.get("body_head_zero", 0.0) - res.get("body_head_ref", 0.0)
    res["by_kind"] = {
        k: {"ref": v["ref"] / max(v["n"], 1),
            "zero": v["zero"] / max(v["n"], 1),
            "delta": (v["zero"] - v["ref"]) / max(v["n"], 1),
            "n": v["n"]}
        for k, v in by_kind.items()}
    return res


def pick_text_probe(entries, eval_idx, n: int) -> list[int]:
    """First eval-split sample of each kind, for the printed GT-vs-pred read.

    Prefers the held-out split (so the training log shows unseen samples) and
    falls back to the whole corpus if a kind is missing there.
    """
    order = ["grounded_real", "grounded_cf", "grounded_distract",
             "abstain", "no_rag_qa", "regular_chat"]
    pools: dict[str, list[int]] = {}
    for i in list(eval_idx) + list(range(len(entries))):
        e = entries[i]
        pools.setdefault(e["kind"], [])
        if e["kind"] in order and i not in pools[e["kind"]]:
            pools[e["kind"]].append(i)
    picked = []
    for kind in order:
        if pools.get(kind):
            picked.append(pools[kind][0])
        if len(picked) >= n:
            break
    return picked


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init", required=True,
                    help="pruned-backbone checkpoint to start from (also the "
                         "anchor source when --resume is used)")
    ap.add_argument("--resume", default="",
                    help="RAG checkpoint (dir or student.pt) to continue from; "
                         "LoRA is re-attached at zero delta on the merged weights")
    ap.add_argument("--start-step", type=int, default=0,
                    help="optimizer step to resume the LR schedule at")
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--tokenizer", default="/workspace/personaplex/moshi_local/tokenizer.model")
    ap.add_argument("--out-dir", default="/workspace/personaplex/ckpt/rag_S21")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="backbone LoRA + new token rows")
    ap.add_argument("--adapter-lr", type=float, default=3e-3,
                    help="the adapter is a fresh module: it needs its own "
                         "learning rate, faster than the LoRA")
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=float, default=128.0)
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument("--ref-mode", default="latent",
                    choices=["latent", "text", "add"],
                    help="latent: RefAdapter slots (needs learning). "
                         "text: write the passage's own tokens into the "
                         "prefill block (the LM reads them natively). "
                         "add: strip the block entirely and add one adapter "
                         "vector per fact token to the frames right after "
                         "<lookup> -- zero extra frames, zero silence "
                         "(MoshiRAG eq. 2)")
    ap.add_argument("--ref-pos", default="pre_body",
                    choices=["pre_body", "pre_lead", "pause"],
                    help="pre_body: rotate <lookup>+ref block to sit right "
                         "before the body (0-frame gap). pre_lead: legacy "
                         "layout, block right after <lookup>")
    ap.add_argument("--ref-jitter", type=int, default=0,
                    help="pre_body only: sample the gap between <ref_end> and "
                         "the body uniformly in [0, N] frames (robustness to "
                         "injection timing; lands only inside existing silence)")
    ap.add_argument("--ref-delay", type=int, default=0,
                    help="add mode: shift the injection this many frames later "
                         "than <lookup> (0 = start on the lookup frame, i.e. "
                         "exactly what the server does)")
    ap.add_argument("--carrier", default="bag", choices=["bag", "tokens"],
                    help="add mode: 'bag' puts the whole card into one native "
                         "text-embedding sum per carrier frame (default); "
                         "'tokens' is the old one-learned-vector-per-token")
    ap.add_argument("--ref-anchor", default="body", choices=["body", "lookup"],
                    help="add mode: where the carrier rides.  'body' moves "
                         "<lookup> to the frame before the answer (measured "
                         "5x higher body sensitivity, and the server can "
                         "prefetch the card long before); 'lookup' leaves it "
                         "right after the user turn, ~85 frames early")
    ap.add_argument("--ref-decay", type=float, default=0.85,
                    help="add mode: geometric decay of the carrier on the "
                         "spoken frames after <lookup> (frame 0 keeps 1.0)")
    ap.add_argument("--ref-dropout", type=float, default=0.0,
                    help="drop the passage (keep the empty block) on this "
                         "fraction of training samples, forcing the model to "
                         "rely on it when present")
    ap.add_argument("--refsel", default="",
                    help="optional .pt of {conv_id: [token ids]} query-selected "
                         "compressed passages (text mode only)")
    ap.add_argument("--ref-cards", default="",
                    help="optional .pt of {conv_id: [token ids]} fact cards "
                         "(build_ref_cards.py); keeps the numbers/names of the "
                         "whole passage inside the same K frames")
    ap.add_argument("--n-slots", type=int, default=16)
    ap.add_argument("--ref-weight", type=float, default=0.0)
    ap.add_argument("--ref-context-weight", type=float, default=0.0)
    ap.add_argument("--ref-slot-weight", type=float, default=0.0)
    ap.add_argument("--lookup-weight", type=float, default=6.0)
    ap.add_argument("--lead-weight", type=float, default=1.0)
    ap.add_argument("--filler-weight", type=float, default=1.0)
    ap.add_argument("--body-weight", type=float, default=2.0)
    ap.add_argument("--body-head", type=int, default=12,
                    help="frames at the start of each body span reported as "
                         "head_delta (the part the passage actually moves); "
                         "0 disables")
    ap.add_argument("--body-head-weight", type=float, default=1.0,
                    help="weight the first --body-head body frames up to this "
                         "multiple (1.0 = uniform, the old behaviour)")
    ap.add_argument("--no-rag-weight", type=float, default=1.0)
    ap.add_argument("--text-padding-weight", type=float, default=0.5)
    ap.add_argument("--audio-weight", type=float, default=0.5)
    ap.add_argument("--first-cb-weight", type=float, default=2.0)
    ap.add_argument("--anchor-weight", type=float, default=0.05)
    ap.add_argument("--overfit", type=int, default=0)
    ap.add_argument("--overfit-kind", default="grounded_cf",
                    help="--overfit takes the first N entries of this kind "
                         "(empty = any)")
    ap.add_argument("--gate-holdout", type=int, default=0,
                    help="overfit gate: also score this many *held-out* "
                         "grounded samples (right/zero/shuffled card) at every "
                         "probe; they can only be answered from the carrier")
    ap.add_argument("--eval-frac", type=float, default=0.05)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-first", type=int, default=0,
                    help="extra eval at this step (early read on the ref gate)")
    ap.add_argument("--probe-every", type=int, default=100,
                    help="ref-ablation probe on the live batch every N steps "
                         "(0 = off)")
    ap.add_argument("--text-probe", type=int, default=2,
                    help="print GT-vs-prediction text for N samples at every "
                         "eval (0 = off; 2 = one grounded_real + one "
                         "grounded_cf)")
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--autocast", default="bfloat16",
                    choices=("none", "bfloat16"))
    args = ap.parse_args()

    # Hard guard: never train the model to emit the reference block.
    if (args.ref_weight or args.ref_context_weight or args.ref_slot_weight) > 0:
        raise SystemExit(
            "ref/ref_slot text weights must stay 0 for the latent-slot design: "
            "the server force-feeds that block, so training it would teach the "
            "model to emit <ref_start>/<ref_end>/slots itself.")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    dtype = torch.bfloat16

    pkg = torch.load(args.tokens, map_location="cpu", weights_only=False)
    entries = pkg["entries"] if isinstance(pkg, dict) else pkg
    meta = pkg.get("meta", {}) if isinstance(pkg, dict) else {}
    logger.info("cache: %d entries | %s", len(entries), json.dumps(meta))

    if args.ref_mode == "add":
        keep = [i for i, e in enumerate(entries) if entry_addable(e)]
        if len(keep) != len(entries):
            logger.warning(
                "ref_mode=add: dropping %d entries with no room between "
                "<lookup> and the body", len(entries) - len(keep))
            entries = [entries[i] for i in keep]
    elif args.ref_pos == "pre_body":
        keep = [i for i, e in enumerate(entries) if entry_rotatable(e)]
        if len(keep) != len(entries):
            logger.warning(
                "ref_pos=pre_body: dropping %d entries whose turn spans cannot "
                "be rotated (body starts inside the filler)",
                len(entries) - len(keep))
            entries = [entries[i] for i in keep]

    import sentencepiece
    tokenizer = sentencepiece.SentencePieceProcessor(model_file=args.tokenizer)

    if args.resume:
        from .rag_probe import load_rag_checkpoint
        student, adapter_loaded, rpkg = load_rag_checkpoint(
            args.resume, torch.device(args.device))
        student = student.to(dtype=torch.float32)
        kept = list(rpkg["kept_layers"])
        base_kwargs = dict(rpkg["base_kwargs"])
        ids = {"<ref_start>": 32000, "<ref_end>": 32001, "<lookup>": 32002}
        args.lookup_id = ids["<lookup>"]
        logger.info("resume: %s | step=%s | text_card=%d",
                    args.resume, rpkg.get("step"), student.text_card)
    else:
        student, init_pkg = load_student_standalone(args.init, device, torch.float32)
        kept = list(student._kept_layers)
        base_kwargs = dict(student._base_kwargs)
        adapter_loaded = None

        ids = add_special_tokens_and_resize(student, tokenizer)
        args.lookup_id = ids["<lookup>"]
        logger.info("control tokens: %s | text_card=%d", ids, student.text_card)

    anchor = None
    if args.anchor_weight > 0:
        # Build at the original 32 000 vocab, then apply the same resize and
        # copy the three tables from the trainable model (identical init, and
        # no random rows in the anchor).
        anchor, _ = load_student_standalone(args.init, device, dtype)
        add_special_tokens_and_resize(anchor, tokenizer)
        anchor.text_emb.load_state_dict(student.text_emb.state_dict())
        anchor.text_linear.load_state_dict(student.text_linear.state_dict())
        anchor.depformer_text_emb.load_state_dict(
            student.depformer_text_emb.state_dict())
        anchor.eval()
        for p in anchor.parameters():
            p.requires_grad_(False)
        logger.info("anchor: frozen init copy in %s, weight %.3f",
                    dtype, args.anchor_weight)

    # Trainable set
    for p in student.parameters():
        p.requires_grad_(False)
    apply_lora(student, r=args.lora_r, alpha=args.lora_alpha,
               dropout=args.lora_dropout)
    freeze_transformer(student, keep_top=None)
    enable_new_token_rows(student)
    student.train()

    adapter = None
    if args.ref_mode == "add":
        if args.carrier == "bag":
            adapter = RefBagAdapter(dim=student.dim,
                                    text_emb=student.text_emb,
                                    n_slots=args.n_slots,
                                    decay=args.ref_decay).to(device)
            logger.info("carrier=bag: the card's own text embeddings are "
                        "summed into the frames after <lookup>, full strength "
                        "on the (muted) lookup frame, decay %.2f afterwards",
                        args.ref_decay)
        else:
            adapter = RefAddAdapter(dim=student.dim,
                                    text_emb=student.text_emb,
                                    n_slots=args.n_slots).to(device)
            logger.info("carrier=tokens: one learned vector per card token")
        if adapter_loaded is not None:
            adapter.load_state_dict(adapter_loaded.state_dict())
            logger.info("resume: additive adapter restored")
        adapter.train()
        logger.info("ref_mode=add: K<=%d vectors added to the frames after "
                    "<lookup>, no prefill block, no silence", args.n_slots)
    elif args.ref_mode == "latent":
        adapter = RefAdapter(dim=student.dim, text_emb=student.text_emb,
                             n_slots=args.n_slots).to(device)
        if adapter_loaded is not None:
            assert int(rpkg["n_slots"]) == args.n_slots, (rpkg["n_slots"], args.n_slots)
            adapter.load_state_dict(adapter_loaded.state_dict())
            logger.info("resume: adapter restored")
        adapter.train()
    else:
        logger.info("ref_mode=text: passage tokens in the prefill block, "
                    "no adapter")
    if args.ref_mode != "add":
        logger.info("ref_pos=%s: %s", args.ref_pos,
                    {"pre_body": "block rotated to sit right before the body",
                     "pause": "card written into the existing pre-body pad "
                              "frames, <lookup> moved to body-K-1 (0 added "
                              "frames)"}.get(args.ref_pos,
                                             "legacy layout after <lookup>"))

    args._refsel = None
    if args.refsel:
        args._refsel = torch.load(args.refsel, map_location="cpu",
                                  weights_only=False)
        logger.info("refsel: %d query-selected passages", len(args._refsel))

    args._ref_cards = None
    if args.ref_cards:
        args._ref_cards = torch.load(args.ref_cards, map_location="cpu",
                                     weights_only=False)["cards"]
        logger.info("ref cards: %d fact cards from %s",
                    len(args._ref_cards), args.ref_cards)

    lora_params = [p for n, p in student.named_parameters()
                   if "parametrizations." in n and not n.endswith(".original")]
    row_params = [p for mod in (student.text_emb, student.text_linear,
                                student.depformer_text_emb)
                  for p in mod.parameters() if p.requires_grad]
    groups = [
        {"params": lora_params, "lr": args.lr, "name": "lora"},
        # weight_decay=0: the rows are mostly frozen (gradient hooks zero them);
        # decay would still shrink the whole base table every step.
        {"params": row_params, "lr": args.lr, "name": "rows",
         "weight_decay": 0.0},
    ]
    if adapter is not None:
        groups.append({"params": list(adapter.parameters()),
                       "lr": args.adapter_lr, "name": "adapter"})
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)

    if args.overfit:
        # A gate needs *grounded* samples: the whole point is whether the card
        # moves the body.  Take the first N of one kind, not the first N rows.
        pool = [i for i, e in enumerate(entries)
                if not args.overfit_kind or e.get("kind") == args.overfit_kind]
        if not pool:
            raise SystemExit(f"no entries of kind {args.overfit_kind!r}")
        n_fit = min(args.overfit, len(pool))
        train_idx = pool[:n_fit] * 8
        eval_idx = []
        logger.info("OVERFIT MODE: %d distinct samples (kind=%s, ids=%s)",
                    n_fit, args.overfit_kind or "any", pool[:n_fit])
    else:
        n_eval = max(1, int(len(entries) * args.eval_frac))
        shuf = list(range(len(entries)))
        random.Random(0).shuffle(shuf)
        eval_idx, train_idx = shuf[:n_eval], shuf[n_eval:]

    steps_per_epoch = max(1, len(train_idx) // (args.batch_size * args.grad_accum))
    total_steps = args.max_steps or int(steps_per_epoch * args.epochs)
    logger.info("train=%d eval=%d | %d steps/epoch -> %d steps",
                len(train_idx), len(eval_idx), steps_per_epoch, total_steps)

    text_probe_ids, text_decode = [], None
    if args.text_probe:
        from .rag_text_probe import make_decoder, print_sample, probe_sample
        import sentencepiece as spm
        text_decode = make_decoder(spm.SentencePieceProcessor(args.tokenizer))
        text_probe_ids = pick_text_probe(entries, eval_idx, args.text_probe)
        logger.info("text probe at each eval: %s",
                    [entries[i]["kind"] for i in text_probe_ids])

    # Gate read (P2): grounded samples the model never trains on.  `--overfit`
    # drives CE to ~0 on its 8 samples (memorisation), where delta/shuf_delta
    # are meaningless; the held-out pair can only be answered from the card.
    hold_batch = None
    if args.overfit and args.gate_holdout:
        _pool = [i for i, e in enumerate(entries)
                 if not args.overfit_kind or e.get("kind") == args.overfit_kind]
        _hi = _pool[args.overfit:args.overfit + args.gate_holdout]
        if _hi:
            hold_batch = collate(
                entries, _hi, device, args.ref_mode,
                getattr(args, "_refsel", None), args.ref_pos,
                ref_cards=getattr(args, "_ref_cards", None),
                ref_delay=args.ref_delay, n_slots=args.n_slots,
                anchor=args.ref_anchor,
                text_pad=student.text_padding_token_id)
            logger.info("[gate] holdout ids %s (%d rows, never trained on)",
                        _hi, hold_batch["codes"].shape[0])

    use_autocast = args.autocast == "bfloat16"
    def _ac():
        return torch.autocast(device.type, dtype=torch.bfloat16,
                              enabled=use_autocast)

    step, micro, t0 = args.start_step, 0, time.time()
    opt.zero_grad()
    order = list(train_idx)
    random.Random(args.seed).shuffle(order)
    cursor = 0
    if args.start_step > 0:
        cursor = ((args.start_step * args.grad_accum * args.batch_size)
                  % max(len(order), 1))
        logger.info("resume: starting at step %d (data cursor %d)", step, cursor)

    while step < total_steps:
        if cursor + args.batch_size > len(order):
            random.Random(args.seed + step).shuffle(order)
            cursor = 0
        idxs = order[cursor:cursor + args.batch_size]
        cursor += args.batch_size
        batch = collate(entries, idxs, device, args.ref_mode, args._refsel,
                        args.ref_pos, args.ref_jitter, args.ref_dropout,
                        getattr(args, "_ref_cards", None),
                        ref_delay=args.ref_delay, n_slots=args.n_slots,
                        anchor=args.ref_anchor,
                        text_pad=student.text_padding_token_id)

        with _ac():
            loss, stats = rag_step(student, adapter, anchor, batch, args)

        (loss / args.grad_accum).backward()
        micro += 1
        if micro < args.grad_accum:
            continue
        micro = 0
        torch.nn.utils.clip_grad_norm_(
            [p for g in groups for p in g["params"]], args.grad_clip)
        # warmup + cosine.  The origin is the *resume* step: a fresh adapter
        # attached at step 1750 must not inherit a schedule that is already
        # 2/3 spent (S24 ran the adapter at 0.33x -> 0.03x of its LR and it
        # never learned anything visible).
        sched0 = int(getattr(args, "start_step", 0) or 0)
        if step < sched0 + args.warmup:
            scale = (step - sched0 + 1) / max(args.warmup, 1)
        else:
            prog = ((step - sched0 - args.warmup)
                    / max(total_steps - sched0 - args.warmup, 1))
            scale = 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))
        for g in groups:
            g["lr"] = {"lora": args.lr, "rows": args.lr,
                       "adapter": args.adapter_lr}[g["name"]] * scale
        if adapter is not None and step % args.log_every == 0:
            # does the write side move at all?  Flat at 0 => the channel is
            # disconnected no matter what the loss says.
            gn = sum(float(p.grad.detach().float().norm()) ** 2
                     for p in adapter.parameters() if p.grad is not None)
            stats["ref_grad"] = gn ** 0.5
        opt.step()
        opt.zero_grad()
        step += 1

        if step % args.log_every == 0:
            logger.info("step %d/%d lr=%.2e %s (%.0fs)", step, total_steps,
                        groups[1]["lr"],
                        " ".join(f"{k}={v:.4f}" for k, v in stats.items()),
                        time.time() - t0)
        # Cheap live read on the ref gate: one extra forward on the batch that
        # is still in memory, with the reference zeroed.  No eval split, no
        # sampling bias -- if this stays at ~0 the adapter is being ignored
        # long before the next checkpoint.
        if args.probe_every and step % args.probe_every == 0:
            with _ac():
                vecs = (adapter(batch["ref_ids"], batch["ref_mask"])
                        if adapter is not None else None)
                ref_ce, _, n_ref, ref_head = body_ce(student, batch, args, vecs)
                if n_ref:
                    zero_ce, _, _, zero_head = body_ce(
                        student, batch, args,
                        torch.zeros_like(vecs) if vecs is not None else None,
                        noref=vecs is None)
                    # "does the *content* matter?" -- give this row another
                    # row's card.  If the body CE does not move, the carrier
                    # is a learned bias, not a reference.
                    shuf_ce = float("nan")
                    if vecs is not None and vecs.shape[0] > 1:
                        v_sw = adapter(batch["ref_ids"].roll(1, 0),
                                       batch["ref_mask"].roll(1, 0))
                        shuf_ce, _, _, _ = body_ce(student, batch, args, v_sw)
                    elif batch.get("ref_mode") == "text":
                        sh_card = shuffle_cards(batch)
                        if sh_card is not None:
                            b_sw = dict(batch)
                            b_sw["codes"] = sh_card
                            shuf_ce, _, _, _ = body_ce(student, b_sw, args, None)
            if n_ref:
                logger.info("[probe] step %d n_ref=%d body_ce_ref=%.4f "
                            "body_ce_zero=%.4f delta=%+.4f head_delta=%+.4f "
                            "shuf_delta=%+.4f",
                            step, n_ref, ref_ce, zero_ce, zero_ce - ref_ce,
                            (zero_head - ref_head)
                            if ref_head == ref_head else float("nan"),
                            shuf_ce - ref_ce)
            if hold_batch is not None:
                with torch.no_grad(), _ac():
                    hv = (adapter(hold_batch["ref_ids"], hold_batch["ref_mask"])
                          if adapter is not None else None)
                    h_ref, _, hn, h_head = body_ce(student, hold_batch, args, hv)
                    h_zero, _, _, hz_head = body_ce(
                        student, hold_batch, args,
                        torch.zeros_like(hv) if hv is not None else None,
                        noref=hv is None)
                    h_shuf = float("nan")
                    if hv is not None and hv.shape[0] > 1:
                        hsv = adapter(hold_batch["ref_ids"].roll(1, 0),
                                      hold_batch["ref_mask"].roll(1, 0))
                        h_shuf, _, _, _ = body_ce(student, hold_batch, args, hsv)
                    elif hold_batch.get("ref_mode") == "text":
                        hs_card = shuffle_cards(hold_batch)
                        if hs_card is not None:
                            hb_sw = dict(hold_batch)
                            hb_sw["codes"] = hs_card
                            h_shuf, _, _, _ = body_ce(student, hb_sw, args, None)
                logger.info("[gate] step %d holdout n_ref=%d body_ce_ref=%.4f "
                            "body_ce_zero=%.4f delta=%+.4f head_delta=%+.4f "
                            "shuf_delta=%+.4f",
                            step, hn, h_ref, h_zero, h_zero - h_ref,
                            (hz_head - h_head) if h_head == h_head
                            else float("nan"),
                            h_shuf - h_ref)
        if eval_idx and (step % args.eval_every == 0
                         or (args.eval_first and step == args.eval_first)):
            ev = evaluate(student, adapter, entries, eval_idx, args, device)
            logger.info("[eval] step %d text=%.4f audio=%.4f lookup_hit=%.3f "
                        "body_ce_ref=%.4f body_ce_zero=%.4f delta=%.4f "
                        "head_delta=%.4f",
                        step, ev["text"], ev["audio"], ev["lookup_hit"],
                        ev["body_ce_ref"], ev["body_ce_zero"],
                        ev["body_ce_zero"] - ev["body_ce_ref"],
                        ev["head_delta"])
            for kind, s in sorted(ev["by_kind"].items()):
                logger.info("[eval]   %-16s n=%4d body_ce_ref=%.4f "
                            "body_ce_zero=%.4f delta=%+.4f",
                            kind, s["n"], s["ref"], s["zero"], s["delta"])
            # GT vs prediction text, ref arm vs zeroed-ref arm.  Same samples
            # at every eval, so the transcripts are directly comparable across
            # steps (and the counterfactual sample is the honest test: its body
            # fact exists *only* in the reference).
            for i in text_probe_ids:
                try:
                    rec = probe_sample(student, adapter, entries[i], device,
                                       text_decode, ref_mode=args.ref_mode,
                                       ref_pos=args.ref_pos,
                                       ref_cards=getattr(args, "_ref_cards", None),
                                       n_slots=args.n_slots,
                                       ref_delay=args.ref_delay,
                                       anchor=args.ref_anchor,
                                       text_pad=student.text_padding_token_id)
                    print_sample(rec)
                except Exception as exc:                  # never kill training
                    logger.warning("text probe failed for entry %d: %s", i, exc)
        if step % args.save_every == 0:
            save_rag_checkpoint(Path(args.out_dir) / f"step_{step}", student,
                                adapter, kept, base_kwargs, args.n_slots,
                                args.tokenizer, step, stats, args.ref_mode,
                                args.ref_pos, ref_anchor=args.ref_anchor)

    save_rag_checkpoint(Path(args.out_dir) / f"step_{step}", student, adapter,
                        kept, base_kwargs, args.n_slots, args.tokenizer, step,
                        stats, args.ref_mode, args.ref_pos,
                        ref_anchor=args.ref_anchor)
    logger.info("done: %d steps", step)


if __name__ == "__main__":
    main()
