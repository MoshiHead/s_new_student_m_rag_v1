"""
rag_stream_check.py — prove the serving prefill reproduces the training forward
==============================================================================

The RAG design lives or dies on one detail: the K reference vectors must land
on the *same* transformer positions during serving as they did in
`forward_train_with_context`.  This script runs both paths on the same token
stream and compares the text logits frame by frame.

    training : forward_train_with_context(codes, ref_slot_spans, vecs)
    serving  : LMGen.step / step_embeddings_with_tokens, one frame at a time

With every token forced, the two must agree everywhere (a streaming cache is
just the causal forward).  A mismatch means the +1 embedding offset or the
forced-frame ordering is wrong — exactly the bug this test exists to catch.

Usage
-----
    # fast, CPU, no checkpoint: random tiny model, same code paths
    python -m moshi_distill.rag_stream_check --tiny

    # real checkpoint + a cached sample (needs ~20 GB GPU)
    python -m moshi_distill.rag_stream_check \
        --ckpt /workspace/personaplex/ckpt/rag_S21/step_250/student.pt \
        --tokens /workspace/personaplex/rag_v3_tokens.pt --sample-index 0
"""

from __future__ import annotations

import argparse
import logging

import torch

from .models.lm import LMModel, LMGen, _delay_sequence
from .ref_adapter import RefAdapter

logger = logging.getLogger(__name__)


def slot_frame(codes: torch.Tensor, rs: int) -> torch.Tensor:
    """The single forced frame the server embeds for every inner slot."""
    return codes[:, :, rs:rs + 1]


def reset_model_streaming(model: LMModel) -> None:
    """Drop the KV cache a previous ``LMGen`` left on the model.

    The transformer's streaming state lives on the *model*, so a batched
    ``forward_train_with_context`` after any streaming run would silently read
    the leftover cache instead of the sequence.  ``reset_streaming`` raises
    when nothing is streaming, hence the guard.
    """
    if getattr(model, "_streaming_state", None) is not None:
        model.reset_streaming()


def _capture_logits(lm_gen: LMGen) -> list[torch.Tensor]:
    """Record the text logits of every streaming call."""
    captured: list[torch.Tensor] = []
    orig = lm_gen.process_transformer_output

    def patched(transformer_out, text_logits, *a, **kw):
        captured.append(text_logits.detach().clone())
        return orig(transformer_out, text_logits, *a, **kw)

    lm_gen.process_transformer_output = patched  # type: ignore[method-assign]
    return captured


@torch.no_grad()
def compare(model: LMModel, codes: torch.Tensor,
            rs: int, re: int, vecs: torch.Tensor,
            tol: float = 1e-4) -> dict:
    """Run training and streaming forwards over `codes`.

    `rs`/`re` are the **ref_slot** span (start of slot 0, exclusive end), i.e.
    `ref_end` sits at frame `re` and `ref_start` at `rs - 1`.  `vecs` is
    [1, K, dim] with K == re - rs.
    """
    B, Kc, T = codes.shape
    assert Kc == model.num_codebooks, (Kc, model.num_codebooks)
    assert vecs.shape[1] == re - rs, (vecs.shape, rs, re)
    spans = [(0, rs, re)]

    reset_model_streaming(model)
    out_train = model.forward_train_with_context(codes, spans, vecs)
    train_logits = out_train.text_logits[:, 0]           # [B, T, V]
    assert train_logits.shape[1] == T, train_logits.shape

    # Ground-truth input embeddings, built exactly like
    # `forward_train_with_context`: emb[0] = initial, emb[p] = delayed[p - 1].
    initial = model._get_initial_token().expand(B, -1, -1)
    delayed = _delay_sequence(model.delays, codes, initial)
    emb_all = model.embed_codes(
        torch.cat([initial, delayed[:, :, :-1]], dim=2))   # [B, T, dim]

    # The server does not replay the delayed history: inside the forced block
    # every frame is the same silence/sine frame, so it embeds one slot frame
    # directly.  Track the largest gap between that shortcut and emb_all.
    slot_emb = model.embed_codes(slot_frame(codes, rs))
    shortcut_gap = 0.0

    device = codes.device
    lm_gen = LMGen(model, device=str(device), use_sampling=False,
                   temp=0.0, temp_text=0.0, check=True)
    lm_gen.streaming_forever(1)
    captured = _capture_logits(lm_gen)

    # offset 0: initialize the cache with the initial token (training's
    # index-0 slot); supplied tokens are intentionally discarded there.
    lm_gen.step()

    for f in range(T):
        text = codes[:, 0, f]
        agent = codes[:, 1:9, f:f + 1]
        user = codes[:, 9:17, f:f + 1]
        # call at offset f+1 writes codes[f] and feeds input position f,
        # i.e. codes[f-1] (plus the adapter vector inside the slot span).
        if rs + 1 <= f <= re:
            j = f - (rs + 1)
            shortcut_gap = max(shortcut_gap, float(
                (emb_all[:, f:f + 1] - slot_emb).abs().max()))
            emb = emb_all[:, f:f + 1] + vecs[:, j:j + 1]
            lm_gen.step_embeddings_with_tokens(
                emb, input_tokens=user, moshi_tokens=agent,
                text_token=text)
        else:
            lm_gen.step(input_tokens=user, moshi_tokens=agent,
                        text_token=text)

    lm_gen._stop_streaming()      # leave the model clean for the next test
    # captured[c] is the logits produced by the call writing frame c
    assert len(captured) == T, (len(captured), T)
    stream_logits = torch.cat(captured, dim=2)[:, 0]      # [B, T, V]
    diff = (stream_logits.float() - train_logits.float()).abs()

    body = slice(re + 1, T)                              # generation after ref
    ref = slice(rs - 1, re + 1)                          # the forced block
    return {
        "max_diff_all": float(diff.max()),
        "max_diff_body": float(diff[:, body].max()) if T > re + 1 else 0.0,
        "max_diff_ref": float(diff[:, ref].max()),
        "argmax_match": float((stream_logits.argmax(-1)
                               == train_logits.argmax(-1)).float().mean()),
        "slot_shortcut_gap": shortcut_gap,
        "ok": float(diff.max()) <= tol,
        "T": T, "rs": rs, "re": re, "K": re - rs,
    }


@torch.no_grad()
def compare_add(model: LMModel, codes: torch.Tensor, lu: int, k: int,
                vecs: torch.Tensor, tol: float = 1e-4) -> dict:
    """Additive mode: ``mode="add"`` training vs ``LMGen.step_with_ref``.

    The span is ``(lu, lu + k)`` (see ``rag_train.ref_add_span``), so training
    writes ``vec[j]`` into ``emb[lu + 1 + j]``.  The streaming call that feeds
    embedding position ``p`` is the one writing ``codes[p]``, so the server
    injects on the ``k`` calls after the one that emitted ``<lookup>`` -- no
    frame is added anywhere.
    """
    B, Kc, T = codes.shape
    assert Kc == model.num_codebooks, (Kc, model.num_codebooks)
    assert lu + k < T, (lu, k, T)
    spans = [(0, lu, lu + k)]

    # A previous `LMGen` leaves the transformer in streaming mode (the KV cache
    # lives on the model, not on the LMGen), and the batched forward below would
    # then read that cache instead of the sequence.
    reset_model_streaming(model)
    out_train = model.forward_train_with_context(codes, spans, vecs)
    train_logits = out_train.text_logits[:, 0]

    lm_gen = LMGen(model, device=str(codes.device), use_sampling=False,
                   temp=0.0, temp_text=0.0, check=True)
    lm_gen.streaming_forever(1)
    captured = _capture_logits(lm_gen)
    lm_gen.step()                      # offset 0: initial token

    for f in range(T):
        text = codes[:, 0, f]
        agent = codes[:, 1:9, f:f + 1]
        user = codes[:, 9:17, f:f + 1]
        if lu + 1 <= f <= lu + k:
            j = f - (lu + 1)
            lm_gen.step_with_ref(input_tokens=user, moshi_tokens=agent,
                                 text_token=text, ref_vec=vecs[:, j:j + 1])
        else:
            lm_gen.step(input_tokens=user, moshi_tokens=agent, text_token=text)

    lm_gen._stop_streaming()      # leave the model clean for the next test
    assert len(captured) == T, (len(captured), T)
    stream_logits = torch.cat(captured, dim=2)[:, 0]
    diff = (stream_logits.float() - train_logits.float()).abs()
    return {
        "max_diff_all": float(diff.max()),
        "max_diff_body": float(diff[:, lu + k + 1:].max()),
        "argmax_match": float((stream_logits.argmax(-1)
                               == train_logits.argmax(-1)).float().mean()),
        "ok": float(diff.max()) <= tol,
        "T": T, "lu": lu, "K": k,
    }


def tiny_model(text_card: int = 64, card: int = 32, dim: int = 64,
               n_q: int = 16) -> LMModel:
    """Mirror the real base_kwargs (S20) at 1/64 scale."""
    delays = [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
    assert len(delays) == n_q + 1
    model = LMModel(device="cpu", dtype=torch.float32,
                    delays=delays, n_q=n_q, dep_q=16,
                    card=card, text_card=text_card, dim=dim, num_heads=4,
                    hidden_scale=2, num_layers=2, context=32,
                    gating="silu", norm="layer_norm", causal=True,
                    depformer_dim=32, depformer_num_heads=4,
                    depformer_dim_feedforward=64, depformer_num_layers=2,
                    depformer_causal=True, depformer_context=8,
                    depformer_multi_linear=True,
                    depformer_weights_per_step=True,
                    depformer_pos_emb="none")
    model.eval()
    return model


def tiny() -> None:
    torch.manual_seed(0)
    text_card, card, dim = 64, 32, 64
    model = tiny_model(text_card, card, dim)

    T = 40
    rs, re = 13, 17          # slots 13..16, ref_start 12, ref_end 17
    codes = torch.randint(0, card, (1, 17, T))
    codes[:, 0] = torch.randint(0, text_card, (1, T))
    codes[:, 0, 11] = 5      # <lookup>
    codes[:, 0, 12] = 6      # <ref_start>
    codes[:, 0, rs:re] = -1  # slot text placeholders
    codes[:, 0, re] = 7      # <ref_end>
    # forced audio, exactly like `_build_ref_slot_frames`
    SIL, SINE = 21, 22
    codes[:, 1:9, rs - 1:re + 1] = SIL
    codes[:, 9:17, rs - 1:re + 1] = SINE
    codes[:, 9, rs:re] = -1                  # user cb9 zero on inner slots
    vecs = torch.randn(1, re - rs, dim)

    res = compare(model, codes, rs, re, vecs, tol=1e-2)  # fp32 batched-vs-step
    for k, v in res.items():
        print(f"{k:16s} {v}")
    if not res["ok"] or res["argmax_match"] < 1.0:
        raise SystemExit("FAIL: streaming != training")

    # ---- additive path: no block, vec[j] on the frame reading lu + j ------
    codes_a = torch.randint(0, card, (1, 17, T))
    codes_a[:, 0] = torch.randint(0, text_card, (1, T))
    codes_a[:, 0, 11] = 5                       # <lookup>
    vadd = torch.randn(1, 4, dim)
    # fresh model: the transformer's KV cache lives on the module, and a model
    # that has already streamed once keeps a little state around (same reason
    # `compare` above needed `_stop_streaming`).
    res_a = compare_add(tiny_model(), codes_a, 11, 4, vadd, tol=1e-2)
    for k, v in res_a.items():
        print(f"[add] {k:16s} {v}")
    if not res_a["ok"] or res_a["argmax_match"] < 1.0:
        raise SystemExit("FAIL: additive streaming != training")
    print("OK: streaming prefill is equivalent to the training forward")
    print("OK: additive injection is equivalent to mode='add' training")


def real(args) -> None:
    from .rag_probe import load_rag_checkpoint
    from .models.lm import LMModel as _LM  # noqa: F401

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, adapter, pkg = load_rag_checkpoint(args.ckpt, device)
    pkg_tok = torch.load(args.tokens, map_location="cpu", weights_only=False)
    e = pkg_tok["entries"][args.sample_index]
    from .rag_train import move_ref_before_body
    codes = e["codes"].long()
    n = int(codes.shape[-1])
    turns = [(k, s, min(x, n)) for k, s, x in e["turns"] if s < n]
    if pkg.get("ref_pos", "pre_lead") == "pre_body":
        codes, turns = move_ref_before_body(codes, turns)
    tmap = {k: (s, x) for k, s, x in turns}
    rs, re = tmap["ref_slot"]
    K = re - rs

    codes = codes.to(device)[None]
    ref_ids = e["ref_ids"].long().to(device)[None]
    mask = torch.ones_like(ref_ids)
    vecs = adapter(ref_ids, mask)[:, :K]
    assert vecs.shape[1] == K

    res = compare(model, codes, rs, re, vecs, tol=0.05)  # bf16 tolerance
    for k, v in res.items():
        print(f"{k:16s} {v}")
    print("sample:", e["kind"], e.get("conv_id"), "| T", res["T"])
    if not res["ok"] or res["argmax_match"] < 0.99:
        raise SystemExit("FAIL: streaming != training")


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--ckpt", type=str)
    ap.add_argument("--tokens", type=str)
    ap.add_argument("--sample-index", type=int, default=0)
    args = ap.parse_args()
    if args.ckpt:
        real(args)
    else:
        tiny()


if __name__ == "__main__":
    main()
