"""
layer_init.py — build a smaller 6.25 Hz student from the TEACHER'S OWN LAYERS
============================================================================

"Smaller without training from scratch": instead of random-initialising the
student's temporal transformer, take a SUBSET of the teacher's transformer
layers (bit-exact copy) and keep the teacher's embeddings + the shared frozen
heads. The student then starts as a functioning PersonaPlex and only has to
learn (a) the 12.5 -> 6.25 -> 12.5 adapter mapping and (b) the serving schedule.

Why this matters (measured on personaplex):
    - a random-init student plateaued at state_cos 0.85 on real data -> salad;
    - the teacher's adjacent-frame hidden cosine is only ~0.65, so the 6.25 Hz
      schedule's one-frame-late slots need the `streaming_fix="augment"` path
      (see models/student.py) to stay usable.

The recipe
----------
    # 1. build the initial 16-layer student checkpoint (no training yet)
    python -m moshi_distill.layer_init --create \
        --moshi-weight /path/model.safetensors \
        --num-layers 16 --streaming-fix augment \
        --out ckpt/layer_init_16L

    # 2. train it (train_distill picks the cfg up from the checkpoint)
    python -m moshi_distill.train_distill \
        --train-data data/train.jsonl --resume ckpt/layer_init_16L \
        --lambda-state 30 --state-warmup-steps 3000 ...

Notes
-----
- Hyper-parameters (dim, heads, FFN size, norm, gating, rope) are DERIVED FROM
  THE TEACHER and validated by a strict per-layer `load_state_dict`, so a
  mismatch fails loudly instead of silently mis-building the model.
- The teacher is only read (values are copied), never aliased: an optimizer
  step on the student cannot touch it (checked in `--self-test`).
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from dataclasses import asdict, replace
from pathlib import Path

import torch

from .models.lm import LMModel
from .models.student import StudentConfig, StudentLM

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Layer selection
# ---------------------------------------------------------------------------

def select_layer_indices(n_src: int, n_dst: int, mode: str = "even") -> list[int]:
    """Which teacher layers to keep.

    "even": evenly spaced across the depth (32 -> 16 gives 0,2,...,30; 32 -> 24
            drops every 4th layer). "first": the first n_dst layers.
    """
    assert 0 < n_dst <= n_src, (n_dst, n_src)
    if n_dst == n_src:
        return list(range(n_src))
    if mode == "first":
        return list(range(n_dst))
    assert mode == "even", mode
    idx, seen = [], set()
    for i in range(n_dst):
        j = round(i * (n_src - 1) / max(1, n_dst - 1))
        while j in seen and j + 1 < n_src:
            j += 1
        seen.add(j)
        idx.append(j)
    return idx


def _effective_ffn_width(layer: torch.nn.Module) -> int:
    """The built FFN's *effective* width (what the layer's weights actually have).

    For the GLU gating used by personaplex this is 2/3 of `dim_feedforward`, so
    it is only a starting point — see `_infer_layer_spec`."""
    sd = layer.state_dict()
    for k, v in sd.items():
        if "gating" in k and k.endswith("linear_out.weight"):     # [dim, eff]
            return int(v.shape[1])
    for k, v in sd.items():
        if "gating" in k and k.endswith("linear_in.weight"):      # [2*eff, dim]
            return int(v.shape[0] // 2)
    for k, v in sd.items():
        if k.endswith(".linear1.weight") or k.endswith("ff.linear1.weight"):
            return int(v.shape[0])                                # [eff, dim]
    raise RuntimeError("could not measure the layer's FFN width")


def _infer_layer_spec(layer: torch.nn.Module, dim: int,
                      num_heads: int) -> tuple[str, str, int]:
    """(norm, gating, dim_feedforward) of a teacher layer, by building probes.

    Guessing from names or single weight shapes is not reliable (the GLU width
    scaling above, several norm implementations, ...), so search a tiny
    candidate set and require the probe's parameter shapes to match the
    teacher's exactly. Returns the *constructor-level* dim_feedforward, which is
    what `hidden_scale = ffn / dim` must reproduce.
    """
    from .modules.transformer import StreamingTransformerLayer

    eff = _effective_ffn_width(layer)
    ffn_candidates: list[int] = []
    for c in (int(eff * 3 / 2), int(eff * 3 / 2) + 1, eff):
        if c > 0 and c not in ffn_candidates:
            ffn_candidates.append(c)
    target = {k: tuple(v.shape) for k, v in layer.state_dict().items()
              if "rope" not in k}
    for ffn in ffn_candidates:
        for norm in ("rms_norm_f32", "rms_norm", "layer_norm"):
            for gating in ("silu", "none"):
                try:
                    probe = StreamingTransformerLayer(
                        d_model=dim, num_heads=num_heads, dim_feedforward=ffn,
                        causal=True, context=None, rope=None, norm=norm,
                        gating=gating)
                except Exception:      # unsupported combo
                    continue
                got = {k: tuple(v.shape) for k, v in probe.state_dict().items()
                       if "rope" not in k}
                if got == target:
                    logger.info("layer-init: inferred norm=%s gating=%s "
                                "dim_feedforward=%d (effective %d)",
                                norm, gating, ffn, eff)
                    return norm, gating, ffn
    raise RuntimeError(
        "could not infer the teacher layer's norm/gating/FFN width; pass them "
        "explicitly to build_layer_init_student(norm=..., gating=...)")


def teacher_transformer_spec(teacher: LMModel) -> dict:
    """Derive the student transformer hyper-parameters from the teacher."""
    layers = teacher.transformer.layers
    dim = int(teacher.dim)
    heads = _infer_num_heads(layers[0])
    norm, gating, ffn = _infer_layer_spec(layers[0], dim, heads)
    return {
        "dim": dim,
        "num_heads": heads,
        "dim_feedforward": ffn,
        "norm": norm,
        "gating": gating,
        "n_src_layers": len(layers),
    }


def _verify_cfg_reproduces_layer(cfg: StudentConfig, layer: torch.nn.Module,
                                 dim: int, num_heads: int) -> None:
    """Build a layer from `cfg` and require its parameter shapes to match the
    teacher's exactly. This is the guard that catches a wrong FFN width / norm /
    gating before an unusable checkpoint is written."""
    from .modules.transformer import StreamingTransformerLayer

    ffn = int(cfg.hidden_scale * cfg.dim)
    probe = StreamingTransformerLayer(
        d_model=dim, num_heads=num_heads, dim_feedforward=ffn, causal=True,
        context=None, rope=None, norm=cfg.norm, gating=cfg.gating,
        layer_scale=cfg.layer_scale)
    want = {k: tuple(v.shape) for k, v in layer.state_dict().items()
            if "rope" not in k}
    got = {k: tuple(v.shape) for k, v in probe.state_dict().items()
           if "rope" not in k}
    if got != want:
        only_want = {k: v for k, v in want.items() if got.get(k) != v}
        only_got = {k: v for k, v in got.items() if want.get(k) != v}
        raise RuntimeError(
            "layer-init cfg does not reproduce the teacher layer "
            f"(dim={dim} heads={num_heads} ffn={ffn} norm={cfg.norm} "
            f"gating={cfg.gating}); teacher-only shapes: {only_want}; "
            f"cfg-only shapes: {only_got}")


def _infer_num_heads(layer: torch.nn.Module) -> int:
    attn = getattr(layer, "self_attn", None)
    assert attn is not None and hasattr(attn, "num_heads"), (
        "cannot read num_heads from the teacher layer")
    return int(attn.num_heads)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_layer_init_student(
    teacher: LMModel,
    num_layers: int = 16,
    mode: str = "even",
    cfg: StudentConfig | None = None,
    device=None,
    dtype=None,
    streaming_fix: str = "augment",
    context: int = 1500,
    compressor_hidden: int | None = None,
    upsampler_hidden: int | None = None,
    phase_hidden: int | None = None,
    copy_embeddings: bool = True,
    trainable_layers: bool = True,
    norm: str | None = None,
    gating: str | None = None,
    verify: bool = True,
) -> tuple[StudentLM, list[int]]:
    """Build a `num_layers`-deep student initialised from the teacher's layers.

    Returns (student, kept_teacher_layer_indices). The student's transformer
    width/heads/FFN/norm/gating match the teacher exactly (validated), its
    embeddings are copied, and `hidden_proj` is the identity because the state
    already lives in the teacher's dimension.
    """
    spec = teacher_transformer_spec(teacher)
    dim, heads, ffn = spec["dim"], spec["num_heads"], spec["dim_feedforward"]
    inferred_norm, inferred_gating = (
        (norm, gating) if norm and gating else (spec["norm"], spec["gating"]))
    if cfg is None:
        cfg = StudentConfig(
            dim=dim,
            num_layers=num_layers,
            num_heads=heads,
            hidden_scale=ffn / dim,          # exact FFN width
            context=context,
            norm=inferred_norm,              # matches the teacher's layers
            gating=inferred_gating,
            positional_embedding="rope",
            max_period=10_000.0,
            layer_scale=None,
            compressor_hidden=compressor_hidden,
            upsampler_hidden=upsampler_hidden,
            streaming_fix=streaming_fix,
            phase_hidden=phase_hidden,
            identity_hidden_proj=True,
        )
    else:
        cfg = replace(cfg)                   # never mutate the caller's cfg
        cfg.streaming_fix = streaming_fix
    assert cfg.dim == dim, (
        f"layer-init needs the student width to equal the teacher's "
        f"({cfg.dim} != {dim}); use dim={dim}")
    assert cfg.num_heads == heads, (cfg.num_heads, heads)
    _verify_cfg_reproduces_layer(cfg, teacher.transformer.layers[0], dim, heads)

    logger.info("layer-init: %d -> %d layers (mode=%s), dim=%d heads=%d ffn=%d",
                spec["n_src_layers"], num_layers, mode, dim, heads, ffn)
    student = StudentLM(cfg, teacher_lm=teacher, device=device, dtype=dtype)
    assert len(student.transformer.layers) == num_layers

    idx = select_layer_indices(spec["n_src_layers"], num_layers, mode)
    src_layers = teacher.transformer.layers
    # Deep-copy the selected teacher layers instead of rebuilding them: the copy
    # carries the teacher's exact norm/gating/rope configuration, so there is no
    # way to mis-derive a hyper-parameter. (Memory: the copy doubles the kept
    # layers transiently; the freshly-built random layers are released.)
    kept = []
    for src_j in idx:
        layer = copy.deepcopy(src_layers[src_j])
        if device is not None or dtype is not None:
            layer = layer.to(device=device, dtype=dtype)
        if getattr(layer, "self_attn", None) is not None:
            layer.self_attn.context = cfg.context   # student cadence, weights intact
        kept.append(layer)
    student.transformer.layers = torch.nn.ModuleList(kept)
    # The teacher's parameters are frozen, and deepcopy carries that flag over:
    # re-enable gradients on the kept layers so they can adapt to the 6.25 Hz
    # pair-summary input (pass trainable_layers=False for a frozen backbone).
    for layer in student.transformer.layers:
        for p in layer.parameters():
            p.requires_grad_(trainable_layers)
    logger.info("layer-init: kept layers are %s",
                "trainable" if trainable_layers else "frozen")

    if copy_embeddings:
        for k in range(student.num_audio_codebooks):
            student.emb[k].load_state_dict(teacher.emb[k].state_dict(), strict=True)
        student.text_emb.load_state_dict(teacher.text_emb.state_dict(), strict=True)
        logger.info("layer-init: copied %d audio embedding tables + text table",
                    student.num_audio_codebooks)

    if verify:
        with torch.no_grad():
            for dst_i, src_j in enumerate(idx):
                sd_dst = student.transformer.layers[dst_i].state_dict()
                sd_src = src_layers[src_j].state_dict()
                assert set(sd_dst) == set(sd_src), "layer key mismatch"
                for k in sd_dst:
                    assert torch.equal(sd_dst[k], sd_src[k]), f"{dst_i}<-{src_j} {k}"
        logger.info("layer-init: verified bit-exact layer copies for %s", idx)
    return student, idx


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _create(args) -> None:
    from huggingface_hub import hf_hub_download
    from .models import loaders

    device = torch.device(args.device)
    weight = args.moshi_weight or hf_hub_download(args.hf_repo, loaders.MOSHI_NAME)
    logger.info("loading teacher from %s", weight)
    teacher = loaders.get_moshi_lm(weight, device=device)
    teacher.eval()

    student, idx = build_layer_init_student(
        teacher, num_layers=args.num_layers, mode=args.mode, device=device,
        dtype=torch.bfloat16, streaming_fix=args.streaming_fix,
        context=args.student_context, compressor_hidden=args.compressor_hidden,
        upsampler_hidden=args.upsampler_hidden, phase_hidden=args.phase_hidden)
    student.eval()

    n_student = sum(p.numel() for p in student.parameters())
    n_train = sum(p.numel() for p in student.parameters() if p.requires_grad)
    logger.info("student: %s params (%.2fB), %s trainable, kept layers %s",
                f"{n_student:,}", n_student / 1e9, f"{n_train:,}", idx)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": student.state_dict(), "cfg": asdict(student.cfg)},
               out / "student.pt")

    # Save -> rebuild -> strict-load round trip: proves the cfg fully describes
    # the saved model (a wrong FFN width / norm / gating would fail here rather
    # than at serving time).
    rebuilt = StudentLM(StudentConfig(**asdict(student.cfg)), teacher_lm=teacher,
                        device=device, dtype=torch.bfloat16)
    rebuilt.load_state_dict(torch.load(out / "student.pt", map_location="cpu")["state_dict"],
                            strict=True)
    print(f"[layer_init] wrote {out / 'student.pt'}")
    print(f"[layer_init] student params: {n_student:,} ({n_student/1e9:.2f}B), "
          f"trainable {n_train:,}")
    print(f"[layer_init] kept teacher layers: {idx}")
    print(f"[layer_init] cfg: dim={student.cfg.dim} layers={student.cfg.num_layers} "
          f"heads={student.cfg.num_heads} ffn={int(student.cfg.hidden_scale*student.cfg.dim)} "
          f"norm={student.cfg.norm} gating={student.cfg.gating} "
          f"streaming_fix={student.cfg.streaming_fix} "
          f"identity_hidden_proj={student.cfg.identity_hidden_proj}")
    print("[layer_init] save/reload round trip: OK")
    logger.info("wrote %s (train with --resume %s)", out / "student.pt", out)


def _self_test() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    import argparse as _ap
    import tempfile

    from .models.student import served_head_states, serving_schedule_states

    torch.manual_seed(0)
    # Mirror the REAL teacher's config (rms_norm_f32 + silu GLU gating): the
    # FFN width inference differs between GLU and plain MLP gating, so the test
    # must cover the GLU path.
    teacher = LMModel(
        delays=[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
        n_q=16, dep_q=16, card=64, text_card=32, dim=64, num_heads=4,
        num_layers=4, hidden_scale=2, existing_text_padding_id=3, context=64,
        causal=True, norm="rms_norm_f32", gating="silu",
        depformer_dim=32, depformer_dim_feedforward=64,
        depformer_num_heads=4, depformer_num_layers=2, depformer_multi_linear=True,
        depformer_weights_per_step=True,
        depformer_weights_per_step_schedule=list(range(16)), depformer_causal=True,
        depformer_layer_scale=None, depformer_context=8, depformer_max_period=10000,
        depformer_gating="silu", depformer_pos_emb="none")
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    spec = teacher_transformer_spec(teacher)
    print("teacher spec:", spec)

    student, idx = build_layer_init_student(teacher, num_layers=2, mode="even",
                                            dtype=torch.float32,
                                            streaming_fix="augment")
    assert idx == [0, 3] or idx == [0, 3], idx       # 4 -> 2 evenly spaced
    print("kept teacher layers:", idx)

    # --- the copied layers must be the teacher's weights --------------------
    for dst, src in enumerate(idx):
        sd_s = student.transformer.layers[dst].state_dict()
        sd_t = teacher.transformer.layers[src].state_dict()
        assert set(sd_s) == set(sd_t), (set(sd_s) ^ set(sd_t))
        assert all(torch.equal(sd_s[k], sd_t[k]) for k in sd_s)
    # embeddings copied too
    assert torch.equal(student.emb[0].weight, teacher.emb[0].weight)
    assert torch.equal(student.text_emb.weight, teacher.text_emb.weight)
    print("bit-exact layer + embedding init: OK")

    # --- the train forward runs and backprops into the adapters ------------
    g = torch.Generator().manual_seed(7)
    codes = torch.randint(0, 64, (2, 17, 16), generator=g)
    codes[:, 0] = torch.randint(1, 31, (2, 16), generator=g)
    codes[:, :, -1:] = -1
    s, out, a = student.forward_train(codes, teacher_lm=teacher, return_states=True)
    assert a.shape == (2, 16, 64) and s.shape == (2, 8, 64)
    print("forward_train ok:", tuple(s.shape), tuple(a.shape))

    # randomise the fix so the test is not vacuous (zero-init would be a no-op)
    with torch.no_grad():
        for seq in student.streaming_fix.phase_proj:
            seq[-1].weight.normal_(0, 0.05, generator=g)

    s2, out2, a_fix = student.forward_train(codes, teacher_lm=teacher,
                                            return_states=True)
    assert not torch.allclose(a_fix, a), "the fix must change the head states"
    print("streaming fix changes head states: OK")

    # --- training must not touch the teacher ------------------------------
    # (runs BEFORE the streaming Gen below: once `streaming_forever` has run,
    # the student's transformer holds a streaming KV state at that batch size.)
    before = copy.deepcopy({k: v.clone() for k, v in teacher.state_dict().items()})
    opt = torch.optim.SGD([p for p in student.parameters() if p.requires_grad], lr=1.0)
    with torch.enable_grad():                 # the harness runs under no_grad
        s3, out3, _ = student.forward_train(codes, teacher_lm=teacher,
                                            return_states=True)
        loss = out3.text_logits[:, 0].nan_to_num().pow(2).mean()
        loss.backward()
    opt.step()
    for k, v in teacher.state_dict().items():
        assert torch.equal(v, before[k]), f"teacher mutated: {k}"
    # the copied layers must actually receive gradients
    layer0_grads = [p.grad for p in student.transformer.layers[0].parameters()]
    assert any(g is not None for g in layer0_grads), "copied layers got no grads"
    assert student.compressor.proj.weight.grad is not None
    assert student.streaming_fix.phase_proj[0][-1].weight.grad is not None
    print("teacher untouched + copied layers receive grads: OK")

    # --- the SERVER's consumed states must equal served_head_states --------
    # NOTE: compute the reference BEFORE creating the streaming Gen — once
    # `streaming_forever` runs, the student's transformer holds a streaming KV
    # state at that batch size and a differently-batched forward would fail.
    from .student_gen import StudentLMGen
    from .models.lm import _delay_sequence
    codes1 = codes[:1]
    initial = teacher._get_initial_token().expand(1, -1, -1)
    delayed = torch.cat([initial, _delay_sequence(teacher.delays, codes1, initial)], dim=2)
    m_ref = student.embed(delayed[:, :, :-1])
    _, a_raw = student.forward_main(m_ref)
    a_norm = teacher.out_norm(a_raw)
    sched = serving_schedule_states(a_norm)          # the server's base states

    gen = StudentLMGen(student, teacher, device="cpu", sample_rate=24000,
                       frame_rate=12.5, audio_silence_frame_cnt=1,
                       use_sampling=False, top_k=1, top_k_text=1)
    gen.streaming_forever(1)
    gen.reset_streaming()
    ok = True
    for o in range(1, 17):
        # Emulate LMGen.step's bookkeeping: at the model call it has already
        # advanced `offset` to the current step number o (prepare_step_input
        # bumps it out of the offset==0 bootstrap, process_transformer_output
        # bumps it at the end of each later step). Driving the backbone
        # directly bypasses that, so set it explicitly.
        gen._streaming_state.offset = o
        t_out, _ = gen._student_forward_codes(delayed[:, :, o - 1: o])
        if o == 1:
            continue          # documented zero-bootstrap slot
        p = o - 1             # embed position consumed at this step
        # server: base = sched[:, p], plus the phase-specific frame injection
        want = sched[:, p] + student.streaming_fix.phase_add(p % 2, m_ref[:, p])
        ok &= torch.allclose(t_out.squeeze(1), want, atol=1e-5)
    assert ok, "server head states != training-path served states"
    print("train/serve head-state agreement (with a non-trivial fix): OK")

    # --- checkpoint round trip / train_distill compatibility --------------
    tmp = Path(tempfile.mkdtemp(prefix="layer_init_test_"))
    torch.save({"state_dict": student.state_dict(), "cfg": asdict(student.cfg)},
               tmp / "student.pt")
    from .layer_init import build_layer_init_student as _b
    cfg = StudentConfig(**torch.load(tmp / "student.pt", map_location="cpu")["cfg"])
    fresh = StudentLM(cfg, teacher_lm=teacher, dtype=torch.float32)
    fresh.load_state_dict(torch.load(tmp / "student.pt", map_location="cpu")["state_dict"],
                          strict=True)
    print("checkpoint round trip (strict): OK")
    print("LAYER-INIT SELF-TEST PASSED")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    c = sub.add_parser("create", help="build the initial student checkpoint")
    c.add_argument("--moshi-weight", default=None)
    c.add_argument("--hf-repo", default="nvidia/personaplex-7b-v1")
    c.add_argument("--num-layers", type=int, default=16)
    c.add_argument("--mode", choices=["even", "first"], default="even")
    c.add_argument("--streaming-fix", choices=["off", "augment"], default="augment")
    c.add_argument("--student-context", type=int, default=1500,
                   help="student steps of context (6.25 Hz)")
    c.add_argument("--compressor-hidden", type=int, default=None)
    c.add_argument("--upsampler-hidden", type=int, default=None)
    c.add_argument("--phase-hidden", type=int, default=None)
    c.add_argument("--device", default="cuda")
    c.add_argument("--out", default="ckpt/layer_init_16L")
    c.set_defaults(func=_create)

    argv = ["create" if a == "--create" else a for a in sys.argv[1:]]
    args = ap.parse_args(argv)
    if args.self_test or args.cmd is None:
        _self_test()
        return
    args.func(args)


if __name__ == "__main__":
    with torch.no_grad():
        main()
