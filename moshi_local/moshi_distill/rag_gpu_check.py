"""GPU equivalence check for the additive path (CUDA graph, real student).

``rag_stream_check --tiny`` covers the arithmetic on the tiny twin, but on GPU
the single-frame streaming loop goes through ``CUDAGraphed`` and the embeddings
are bf16.  This script loads the real student and proves two things:

  * ``LMGen.step_with_ref(ref_vec=0)`` is **bit-identical** to ``LMGen.step``
    (so the additive path costs nothing when no reference is queued), and
  * a real adapter vector on one frame does reach the text logits.

It needs the whole student on the GPU (~6 GB), so run it when nothing else is
training:

  PYTHONPATH=/workspace/personaplex /workspace/menv/bin/python \
      -m moshi_local.moshi_distill.rag_gpu_check \
      --ckpt /workspace/personaplex/ckpt/rag_S24/step_3000/student.pt
"""
import argparse

import torch

from .models.lm import LMGen
from .rag_train import load_student_standalone
from .ref_adapter import RefBagAdapter

DEV = "cuda"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="student.pt (its ref card size is not used)")
    ap.add_argument("--tokens", default="/workspace/personaplex/rag_v3_tokens.pt")
    ap.add_argument("--cards", default="/workspace/personaplex/rag_v3_cards.pt")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--lookup-at", type=int, default=2,
                    help="streaming step that receives the first vector")
    ap.add_argument("--device", default=DEV)
    args = ap.parse_args()

    dtype = torch.bfloat16
    student, pkg = load_student_standalone(args.ckpt, args.device, dtype)
    student.eval()
    print("student dim=%d layers=%d" % (student.dim, len(pkg["kept_layers"])))

    adapter = RefBagAdapter(student.dim, student.text_emb, n_slots=16).to(
        device=args.device, dtype=dtype).eval()

    entries = torch.load(args.tokens, map_location="cpu",
                         weights_only=False)["entries"]
    cards = torch.load(args.cards, map_location="cpu",
                       weights_only=False)["cards"]
    i = next(j for j, e in enumerate(entries)
             if e["kind"] == "grounded_cf" and e["has_ref"])
    e = entries[i]
    n = int(e["codes"].shape[-1])
    t = {k: (s, min(x, n)) for k, s, x in e["turns"]}
    lu = t["lookup"][0]
    codes = e["codes"][:, lu:lu + args.frames].unsqueeze(0)
    codes = codes.to(args.device).long()
    codes[:, 0] = codes[:, 0].clamp(0, student.text_card - 1)
    print("codes %s (from frame %d, i.e. <lookup>)" % (tuple(codes.shape), lu))

    ids = cards[int(e["conv_id"])][:4]
    vecs = adapter.encode_ids(ids, args.device)      # [4, dim]
    print("card ids %s -> per-vector norm %.3f" % (
        ids, float(vecs[0].norm())))

    def run(ref_at=None, vec=None):
        lm = LMGen(student, device=args.device, use_sampling=False, temp=0.0,
                   temp_text=0.0, check=True)
        lm.streaming_forever(1)
        captured = []
        orig = lm.process_transformer_output

        def spy(transformer_out, text_logits, *a, **kw):
            captured.append(text_logits.detach().float().clone())
            return orig(transformer_out, text_logits, *a, **kw)

        lm.process_transformer_output = spy       # type: ignore[method-assign]
        lm.step()                                 # consume the initial token
        for f in range(codes.shape[-1]):
            tx = codes[:, 0, f]
            a = codes[:, 1:9, f:f + 1]
            u = codes[:, 9:17, f:f + 1]
            if ref_at is not None and f == ref_at:
                lm.step_with_ref(input_tokens=u, moshi_tokens=a,
                                 text_token=tx, ref_vec=vec)
            else:
                lm.step(input_tokens=u, moshi_tokens=a, text_token=tx)
        lm._stop_streaming()
        return torch.cat(captured, dim=2)

    base = run()
    zero = run(ref_at=args.lookup_at,
               vec=torch.zeros(1, 1, student.dim, device=args.device,
                               dtype=dtype))
    dz = float((base - zero).abs().max())
    print("zero vector vs step       : %.3e" % dz)
    assert dz == 0.0, "step_with_ref(0) is not identical to step on cuda"

    for scale in (1.0, 10.0):
        v = vecs[0].view(1, 1, -1) * scale
        real = run(ref_at=args.lookup_at, vec=v)
        d = (base - real).abs()
        m = float(d[:, :, args.lookup_at + 1:].max())
        print("vec x%-4.0f max|dlogit| %.4f | after that frame %.4f" % (
            scale, float(d.max()), m))
        assert m > 1e-3, "the vector does not reach the logits"
    print("GPU_CHECK_OK")


if __name__ == "__main__":
    main()
