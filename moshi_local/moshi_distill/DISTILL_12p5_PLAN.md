# Distillation v2 — 12.5 Hz layer-pruned student (revised plan)

Decision: **stay at 12.5 Hz**. Drop the 6.25 Hz temporal compression.

Reason: the 6.25 Hz path needs a 2-frame compressor + upsampler, and the
upsampler's held/reused states are a known source of **stutter — the model
repeating the previous step** — and of spectral distortion from the rate
change. At 12.5 Hz there is no rate adapter at all: the student is the same
`LMModel` interface with fewer layers, so the audio path is identical to the
teacher's. (The 6.25 Hz plan was also never actually trained — no
compressor/upsampler checkpoint exists.)

---

## 1. What the previous 12.5 Hz run did (why intelligence dropped)

* `backbone_distill.py collect` → teacher rollouts (tokens) from
  `train/temp_data/train_split.jsonl` (65 k chunks, **old synthetic RAG only**).
* `backbone_distill.py fit` → `--keep-layers 16 --strategy spaced`,
  **full fine-tune of the 16-layer `LMModel` (3.68 B trainable)**, `adamw8bit`,
  KD on hidden + text KL + audio KL. Checkpoints `ckpt/backbone_S16/*`.
* No LoRA, no backbone freeze, no broad data.

So the drop in intelligence = **depth pruning 32→16 + full FT on a narrow
distribution**. The rate was fine; the adaptation and data were not.

## 2. Revised method (12.5 Hz)

| aspect | old | v2 |
|---|---|---|
| rate | 12.5 Hz | **12.5 Hz (unchanged)** |
| depth | 16/32 | **20/32** (even spacing; 5.9 B vs 5.08 B) |
| init | layer-init from teacher | same (layer-init) |
| backbone training | **full FT (3.68 B)** | **LoRA + top-N layers**, rest frozen |
| data | 65 k old RAG rollouts | **broad**: rag_v3 safe + general conversation |
| KD | hidden+text+audio | same (already good) |

Why this fixes it:
* **No rate adapter** → no upsampler reuse artifact, no 6.25 Hz distortion.
* **LoRA + freeze** keeps the teacher-copied language/knowledge weights; we only
  heal the pruned representation with bounded low-rank updates (instead of
  overwriting it with a narrow distribution).
* **20 layers** recovers some of the depth capacity lost at 16.
* **Broad data** is the dominant lever — a pruned+LoRA model still narrows if it
  only ever sees RAG QA.

## 3. Code changes needed (`backbone_distill.py`)

The `fit` path already has the pieces but doesn't expose them:

* `finetune_top_layers(student, n_trainable)` exists but is **unused** → wire it
  to `--trainable-layers N` (and `--freeze-layers` to freeze the whole
  transformer, training only embeddings/heads).
* Add **LoRA** on `student.transformer` via `peft` (already used in
  `moshi/train_rag.py`): `--lora --lora-r 64 --lora-alpha 128 --lora-lr 2e-4`,
  target `in_proj/out_proj/fc1/fc2/linear/proj`. Keep the depformer frozen.
* Add a separate `--adapter-lr` for LoRA vs `--lr` for the (few) trainable
  transformer layers / embeddings / heads.
* Everything else (`BackboneDistillWeights`, stage 1/2, eval) is unchanged.

Nothing changes in `collect`.

## 4. Data (the real lever)

`collect` turns a manifest of user audio into teacher rollouts (agent stream +
text). Use a **broad** manifest, not just RAG:

* rag_v3 **safe** set — `no_rag_qa` 3,000 + `grounded_real` 5,000 +
  `regular_chat` 1,500 (exclude `grounded_cf`/`distract`/`abstain` bodies; they
  are fabricated/refusal and only meaningful with injected context), **plus**
* as much **general conversation / open-domain speech** as available (ideally a
  sample of the base's own distribution; otherwise an open dialogue/ASR corpus
  run through the teacher, or teacher generations on diverse prompts → TTS →
  align).
* Keep ~20-30 % general even for a demo run.

Since `collect` is the bottleneck (teacher rollouts), batch across streams
(`--rollout-batch-size 32-64`) as the demo plan notes.

## 5. Concrete run (code changes done + self-tested)

`backbone_distill.py fit` now exposes `--freeze-layers`, `--trainable-layers N`,
and `--lora` (LoRA via `torch.nn.utils.parametrize` on attention `in_proj_weight`
/ `out_proj` and the gated FFN `linear_in/out`; merged into a plain state_dict on
save). LoRA identity-at-init, gradient flow, and merge→plain-LMModel load were
verified.

```bash
SNAP=/workspace/.cache/huggingface/hub/models--nvidia--personaplex-7b-v1/snapshots/fdaf4090a61cb315c138a1faee287ffd6c716309
PY=/workspace/menv/bin/python
cd /workspace/personaplex

# 1) teacher rollouts on the BROAD manifest (9,500: grounded_real 5000 +
#    no_rag_qa 3000 + regular_chat 1500). This is the slow step.
$PY -u -m moshi_local.moshi_distill.backbone_distill collect \
    --train-data /workspace/data/rag_v3/broad_train.jsonl \
    --moshi-weight $SNAP/model.safetensors \
    --mimi-weight  $SNAP/tokenizer-e351c8d8-checkpoint125.safetensors \
    --tokenizer    $SNAP/tokenizer_spm_32k_3.model \
    --duration-sec 20 --batch-size 8 --rollout-batch-size 32 \
    --max-samples 9500 --temp 0.8 --top-k 250 \
    --device cuda --out /workspace/personaplex/rollouts_v2.pt

# 2) fit: 20 layers, LoRA + top-4 layers trainable, NOT full FT
$PY -u -m moshi_local.moshi_distill.backbone_distill fit \
    --tokens /workspace/personaplex/rollouts_v2.pt \
    --keep-layers 20 --strategy spaced \
    --trainable-layers 4 --lora --lora-r 64 --lora-alpha 128 --lora-lr 2e-4 \
    --lr-s1 2e-5 --lr-s2 1e-5 --stage1-frac 0.2 \
    --epochs 8 --batch-size 4 --grad-accum 4 --max-steps 4000 \
    --student-dtype float32 --autocast bfloat16 --optim adamw8bit \
    --save-dtype bfloat16 --held-out 64 \
    --moshi-weight $SNAP/model.safetensors \
    --out-dir /workspace/personaplex/ckpt/backbone_S20_lora
```

Escalate `--trainable-layers` / `--lora-r` only if the held-out eval is short of
the teacher; do **not** fall back to full FT.

## 6. Eval gates (detect forgetting and stutter)

* Held-out **cos_hidden / text_agree on general (non-RAG) prompts** — the gate
  the previous run lacked. Compare S20-LoRA vs S16-fullFT vs teacher.
* **Closed-loop rollouts** (60-120 s) from held-out user audio: listen for
  repetition/looping and long silences (the stutter the user is worried about —
  and confirm 12.5 Hz avoids it).
* ASR-WER + speaker similarity.
* Log n-gram repetition and longest repeated run.

## 7. RAG stage implication (12.5 Hz)

Because the student is a full-rate `LMModel`, the RAG adaptation uses the
**full-rate path**, not `StudentLM`:

* Sidecars from rag_v3 are already 12.5 Hz and use `<lookup>` /
  `<ref_start> … <ref_end>` slot frames → compatible.
* Use `moshi_distill/interleaver.py` (splice wired) + `RefAdapter`
  (`ref_adapter.py`, prefill-only) + `train_latent.py`-style head/losses, or the
  `moshi/train_rag.py` LoRA+adapter path.
* The `distill/data.py --use-rag` (StudentLM) path is **not** used for 12.5 Hz.

Order is unchanged: **distill (this doc) → then RAG** on the frozen distilled
student; never plain-distill after RAG.

---

### TL;DR
* Go **12.5 Hz** (pruned `LMModel`) to avoid the 6.25 Hz upsampler stutter.
* Re-distill with **20 layers + LoRA/top-N (not full FT) + broad data** — the
  previous 16-layer full-FT on narrow RAG is what lost the intelligence.
* Wire `--trainable-layers` / `--lora` into `backbone_distill.py fit`
  (`finetune_top_layers` already exists, just unused).
* Gate on held-out **general** hidden/text agreement + closed-loop audio.
