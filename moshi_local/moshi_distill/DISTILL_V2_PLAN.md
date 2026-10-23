# Re-distillation v2 — retain base intelligence (LoRA / partial FT) + new data

Answers: *was the last student fully finetuned? should we LoRA instead? should we
delete it and re-distill on the new data?*

---

## 1. What the previous run actually did (verified)

* Method: **layer-init + full fine-tune**, no LoRA.
  * `layer_init.py` selects 16 of the teacher's 32 transformer layers
    (`strategy=spaced` → e.g. `[0,2,4,…,30]`, or contiguous
    `[0..11,28..31]`), copies those weights + embeddings into the student, then
    saves it.
  * `backbone_distill.py fit` / `train_distill.py` then trained it:
    `kept 16 layers`, **trainable params 3.68 B**, `optim adamw8bit`,
    two stages (stage 1 freezes emb/heads; stage 2 unfreezes them).
  * Fit log: `cos_hidden≈0.966`, `text_agree≈0.959`, `text_agree_nonpad≈0.896`.
* Student: **5.08 B params, 16 layers** (of 32), saved at
  `ckpt/backbone_S16/step_1984/student.pt`, `backbone_S16_spaced_fix/step_355`.
* Data: `train/temp_data/train_split.jsonl` = **65,263 chunks**, all from
  `new_audios/` — i.e. the **old synthetic RAG corpus only** (grounded QA +
  short small-talk). No broad/general conversation data.

**So: yes — it was layer-initialised and then fully finetuned (3.68 B trainable),
on narrow RAG-only data.**

## 2. Why it lost the base model's intelligence

1. **Full fine-tune on a narrow distribution** → catastrophic forgetting: the
   teacher-copied language/knowledge weights are overwritten to fit short
   grounded QA + scripted small-talk.
2. **Depth pruning 32→16** → real capacity loss (knowledge + reasoning).
3. **Data breadth**, not the method, is the dominant factor: 65 k samples of one
   synthetic distribution cannot preserve a base trained on a huge corpus.
4. 6.25 Hz temporal compression adds further information loss.

LoRA alone does **not** fix (2)–(4). It fixes (1).

## 3. Full FT vs LoRA / partial FT — recommendation

**Yes, stop fully finetuning the backbone. Use a partial-freeze / LoRA scheme**
that trains the *new* modules fully and adapts the *pretrained* backbone cheaply.

The student must learn three new things: `12.5→6.25` compression, the 6.25 Hz
temporal dynamics, and the 6.25→12.5 upsampling. Those live in
`compressor / upsampler / hidden_proj / out_norm` — **train those fully**. The
transformer backbone is copied from the teacher and already holds the language
knowledge — **protect it**.

Recommended, in increasing aggressiveness:

| Backbone handling | What trains | When to use |
|---|---|---|
| `--student-freeze-layers` (freeze all transformer) | adapters only (+emb/heads stage 2) | safest; use first |
| `--student-freeze-layers --trainable-layers 4` | adapters + top 4 layers | if adapters-only underfits the rate change |
| **+ LoRA** on attn/FFN (peft) | adapters + low-rank deltas everywhere | best capability/forgetting trade-off |
| full FT | everything | **avoid** (this is what lost the intelligence) |

Also: **keep more layers** so you are not fighting depth loss on top of
forgetting. Arbitrary depths are supported (`select_layer_indices` even-spaces
`round(i*(31)/(N-1))`), and the real sizes are:

| student layers | params | note |
|---|---|---|
| 16 (previous) | 5.08 B | too much capacity loss |
| **20 (recommended)** | **5.91 B** | +0.83 B vs old, −0.82 B vs 24 — good balance |
| 24 | 6.73 B | max retention; larger/slower |
| 32 | 8.37 B | = teacher depth; heaviest |

At 6.25 Hz you already halve temporal steps, so 20 layers is usually fine on the
target hardware. Pick 16 only under a hard Jetson budget, 24 only if the
held-out general eval still shows 20 is weak. Layer count helps, but **data
breadth + freeze/LoRA matter more** — 20 layers fully finetuned on narrow data
will still forget.

LoRA is not in the current trainer (no LoRA code in `backbone_distill.py` /
`train_distill.py`), but `peft` is already used elsewhere (`moshi/train_rag.py`),
so wrapping `student.transformer` attention/FFN with `get_peft_model` is a small
addition. The built-in freeze path works today without it.

## 4. Data — the real lever

Intelligence is bounded by the distillation distribution. The new `rag_v3`
(9,500 safe records) is broader than before (real QA + greetings/small-talk) but
still synthetic/short. Plan:

* **Core target behaviour** — the new `rag_v3` Stage-1 subset (9,500):
  `no_rag_qa` 3,000 + `grounded_real` 5,000 + `regular_chat` 1,500.
  (Exclude `grounded_cf/distract/abstain` bodies — fabricated/refusal; they are
  only meaningful with injected context.)
* **Breadth for intelligence** — add as much *general* conversation/QA/knowledge
  audio as you can and apply **teacher KD** on it:
  * ideally a sample of the base's original training distribution, or
  * an open conversation/ASR corpus (e.g. DailyTalk/LibriSpeech-style speech with
    transcripts) run through the teacher, or
  * the teacher's own generations on diverse prompts → TTS → align.
  * The old 65 k RAG set can stay as extra volume but is *not* breadth.
* Weight the KD terms (`--lambda-kd-text/audio`, `--lambda-hidden`) on the broad
  set so general ability is explicitly preserved, and the task terms on rag_v3.

## 5. Concrete re-distillation run

Delete/ignore the old student and rebuild from the teacher:

```bash
SNAP=/workspace/.cache/huggingface/hub/models--nvidia--personaplex-7b-v1/snapshots/<hash>
PY=/workspace/menv/bin/python
cd /workspace/personaplex

# 1) layer-init a 20-layer student from the teacher (even spacing), NOT 16
$PY -m moshi_local.moshi_distill.layer_init \
    --moshi-weight $SNAP/model.safetensors --num-layers 20 --mode even \
    --out ckpt/layer_init_20L

# 2) distill: freeze the backbone, train adapters fully, adapt top layers,
#    keep emb/heads frozen in stage 1 and unfreeze in stage 2 (default)
$PY -u -m moshi_local.moshi_distill.train_distill \
    --train-data /workspace/data/rag_v3/audio/train_distill.jsonl \
    --student-init ckpt/layer_init_20L/student.pt \
    --student-freeze-layers --trainable-layers 4 \
    --adapter-lr 3e-4 --lr 2e-5 \
    --lambda-task-text 1 --lambda-task-audio 1 \
    --lambda-hidden 1 --lambda-kd-text 1 --lambda-kd-audio 1 --lambda-state 10 \
    --optim adamw --dtype bfloat16 --grad-checkpointing \
    --max-steps 20000 --save-every 2000 --eval-every 1000 \
    --moshi-weight $SNAP/model.safetensors --mimi-weight $SNAP/tokenizer-*.safetensors \
    --tokenizer $SNAP/tokenizer_spm_32k_3.model \
    --out-dir ckpt/distill_v2_20L_lora
```

Then add LoRA (small patch) and/or raise `--trainable-layers` until the held-out
eval matches the teacher; do **not** fall back to full FT.

## 6. Eval gates (detect forgetting)

* Held-out **text-agreement / cos_hidden** vs teacher (existing eval) on data
  **not** in the narrow RAG set (general prompts) — this is the intelligence
  proxy that the previous run never checked.
* A small general-chat/perplexity probe: student vs teacher text log-probs on
  open-domain transcripts. If the student collapses to short-answer mode, it is
  overfit → more freezing / more breadth.
* Audio intelligibility (ASR-WER) and persona/turn-taking.

## 7. Order with RAG

Unchanged from `RAG_TRAIN_PLAN.md`: **re-distill first (this doc), then RAG
adaptation on the frozen distilled student.** Never plain-distill after RAG.

---

### TL;DR
* Last run = layer-init **+ full fine-tune** of a **16-layer** student on
  **narrow RAG-only** data → that is why it lost base intelligence.
* **Yes, switch to LoRA / partial freeze**: train the rate adapters fully, keep
  the teacher-copied backbone frozen (or LoRA it), keep **more layers (24)**.
* **Delete the old student and re-distill on the new data**, but the biggest win
  is **adding broad general conversation data with teacher KD** — freezing alone
  will not restore intelligence that 16 layers + narrow data removed.
