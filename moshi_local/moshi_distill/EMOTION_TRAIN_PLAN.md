# Emotion training (DepFormer + emotion module)

Follows `EMOTION_PLAN.md` §5 Option B, tightened to the requirement:
**freeze the temporal transformer and train only the DepFormer + an emotion
module, with audio-based losses so the model learns the intensity and prosody
of each emotion.**

---

## 1. What is frozen / trained

| module | state |
|---|---|
| Mimi (tokenizer) | frozen |
| temporal transformer (20 layers) | **frozen** |
| audio input embeddings `emb`, `text_emb`, `out_norm`, `text_linear` | **frozen** |
| **DepFormer** — `depformer_in`, `depformer_emb`, `depformer_text_emb`, `depformer` (6 layers), `linears` | **trained** (1.398 B params) |
| **EmotionAdapter** (`emotion_id -> depformer_dim`) | **trained** (0.27 M params) |

The frozen backbone is kept in **bf16** (`--frozen-dtype bfloat16`, the default);
only the DepFormer is fp32. In cached mode the backbone is never even executed
during training, so this costs nothing.

## 2. How the emotion conditions generation

The frozen temporal state already carries the prosody of the *past* frames
(teacher forcing feeds the real codes back in), so the emotion cannot be
injected through the text stream. Instead the adapter vector is **added to every
DepFormer input frame**:

```
depformer_input[b, t, cb, :] += emo_vec[b, :]        # all T frames x all 16 codebooks
```

`emo_vec = EmotionAdapter(emotion_id)`; the projection is zero-initialised so
step 0 is bit-identical to the untouched base model.

## 3. Objective (audio-first)

```
L = lambda_audio    * CE_audio(DepFormer logits, target codes)      # main
  + lambda_contrast * relu(margin + CE_true - CE_perm)              # make the emotion matter
  + lambda_anchor   * mean||theta - theta_0||^2                     # optional drift guard
```

* **CE_audio** — per-codebook weighted cross-entropy over the 16 target
  codebooks. Codebook 0 (the coarse *semantic + prosody* stream) carries most of
  the pitch/energy envelope, i.e. the **intensity** of the emotion, so it is
  up-weighted (`--first-codebook-mult`, default 2.0). `--codebook-decay` gives an
  optional exponential falloff over the refinement codebooks.
* **contrastive** — the true emotion must explain the target codebooks better
  than a permuted one. This is what stops the DepFormer from just copying the
  prosody it sees in the context and ignoring the emotion knob.
* **anchor** — optional proximal term (`--lambda-anchor`, default off).

There is **no text CE term**: the temporal transformer that produces the text
stream is frozen, so it carries no gradient.

## 4. Frozen-transformer cache (why training is fast)

`transformer_out` depends only on the codes, so the 20-layer backbone forward is
run **once** and cached (`--cache-dir`). Training then only runs the 6-layer
DepFormer. The cache is ~`n_samples x T x 4096` in bf16 (≈12 GB for the full
8 882-clip set) and is built on the first run, reused afterwards.

## 5. Data

`emotion_preprocess.sh` → `/workspace/data/emotion_v1/audio/`:
`<id>.wav` (24 kHz stereo; ch0 = agent = **real emotional clip**, ch1 = user =
Chatterbox neutral question) + `<id>.sidecar.json` + `train.jsonl`.

The label is in `text_conditions.emotion_id` and `<emotion>` is already
prepended to the system prompt. Emotion order (fixed, used by the adapter):

```
0 neutral  1 happy  2 sad  3 angry  4 fearful  5 disgust  6 surprised  7 calm
```

## 6. Run

```bash
cd /workspace/personaplex
PY=/workspace/menv/bin/python
SNAP=/workspace/.cache/huggingface/hub/models--nvidia--personaplex-7b-v1/snapshots/fdaf4090a61cb315c138a1faee287ffd6c716309

# full run (builds the cache on first launch, then trains)
$PY -m moshi_local.moshi_distill.train_emotion \
    --student-checkpoint ckpt/backbone_S20_lora/step_1800 \
    --mimi-weight "$SNAP/tokenizer-e351c8d8-checkpoint125.safetensors" \
    --tokenizer   "$SNAP/tokenizer_spm_32k_3.model" \
    --train-data /workspace/data/emotion_v1/audio/train.jsonl \
    --eval-data  /workspace/data/emotion_v1/audio/train.jsonl \
    --cache-dir  /workspace/data/emotion_v1/cache \
    --out-dir    ckpt/emotion_v1 \
    --batch-size 8 --max-steps 6000 --warmup-steps 200 \
    2>&1 | tee /workspace/data/emotion_v1/logs/train_emotion.log
```

Output per checkpoint: `ckpt/emotion_v1/step_N/student.pt` (same format as the
backbone, loadable by `server_student.py`) + `emotion.pt` (the adapter).

### Memory / speed (measured, A100 80 GB)
* batch 8: fits with ~55 GB free alongside the Whisper aligner's 24 GB.
* ~7–9 s/step at batch 8 while the aligner is running; faster with a free GPU.
* Always set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
* If you must share the GPU with the aligner, drop to `--batch-size 4`.

## 7. Serving the trained emotion

`emotion_module.set_emotion(model, adapter, emotion_id)` writes the adapter
vector into the bias of every `depformer_in` linear. Because `Linear` computes
`x @ W.T + b`, this reproduces the training-time injection **exactly**, with no
change to the generation loop, and the in-place `copy_` means a captured CUDA
graph sees the new emotion on the next frame.

Wire it into `server_student.py` next to the voice/text-prompt handling: resolve
an `emotion` query param -> id, then `set_emotion(...)` before
`step_system_prompts_async`. (Not done yet — flag it when you want it.)

## 8. Evaluation

`train_emotion.py` logs, every `--eval-every`:

* `audio_ce` — weighted DepFormer CE on held-out batches (should fall);
* `audio_ce_perm` — the same CE with a permuted emotion;
* `emotion_reliance = audio_ce_perm - audio_ce` — the controllability gap.
  Starts at 0.0 (adapter is a no-op) and must grow: it is the direct evidence
  that the emotion knob actually changes the generated audio.

Listening check: run the student with `set_emotion` for each id on the same
prompt and compare prosody.
