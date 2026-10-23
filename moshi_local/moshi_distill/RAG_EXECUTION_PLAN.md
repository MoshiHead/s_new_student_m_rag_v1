# RAG Execution Plan — 12.5 Hz pruned Moshi student

**Status:** plan only, nothing implemented yet.
**Scope:** (1) properly re-distill the 20-layer 12.5 Hz student, (2) add prefill-only
RAG latent conditioning + `<lookup>` emission, (3) serving, (4) eval.
**Out of scope:** everything emotion-related (`train_emotion.py`, `emotion_*`,
`ckpt/emotion_*`, emotion probe logs) — not touched, not read.

---

## 0. Decisions (confirmed 2026-10-21)

| # | Decision | Outcome |
|---|----------|---------|
| A | Stage order | **RAG on the current S20 first.** Skip the separate re-distill for now; treat it as a fallback (§4) if the Stage-A gate fails. |
| B | Reference adapter | **Mean-pool `RefAdapter`**, gated by a 500-sample ref-ablation A/B before the long run. |
| C | Serving base | **Extend `server_student.py`**; port STT/VAD/retrieval/lookup-wait from `server_s.py`, do not port `RefResampler`/`body_gate`. |

Fixed defaults: window `duration_sec=48` (600 frames, covers p99 = 41.4 s),
turn-end truncation with `tail_sec=0.8` (10 frames), `K=16` slots.

### Consequence of choosing A (read this before Stage A)

Stage A now starts from S20 and trains on `rag_v3`, not on RAG-free data. Two
things follow:

1. **Rambling is only partly fixed.** Turn-end truncation is applied to the
   `rag_v3` samples too, so the model *does* get a stop signal on this
   distribution, and the grounded/abstain classes teach it to stop after the
   answer. But the S20 prior (14 s of teacher improv per sample) is still in the
   weights, so out-of-distribution smalltalk may still ramble. That is the one
   thing a proper re-distill fixes and Stage A cannot.
2. **Ungrounded factual answers are directly trained against** — `grounded_*`
   teaches "lookup → grounded body", `abstain` teaches "say you don't know". This
   is the part Stage A *can* fix, and it is the part that matters most for the demo.

So Stage A is a real test of the RAG pathway, not a shortcut that hides defects.
If the Stage-A gate (§5.7) fails on stop behaviour only, run §4 as Stage B and
re-run Stage A on the new base. If it fails on grounding, the bug is in the RAG
pipeline, and Stage B would not have helped.

---

## 1. Verified current state (facts, with paths)

### 1.1 Checkpoint

`/workspace/personaplex/ckpt/backbone_S20_lora/step_1800/student.pt`

```
keys           : state_dict, kept_layers, save_dtype, stats, base_kwargs
kept_layers    : [0,1,2,4,6,8,10,12,14,15,17,19,21,23,25,27,28,29,30,31]   (20 of 32)
save_dtype     : bfloat16
step           : 1800
base_kwargs    : dim 4096, text_card 32000, existing_text_padding_id 3, n_q/dep_q 16,
                 card 2048, heads 32, context 3000, depformer 6x1024, delays
                 [0,0,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1]
```

Fit metrics at step 1800: `cos_hidden 0.9751`, `text_agree 0.9585`,
`text_agree_nonpad 0.9124`. Training wall time ≈ 5.9 h.

### 1.2 Tokenizer

`/workspace/personaplex/moshi_local/tokenizer.model` — 32 003 pieces;
`<ref_start> = 32000`, `<ref_end> = 32001`, `<lookup> = 32002`.
The teacher (and therefore the current student) has `text_card = 32000`, i.e. it
can never emit those three ids. This is why Stage 2 must be **CE only, no teacher KD**
(see §4.4).

### 1.3 Data

`/workspace/data/rag_v3/audio/` — 14 000 WAVs + `*.sidecar.json`. Filenames start
with `-` (negative Telegram ids), so **never `ls *.json`** on that directory; use
`find . -name '*.sidecar.json'`.

Sidecar schema (verified on live files):

```json
{"alignments":[[word,[s,e],"ASSISTANT"|"USER"],...],
 "duration": 33.5,
 "segment_meta": {"turns":[{"kind":"regular|lookup|lead|filler|body",
                            "start_sec":0.0,"end_sec":7.18}, ...]},
 "text_conditions": {"reference":"...", "system_prompt":"..."},
 "metadata": {"rag_kind":"...", "persona":{...}}}
```

Counts by `metadata.rag_kind`: `grounded_real` 5 000, `grounded_cf` 3 000,
`no_rag_qa` 3 000, `regular_chat` 1 500, `grounded_distract` 1 000, `abstain` 500.

Durations: min 14.0 s, p50 33.5 s, p90 37.7 s, p99 41.4 s, max 58.1 s.

`/workspace/data/rag_v3/broad_train.jsonl` — 9 500 lines, keys `path`/`duration`
only; the `rag_kind` must be read from the sidecar. Stage-1 subset =
`no_rag_qa` + `regular_chat` = 4 500.

`/workspace/personaplex/rollouts_v2.pt` — 5 000 × `[1,17,860]`. 860 = 600 audio
frames (48 s) + ~260 persona-prompt frames. **Do not reuse** (see §2).

### 1.4 Code, as it actually is

| File | State |
|---|---|
| `distill/data.py` | `DistillInterleavedTokenizer(use_rag=False, ...)`. Returns `Sample` with `segment_meta=None`; parses sidecar once (already loads `metadata.system_prompt`). No `return_meta` flag. |
| `interleaver.py` | `splice_rag_frames()` inserts `<lookup>` then `<ref_start> [K zero] <ref_end>`, records `kind="ref"` (whole block) **and** `kind="ref_slot"` (outer frames excluded). `build_ref_slot_frames()` puts slots in user codebook 8. |
| `ref_adapter.py` | Locked mean-pool adapter over the LM's own `text_emb`; `forward(ids, mask) -> [B,K,dim]`, `inject(emb, spans, vecs, mode="add")`, `encode_batch(refs, tok, device)`, `spans_from_metas(metas, refs, offset=0)`. Self-test passes. |
| `rag_loss_new.py` | `make_segment_weight_tensor(...)`, `compute_text_loss_rag(...)`, `compute_audio_loss(...)`, `compute_rag_losses(...)`. `kind_weight` has `ref` but **not** `ref_slot`; defaults `ref_weight=5.0` → would train the model to *emit* the ref block. |
| `models/lm.py` | `LMModel.forward_train()`, `embed_codes()`, `forward_embeddings()`, `forward_depformer_training()`; `LMGen.step_embeddings()` writes **dummy** tokens into the cache (unusable for a multi-frame ref block). No context hook anywhere. |
| `backbone_distill.py` | `rollout_tokens_batched()`, `collect_tokens()`, `fit()`, `train_step()`, `forward_distill()`, `loss_text/audio/hidden/feat`, `build_student`, `apply_lora`, `freeze_transformer`, `plain_state_dict`, `_save_student`, `load_student_backbone`, `load_student_weights`. `frame_valid` = `codes[:,8] != -1`; ends at the sample length, **no truncation**. |
| `train_latent.py` | Reference only. Has `add_special_tokens_and_resize()` (correctly resizes `text_emb`, `text_linear`, **and** `depformer_text_emb`, moves the pad/initial row 32000→32003, gradient-hooks all non-new rows). But instantiates nothing: no `RefAdapter`, no gate, and asserts a `RefResampler` with an external HF encoder. **Do not run.** |
| `server_s.py` | Has STT (`faster-whisper`), VAD, retrieval, `<lookup>` detection and `MAX_LOOKUP_WAIT_FRAMES = 1.5 * frame_rate` host fallback. But also `RefResampler` + per-frame `body_gate` streaming sum → wrong design. |
| `server_student.py` | Loads the 12.5 Hz S20 correctly; no RAG, no STT, no VAD. |
| `rag_index/manifest.json` | 30-chunk demo (`RAG.txt`, all-MiniLM-L6-v2). Smoke only. |

---

## 2. Why the current S20 rambles and answers facts from memory

1. **Free-run tail.** `rollout_tokens_batched` generates for the whole window
   (`T = plen + 600`) and never stops at the last assistant turn. Average
   conversation = 33.5 s, so ~14 s of every training sample is the teacher
   improvising past the end of the dialogue. The student learned that as the
   target distribution → it does not stop.
2. **Grounded-heavy data.** The 65 k manifest the S20 was actually trained on
   (`/workspace/train/temp_data/train_split.jsonl`, first 5 k lines sampled were
   all `body`-grounded) teaches "answer factual questions from memory, confidently".
   There is no `<lookup>`, no abstain, no "I don't know" in it.
3. **Prompt frames counted as targets.** The persona block (~260 frames) is
   included in the text KD target, so capacity goes into reciting the system prompt.
4. **No turn-end signal.** Nothing in the loss says "the answer ends here".

Stage 1 fixes 1, 3, 4 and changes the data mix to RAG-free (fix for 2's failure
mode: the teacher never sees a reference, so it cannot teach grounded answers —
and we do not want it to).

---

## 3. Locked design (recap, so the code matches)

* **Prefill-only, no per-step module.** Reference → `K` latent vectors, written
  once into `K` reserved frames: `<lookup> <ref_start> [K slots] <ref_end>`,
  placed immediately after `<lookup>`, before the response body. The slots enter
  the KV cache at the normal 80 ms cadence. Zero steady-state cost.
* **Adapter uses the LM's own `text_emb`**, no external HF encoder.
* **`<lookup>` is a text token**, trained with weighted CE (not a separate head).
  No `context_gate_net`, no `streaming_sum`, no per-frame gate.
* **`<ref_start>` / slots / `<ref_end>` are never targets.** `ref_weight = 0`,
  `ref_slot_weight = 0`. At serving the server force-feeds them.
* **`K = 16`** slots → 1.28 s of think-pause, inside the 1.5 s lookup wait.
* Stage 1 uses **RAG-free data only**; Stage 2 adds the RAG data. Never run plain
  Stage 1 again after Stage 2 (unless KD is masked over grounded body spans).

---

## 4. Stage 1 — proper re-distill (target ≈ 9 h GPU)

### 4.1 Manifest

Build `/workspace/data/rag_v3/stage1_train.jsonl` by filtering `broad_train.jsonl`
on `metadata.rag_kind in {"no_rag_qa", "regular_chat"}` (4 500 files). Sidecar read
is ~10 s for 9 500 files. Keep a held-out 200 (split by conversation id, not file)
for the Stage-1 gate.

### 4.2 Truncation at the last turn end (the core fix)

`segment_meta.turns[*].end_sec` gives the true end of the dialogue. For every
collected rollout:

```
keep = plen + ceil((max(end_sec) + tail_sec) * 12.5)     # tail_sec = 0.8 -> 10 frames
out  = out[:, :, :keep]
```

`plen` comes from `forced_prompt_len()` (already implemented and used).
The slice is exact: the transformer is causal and the depformer is delayed by a
fixed 1 frame, so frames ≥ `keep` never influence frames < `keep`.
Result: every training sample ends 10 frames after the last word, instead of 14 s
of improv.

Implementation: add `return_meta=True` to `DistillInterleavedTokenizer` (parse
`parse_segment_meta(...)` without splicing), carry `sample.segment_meta` through
`Batch.collate` (already supported), and truncate inside `collect_tokens()`.
Do **not** touch `interleaver.py` for this.

### 4.3 Loss masking

* `frame_valid` already excludes right-padded frames (`codes[:,8] == -1`). Verify
  `zero_token_id == -1` on the pod; if it is `0`, real silence frames at the end of
  speech would be masked — assert and log.
* Text KD: additionally zero frames `< plen` (persona block). Hidden/feature KD
  keeps them (they teach persona conditioning). Audio KD keeps them (they are
  silence/sine, negligible).
* Leave `BackboneDistillWeights` as is (`hidden 1, feat 1, text 1, audio 0.5`,
  `audio_frame_keep 0.25`).

### 4.4 Recipe (same shape as S20 so the comparison is clean)

| Knob | Value |
|---|---|
| kept layers | `[0,1,2,4,6,8,10,12,14,15,17,19,21,23,25,27,28,29,30,31]` (from S20) |
| init | fresh `build_student(teacher, kept, base_kwargs)` (teacher copy), **not** S20 |
| LoRA | r 64, alpha 128, lr 2e-4, dropout 0 |
| base lr | s1 2e-5 → s2 1e-5, warmup 100, stage1_frac 0.2 |
| batch | 8, grad_accum 1 |
| steps | 2 000 (4 500 samples × 4 epochs = 2 250; cap 2 000) |
| dtype | student fp32 master, bf16 autocast, bf16 save |
| window | `duration_sec=48`, `tail_sec=0.8` |

### 4.5 Commands

```bash
cd /workspace/personaplex
source /workspace/menv/bin/activate

# 0) manifest (new script, ~1 min)
python -m moshi_local.moshi_distill.make_stage1_manifest \
    --broad /workspace/data/rag_v3/broad_train.jsonl \
    --audio-dir /workspace/data/rag_v3/audio \
    --out /workspace/data/rag_v3/stage1_train.jsonl

# 1) collect teacher rollouts (~5 h at 48 s windows, ~12x real time)
python -m moshi_local.moshi_distill.backbone_distill collect \
    --train-data /workspace/data/rag_v3/stage1_train.jsonl \
    --duration-sec 48 --batch-size 4 --rollout-batch-size 32 \
    --max-samples 4500 --temp 0.8 --top-k 250 \
    --moshi-weight /workspace/personaplex/ckpt/personaplex-7b-v1/model.safetensors \
    --out /workspace/personaplex/rollouts_stage1_s48.pt

# 2) fit (~3.5 h)
python -m moshi_local.moshi_distill.backbone_distill fit \
    --tokens /workspace/personaplex/rollouts_stage1_s48.pt \
    --keep 0,1,2,4,6,8,10,12,14,15,17,19,21,23,25,27,28,29,30,31 \
    --epochs 4 --max-steps 2000 --batch-size 8 \
    --lr-s1 2e-5 --lr-s2 1e-5 --stage1-frac 0.2 \
    --lora --lora-r 64 --lora-alpha 128 --lora-lr 2e-4 \
    --student-dtype bfloat16 --autocast bfloat16 --save-dtype bfloat16 \
    --eval-every 100 --save-every 250 --held-out 40 \
    --out-dir /workspace/personaplex/ckpt/backbone_S21
```

(Exact `--moshi-weight` path must be confirmed on the pod; `serve_s20.sh` has the
current one.)

### 4.6 Stage-1 gate (must pass before Stage 2)

| Metric | Threshold |
|---|---|
| `cos_hidden` (held-out) | ≥ 0.975 (≥ S20) |
| `text_agree_nonpad` | ≥ 0.915 (≥ S20) |
| stop behaviour: frames generated after last turn end, over 100 prompts | ≤ 12 frames (1 s) p90 |
| manual listening: 10 personas × 2 prompts | no loop, no ramble, greeting present |
| factual questions with no reference | answers "I'd need to check" / asks, **not** invented facts |

If "stop behaviour" fails, increase `tail_sec` to 0 (hard cut) and re-fit — do not
proceed.

---

## 5. Stage 2 — RAG adaptation (target ≈ 10 h, CE only)

### 5.1 Pre-tokenize once (new `rag_cache.py`, ~1 h)

14 000 samples × ~700 frames × 17 int16 ≈ 340 MB. Cache:
`codes [1,17,T]`, `ref_ids [≤256]`, `meta` (turns after splicing),
`plen`, `kind`, `conv_id`. This removes Mimi encoding from the epoch loop and makes
ablations cheap. Splicing happens here, via the existing
`DistillInterleavedTokenizer(use_rag=True)`.

### 5.2 Vocabulary resize

Call `add_special_tokens_and_resize(student, tokenizer)` **once** after loading the
S20 checkpoint, then `torch.nn.Parameter`-enable the three new rows
(`text_emb.weight`, `text_linear.weight/bias`, `depformer_text_emb.weight`) —
the function only installs gradient hooks, it does **not** set `requires_grad`.
Assert afterwards: `text_card == 32003`, `text_initial_token_id == 32003`,
`text_padding_token_id == 3`, `text_linear.out_features == 32003`,
`text_emb.num_embeddings == 32004`.

### 5.3 Trainable set

| Group | Trainable | LR |
|---|---|---|
| `RefAdapter` | full | 1e-3 |
| LoRA r64 on transformer | yes | 1e-4 |
| 3 new token rows (3 tables) | yes | 1e-4 |
| everything else | frozen | — |

Rationale: S20 is a good acoustic/linguistic prior; we only want to add the
lookup→slots→grounded-body pathway.

### 5.4 Forward with context

Add to `models/lm.py`:

```python
def forward_train_with_context(self, codes, ref_spans, ref_vectors, mode="add"):
    B, K, T = codes.shape
    initial = self._get_initial_token().expand(B, -1, -1)
    delayed = _delay_sequence(self.delays, codes, initial)
    delayed = torch.cat([initial, delayed], dim=2)
    emb = self.embed_codes(delayed[:, :, :-1])          # [B, T, dim]
    emb = RefAdapter.inject(emb, ref_spans, ref_vectors, mode=mode)   # spans already +1
    transformer_out, text_logits = self.forward_embeddings(emb)
    logits = self.forward_depformer_training(delayed[:, :, 1:], transformer_out)
    ... same undelay as forward_train ...
```

Offsets: `spans_from_metas(metas, refs, offset=1)` because `delayed` prepends the
initial token (code frame `s` → emb index `s+1`). Sanity test: run two passes with
ref A and ref B on the same codes; assert `hidden[:, ref_end_frame:]` differs by
> 1e-3 and `hidden[:, :lookup]` is bit-identical.

### 5.5 Losses

```python
seg_w = make_segment_weight_tensor(
    B, T, metas, device,
    ref_weight=0.0,            # never train <ref_start>/<ref_end>
    ref_context_weight=0.0,
    ref_slot_weight=0.0,       # REQUIRES adding the "ref_slot" key (see §7)
    lookup_weight=6.0,         # rare token, needs the boost
    lead_weight=1.0, filler_weight=1.0,
    body_weight=2.0, no_rag_weight=1.0,
)
text_loss  = compute_text_loss_rag(text_logits, codes[:, :1], text_mask,
                                   seg_w, text_padding_ids={3, 0})
audio_loss = compute_audio_loss(audio_logits, codes[:, 1:1+8], audio_mask,
                                synthetic_keep=~ref_or_slot_mask)
loss = text_loss + 0.5 * audio_loss + 0.05 * anchor
```

`anchor` = `L_hidden` against a frozen S20 copy on the same batch (cheap, one extra
forward) so the backbone does not drift away from its distilled behaviour. Weight
0.05, drop to 0 if it fights the grounding.

**No teacher KD.** The teacher's `text_card` is 32 000 (cannot represent
`<lookup>`) and it never saw the reference, so its body logits encode exactly the
ungrounded behaviour we are trying to remove.

### 5.6 Schedule & commands

* batch 8, 3 epochs (≈5 250 steps), cosine, warmup 200.
* loss logging per segment kind (`lookup`, `body`, `regular`) so a dead `<lookup>`
  signal is visible at step 50.
* eval every 250 steps on 300 held-out grounded + 100 abstain.
* save `step_*, adapter.pt` (adapter + tokenizer config + K + text_card).

```bash
python -m moshi_local.moshi_distill.rag_cache \
    --audio-dir /workspace/data/rag_v3/audio \
    --train /workspace/data/rag_v3/broad_train.jsonl \
    --out /workspace/personaplex/rag_v3_tokens.pt --duration-sec 48

python -m moshi_local.moshi_distill.rag_train \
    --init /workspace/personaplex/ckpt/backbone_S20_lora/step_1800/student.pt \
    --tokens /workspace/personaplex/rag_v3_tokens.pt \
    --tokenizer /workspace/personaplex/moshi_local/tokenizer.model \
    --adapter meanpool --n-slots 16 \
    --epochs 3 --batch-size 8 --lora-r 64 --lora-lr 1e-4 --adapter-lr 1e-3 \
    --out-dir /workspace/personaplex/ckpt/rag_S21 --eval-every 250
```

### 5.7 Stage-2 kill criteria

| Signal | Meaning | Action |
|---|---|---|
| `lookup` CE flat after 300 steps, emit rate 0 at inference | weights wrong (`ref_weight`≠0, or `ref_slot` still 1.0) | stop, fix weights |
| body CE improves but ref-ablation delta ≈ 0 | adapter ignored | switch to perceiver |
| audio CE rises > 30 % over S20 | drift | lower LoRA lr, raise anchor weight |
| `<lookup>` emit rate > 40 % frames | runaway | raise `lookup_weight`, check span frame indices |

---

## 6. Stage 3 — serving

### 6.1 New `LMGen` method (required)

`step_embeddings()` writes dummy tokens into the cache, which corrupts later steps.
Add:

```python
def step_embeddings_with_tokens(self, embeddings, text_token=None,
                                moshi_tokens=None, input_tokens=None):
    prepared = self.prepare_step_input(input_tokens, moshi_tokens, text_token)
    if prepared is None: ...
    transformer_out, text_logits = state.graphed_embeddings(embeddings)
    return self.process_transformer_output(...)
```

Timing convention (same as training): the token written at step `t` becomes the
*input* at step `t+1`. So for the block
`<ref_start> s0 s1 ... s15 <ref_end>`:
1. force-feed the tokens for the whole block into the cache, silence audio
   (`moshi_tokens = SINCE_FRAME`), one per step;
2. for each slot frame, call `step_embeddings_with_tokens(embed_codes(token) + vec_j)`
   at the step *after* its token was written;
3. return to the normal `step()` loop.

Unit test (must exist before wiring the server): 40 forced steps on the student,
same tokens, `vec_j = 0` vs `vec_j = adapter(ref)`; assert the last hidden state
differs and that no step raises the `check` assertion.

### 6.2 Server = `server_student_rag.py` (copy of `server_student.py` + ports)

Port from `server_s.py`, keeping the student's 12.5 Hz loop:

* `faster-whisper` streaming STT (installed: 1.2.1 in `/workspace/menv`) +
  existing VAD.
* Retrieval: encode transcript tail → top-1 passage.
* `<lookup>` detection on the emitted text token; on hit, freeze the main loop,
  feed silence, run retrieval, wait ≤ `1.5 * 12.5 = 18` frames, then
  `MAX_LOOKUP_WAIT_FRAMES` host fallback if retrieval is late (same policy as
  `server_s.py`).
* Force the ref block + injected slots, then resume.
* **Do not port** `RefResampler`, `body_gate`, `streaming_sum`, or any per-frame
  reference path.

### 6.3 Index

`rag_index/` (30 chunks) is a smoke test. New `rag_index_v3.py`: passages =
`text_conditions.reference` (plus the surrounding doc text if available),
embeddings `all-MiniLM-L6-v2` (verify installed; else numpy cosine, 14 k × 384 is
trivial), split by conversation id so eval passages are unseen. Gate: hit@1 ≥ 0.85
on 300 held-out queries built from the assistant's body text. If hit@1 is lower,
the model cannot be grounded and no amount of training helps — fix retrieval first.

---

## 7. File-by-file change list

| File | Change | Risk |
|---|---|---|
| `distill/data.py` | `return_meta` flag; parse `segment_meta` without splicing | low |
| `backbone_distill.py` | `collect_tokens(truncate=True, tail_sec=0.8)`; prompt mask for text KD; `--no-truncate`, `--tail-sec` | low |
| `make_stage1_manifest.py` | new, ~40 lines | none |
| `interleaver.py` | none required (`ref`/`ref_slot` spans already recorded) | — |
| `rag_loss_new.py` | add `"ref_slot"` to `kind_weight`; make `ref_weight` default **0.0**; assert in `compute_rag_losses` | **high if missed** |
| `models/lm.py` | `forward_train_with_context`, `step_embeddings_with_tokens` | medium |
| `rag_cache.py` | new: pre-tokenize + splice + cache | low |
| `rag_train.py` | new: resize, adapter, LoRA, weighted CE, eval hooks | medium |
| `ref_adapter.py` | optional `--adapter perceiver` behind a flag; mean-pool unchanged | low |
| `rag_index_v3.py` | new: passages + embeddings + held-out split | low |
| `server_student_rag.py` | new, from `server_student.py` + ported blocks | medium |
| `serve_s21_rag.sh` | new launcher (32 k tokenizer, port 8999) | none |
| `eval_rag.py` | new: lookup P/R, grounded accuracy, abstain rate, stop behaviour | low |

Do **not** modify: `train_latent.py` (keep as reference), `server_s.py` (keep as
reference), `train_emotion.py` / `emotion_*` (out of scope), `rollouts_v2.pt`.

---

## 8. Timeline / GPU budget

| Slot | Work | Wall |
|---|---|---|
| 0 | patch §7 Stage-1 parts, self-tests, manifest, 50-sample smoke | 2 h |
| 1 | Stage 1 collect (4 500 × 48 s @ ~12× RT) | ~5 h |
| 2 | Stage 1 fit (2 000 steps) + gate eval | ~4 h |
| 3 | patch Stage-2 parts, pre-tokenize, 5-step smoke, 500-sample adapter A/B | ~3 h |
| 4 | Stage 2 train (3 epochs, 5 250 steps) + eval | ~7 h |
| 5 | index build + serving + latency + eval + demo recording | ~6 h |
| | **total** | **~27 h GPU, ~2.5 days wall** |

One A100-80GB, `nvidia-smi` currently idle at 0 MiB, 1 TB RAM, 668 TB free on
`/workspace`. No multi-GPU needed.

---

## 9. Risks

1. **`zero_token_id` assumption.** `frame_valid` masks `codes[:,8] == zero_token_id`.
   If the pod's model uses `0` (not `-1`), real silence frames are dropped from
   every loss. Check once, log it, assert.
2. **`forced_prompt_len` miss.** It matches `SILENCE_TOKENS`/`SINE_TOKENS` exactly.
   If a sample's prompt block differs (e.g. no system prompt in the sidecar),
   `plen=0` and truncation lands 260 frames too early. Assert `plen > 0` for every
   Stage-1 sample; fallback `plen = T - 600`.
3. **Splicing shifts spans.** `splice_rag_frames` inserts `<lookup>` first and then
   the ref block, shifting spans twice. The existing `shift_spans_from` handles it,
   but the `ref`/`ref_slot` spans are appended *after* the shift, so they must not
   be shifted again. Add an assertion that `meta.ref_slot.end_frame <= T` and that
   `body_start > ref_slot.end_frame`.
4. **Stage-2 without KD drifts audio.** Mitigated by audio CE on the 8 agent
   codebooks + 0.05 hidden anchor + a hard eval gate on audio CE.
5. **`<lookup>` never fires.** The token sits in a 32 k softmax and occurs once per
   conversation. `lookup_weight=6` + per-kind logging + a "force-emit at step 50"
   diagnostic that overfits 4 samples; if the model cannot overfit 4, the bug is
   in the pipeline, not the schedule.
6. **Serving embedding/token desync.** The single most likely silent failure —
   mitigated by the 40-step unit test in §6.1.
7. **Retrieval quality.** Gate hit@1 ≥ 0.85 before Stage 2; otherwise everything
   downstream measures the index, not the model.
8. **Eval leakage.** Split by conversation id, and check no `reference` string from
   eval appears in train (exact + 8-gram overlap).

---

## 10. Immediate next actions (chosen order)

**Stage A — RAG on S20 (now).**

1. Patch `rag_loss_new.py` (`ref_slot` key, zeroed ref weights) and `models/lm.py`
   (`forward_train_with_context`, `step_embeddings_with_tokens`); run the self-tests.
2. Write `rag_cache.py`; pre-tokenize `rag_v3` with splicing + turn-end truncation.
   Assert: `plen > 0`, `ref_slot.end_frame <= T`, `body_start > ref_slot.end_frame`,
   truncation lands within 10 frames of the last turn end.
3. Write `rag_train.py`; overfit 4 samples first (must reach `<lookup>` emit rate 1.0
   and near-zero body CE). If 4 samples cannot be overfit, the pipeline is wrong.
4. 500-sample adapter A/B: body-token logprob delta with ref vs zero ref. Gate:
   delta must be clearly above seed noise, else switch to the perceiver.
5. Full run (3 epochs, 5 250 steps, ~7 h) → §5.7 gate table.

**Stage B — proper re-distill (only if the Stage-A stop gate fails).**
Manifest → collect → fit → §4.6 gate → re-run Stage A on the new base.

**Stage C — serving + eval.** `rag_index_v3.py`, `server_student_rag.py`,
`serve_s21_rag.sh`, `eval_rag.py`; 40-step cache/embedding unit test before wiring
the server.
