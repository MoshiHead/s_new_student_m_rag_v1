# RAG / Mid-Stream Context Injection — Plan & Dataset Design

Author: engineering analysis, based on the pod at `157.157.221.29` (`/workspace`) and
the local package copy in this folder.

---

## 0. TL;DR

* The model already has the two pieces you need: streaming **latent ref-slots**
  (`<ref_start> … <ref_end>`) in the user-audio channel and a small
  **projection/gate/adapter** (`RefResampler`, `context_gate_net`, `rag_head`).
* **Recommended injection = "prefill-only latent slots":** encode the retrieved
  reference **once** with an adapter, write the resulting K vectors into the K
  reserved frames right after `<lookup>`, and let the model consume them at the
  normal frame cadence while it is already speaking the filler. **No new
  per-step module** → the 80 ms audio budget (and the student's 160 ms temporal
  budget) is untouched. The only per-step optional extra is a 1-D trigger head.
* **Dataset problem:** every current RAG example uses a *real* reference and a
  *real* answer from SQuAD/HotpotQA/NQ/FinanceQA. The base model already knows
  many of those answers, so the cheapest way to lower body CE is to **ignore the
  injected context** and answer from parametric memory. Grounding overlap is
  ~100%, but there is no counterfactual supervision.
* **Better dataset (your idea, formalised):** keep the *queries* from the
  existing sets, but **fabricate the reference** so the answer is *not* in the
  model's memory (counterfactual facts), plus distractor / unanswerable /
  no-retrieval negatives, plus regular (non-RAG) conversation. Target mix below.

---

## 1. What is actually on the pod (inventory)

### 1.1 Code

| Area | Files | Notes |
|---|---|---|
| RAG (full-rate Moshi) | `moshi/train_rag.py`, `moshi/rag_index.py`, `moshi/rag_loss.py`, `moshi/interleaver.py`, `moshi/build_index.py` | v1 "ReferenceAdapter" (mean-pool `text_emb` → MLP → streaming-sum) + `<ref>` block slot variant |
| RAG (latent, later) | `moshi_distill/train_latent.py`, `moshi_distill/rag_loss_new.py`, `moshi_distill/interleaver.py` | `RefResampler` (external HF encoder + 16 cross/self-attn queries), `context_gate_net`, `rag_head`, contrastive correct-vs-wrong-ref loss |
| Distillation (student) | `moshi_distill/backbone_distill.py`, `distill_loss.py`, `models/student.py`, `student_gen.py`, `server_student.py`, `train_distill.py`, `distill/data.py` | 6.25 Hz `StudentLM` (compressor 12.5→6.25, upsampler 6.25→12.5), frozen teacher heads |
| Dataset build | `/workspace/build_dataset.py`, `build_dataset_fin.py`, `extended*.py`, `generic.py`, `combined_meta.py`, `train/build_sidecard.py` | LLM 3-pass transcript builder → persona extension → TTS/Whisper align → sidecar |
| Tokenizer | `moshi_local/tokenizer.model` | vocab **32003**: `<ref_start>`=32000, `<ref_end>`=32001, `<lookup>`=32002. The 32k base `tokenizer_spm_32k_3.model` does **not** have them. |

### 1.2 Data on the pod

| File | Records | What |
|---|---|---|
| `data/hotpot_transcripts.jsonl` | 5,973 | HotpotQA/SQuAD → lead/filler/reference/body |
| `data/nq_train_40k.jsonl` | 5,005 | Natural Questions |
| `data/nq_transcripts_1_4090.jsonl` | 485 | NQ |
| `data/fin_transcripts.jsonl` | 2,370 | FinanceQA (only **464 unique queries** — very repetitive) |
| `data/extended_hotpot.jsonl`, `extended_nq_*.jsonl` | equals sources | + persona / system_prompt / full conversation / qa_turn_index |
| `new_e.jsonl` | ~large (277 MB) | same extended schema |
| `kta.jsonl` (85 MB), `train/all_entries.jsonl` (34 MB) | large | finance + persona |
| `new_audios/data/train.jsonl` | **91,160 chunks** | `{path, duration}` manifest → `data/rag_from_new_e/audio/*.wav`; each `.wav` has `.new.combined.json` + `.sidecar.json` |

### 1.3 Data example (one real record)

Sidecar (what the tokenizer consumes):

```json
{
  "alignments": [["Hi",[0.25,0.56],"ASSISTANT"], ["there!",[0.594,0.904],"ASSISTANT"], ...],
  "duration": 36.17,
  "segment_meta": {"turns":[
      {"kind":"regular","start_sec":0.25,"end_sec":7.449},
      {"kind":"regular","start_sec":10.89,"end_sec":14.786},
      {"kind":"lookup", "start_sec":18.59,"end_sec":18.75},
      {"kind":"lead",   "start_sec":18.75,"end_sec":21.87},
      {"kind":"filler", "start_sec":22.253,"end_sec":25.018},
      {"kind":"body",   "start_sec":25.403,"end_sec":35.636}]},
  "text_conditions": {
      "reference": "Blue jeans were invented in 1871 by Jacob W. Davis in partnership with Levi Strauss & Co. ...",
      "system_prompt": "You are Drew Kimura, a Knowledge Researcher at Verity Intelligence Hub. ..."},
  "metadata": {"persona": {...}, "conversation": [...], "qa_turn_index": 3}
}
```

So a training example is: **greeting + smalltalk + (user query) → `<lookup>` → lead →
filler → `<ref>` block → body**, where only the body is required to be grounded.

### 1.4 Current loss wiring

* `moshi_distill/distill/data.py` deliberately **strips RAG** (`segment_meta=None`,
  `reference_text=""`) and only splices the `<system>` persona. So student
  distillation today is **RAG-free** weighted CE + teacher distillation.
* `moshi_distill/interleaver.py` defines `_build_ref_slot_frames` (K latent slots
  in the **user audio channel cb 8**, marked with `zero_token_id`) but the ref
  splice is **not wired into `__call__`** — the latent-RAG path is mid-refactor.
* `moshi_distill/train_latent.py` still imports the external HF `AutoModel` text
  encoder inside `RefResampler` (an "external module" you said you want to avoid).

---

## 2. Mid-stream injection plan (budget-safe)

### 2.1 Constraints

* Audio everywhere runs at **12.5 Hz = 80 ms/frame**. The student temporal
  transformer runs at **6.25 Hz = 160 ms/step** (compressor/upsampler bridge the
  two). Whatever we add must not add a *new per-frame neural module*.
* Retrieval returns a text chunk (typ. 40–400 words). At inference it arrives at
  the query boundary, exactly when the model starts the lead/filler (~1–3 s in
  the data). We can use that pause to ingest context.

### 2.2 Mechanism A — text-prefix injection (fallback, zero new params)

`rag_index.inject_as_text_prompt()` already does this: feed the reference tokens
through `lm_gen.step()` with silence audio. Mirrors the `<system>` convention.

* ✅ zero new modules, no training of a mechanism.
* ❌ consumes the **text codebook**, which the student is trained `keep_main_only`
  to use for *its own speech* → fights the text objective.
* ❌ length × 80 ms of pause needed; long refs are impossible.

### 2.3 Mechanism B — prefill-only latent slots **(recommended)**

Convention (tokenizer already has the tokens):

```
… lead … filler … <lookup> <ref_start> [slot_0 … slot_{K-1}] <ref_end> body …
```

* At build/train time the interleaver splices `K` reserved frames (`K` derived
  from ref length, e.g. 4-word/slot, capped at e.g. 16).
* Slot frames carry `zero_token_id` in **cb 8 (user audio channel)** so
  `ScaledEmbedding` zeroes them; the continuous context vectors are then
  **written into the summed embedding** at those positions (exactly what
  `_build_ref_injection_tensors` is set up to do).
* The **adapter/projection** turns the reference into those `K` vectors:
  * reuse the LM's own `text_emb` (mean-pool over ref tokens) → small MLP, like
    `ReferenceAdapter` in `moshi/train_rag.py` (no external encoder), **or**
  * a `K`-query mini-Perceiver built only from `nn.MultiheadAttention` +
    `nn.Linear` (still an adapter, weights trained from scratch, no external
    model). This preserves more detail than a single mean-pool vector.
* The model then processes the slot frames at the **normal 80 ms cadence** while
  it is already speaking the filler → the vectors land in the KV cache. At body
  time the ordinary causal self-attention reads them.

Cost accounting:

| Cost | Where | Per-step? |
|---|---|---|
| Adapter forward on ≤400 tokens | once per retrieval, during the pause | **No** (one-off, ~ms) |
| K slot frames | K × 80 ms of streaming time, overlapped with filler audio | **No** new compute (same forward you already run) |
| KV growth by K (≤16) | normal attention | negligible vs context (~3000 frames) |
| Optional `rag_head` trigger (dim→1) | every temporal step | tiny linear |
| Optional `context_gate_net` on `[ctx;embed]` | every frame where active | small MLP — **drop it if you want literally zero new per-step cost** |

This satisfies "no external module other than an adapter/projection layer" and
keeps the 80 ms budget: **the only work added to a steady-state step is zero, or
a `dim→1` head.** The projection is amortised into the thinker pause.

Recommended refinements:

1. **Drop the external HF encoder** from `RefResampler`; feed the LM's own
   `text_emb` (in-distribution, no OOV/domain shift).
2. **Splice slots after `<lookup>`**, not after filler
   (`ref_slots_after_query=True`) so the context sits immediately before body.
3. Keep `K` **even** if you also feed the slots through the 6.25 Hz compressor
   (pair alignment); or insert at 12.5 Hz and record the span in original frame
   indices so the slice is exact.
4. Default to **no `context_gate_net`** (prefill carries the context). Keep a
   gate only if ablations show the model under-uses the slots; then make its
   hidden width small (e.g. `dim/4`).

### 2.4 Streaming / KV-cache details for the 6.25 Hz student

* Insert the `K` slot frames as **12.5 Hz frames between `<lookup>` and body**.
  The compressor pairs frames `(2t, 2t+1)`, so the slots are simply part of the
  normal stream — no new code path.
* The student's attention `context = 1500` steps = 3000 audio frames = **240 s**.
  A `K ≤ 16` slot block never approaches eviction; no special retention logic
  needed.
* On a **new query in the same session**, write new vectors into the next ref
  block. No cache surgery required (the old block ages out naturally).

### 2.5 Training objectives

Start from the existing `rag_loss_new.py` weighted CE and add, in order:

1. **Weighted text CE** — body weight up, ref-slot frames weight **0** (they are
   continuous placeholders, nothing to predict).
2. **Reliance / contrastive loss** (already in `train_latent.py`): body
   log-prob with the *correct* reference minus a body log-prob with a *shuffled*
   reference, margin `m`. This is the single most important objective — it is
   what forces the model to actually read the context instead of guessing.
3. **Trigger BCE** (`rag_head`, `compute_rag_trigger_loss`): positive on the
   `<lookup>` frame of grounded turns, negative on every other frame — teaches
   *when* to retrieve.
4. Optional **abstain loss**: for "unanswerable" examples, the target body is an
   honest "I couldn't find that in the material" → discourages hallucination.

Order: freeze everything, train adapter + (optional gate) + trigger head first
with reference always present; then mix in negatives (no-RAG turns, shuffled
refs, distractor refs); then LoRA/unfreeze lightly.

### 2.6 Inference flow

```
user finishes query
   └─ trigger fires (<lookup>) / or fixed policy
        └─ retrieve top-k chunks  (cb0 histogram in rag_index.py, or text embed)
             └─ adapter(reference) once  → K vectors
                  └─ emit <ref_start> [K slots] <ref_end> frames at 80 ms cadence
                       (model simultaneously speaks lead/filler)
                        └─ body generation reads slots from KV cache
```

---

## 3. Dataset analysis — and a better dataset

### 3.1 Measurements from the pod

| Set | n | ref words (mean/p90) | body words | grounded ≥0.15 | ref ⊆ body | dup queries |
|---|---|---|---|---|---|---|
| hotpot | 5,973 | 42 / 55 | 24.5 | 100% | 0.2% | 75 |
| nq40k | 5,005 | 42 / 56 | 23.7 | 100% | 1.3% | 16 |
| fin | 2,370 | 25 / 33 | 16.2 | 99% | 3.4% | **1,906 of 2,370** |
| new_e | ~20k+ | 42 / 57 | 23.7 | 100% | 1.1% | 196 |

### 3.2 Problems

1. **Parametric shortcut (the big one).** The reference is a real Wikipedia/QA
   passage and the body restates its answer. HotpotQA/NQ answers are in the base
   model's memory. The body CE can be minimised by ignoring the injected slots →
   the model learns to answer from memory and the RAG path is dead weight.
2. **29th query too short / too clean.** References average ~42 words, p90 ≈ 55.
   Real retrieval returns noisy, distractor-laden chunks (often 150–400 words).
   The model never learns needle-in-haystack selection.
3. **Finance set is template-collapsed** (464 unique queries, 80% duplicates).
   Useful for a narrow domain but must be capped by *unique query*, not raw rows.
4. **Almost no true negatives.** ~100% of examples are grounded turns; the model
   has no signal for "don't retrieve", "context is irrelevant", or "answer is not
   in the context". The trigger head would be trivially always-on.
5. **No regular conversation / smalltalk fraction in the audio manifests** (only
   greetings + a couple of smalltalk turns baked into the extended transcripts).
   Pure-RAG training erodes natural chit-chat and turn-taking.
6. **Mechanism is mid-refactor**: `interleaver.py` defines ref slots but does not
   splice them; `distill/data.py` strips references; `train_latent.py` still uses
   an external encoder. Any dataset format must be pinned to the convention we
   finalise first.

### 3.3 Proposed dataset: "fabricated references, real queries"

Your idea, made concrete:

> **Keep the user queries from the existing sets (they are realistic and already
> have TTS audio), but synthesise the reference and the grounded body so the
> model cannot answer from memory.**

For each source record `(q, real_ref, real_answer)`:

1. **Counterfactual reference** `r'`: rewrite `real_ref` changing the key fact
   (a date, number, name, place) to a plausible alternative, keeping it fluent,
   self-consistent and 80–250 words.
2. **Body `b'`**: answer `q` using **only** `r'` (so it states the counterfactual
   fact). Lead + filler stay generic and factual-free.
3. **Sanity checks**: `b'` grounded in `r'`; `b'` does **not** match the real
   answer; `q` unchanged.

Because the gold answer is now *not* in the model's memory, the only way to lower
body CE is to actually use the injected slots → this is the anti-shortcut signal
the current data lacks.

Four example classes, one schema, mix by design:

| Class | `rag_kind` | Reference | Body | Purpose |
|---|---|---|---|---|
| Grounded (counterfactual) | `grounded_cf` | fabricated, contains answer | derived from ref | force context use |
| Grounded (distractor/multihop) | `grounded_distract` | `r'` + 2 irrelevant chunks | derived from ref | needle-in-haystack |
| Unanswerable | `abstain` | plausible but **no** answer | "I couldn't find that in the material." | abstention, no hallucination |
| No-retrieval turn | `no_rag` | none / no `<lookup>` | answered from parametric memory | teach the trigger **not** to fire; preserve chat |

Target mix (by unique query, weighted, not raw rows):

```
40%  grounded_cf
15%  grounded_distract
10%  abstain
35%  no_rag  (regular conversation: greetings, smalltalk, chit-chat, open QA)
```

Additional construction rules:

* **Reference length**: 80–400 words, with 1–3 distractor paragraphs for the
  distractor class, so context use generalises to real retrieval.
* **Query reuse**: source `q` from hotpot/nq/fin; dedupe by normalised query
  (kills the finance 80%-duplicate problem).
* **Multi-turn sessions**: 2–3 queries per session, some grounded, some not, with
  the reference changing between turns → trains mid-session injection.
* **Perturbation strength**: for counterfactuals, make sure the fabricated fact
  is not inferable from world knowledge (a probe pass over the base model should
  show it does not produce the fabricated answer).
* **Schema unchanged** (so TTS/Whisper/sidecar pipeline is reused):
  `{id, persona, system_prompt, conversation, qa_turn_index, reference, lead,
  filler, body, raw_response, rag_kind}` — `rag_kind` rides in metadata and the
  sidecar builder copies it to `segment_meta`/`text_conditions`.

### 3.4 Audio implication

The **user query audio can be reused** unchanged. Only the grounded assistant
turn (lead + filler + body, ~15–30 words) changes, so it needs re-TTS; the
greeting/smalltalk audio is reusable. This is exactly the `build_dataset.py` →
`extended.py` → TTS/Whisper (`generic.py`) → `build_sidecard.py` chain, so a new
data builder can emit the same schema and drop into the existing pipeline.

### 3.5 Evaluation we must add (otherwise we can't tell it worked)

1. **Context-ablation gap**: same query with correct vs empty vs wrong reference;
   body log-prob / ASR WER gap should be large. (This is the contrastive metric
   at eval time.)
2. **Memory-leak probe**: base model on the fabricated answers — must be wrong.
3. **Trigger P/R**: `<lookup>` frame detection on grounded vs no-RAG turns.
4. **Abstain rate** on the unanswerable class.

---

## 4. Implementation roadmap

1. **Pin the convention.** Decide Mechanism B; wire the ref-slot splice in
   `moshi_distill/interleaver.py` (`<lookup> <ref_start> [K zero] <ref_end>`
   after the query; `ref_slot` span; K from ref length, even).
2. **Adapter without external encoder.** Add `RefAdapter` (mean-pool `text_emb`
   → 2-layer MLP → K slots, or K-query mini-Perceiver from `nn.Linear` +
   `nn.MultiheadAttention`). Reuse `context_gate_net`/`rag_head` wiring from
   `train_latent.py`, but make the gate optional.
3. **Stop stripping RAG in distillation.** Add a `use_rag` flag to
   `distill/data.py`; when on, emit `segment_meta` + `reference_text` and splice
   slots.
4. **Trainer.** Extend `train_distill.py` `_embed_and_transform` to write adapter
   vectors into slot frames, and add weighted CE (slots weight 0) + reliance loss
   + trigger BCE.
5. **Serving.** In `student_gen.py` / `server_student.py`, on trigger run the
   adapter once and stream the K slot frames during the filler.
6. **Dataset builder** `build_rag_v2.py`: load existing queries, synthesise
   counterfactual references + bodies via the LLM (reuse `build_dataset.py`
   prompts, add a "change the key fact" prompt), mix with no-RAG conversation,
   emit the unchanged schema, then reuse the TTS/align/sidecar chain.
7. **Eval** (§3.5).

---

## 5. Open decisions

* Adapter type: single pooled vector vs K-query mini-Perceiver.
* Slot channel: user-audio cb 8 (current scaffold) vs a dedicated text slot.
* Keep `context_gate_net` (tiny per-step cost) or prefill-only (zero).
* How strong to make counterfactuals (fully fabricated vs real-with-one-fact-swapped).
* Training the student jointly with distillation, or RAG-finetune after distillation.

---

## 6. Implementation status

Decisions locked in (user): **prefill-only latent slots**, adapter =
**LM `text_emb` mean-pool + MLP**, **no `context_gate_net`**.

| Step | Status | Artifact |
|---|---|---|
| Dataset builder (counterfactual + regular conversation) | **done, tested** | `build_rag_v2.py`, `RAG_DATASET.md` |
| Sidecar honors `rag_kind` (no `<lookup>` for no_rag/regular) | **done, unit-tested** | `train/build_sidecard.py` |
| `RefAdapter` (+ slot injection helper) | **done, tested** | `ref_adapter.py` |
| Interleaver ref-slot splice (`<lookup> <ref_start> [K] <ref_end>`) | **done, unit-tested** | `interleaver.py` (`splice_rag_frames`) |
| Distillation tokenizer opt-in RAG (`use_rag`) | **done, tested** | `distill/data.py`, `train_distill.py --use-rag` |
| `train_distill.py` RAG path (inject adapter + reliance + trigger, KD mask on body) | next | `train_distill.py`, `distill_loss.py` |
| Serving: adapter once on retrieval, stream K slots | next | `student_gen.py` / `server_student.py` |

Note: the latent-RAG path was mid-refactor — `interleaver.py` defined
`_build_ref_slot_frames` but never spliced, and `distill/data.py` deliberately
stripped RAG. Both are now wired through the shared `splice_rag_frames` helper,
and the distillation tokenizer opts in with `use_rag=True` (default off, so the
plain distillation path is byte-for-byte unchanged).

**Backups on the pod:** `interleaver.py.orig`, `distill/data.py.bak_rag`,
`train_distill.py.bak_rag`, `train/build_sidecard.py.orig`.
