# Distill first, or RAG first? — training-order plan

**Decision: distill first, then do RAG adaptation on the frozen distilled
student. Do *not* do RAG first, and do not "RAG then distill".**

Optional third stage: a short joint refresh (distill + RAG together) if the RAG
stage causes measurable chat/audio drift.

---

## 1. Why this order

The teacher (PersonaPlex) is **frozen and has no retrieval context**. Therefore:

* The teacher can supply `L_KD` / `L_hidden` targets only for *teacher-like*
  behaviour — speech, turn-taking, persona, factual QA from parametric memory.
  It **cannot** produce logits for a grounded, counterfactual body: on those
  frames its distribution is the parametric (wrong) answer.
* Grounding is therefore learned only through **task CE** on the grounded body
  (self-supervised on the spoken counterfactual) + the **reliance contrastive**
  loss. That signal needs a backbone that can already produce fluent speech and
  text — otherwise you are teaching grounding to a model that cannot yet talk.

Consequences:

| Order | Verdict |
|---|---|
| **Distill → RAG** | ✅ student base is competent; RAG adds a small adapter/head; regressions are attributable and rollback-able |
| RAG → distill | ❌ RAG on an un-distilled student = grounding on a broken backbone; and the later KD pass would **overwrite the grounded behaviour** (teacher has no context) — it teaches the model to answer from memory again |
| RAG on the 7B teacher, then distill | ❌ the student has a different rate/size; the teacher cannot accept the injected context, so the reliance behaviour does not transfer through hidden/logit KD |
| Fully joint from scratch | ⚠️ viable but harder: random adapter/trigger early, slower convergence, noisy attribution. Best used as a *short refresh*, not the primary run |

---

## 2. What "distill first" must include (and must exclude)

The distillation stage has **no context injection**, so the counterfactual /
refusal bodies must **not** be distilled — that would teach the student to
hallucinate the fabricated fact or to refuse from parametric memory.

**Stage 1 distillation subset (safe, real behaviour only):**

| kind | count | why |
|---|---|---|
| `no_rag_qa` | 3,000 | parametric factual QA — real, distillation-consistent |
| `grounded_real` | 5,000 | real reference + real body — true facts, full KD OK |
| `regular_chat` | 1,500 | greetings / small-talk |
| **total** | **9,500** | |

Exclude `grounded_cf` / `grounded_distract` / `abstain` (4,500) from Stage 1 —
their bodies are fabricated/refusal and are only meaningful *with* injected
context. (If you want their greeting/lead/filler coverage, include them but mask
KD + task CE over the body span so only lead/filler/chat are distilled.)

Build the manifest by filtering the full `train.jsonl` on `rag_kind` (sidecar
metadata) → `train_distill.jsonl` (9,500 lines).

---

## 3. Staged plan

### Stage 0 — data (done)
14,000 transcripts + audio + sidecars (`/workspace/data/rag_v3/audio/train.jsonl`).

### Stage 1 — Student distillation (base model)
`train_distill.py` / `backbone_distill.py`, teacher frozen.

* Data: the 9,500 `train_distill.jsonl`.
* Loss: `L_task` (text/audio CE) + `L_hidden` + `L_KD` + `L_audio` (existing
  `distill_loss.py`), full KD (no masking needed — no grounded bodies here).
* Trainable: compressor, student transformer, upsampler, hidden projection.
* Gate to Stage 2: audio intelligibility (ASR-WER vs teacher), text quality,
  chat/persona, turn-taking. **If you already have a good distilled student
  (`ckpt/backbone_S16`, `distill_v1`), reuse it and skip to Stage 2.** Do not
  start RAG on a student that fails this gate.

### Stage 2 — RAG adaptation (grounding)
Init from the Stage-1 student. Add `RefAdapter` + `rag_head` (trigger).

* Data: full 14,000 mix —
  * positives: `grounded_cf` 3,000, `grounded_distract` 1,000, `abstain` 500,
    `grounded_real` 5,000;
  * negatives (trigger must stay off): `no_rag_qa` 3,000, `regular_chat` 1,500.
* Injection: prefill-only latent slots — `RefAdapter` writes K vectors into the
  `<ref_slot>` frames; **no per-step module** (80 ms budget preserved).
* Loss:
  * weighted text CE + audio CE (`rag_loss_new`), ref-slot frames weight 0;
  * **reliance contrastive** (correct vs shuffled reference body log-prob);
  * **trigger BCE** (`rag_head`, +1 at `<lookup>` on grounded turns, 0 elsewhere);
  * **teacher KD only on non-grounded frames** — mask KD over the body span of
    grounded turns (the teacher's grounded-body logits are wrong).
* Trainable: `RefAdapter` + `rag_head` first (backbone frozen); if grounding is
  weak, add **LoRA** on attention/FFN at low LR. Keep the backbone mostly frozen
  so chat/audio do not drift.
* Gate to Stage 3/serving: context-ablation gap (correct vs empty vs wrong ref),
  trigger precision/recall, abstain rate, memory-leak probe. Also re-check chat
  WER to confirm no drift.

### Stage 3 — optional joint refresh (anti-forgetting)
Only if Stage 2 measurably degraded chat/audio. Run a **short** joint pass on the
full 14,000 with the kind-aware losses (KD masked on grounded bodies), low LR,
LoRA + adapter trainable. This is the "distill + RAG at the same time" mode — it
works, but it is a *refresh*, not the first thing you do.

### Stage 4 — eval / freeze
Eval suite (context-ablation, trigger P/R, abstain, WER, memory-leak), then
freeze and export.

---

## 4. The one mistake to avoid

**Never run a plain distillation pass *after* RAG training.** `L_KD` pushes the
student back toward the teacher's context-free distribution, which is exactly
the parametric-memory answer — it erases the grounding you just trained. If you
must re-distill after RAG, you have to mask KD over every grounded body span
(Stage 3).

---

## 5. Why sequential beats fully-joint as the primary plan

* The adapter/trigger start random; joint-from-scratch lets them inject noise
  into the student while it is still learning to speak.
* Mixing a heavy KD objective with a small grounding objective early on makes it
  hard to tell whether a regression is the backbone or the RAG path.
* Distillation is the expensive phase; RAG is cheap (adapter ≈2M params + a head
  + optional LoRA). Doing them separately means the expensive phase runs once,
  and the cheap phase can be iterated quickly.

## 6. Practical notes

* If a good distilled student already exists, Stage 1 is effectively done —
  start at Stage 2 on that checkpoint.
* The `RefAdapter`/`<ref_slot>` frames are at 12.5 Hz pre-compressor, so the
  student's 2-frame compressor sees them as ordinary pairs (keep K even).
* Keep the trigger head at the student's 6.25 Hz hidden state (160 ms) — fine for
  a "start retrieval" decision.
* Checkpoint after every stage; keep Stage-1 student and Stage-2 adapter
  separately so you can roll back.
