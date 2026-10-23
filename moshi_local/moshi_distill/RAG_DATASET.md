# RAG v2 dataset — builder & distillation handoff

`build_rag_v2.py` generates the next-generation RAG dataset: **real queries,
fabricated (counterfactual) references**, plus **regular conversation** so the
same data can also drive plain distillation of the student.

Full design rationale: `RAG_PLAN.md`. This file is the operator how-to.

---

## 1. Why counterfactual + regular

| Problem with the v1 data | Fix here |
|---|---|
| Real references, real answers from SQuAD/NQ/HotpotQA → base model already knows the answer, so it can **ignore the injected context**. | `grounded_cf`: the key fact is **changed**, so the model can only answer by reading the injected reference. |
| No negatives → trigger head is trivially always-on; no abstention; context misuse untested. | `abstain`, `no_rag_qa`, `grounded_distract`. |
| Almost no free-form chat → RAG-only training erodes natural dialogue. | `regular_chat` (greeting + small-talk, no QA). |
| Finance set is 80% duplicate queries. | de-duplicated by normalised query. |

## 2. Classes and default mix

| `rag_kind` | Reference | Body | Default |
|---|---|---|---|
| `grounded_cf` | counterfactual fact | derived only from ref | 35% |
| `grounded_distract` | cf ref + 2 unrelated passages | derived only from ref | 12% |
| `abstain` | plausible but unrelated ref | honest "not in the material" | 8% |
| `no_rag_qa` | *(none)* | parametric answer | 20% |
| `regular_chat` | *(none)* | greeting + small-talk only | 25% |

`--ratios grounded_cf=0.35,grounded_distract=0.12,abstain=0.08,no_rag_qa=0.20,regular_chat=0.25`

## 3. Usage

```bash
# Real run — a local LLM fabricates fluent counterfactuals:
/workspace/menv/bin/python build_rag_v2.py \
    --sources data/hotpot_transcripts.jsonl,data/nq_train_40k.jsonl,data/fin_transcripts.jsonl \
    --out data/rag_v2_transcripts.jsonl \
    --model google/gemma-3-12b-it --load-in-4bit

# No model — deterministic rule-based counterfactuals (pipeline validation,
# ~70% of general refs + ~100% of finance have a swappable number; "weak"
# counterfactuals are dropped by default):
/workspace/menv/bin/python build_rag_v2.py \
    --sources data/hotpot_transcripts.jsonl,data/nq_train_40k.jsonl \
    --out /tmp/rag_v2_test.jsonl --no-llm --max-total 200

# Preview only:
/workspace/menv/bin/python build_rag_v2.py --sources ... --out "" --dry-run
```

Key flags: `--max-total`, `--limit-per-source`, `--smalltalk-turns`,
`--seed`, `--dedupe/--no-dedupe`, `--drop-weak-cf/--keep-weak-cf`,
`--no-llm`, `--model`, `--load-in-4bit`, `--preview-n`.

Output schema = the existing `data/extended_*.jsonl` schema **plus `rag_kind`**:

```json
{ "id": ..., "persona": {...}, "system_prompt": "...",
  "conversation": [...], "qa_turn_index": 3,
  "reference": "...", "raw_response": "[LEAD] ... [FILLER] ... [BODY] ...",
  "user": "...", "lead": "...", "filler": "...", "body": "...",
  "rag_kind": "grounded_cf" }
```

`regular_chat` has `qa_turn_index = null`, empty `reference`/`body`.

## 4. End-to-end pipeline (unchanged downstream)

```
build_rag_v2.py  ->  data/rag_v2_transcripts.jsonl
   -> TTS / recording                -> <id>.wav + <id>.new.combined.json
   -> generic.py (Whisper timestamps)   # word alignments per speaker
   -> combined_meta.py                  # merge audio annotations + metadata
   -> train/build_sidecard.py           # sidecar + data/train.jsonl manifest
   -> train_distill.py / train_latent.py
```

`train/build_sidecard.py` was patched to honour `rag_kind`:

* grounded / abstain → `lookup` + `lead` + `filler` + `body` (ref injected)
* `no_rag_qa` / `regular_chat` → plain `regular` turn, **no `<lookup>`**, so the
  trigger head learns *not* to fire.

## 5. Using the same data for distillation

The same audio + sidecars feed both trainers. `distill/data.py` already ignores
`reference`/`segment_meta` and uses `alignments` + `system_prompt`, so every
record (including `no_rag_qa` and `regular_chat`) is a valid distillation sample.

**Important KD rule for grounded turns.** The frozen teacher (PersonaPlex) has no
access to the injected reference, so on `body` frames its logits reflect
parametric memory. For grounded kinds (`grounded_cf`, `grounded_distract`,
`abstain`) the body text distribution can *contradict* the teacher. Therefore:

* `regular_chat`, `no_rag_qa` → full teacher KD + task CE (distillation-safe).
* `grounded_*` / `abstain` → **mask teacher KD over the body span**; train the
  body with task CE only (the reliance/contrastive loss supplies the grounding
  signal). `rag_loss_new.compute_rag_losses` already builds per-frame weights and
  `segment_meta` carries the body spans, so masking KD by span is straightforward.

If you skip this, the counterfactual body CE and the teacher KD fight each other.

## 6. Recommended validation before training

1. `--no-llm --dry-run` (done; invariants verified):
   each grounded record has a non-empty reference; `no_rag_qa`/`regular_chat`
   have none; `regular_chat` has `qa_turn_index = null`; no `weak` cf leaks.
2. Spot-check counterfactual strength: the fabricated answer must **not** be the
   real answer. Feed a sample of fabricated references to the *base* model and
   confirm it does not reproduce the fabricated fact (`memory-leak probe`).
3. After TTS/align, run `build_sidecard.py --dry-run` and confirm turn kinds per
   `rag_kind`.
4. After training, run the context-ablation eval (correct vs empty vs wrong ref).
