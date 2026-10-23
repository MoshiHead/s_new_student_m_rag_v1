# RAG + Distillation data plan (v3)

Status: **audio generation running** (6 workers, 1× A100).
* transcripts: `/workspace/data/rag_v3_transcripts.jsonl` — **14,000** records
* audio plan: `/workspace/data/rag_v3_audio_plan.jsonl`
* prepare done: `/workspace/data/rag_v3/audio/` — 8,000 reuse symlinks + metadata,
  4,500 splice jobs, 1,500 TTS jobs
* remaining: splice + TTS → Whisper align → sidecar

---

## 1. How the existing data/audio was created

```
build_dataset.py / build_dataset_fin.py   (HotpotQA/SQuAD, Natural Questions, FinanceQA)
    real query -> LLM pass1 spoken rewrite -> pass2 reference extraction
    -> pass3a lead+filler -> pass3b grounded body
extended_new.py       + persona + system_prompt + conversation + qa_turn_index
extend_dataset.py     ChatterboxTTS + pydub -> stereo WAV (Left=assistant, Right=user)
generic.py            faster-whisper word timestamps -> alignments
train/build_sidecard.py -> <id>.sidecar.json + train.jsonl
```

Measured: **91,160** records with WAV+sidecar (≈89k aligned, 72,521 de-badged);
reference is **metadata only — never spoken**. Source audio is **24 kHz stereo**.
`extend_dataset.py` had a latent `mono_to_channel` off-by-one bug — **fixed**.

## 2. Final volume (14,000)

| `rag_kind` | count | audio method |
|---|---|---|
| `grounded_cf` | 3,000 | **splice** body-only TTS |
| `grounded_distract` | 1,000 | **splice** |
| `abstain` | 500 | **splice** |
| `regular_chat` | 1,500 | full TTS (new conversation) |
| `grounded_real` | 5,000 | reuse source WAV |
| `no_rag_qa` | 3,000 | reuse source WAV |
| **total** | **14,000** | 4,500 splice / 1,500 TTS / 8,000 reuse |

## 3. Audio strategy (two optimizations)

**Optimization 1 — body splice.** For the grounded classes the only *spoken*
text that changes is the body sentence (greeting / small-talk / query /
lead / filler are identical, and the reference is metadata). So instead of
re-TTS'ing the whole conversation we Chatterbox-synthesize **only the body** and
replace `[body_start, body_end]` (from the source sidecar's `segment_meta`) in
the source WAV, keeping everything else including any tail. Same voice, short
crossfades, cut inside the sentence gap → no clicks. Then Whisper re-aligns
(needed anyway because the tail shifts). Measured **~14 s/record vs ~90 s** full.

**Optimization 2 — trimmed TTS volume.** Only `regular_chat` needs
full-conversation synthesis, reduced to 1,500.

Reuse stays at 8,000: the rebuilt conversation is byte-identical to the source,
so the source WAV + alignments are carried over (verified 8,000/8,000 exact).

Pipeline (idempotent / resumable):

```bash
PY=/workspace/menv/bin/python
$PY /workspace/make_rag_v3_audio.py --stage prepare          # done
bash /workspace/run_tts_v3.sh 6                              # RUNNING (splice then tts)
$PY /workspace/make_rag_v3_audio.py --stage align --workers-per-gpu 6
$PY /workspace/make_rag_v3_audio.py --stage sidecar
```

`run_tts_v3.sh` shards both inputs across workers and forces `HF_HUB_OFFLINE=1`
(weights cached) — without it concurrent workers stall on unauthenticated HF Hub
limits. Chatterbox lives in an isolated venv `/workspace/ttsenv` so it cannot
perturb the training env. Everything is at **24 kHz** to match the source.

Estimated wall time on 1× A100 (measured ~0.5 rec/s splice, ~0.09 rec/s TTS):
splice ≈ 2.5 h + TTS ≈ 4.6 h + align ≈ 1–2 h ≈ **8–9 h** (well inside 12–15 h).

## 4. Joint distillation + RAG training (one model, one run)

| loss | applies to | note |
|---|---|---|
| audio CE | all | unchanged |
| text CE (generated) | all | ref-slot frames weight 0 |
| teacher KD | `regular_chat`, `no_rag_qa`, grounded **lead/filler** | full |
| teacher KD on **body** | **masked for `grounded_*`/`abstain`** | teacher has no context → its logits contradict the counterfactual |
| `RefAdapter` injection | grounded (`body` active) | prefill-only; K vectors into `<ref_slot>` frames |
| reliance contrastive | grounded | correct-vs-shuffled ref body log-prob margin |
| trigger BCE (`rag_head`) | all | +1 at `<lookup>` only on grounded turns |

Batch mix via weighted sampler / `interleave_iterators`. 8k reused + 1.5k chat
+ 3k no-RAG = full-KD conversation; only ~4.5k counterfactual/abstain bodies use
task-CE-only. Adapter wiring already implemented (`ref_adapter.py`,
`interleaver.splice_rag_frames`, distill `use_rag=True`).

## 5. Artifacts

| path | what |
|---|---|
| `/workspace/data/rag_v3_transcripts.jsonl` | 14,000 transcripts (`rag_kind`) |
| `/workspace/data/rag_v3_audio_plan.jsonl` | per-record action: reuse/splice/tts |
| `/workspace/data/rag_v3/audio/` | WAVs + `.new.combined.json` (+ `.sidecar.json` after align) |
| `/workspace/build_rag_v3.py` | builder |
| `/workspace/regular_convo.py` | persona small-talk generator |
| `/workspace/make_rag_v3_audio.py` | audio driver (prepare/splice/tts/align/sidecar) |
| `/workspace/splice_body_tts.py` | body-only TTS + splice |
| `/workspace/run_tts_v3.sh` | parallel launcher |
| `/workspace/extend_dataset.py.bak_rag` | pre-fix backup |

## 6. Monitor / next

```bash
tail -f /workspace/data/rag_v3/audio/tts_logs/splice_0.log /workspace/data/rag_v3/audio/tts_logs/tts_0.log
find /workspace/data/rag_v3/audio -type f -name '*.wav' | wc -l    # target 14,000
```
Then `--stage align` → `--stage sidecar`, and the model-side trainer injection.
