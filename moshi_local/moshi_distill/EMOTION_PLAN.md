# Emotion conditioning plan (speech model)

Goal: let the model **speak with a controllable emotion** (happy / sad / angry / …)
via an external module that picks the emotion and conditions audio generation.

Two things are needed: (a) **emotional TTS + emotion-labelled data**, and (b) a
**conditioning path** into the model. Both are described below, with a phased
plan that starts with a no-training test.

---

## 1. Emotion taxonomy

Start with 6 discrete labels (Ekman + neutral):

```
neutral, happy, sad, angry, surprised, fearful
```

Optionally add `calm`, `disgust`, `excited`, `tired`. Keep the set small and
balanced; each label needs ~equal data. Also store a continuous
`(valence, arousal)` per clip if the TTS supports it (useful for interpolation).

## 2. What the model can already condition on

* **Voice prompt** — `LMGen.load_voice_prompt_embeddings(path)` loads
  `{embeddings, cache}` (per-frame LM embeddings + KV cache) and replays them, so
  the model continues in that *voice and prosody*. **This is the cheapest emotion
  channel** (prosody is largely carried here).
* **System/text prompt** — the persona text (`text_prompt_tokens`), already used.
* **`<ref_slot>` + adapter** — the RAG machinery (`interleaver.splice_rag_frames`
  + `RefAdapter.inject`) writes continuous vectors into reserved frames. **Reuse
  this for an "emotion slot"** if we want a learned emotion embedding.

## 3. Emotional TTS (for labelled data)

Current Chatterbox is weak for discrete emotion (`exaggeration` only). Options,
best → simplest:

| TTS | emotion control | notes |
|---|---|---|
| **EmotiVoice** | explicit `happy/sad/angry/neutral/…` + multi-speaker | purpose-built; best for labelled data |
| **CosyVoice2** | natural-language instruction ("speak angrily") | strong, instruction-based |
| **XTTS-v2** | emotion via a reference clip (cloning) | easy if you have emotional refs |
| **Parler-TTS** | text description of voice/style | controllable descriptions |
| **StyleTTS2** | style vectors (needs a style encoder) | research-y |
| Chatterbox (installed) | `exaggeration` 0.1–1.0 + ref audio | keep as fallback / quick test |

Recommendation: **EmotiVoice** (or CosyVoice2) to synthesise the dataset;
keep Chatterbox for the fast iteration loop.

## 4. Data generation (the main lever)

**Disentangle emotion from content**: synthesise the **same text** under each
emotion, so the model learns emotion as a *style factor*, not a property of the
words.

1. Take a text set (existing rag_v3 `regular_chat` + short answers, plus generic
   neutral sentences) and synthesise each utterance in all 6 emotions with the
   chosen TTS → `(wav, text, emotion)`.
2. Keep one **speaker per voice** but vary emotion (and vice-versa) so both are
   controllable.
3. Align (Whisper) and build sidecars exactly like the existing pipeline
   (`generic.py` → `train/build_sidecard.py`), adding `emotion` to the metadata.
4. Balance: ~equal counts per emotion; ~500–1,500 utterances per emotion to
   start (emotion is a low-dimensional style, it needs less data than RAG).

## 5. Conditioning mechanism — pick one

### Option A — per-emotion voice prompts (no training; try first)
For each emotion, record/precompute a **voice-prompt** from an emotional
reference clip (`save_voice_prompt_embeddings`), then at inference select the
voice prompt matching the desired emotion. The model already continues in the
voice-prompt prosody.

* ✅ zero training, immediate test of "does emotion transfer via the voice prompt".
* ❌ limited precision; emotion is entangled with the specific reference voice.

### Option B — learned emotion adapter + emotion slot (recommended for control)
Mirror the RAG design:

* Add an **emotion slot**: a few reserved frames (reuse `<ref_slot>` style) or a
  prefix to the voice-prompt embeddings.
* **Emotion adapter** = tiny MLP/embedding table: `emotion_id → K vectors`
  (like `RefAdapter`, ~1–2 M params). Inject into the slot frames (or add to the
  voice-prompt embedding).
* Train the adapter (+ LoRA on the backbone) on the emotion-labelled data with
  the existing weighted-CE / KD losses; the emotion condition is the only
  difference between the neutral and emotional renders of the same text, so the
  adapter learns to move prosody.

* ✅ precise, interpolatable (valence/arousal), separable from speaker.
* ❌ needs training + data.

### Option C — emotion text tag (cheapest to wire)
Prepend `<emotion:happy>` (a new special token, like `<lookup>`) to the system
prompt; train the model to follow it.

* ✅ no new inference module; works through the text channel.
* ❌ competes in the text stream; less direct control than a dedicated embedding.

**Recommendation: A → then B.** Do A to confirm emotion transfers at all, then B
for a controllable, speaker-independent emotion knob.

## 6. Training (Option B)

* Backbone: the distilled student (`ckpt/backbone_S20_lora`), frozen + **LoRA**
  (as already supported) — emotion is a small style change, do **not** full-FT.
* Trainable: **emotion adapter** + LoRA (attn/FFN) + (stage 2) emb/heads.
* Loss: task CE (text/audio) + teacher KD (the teacher render is the emotional
  audio; mask KD where the emotion condition is active if the teacher lacks it) +
  a **controllability loss**: body log-prob gap between the correct emotion
  condition and a wrong emotion (contrastive, exactly like the RAG reliance loss).
* Curriculum: neutral first, then all emotions; balance batches by emotion.

## 7. Inference (external emotion module)

```
user turn / planned response
        │
   emotion module  ──► emotion id (or valence/arousal)
        │                ├─ Option A: select emotional voice prompt
        │                └─ Option B: emotion adapter → K vectors → emotion slot
        ▼
   StudentLMGen / LMGen step  → emotional speech
```

The "external module" can be:
* a **UI control** (user picks happy/sad/angry),
* a **text sentiment/emotion classifier** on the user's message,
* an **LLM tag**: the same LLM that writes the reply also emits an emotion tag,
* optionally a **prosody/emotion classifier on the user's audio** (react to the
  user's mood).

Wire it in `server_student.py` next to the existing voice/text-prompt handling:
resolve an `emotion` query param → load the matching voice prompt (A) or set the
emotion slot (B), before `step_system_prompts_async`.

## 8. Evaluation

* **Emotion classifier (SER)** on the generated audio (e.g. an off-the-shelf
  wav2vec2 emotion model): accuracy = does `happy` condition → `happy` output.
* **Confusion matrix** across the 6 emotions.
* **Controllability gap**: same text, 6 conditions → measure classifier/embedding
  separation (analogue of the RAG context-ablation gap).
* **Speaker/voice preservation**: speaker-similarity vs the neutral render
  (emotion should not destroy identity).
* **Human MOS** on naturalness per emotion.
* **Latency**: the emotion adapter is one MLP forward per turn — negligible.

## 9. Phased plan

| Phase | Work | Cost |
|---|---|---|
| 0 | Pick TTS + emotion set; synthesise ~300 clips (6 emotions × short texts) | hours |
| 1 | **Option A**: precompute per-emotion voice prompts; test transfer on the current student | no training |
| 2 | If transfer is weak: build the emotion adapter + slot; **Option B** | LoRA train |
| 3 | Synthesise the full balanced emotion dataset (~1k/emotion) | TTS hours |
| 4 | Train adapter+LoRA with the contrastive controllability loss | ~hours |
| 5 | Wire the emotion module into `server_student.py`; evaluate (SER, MOS, latency) | hours |

## 11. Phase 1 results (EmotiVoice + voice-prompt transfer test)

**Done on the pod (Oct 13):**
* Installed **EmotiVoice** (`/workspace/EmotiVoice`, models 5.0 G + simbert 754 M) in an isolated
  `/workspace/emotivenv` (torch cu121 + transformers 4.40.2); generator
  `emotivoice_gen.py` produces per-utterance emotion/speaker wavs.
* Generated a Phase-0 set (`/workspace/emotion_data`, 30 clips: 3 texts × 5 emotions × 2 speakers).
* **Transfer test**: used each emotional clip as the student's **voice prompt**,
  generated 8 s (`/workspace/emotion_transfer/*.wav`), and evaluated.

**Findings:**
1. **EmotiVoice's text style-prompt emotion is weak.** Measured prosody of the
   references is near-identical across intended emotions (f0_mean ≈ 235–253 Hz,
   f0_std ≈ 25–53, rms ≈ 0.09, dur ≈ 2.1–2.2 s) — even with rich prompts
   ("A very sad, slow, low-pitched, subdued voice."). The style embedding has
   little effect (content embedding likely dominates).
2. **SER evaluators are unreliable here.**
   * `superb/wav2vec2-base-superb-er` labels *every* EmotiVoice clip "angry"
     (sad→ang 0.90, neutral→ang 0.81) — unusable on these synthetic voices.
   * `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` fails to load its
     classifier/projector weights under transformers 4.40.2 (all outputs ≈ 0).
3. Student outputs vary in prosody (f0_mean 184–240, rms 0.036–0.076) but **not
   systematically by the voice-prompt emotion** — consistent with the references
   not being emotional.

**Conclusion:** the Phase-1 transfer test is **inconclusive**, blocked by
(a) weak emotional references and (b) no trustworthy SER. It is *not* evidence
that the model can't do emotion.

**Next actions (revised):**
1. **Use real emotional speech** for references (RAVDESS / CREMA-D / ESD — labels
   happy/sad/angry/neutral, multiple speakers). Guaranteed distinct emotions, no
   TTS dependency — redo the transfer test with these.
2. **Fix / replace the TTS for the dataset**: debug EmotiVoice's style conditioning
   (or use CosyVoice2 instruction-based) — EmotiVoice as-called is too flat.
3. **Fix the evaluator**: get a working SER (pin the transformers version the
   audeering model expects, or `3loi/SER-Odyssey-Baseline-WavLM`, or human MOS).

The rest of the plan (§5 Option B adapter, §6 training, §7 inference, §8 eval) is
unchanged; the voice-prompt/adapter channel still needs emotion that is actually
present in the data.
