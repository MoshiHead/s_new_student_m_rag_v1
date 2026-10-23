# PersonaPlex 6.25 Hz Temporal Distillation

## 1. Goal

Distill PersonaPlex into a smaller streaming model suitable for eventual
real-time deployment on Jetson Orin.

The main change is:

- Keep Mimi unchanged at 12.5 Hz.
- Compress the 12.5 Hz temporal representation into 6.25 Hz.
- Run the expensive Temporal Transformer at 6.25 Hz.
- Distill the original PersonaPlex Temporal Transformer into a smaller
  student model.
- Keep audio generation streaming and compatible with the student's
  lower-rate temporal states.
- Ignore KV-cache compression for this stage.
- Ignore Jetson-specific optimization for this stage.
- The lip-sync model is independent and can be retrained later.


## 2. High-Level Architecture

### Teacher

Audio
  ↓
Mimi
  ↓
12.5 Hz audio tokens
  ↓
PersonaPlex Temporal Transformer (~7B)
  ↓
Temporal hidden states
  ↓
DepFormer
  ↓
Audio codebooks
  ↓
Mimi decoder
  ↓
Generated audio


### Student

Audio
  ↓
Mimi
  ↓
12.5 Hz audio tokens
  ↓
12.5 → 6.25 Hz temporal compressor
  ↓
6.25 Hz compact Temporal Transformer
  ↓
Student temporal states
  ↓
Audio generation / DepFormer
  ↓
Audio codebooks
  ↓
Mimi decoder
  ↓
Generated audio


The critical architectural change is:

    Teacher:
        Temporal Transformer @ 12.5 Hz

    Student:
        Temporal Transformer @ 6.25 Hz


Mimi remains at:

    12.5 Hz


## 3. Why Keep Mimi at 12.5 Hz?

Do not modify Mimi during the first experiment.

Mimi continues to process audio at its original temporal resolution:

    m0, m1, m2, m3, m4, ...

The student compressor consumes two consecutive Mimi frames:

    (m0, m1) → z0
    (m2, m3) → z1
    (m4, m5) → z2
    ...

Therefore:

    Mimi:
        12.5 frames/sec

    Student Temporal Transformer:
        6.25 frames/sec


One student temporal step represents approximately:

    2 × 80 ms = 160 ms


This allows the expensive Temporal Transformer to operate at half
the original temporal rate without changing the audio encoder.


## 4. Teacher Data Extraction

Use the original PersonaPlex model as a frozen teacher.

For each training conversation:

    Audio
      ↓
    Mimi
      ↓
    12.5 Hz representations
      ↓
    PersonaPlex Temporal Transformer
      ↓
    Temporal hidden states
      ↓
    DepFormer
      ↓
    Audio outputs


Save the required teacher information.

For every temporal position:

    Mimi representations:
        m0, m1, m2, ...


    Temporal Transformer hidden states:
        h0, h1, h2, ...


    Output logits / distributions:
        y0, y1, y2, ...


The teacher is completely frozen.


## 5. 12.5 → 6.25 Hz Temporal Compressor

Do not discard every second Mimi frame.

Instead, combine every two consecutive 12.5-Hz frames.

Initial implementation:

    [m(2t), m(2t+1)]
             ↓
       concatenate
             ↓
        Linear / MLP
             ↓
            zt


Mathematically:

    z_t = Compressor([m_2t ; m_(2t+1)])


where:

    z_t

is the 160-ms representation consumed by the student Temporal
Transformer.


The compressor must be causal.

For z_t, only use:

    m_2t
    m_(2t+1)

and never:

    m_(2t+2)
    m_(2t+3)
    ...


## 6. Student Temporal Transformer

Create a smaller Temporal Transformer.

Initial target:

    approximately 1B–3B parameters


Start with the same general architecture as PersonaPlex/Moshi.

Reduce model size primarily through:

    - fewer Transformer layers
    - smaller FFN dimension

Potential later reductions:

    - hidden dimension
    - attention heads


Do not aggressively redesign the Transformer in the first experiment.

The purpose of the first version is to isolate the effect of:

    12.5 Hz → 6.25 Hz


from other architectural changes.


## 7. Teacher Hidden-State Alignment

The teacher produces:

    h0, h1, h2, h3, h4, h5, ...


The student produces:

    s0, s1, s2, ...


Align them as:

    s0 ← {h0, h1}
    s1 ← {h2, h3}
    s2 ← {h4, h5}
    ...


Initial teacher target:

    H_t = (h_2t + h_(2t+1)) / 2


Therefore:

    H0 = (h0 + h1) / 2
    H1 = (h2 + h3) / 2
    H2 = (h4 + h5) / 2
    ...


If teacher and student hidden dimensions differ:

    Teacher hidden
          ↓
    learned projection
          ↓
    Student hidden dimension


## 8. Hidden-State Distillation

Train the student temporal representation to reproduce the compressed
teacher representation.

    L_hidden =
        MSE(StudentHidden_t, Projection(H_t))


The purpose is to preserve the information contained in the original
12.5-Hz Temporal Transformer while representing it at 6.25 Hz.


## 9. Output Distillation

The student should also learn the teacher's output behavior.

For aligned positions:

    Teacher logits
          ↓
       softmax(T)
          ↓
    Teacher distribution


    Student logits
          ↓
       softmax(T)
          ↓
    Student distribution


Use:

    L_KD = KL(P_teacher || P_student)


Use temperature-based knowledge distillation.


## 10. Normal Training Loss

Keep the original task/generation loss.

    L_task


Total initial loss:

    L =
        L_task
        + λ_hidden * L_hidden
        + λ_KD * L_KD


The student therefore learns both:

    1. The original task
    2. The internal representation of PersonaPlex
    3. PersonaPlex's output behavior


# 11. Audio Generation / DepFormer

This needs to be handled separately from the temporal-rate reduction.

PersonaPlex has two important components:

    Temporal Transformer
    DepFormer


The Temporal Transformer operates at the main temporal rate and produces
the high-level temporal state.

The DepFormer handles dependencies between the multiple Mimi audio
codebooks.

The goal of the first experiment should NOT be to completely redesign
DepFormer.


## 11.1 Initial Strategy: Keep DepFormer

Use the existing PersonaPlex/Moshi-style DepFormer as much as possible.

Student:

    Student Temporal Transformer
             │
             │ 6.25 Hz state
             ▼
         audio generation
             │
             ▼
          DepFormer
             │
             ▼
       Mimi codebooks
             │
             ▼
        Mimi decoder
             │
             ▼
            audio


However, the student is now producing one temporal state every 160 ms,
while the original audio system expects temporal updates at 80 ms.


Therefore a temporal-rate adapter is required between the student's
6.25-Hz representation and the audio-generation path.


# 12. 6.25 Hz → 12.5 Hz Audio-State Adapter

Use a small learned upsampler after the student Temporal Transformer.

Student:

    s0, s1, s2, ...


Upsampler:

    s0 → a0, a1
    s1 → a2, a3
    s2 → a4, a5
    ...


where:

    a_i

are 12.5-Hz audio-conditioning states.


Architecture:

    Student Temporal Transformer
              │
            6.25 Hz
              │
              ▼
       temporal upsampler
              │
            12.5 Hz
              │
              ▼
          audio head
              │
              ▼
          DepFormer
              │
              ▼
       Mimi audio codes


The simplest first implementation is:

    s_t
      ↓
    Linear
      ↓
    two 12.5-Hz states


For example:

    [s_t]
      ↓
    MLP
      ↓
    [a_2t, a_(2t+1)]


The upsampler can later be replaced with a causal temporal convolution,
GRU, attention-based module, or other learned interpolation mechanism.


# 13. Important Distinction: Semantic Rate vs Audio Rate

The student does NOT need to generate audio at 6.25 Hz.

Instead:

    Semantic / temporal reasoning:
        6.25 Hz


    Audio conditioning:
        12.5 Hz


    Mimi audio representation:
        12.5 Hz


Therefore the architecture becomes:

    Audio
      ↓
    Mimi
      │
      │ 12.5 Hz
      ▼
    Compressor
      │
      │ 6.25 Hz
      ▼
    Student Temporal Transformer
      │
      │ 6.25 Hz
      ▼
    Audio-State Upsampler
      │
      │ 12.5 Hz
      ▼
    DepFormer / audio generation
      │
      ▼
    Mimi decoder
      │
      ▼
    Audio


This preserves the original audio temporal resolution while reducing
the number of expensive Temporal Transformer steps.


# 14. DepFormer Training Strategy

Initially, keep DepFormer relatively close to the teacher.

There are two possible configurations.

### Option A — Frozen DepFormer

Start with:

    Student Temporal Transformer
            ↓
      6.25 → 12.5 adapter
            ↓
      frozen teacher DepFormer
            ↓
        audio codes


This is the cleanest first experiment because it isolates whether the
student can provide useful temporal conditioning.


### Option B — Trainable DepFormer

If the frozen DepFormer cannot adapt sufficiently to the student's
representation:

    Student Temporal Transformer
            ↓
      6.25 → 12.5 adapter
            ↓
       trainable DepFormer
            ↓
        audio codes


The DepFormer can then learn to interpret the student's new temporal
representation.


Recommended order:

    1. Try frozen DepFormer.
    2. If necessary, unfreeze/train DepFormer.
    3. Only later consider distilling/compressing DepFormer itself.


# 15. Audio Generation Loss

The final audio-generation objective should still train the student
toward the correct Mimi audio codebooks.

Conceptually:

    Student temporal state
            ↓
        DepFormer
            ↓
       predicted codes


Train against the teacher/ground-truth audio codes using the original
codebook prediction objective.

For multiple codebooks:

    L_audio =
        Σ_k L_codebook_k


where k indexes the Mimi codebooks.


The first implementation should retain the original codebook weighting
used by the PersonaPlex training setup wherever possible.


# 16. Recommended Initial Audio Architecture

Use:

    Mimi
      ↓
    12.5 Hz
      ↓
    2-frame compressor
      ↓
    6.25 Hz
      ↓
    Small Temporal Transformer
      ↓
    6.25 → 12.5 Hz learned upsampler
      ↓
    DepFormer
      ↓
    Mimi audio codebooks
      ↓
    Mimi decoder
      ↓
    Audio


Do not change Mimi.


Do not initially compress DepFormer.


Do not initially change the number of Mimi codebooks.


Do not initially change the Mimi decoder.


The main experiment should be the temporal-rate reduction.


# 17. Full Training Architecture

                    TEACHER
                    =======

Audio
  │
  ▼
Mimi
  │
  │ 12.5 Hz
  ▼
PersonaPlex Temporal Transformer
  │
  ├── hidden states
  │
  └── audio conditioning
          │
          ▼
      DepFormer
          │
          ▼
      Mimi codes
          │
          ▼
      Audio output


                    STUDENT
                    =======

Audio
  │
  ▼
Mimi
  │
  │ 12.5 Hz
  ▼
2-frame Compressor
  │
  │ 6.25 Hz
  ▼
Small Temporal Transformer
  │
  ├───────────────┐
  │               │
  │               ▼
  │        Hidden-state KD
  │
  ▼
6.25 → 12.5 Hz Audio-State Upsampler
  │
  │ 12.5 Hz
  ▼
DepFormer
  │
  ▼
Mimi audio codes
  │
  ▼
Mimi decoder
  │
  ▼
Audio


# 18. Training Loss

Initial total loss:

    L_total =
        L_task
        + λ_hidden L_hidden
        + λ_KD L_KD
        + λ_audio L_audio


Where:

    L_task
        = original conversational/generation objective

    L_hidden
        = compressed teacher hidden-state matching

    L_KD
        = teacher → student output distillation

    L_audio
        = Mimi codebook/audio generation loss


The exact weighting should initially be kept close to the existing
PersonaPlex training objective, adding the new distillation losses
incrementally.


# 19. Training Stages

## Stage 1 — Teacher Extraction

Freeze PersonaPlex.

Run the complete teacher pipeline:

    Audio → Mimi → Temporal Transformer → DepFormer → Audio


Store:

    Mimi representations
    Temporal hidden states
    Teacher logits/output distributions
    Audio/Mimi target codes


No student training yet.


## Stage 2 — 12.5 → 6.25 Compressor

Implement:

    2 × 12.5-Hz frames
             ↓
       small compressor
             ↓
         1 × 6.25-Hz frame


Keep it causal.


## Stage 3 — Student Temporal Transformer

Replace the 12.5-Hz teacher Temporal Transformer with a smaller
6.25-Hz student.

Initial target:

    1B–3B parameters


Keep the architecture structurally similar to the teacher.


## Stage 4 — Hidden-State Distillation

Train:

    Student state
        ≈
    pooled teacher state


Use:

    L_hidden


## Stage 5 — Output Distillation

Add:

    L_KD


Train the student to reproduce the teacher's output distribution.


## Stage 6 — Audio Generation

Connect:

    Student
      ↓
    6.25 → 12.5 upsampler
      ↓
    DepFormer
      ↓
    Mimi codebooks
      ↓
    Mimi decoder


Initially keep DepFormer frozen if possible.


## Stage 7 — Joint Training

Train:

    2-frame compressor
    Student Temporal Transformer
    6.25 → 12.5 audio-state upsampler


Optionally train:

    DepFormer


Keep:

    Mimi frozen


The teacher remains frozen throughout.


# 20. Initial Trainable / Frozen Components

### Frozen

    ✓ Teacher PersonaPlex
    ✓ Teacher Mimi
    ✓ Student Mimi
    ✓ Mimi decoder
    ✓ DepFormer initially


### Trainable

    ✓ 12.5 → 6.25 compressor
    ✓ Student Temporal Transformer
    ✓ Teacher → student hidden projection
    ✓ 6.25 → 12.5 audio-state upsampler


Potentially train later:

    ✓ DepFormer


# 21. What NOT to Do Yet

Do not implement:

    ✗ KV-cache compression
    ✗ Hierarchical memory
    ✗ Long-context compression
    ✗ 3.125 Hz
    ✗ INT4
    ✗ INT8
    ✗ TensorRT optimization
    ✗ Jetson-specific kernels
    ✗ Speculative decoding
    ✗ Mimi modification
    ✗ DepFormer compression
    ✗ Lip-sync-specific training
    ✗ Complex temporal memory mechanisms


First establish the basic:

    12.5 Hz Mimi
          ↓
    6.25 Hz Temporal Transformer
          ↓
    12.5 Hz audio conditioning
          ↓
    DepFormer
          ↓
    Mimi audio


# 22. Final Target Architecture

                         AUDIO
                           │
                           ▼
                     ┌───────────┐
                     │   Mimi    │
                     └─────┬─────┘
                           │
                        12.5 Hz
                           │
                           ▼
                  ┌─────────────────┐
                  │ 2-frame causal  │
                  │ compressor      │
                  └────────┬────────┘
                           │
                         6.25 Hz
                           │
                           ▼
                  ┌─────────────────┐
                  │ Compact         │
                  │ Temporal        │
                  │ Transformer     │
                  │ 1–3B            │
                  └────────┬────────┘
                           │
                         6.25 Hz
                           │
                 ┌─────────┴─────────┐
                 │                   │
                 ▼                   ▼
          Hidden/semantic       Audio-state
             state              upsampler
                                     │
                                   12.5 Hz
                                     │
                                     ▼
                               ┌───────────┐
                               │ DepFormer │
                               └─────┬─────┘
                                     │
                               Mimi codebooks
                                     │
                                     ▼
                               Mimi decoder
                                     │
                                     ▼
                                   AUDIO


Core principle:

    Keep the AUDIO representation at 12.5 Hz.

    Reduce only the expensive TEMPORAL REASONING rate:

        12.5 Hz → 6.25 Hz

    Then reconstruct 12.5-Hz conditioning states for the existing
    audio-generation pipeline.


# 23. First Implementation Goal

The first working model should therefore be:

    Mimi
      ↓
    [12.5 Hz]
      ↓
    [2-frame compressor]
      ↓
    [6.25 Hz]
      ↓
    [small Temporal Transformer]
      ↓
    [6.25 → 12.5 learned upsampler]
      ↓
    [existing DepFormer]
      ↓
    [Mimi codebooks]
      ↓
    [Mimi decoder]


Training:

    L_total =
        L_task
        + λ_hidden L_hidden
        + λ_KD L_KD
        + λ_audio L_audio


Teacher:

    Original PersonaPlex
    completely frozen


Student trainable components:

    Compressor
    Small Temporal Transformer
    Hidden projection
    Audio-state upsampler
    DepFormer only if necessary


Later optimization can independently address:

    model pruning
    quantization
    KV-cache compression
    memory optimization
    TensorRT
    Jetson Orin deployment