"""
emotion_module.py — discrete emotion conditioning for the DepFormer
===================================================================

Design (EMOTION_PLAN.md §5 Option B, tightened for "train only the DepFormer"):

The temporal transformer is **frozen**, so the emotion cannot be injected by
changing the text stream alone.  Instead the emotion conditions the DepFormer
*directly*: an embedding table maps

    emotion_id -> depformer_dim

and that vector is added to **every** DepFormer input frame — all T frames x
all `dep_q` codebook slots — right before the DepFormer stack runs:

    depformer_input[b, t, cb, :] += emo_vec[b, :]

Why this works, and why the contrastive term matters
----------------------------------------------------
The frozen temporal state at frame `t` already carries the prosody of the
*previous* frames (teacher forcing feeds the real target codes back in).  So
the emotion vector is most informative at the **start of the agent turn**,
where the context only holds the user turn + system prompt.  The plain audio CE
therefore *can* be satisfied by copying the prosody it sees in the context and
ignoring the emotion vector.  The optional contrastive term removes that
shortcut: it requires the true emotion to explain the target codebooks better
than a permuted one.

Serving
-------
`EmotionAdapter` + the trained DepFormer are saved by `train_emotion.py` as
`student.pt` (unchanged format, loadable by server_student.py) + `emotion.pt`.
At inference, resolve an `emotion` query param -> id, run the adapter once, and
add the vector inside the DepFormer step (one add per frame; no extra module).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn


class EmotionAdapter(nn.Module):
    """emotion_id -> [B, n_codebooks, depformer_dim] conditioning vectors.

    **One vector per codebook**, which is the point: each codebook controls a
    different aspect of the audio (0 = coarse semantic/prosody, 1..7 = finer and
    finer acoustic detail such as timbre and voice quality).  A single shared
    vector forced every codebook to move together — either the emotion barely
    showed (restricted to codebook 0) or the refinement codebooks were scrambled
    (all slots).  Per-codebook vectors let the model learn *which* aspects to
    move, and how much.

    `cb_gate` starts at [1, 0, 0, ...]: at step 0 only codebook 0 is modulated,
    i.e. the known-coherent regime, and training opens the other codebooks only
    if that genuinely helps (the reconstruction loss is what decides).
    """

    def __init__(self, n_emotions: int, dim: int, n_codebooks: int = 16,
                 hidden: int = 256, out_scale_init: float = 1.0,
                 init: str = "small", target_rms: float = 0.0):
        super().__init__()
        self.n_emotions = int(n_emotions)
        self.dim = int(dim)
        self.n_codebooks = int(n_codebooks)
        self.target_rms = float(target_rms)
        self.emb = nn.Embedding(self.n_emotions, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.fc = nn.Linear(hidden, self.n_codebooks * dim)
        self.out_scale = nn.Parameter(torch.tensor(float(out_scale_init)))
        gate = torch.zeros(self.n_codebooks)
        gate[0] = 1.0
        self.cb_gate = nn.Parameter(gate)

        nn.init.normal_(self.emb.weight, std=0.02)
        if init == "zero":
            nn.init.zeros_(self.fc.weight)
        else:
            nn.init.normal_(self.fc.weight, std=0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, emotion_ids: torch.Tensor) -> torch.Tensor:
        """[B] long -> [B, n_codebooks, dim]."""
        x = self.emb(emotion_ids)
        x = self.fc(self.norm(x)).view(-1, self.n_codebooks, self.dim)
        if self.target_rms > 0.0:
            # FIXED magnitude.  Measured, the adapter was only ~10% of the
            # DepFormer input scale, i.e. ~4% of the logit scale -- far too weak
            # to be heard, and the audio CE happily shrank `out_scale` further
            # (it fell to 0.54 in the v3 run) so the emotion could be ignored.
            # Renormalising removes that escape hatch: the model must adapt to
            # the emotion rather than turn it off.
            r = x.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
            x = x / r * self.target_rms
            return x * self.cb_gate.view(1, -1, 1)
        return x * self.out_scale * self.cb_gate.view(1, -1, 1)

    # -- persistence --------------------------------------------------------

    def save(self, path: str | Path, emotion_order: list[str]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.state_dict(),
            "n_emotions": self.n_emotions,
            "dim": self.dim,
            "n_codebooks": self.n_codebooks,
            "target_rms": self.target_rms,
            "emotion_order": list(emotion_order),
        }, path)

    @classmethod
    def load(cls, path: str | Path, map_location="cpu") -> "EmotionAdapter":
        pkg = torch.load(path, map_location=map_location, weights_only=False)
        m = cls(pkg["n_emotions"], pkg["dim"], pkg.get("n_codebooks", 16),
                target_rms=pkg.get("target_rms", 0.0))
        m.load_state_dict(pkg["state_dict"])
        m.eval()
        return m


# ---------------------------------------------------------------------------
# DepFormer-side plumbing (shared by the trainer and the serving hook)
# ---------------------------------------------------------------------------

def depformer_inputs(model, seq: torch.Tensor, transformer_out: torch.Tensor
                     ) -> torch.Tensor:
    """Build the [B, T, Ka, depformer_dim] DepFormer input exactly as
    ``LMModel.forward_depformer_training`` does, WITHOUT running the stack."""
    B, _, T = seq.shape
    Ka = model.dep_q
    ins = []
    for cb in range(Ka):
        if model.depformer_multi_linear:
            li = cb
            if model.depformer_weights_per_step_schedule is not None:
                li = model.depformer_weights_per_step_schedule[cb]
            lin = model.depformer_in[li]
        else:
            lin = model.depformer_in[0]
        # the frozen backbone may be kept in bf16 while the DepFormer trains in
        # fp32, so cast the state to the projection's dtype here
        tr = lin(transformer_out.to(lin.weight.dtype))
        if cb == 0:
            tok = model.depformer_text_emb(seq[:, 0])
        else:
            tok = model.depformer_emb[cb - 1](seq[:, cb + model.audio_offset - 1])
        ins.append(tok + tr)
    return torch.stack(ins, 2)                       # [B, T, Ka, D]


def depformer_logits(model, seq: torch.Tensor, transformer_out: torch.Tensor,
                     emo_vec: torch.Tensor | None = None,
                     emo_slots: int | None = None) -> torch.Tensor:
    """DepFormer audio logits [B, Ka, T, card] (NOT yet undelayed).

    ``emo_vec`` is [B, Ka, depformer_dim] — a **separate vector per codebook**,
    broadcast over frames and added to that codebook's DepFormer input.  A
    legacy [B, depformer_dim] is accepted and expanded to all codebooks.
    ``emo_vec=None`` is bit-identical to the frozen base model.

    ``emo_slots`` optionally masks the injection to the first N codebooks (the
    adapter's own `cb_gate` normally handles this, learnably).
    """
    B, _, T = seq.shape
    Ka = model.dep_q
    dep = depformer_inputs(model, seq, transformer_out)          # [B, T, Ka, D]
    if emo_vec is not None:
        v = emo_vec.to(dep.dtype)
        if v.dim() == 2:                       # legacy [B, D]
            v = v[:, None, :].expand(-1, Ka, -1)
        v = v[:, None, :, :]                   # [B, 1, Ka, D]
        if emo_slots is not None and emo_slots < Ka:
            m = torch.zeros(1, 1, Ka, 1, dtype=dep.dtype, device=dep.device)
            m[:, :, :emo_slots] = 1.0
            v = v * m
        dep = dep + v
    dep = dep.reshape(B * T, Ka, -1)
    out = model.depformer(dep)
    logs = [model.linears[cb](out[:, cb]).view(B, T, -1) for cb in range(Ka)]
    return torch.stack(logs, 1)


# ---------------------------------------------------------------------------
# Serving: apply an emotion during streaming generation
# ---------------------------------------------------------------------------

def set_emotion(model, adapter, emotion_id: int, device="cuda") -> None:
    """Write the emotion vector into the DepFormer input projections.

    The training-time injection adds `emo_vec` to
    `depformer_in(transformer_out) + token_emb(...)`.  `nn.Linear` computes
    `x @ W.T + b`, so storing `emo_vec` as the bias of every `depformer_in`
    linear reproduces that sum EXACTLY — with zero changes to the generation
    loop.  The write is in-place (`copy_`) so a captured CUDA graph picks up the
    new emotion on the next frame without re-capturing.

    Call once before/while serving; `emotion_id` indexes the trained
    `emotion_order` (see `EmotionAdapter.save`).
    """
    import torch.nn as nn
    with torch.no_grad():
        vec = adapter(torch.tensor([int(emotion_id)], device=device))[0]  # [Ka, D]
    for i, lin in enumerate(model.depformer_in):
        v = vec[i] if i < vec.shape[0] else vec[0]
        if getattr(lin, "bias", None) is None:
            lin.bias = nn.Parameter(torch.zeros_like(v))
        lin.bias.copy_(v)


def clear_emotion(model) -> None:
    """Zero the DepFormer-input bias written by `set_emotion`.

    Must be called after any generation probe: otherwise the last emotion's
    vector stays baked into `depformer_in.bias` and every subsequent TRAINING
    step would add it on top of the per-sample `emo_vec` (double conditioning,
    plus a constant offset for all emotions).
    """
    for lin in model.depformer_in:
        if getattr(lin, "bias", None) is not None:
            with torch.no_grad():
                lin.bias.zero_()
