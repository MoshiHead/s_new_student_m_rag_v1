"""
emotion_probe.py — objective probes answering "is the emotion training doing
anything at all?"
=============================================================================

Two complementary probes, both run from inside the training loop:

1. `emotion_discrimination` (cheap, every eval)
   Runs the DepFormer under EVERY candidate emotion and scores the true
   codebooks.  Produces:
     * `emotion_acc`   — fraction of held-out samples where the TRUE emotion
                         gets the LOWEST audio CE (chance = 1/n_emotions).
     * `ce_gap`        — mean CE(wrong emotions) - CE(true emotion).
   These are the discriminative form of `emotion_reliance`: if the adapter is
   doing nothing, `emotion_acc` sits at chance and `ce_gap` at 0.

2. `run_generation_probe` (every `--gen-every` steps)
   Actually GENERATES speech for each emotion from the SAME prompt with the
   SAME seed, decodes with Mimi, saves the wavs, and reports pitch/energy per
   emotion.  This is the only probe that proves the emotion is audible, and it
   is the one you can listen to.
"""

from __future__ import annotations

import contextlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .emotion_module import depformer_logits, set_emotion, clear_emotion
from .models.lm import LMGen, SILENCE_TOKENS


# ---------------------------------------------------------------------------
# Per-sample weighted audio CE (so we can build a B x E counterfactual matrix)
# ---------------------------------------------------------------------------

def audio_ce_per_sample(logits: torch.Tensor, codes: torch.Tensor,
                        mask: torch.Tensor,
                        codebook_weights: torch.Tensor | None = None,
                        first_codebook_mult: float = 1.0) -> torch.Tensor:
    """Weighted per-sample CE -> [B] (same weighting as `train_emotion.audio_ce`)."""
    target = torch.where(mask, codes, torch.zeros_like(codes))
    w = mask.float()
    if codebook_weights is not None:
        w = w * codebook_weights.to(w.device).view(1, -1, 1)
    if first_codebook_mult != 1.0:
        w = w.clone()
        w[:, 0] *= first_codebook_mult
    B = logits.shape[0]
    lf = torch.nan_to_num(logits.reshape(-1, logits.size(-1)).float(),
                          nan=0.0, posinf=0.0, neginf=0.0)
    per = F.cross_entropy(lf, target.reshape(-1), reduction="none").view(B, -1)
    wf = w.reshape(B, -1)
    return (per * wf).sum(-1) / wf.sum(-1).clamp(min=1e-8)


@torch.no_grad()
def emotion_discrimination(codes, t_out, emo_ids, model, adapter, args,
                           cb_weights, n_emotions):
    """Score the true codebooks under every candidate emotion.

    Returns (n_correct, n_total, ce_true, ce_wrong, matrix[B,E]).
    """
    from .train_emotion import depformer_input_seq, undelay_audio
    Ka = model.dep_q
    audio_codes = codes[:, model.audio_offset:model.audio_offset + Ka]
    seq = depformer_input_seq(model, codes)
    B = codes.shape[0]
    ces = []
    for e in range(n_emotions):
        ids = torch.full((B,), e, dtype=torch.long, device=codes.device)
        lg = depformer_logits(model, seq, t_out, adapter(ids),
                              emo_slots=(getattr(args, "emo_slots", 0) or None))
        lg, mk = undelay_audio(model, lg, codes)
        ces.append(audio_ce_per_sample(lg, audio_codes, mk, cb_weights,
                                       args.first_codebook_mult))
    ce = torch.stack(ces, dim=1)                                  # [B, E]
    diag = ce.gather(1, emo_ids.view(-1, 1)).squeeze(1)            # [B]
    n_correct = int((ce.argmin(dim=1) == emo_ids).sum().item())
    ce_wrong = (ce.sum(dim=1) - diag) / max(1, n_emotions - 1)
    return n_correct, B, float(diag.mean()), float(ce_wrong.mean()), ce


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

# NOTE: prosody scoring (f0/rms) was removed on purpose — the wavs are for you
# to listen to. `torchaudio.functional.detect_pitch_frequency` in particular is
# pathologically slow here (CPU NCCF over the whole waveform) and made the probe
# look hung.

class _NoGraph:
    """Passthrough stand-in for CUDAGraphed (probe runs ungraphed)."""

    def __init__(self, fn, disable=False):
        self.fn = fn

    def __call__(self, *a, **k):
        return self.fn(*a, **k)


@contextmanager
def _no_cuda_graphs():
    from .models import lm as lm_mod
    orig = lm_mod.CUDAGraphed
    lm_mod.CUDAGraphed = _NoGraph
    try:
        yield
    finally:
        lm_mod.CUDAGraphed = orig


@torch.no_grad()
def generate_grid(model, adapter, mimi, user_codes, sys_prompt_tokens,
                  emotions, n_greet, n_ans, voice_path=None, device="cuda",
                  seed=0, emo_slots=None):
    """One CONVERSATIONAL generation per row, with the emotion on the agent.

        system prompt -> agent GREETS (user silent)
                      -> user ASKS   (the sample's real query)
                      -> agent ANSWERS

    The greeting is the agent's first turn, exactly as in the training data, so
    the query lands where it is supposed to and the agent answers instead of
    greeting.  Returns {name: {"agent": wav, "mix": wav}} where `mix` is the
    agent+user conversation, i.e. something you can actually listen to.
    """
    model.eval()
    E = len(emotions)
    names = ["base"] + list(emotions)
    # Row 0 carries a ZERO emotion vector = the untouched base model.  That is
    # the control: if `base` is coherent and the emotion rows are not, the
    # problem is the conditioning; if `base` is also bad, it is the model.
    emo_mat = torch.zeros(len(names), adapter.n_codebooks, adapter.dim,
                          device=device)
    if E:
        emo_mat[1:] = adapter(torch.arange(E, device=device))

    # Per-row emotion injection, exactly where training puts it: on the
    # DepFormer input, after the transformer_out projection and the token emb.
    orig_forward_depformer = model.forward_depformer

    def _patched_forward_depformer(cb_index, sequence, transformer_out):
        lm = model
        lin = (lm.depformer_in[cb_index] if lm.depformer_multi_linear
               else lm.depformer_in[0])
        dep = lin(transformer_out.to(lin.weight.dtype))
        tok = (lm.depformer_text_emb(sequence[:, 0]) if cb_index == 0
               else lm.depformer_emb[cb_index - 1](sequence[:, 0]))
        dep = dep + tok
        if model._emo_vec is not None:
            nslots = getattr(model, "_emo_slots", None)
            if nslots is None or cb_index < nslots:
                v = model._emo_vec
                if v.dim() == 3:                      # [B, Ka, D] per-codebook
                    if cb_index < v.shape[1]:
                        dep = dep + v[:, cb_index][:, None, :].to(dep.dtype)
                else:                                 # legacy [B, D]
                    dep = dep + v[:, None, :].to(dep.dtype)
        logits = lm.linears[cb_index](lm.depformer(dep))
        return logits[:, None]

    model.forward_depformer = _patched_forward_depformer
    model._emo_slots = emo_slots
    silence = torch.as_tensor(SILENCE_TOKENS, dtype=torch.long,
                              device=model.device).view(1, 8, 1)
    out = {}
    n_prompt = len(sys_prompt_tokens)
    # the cached codes ALREADY contain the system-prompt frames at the front
    uc = user_codes.long().to(model.device)[:, :, n_prompt:]
    n_q = uc.shape[-1]
    spf = int(mimi.sample_rate / mimi.frame_rate)
    try:
        with _no_cuda_graphs():
            lm_gen = LMGen(model, model.device, use_sampling=True, temp=0.8,
                           temp_text=0.7, top_k=250, top_k_text=25,
                           sample_rate=mimi.sample_rate,
                           frame_rate=mimi.frame_rate)
            lm_gen.text_prompt_tokens = sys_prompt_tokens
            if voice_path and os.path.exists(voice_path):
                try:
                    lm_gen.load_voice_prompt_embeddings(voice_path)
                except Exception:
                    pass
            q_wav = mimi.decode(uc.clamp(0, model.card - 1))[0, 0].float().cpu().numpy()
            for row, name in enumerate(names):
                model._emo_vec = emo_mat[row:row + 1]     # one row = batch 1
                torch.manual_seed(seed)                   # identical sampling noise
                # BATCH 1 ON PURPOSE: the voice-prompt replay and the silence
                # frames are written for a single stream.
                with lm_gen.streaming(1):
                    lm_gen.step_system_prompts(mimi)
                    chunks = []
                    for phase, count in (("greet", n_greet), ("query", n_q),
                                         ("answer", n_ans)):
                        for t in range(count):
                            ui = (uc[:, :, t:t + 1] if phase == "query"
                                  else silence)
                            tok = lm_gen.step(input_tokens=ui)
                            if tok is None:
                                break
                            chunks.append(tok[:, 1:9])
                if not chunks:
                    out[name] = {"agent": np.zeros(1, np.float32),
                                 "mix": np.zeros(1, np.float32)}
                    continue
                codes = torch.cat(chunks, dim=-1).clamp(0, model.card - 1)
                agent = mimi.decode(codes)[0, 0].float().cpu().numpy()
                mix = agent.copy()
                s = int(n_greet * spf)                    # query sits after the greeting
                e = min(len(mix), s + len(q_wav))
                if e > s:
                    mix[s:e] += q_wav[:e - s]
                out[name] = {"agent": agent, "mix": mix}
    finally:
        model.forward_depformer = orig_forward_depformer
        model._emo_vec = None
        torch.cuda.empty_cache()
        model.train()
    for name in names:
        out.setdefault(name, {"agent": np.zeros(1, np.float32),
                              "mix": np.zeros(1, np.float32)})
    return out


_ASR = None


def _asr(wav, sr):
    """Round-trip transcription (loaded once, reused)."""
    global _ASR
    if _ASR is None:
        from faster_whisper import WhisperModel
        _ASR = WhisperModel("medium", device="cuda", compute_type="float16")
    import numpy as _np
    w = _np.asarray(wav, dtype=_np.float32).reshape(-1)
    segs, _ = _ASR.transcribe(w, language="en", beam_size=5, vad_filter=True)
    return " ".join(s.text.strip() for s in segs).strip() or "<EMPTY>"


def _agent_start_frame(target_codes) -> int:
    """First frame of the agent's turn in the target codes.

    The agent channel is silence while the user talks (the assembled audio puts
    the user prompt first), so the first non-silence codebook-0 token marks
    where the agent starts.  Returns 0 if it cannot be determined, in which case
    callers fall back to using the whole waveform.
    """
    if target_codes is None:
        return 0
    cb0 = target_codes[0].tolist() if target_codes.dim() == 2 else target_codes[0][0].tolist()
    sil = int(SILENCE_TOKENS[0])
    for i, v in enumerate(cb0):
        if v == -1:
            break
        if v != sil:
            return i
    return 0


def run_generation_probe(model, adapter, mimi, tokenizer, user_codes,
                         system_prompt, emotions, out_dir: Path, step: int,
                         n_greet: int = 40, n_ans: int = 100,
                         voice_path: str | None = None, device="cuda",
                         emo_slots=None, target_codes=None, do_asr: bool = True):
    """Generate a CONVERSATION from an EXACT training example, one row per emotion.

    The agent greets (its first turn, as in the training data), the user asks
    that sample's real query, then the agent answers -- so the query lands where
    it belongs and the model answers instead of greeting.  Saves, per step:
      user_turn.wav / target.wav / target_convo.wav   (the real example)
      <emotion>.wav          the agent channel only   (emotion comparison)
      <emotion>_convo.wav    agent+user mixed         (listen to this)
      preview_all.wav        all rows' answers back to back
      transcripts.txt        TARGET vs each emotion's ASR
    """
    import soundfile as sf

    sys_tokens = tokenizer.encode(f"<system> {system_prompt} <system>")
    grid = generate_grid(model, adapter, mimi, user_codes, sys_tokens,
                         emotions, n_greet, n_ans, voice_path=voice_path,
                         device=device, emo_slots=emo_slots)
    d = Path(out_dir) / f"gen_step_{step}"
    d.mkdir(parents=True, exist_ok=True)

    spf = int(mimi.sample_rate / mimi.frame_rate)
    n_prompt = len(sys_tokens)
    uc = user_codes.long().to(model.device)[:, :, n_prompt:].clamp(0, model.card - 1)

    # the exact user query and the ground-truth agent answer, for reference
    with torch.no_grad():
        q_wav = mimi.decode(uc)[0, 0].float().cpu().numpy()
    sf.write(str(d / "user_turn.wav"), q_wav, mimi.sample_rate, subtype="PCM_16")
    tgt_txt = ""
    if target_codes is not None:
        tc = target_codes.long().to(model.device).clamp(0, model.card - 1)
        if tc.dim() == 2:
            tc = tc[None]
        with torch.no_grad():
            tw = mimi.decode(tc)[0, 0].float().cpu().numpy()
        sf.write(str(d / "target.wav"), tw, mimi.sample_rate, subtype="PCM_16")
        tmix = tw.copy()
        e = min(len(tmix), len(q_wav))
        tmix[:e] += q_wav[:e]
        sf.write(str(d / "target_convo.wav"), tmix, mimi.sample_rate,
                 subtype="PCM_16")
        if do_asr:
            # ASR the WHOLE waveform: the agent channel is silent while the user
            # talks, so Whisper locates the agent's speech itself.
            tgt_txt = _asr(tw, mimi.sample_rate)

    a0 = int((n_greet + uc.shape[-1]) * spf)      # start of the agent's ANSWER
    parts, lines = [], []
    if tgt_txt:
        lines.append(f"TARGET   : {tgt_txt}")
    for name, g in grid.items():
        sf.write(str(d / f"{name}.wav"), g["agent"], mimi.sample_rate,
                 subtype="PCM_16")
        sf.write(str(d / f"{name}_convo.wav"), g["mix"], mimi.sample_rate,
                 subtype="PCM_16")
        parts.append(g["agent"][a0:])             # preview = the answers only
        parts.append(np.zeros(int(0.35 * mimi.sample_rate), dtype=np.float32))
        if do_asr and name != "base":
            lines.append(f"{name:9s}: {_asr(g['agent'], mimi.sample_rate)}")
    if parts:
        sf.write(str(d / "preview_all.wav"), np.concatenate(parts),
                 mimi.sample_rate, subtype="PCM_16")
    if lines:
        (d / "transcripts.txt").write_text("\n".join(lines) + "\n")

    names = list(grid)
    ref = grid[names[0]]["agent"]
    diffs = [float(np.abs(grid[n]["agent"][:len(ref)] - ref[:len(ref)]).max())
             for n in names[1:] if len(grid[n]["agent"]) == len(ref)]
    max_diff = round(max(diffs), 5) if diffs else 0.0
    return {"step": step, "dir": str(d), "max_wav_diff": max_diff,
            "emotions_identical": max_diff < 1e-4, "target": tgt_txt}
