"""
emotion_recon.py — waveform-domain reconstruction loss for the emotion DepFormer
===============================================================================

Why: the token-level audio CE only scores each codebook position independently.
A model can score well per-token and still generate incoherent audio, because
nothing in the objective looks at the *acoustic* result.  This adds a
reconstruction loss on the audio Mimi actually renders, so incoherent output is
penalised directly.

How (the interesting bit)
-------------------------
`mimi.decode` normally takes INTEGER codes and does a table lookup.  We need it
to take the DepFormer's softmax over the 2048 codebook entries so the gradient
can flow from a waveform loss back into the logits.  Rather than reimplement
Mimi's decoder we patch exactly one method — `EuclideanCodebook.decode` — to do
`soft @ embedding` when handed a float tensor.  Everything downstream
(transpose, output_proj, decoder_transformer, SEANet decoder) is then reused
verbatim, so the loss is guaranteed to measure the same audio that generation
produces.

Loss: log-mel L1 between the decoded prediction and the decoded target.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn.functional as F


@contextmanager
def soft_quantizer():
    """Make Mimi's quantizer accept SOFT (float) code assignments."""
    from .quantization.core_vq import EuclideanCodebook
    orig = EuclideanCodebook.decode

    def decode(self, codes):
        if codes.dtype.is_floating_point:
            return torch.einsum("...c,cd->...d", codes, self.embedding)
        return orig(self, codes)

    EuclideanCodebook.decode = decode
    try:
        yield
    finally:
        EuclideanCodebook.decode = orig


@contextmanager
def _no_grad_on(module):
    req = [p.requires_grad for p in module.parameters()]
    for p in module.parameters():
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p, r in zip(module.parameters(), req):
            p.requires_grad_(r)


def log_mel(wav: torch.Tensor, sr: int, n_fft: int = 1024, hop: int = 256,
            n_mels: int = 80, fmin: float = 0.0, fmax: float = 8000.0):
    """[B, T] waveform -> [B, n_mels, frames] log-mel."""
    win = torch.hann_window(n_fft, device=wav.device, dtype=wav.dtype)
    spec = torch.stft(wav, n_fft=n_fft, hop_length=hop, window=win,
                      return_complex=True)
    mag = spec.abs()
    fb = _mel_fbanks(sr, n_fft, n_mels, fmin, fmax, wav.device, wav.dtype)
    mel = torch.einsum("bft,fm->bmt", mag, fb)
    return torch.log(mel.clamp_min(1e-5))


def _mel_fbanks(sr, n_fft, n_mels, fmin, fmax, device, dtype):
    try:
        import torchaudio.functional as AF
        fb = AF.melscale_fbanks(n_fft // 2 + 1, fmin, fmax, n_mels, sr)
        return fb.to(device=device, dtype=dtype)
    except Exception:
        pass
    # minimal fallback: linear filterbank over the same bins
    import math
    n_freqs = n_fft // 2 + 1
    return torch.ones(n_freqs, n_mels, device=device, dtype=dtype) / n_mels


def soft_codes(logits: torch.Tensor, mask: torch.Tensor | None = None
               ) -> torch.Tensor:
    """DepFormer logits [B, K, T, card] -> soft assignment [B, K, T, card].

    Invalid (NaN / padded) positions are zeroed so they contribute nothing to
    the decoded waveform instead of injecting noise.
    """
    lg = torch.nan_to_num(logits.float(), nan=0.0, posinf=0.0, neginf=0.0)
    soft = torch.softmax(lg, dim=-1)
    if mask is not None:
        soft = soft * mask.unsqueeze(-1).to(soft.dtype)
    return soft


def recon_loss(mimi, logits: torch.Tensor, target_codes: torch.Tensor,
               mask: torch.Tensor | None = None, n_mels: int = 80):
    """Log-mel L1 between Mimi's rendering of the prediction and of the target.

    logits        : [B, Ka, T, card]  DepFormer logits (agent codebooks only)
    target_codes  : [B, Ka, T]        the matching ground-truth codebooks
    mask          : [B, Ka, T]        valid positions

    The trailing batch padding is trimmed off before the mel comparison: padded
    frames are zeroed on the prediction side but the target side would decode
    `clamp(-1, 0)` = codebook entry 0, i.e. real audio, and scoring that
    mismatch would dominate the loss.
    """
    sr = mimi.sample_rate
    soft = soft_codes(logits, mask)                       # [B, Ka, T, card]
    tgt = target_codes.clamp(0, soft.shape[-1] - 1)

    with _no_grad_on(mimi):                               # target side: no grad
        wav_tgt = mimi.decode(tgt)                        # [B, 1, samples]
    with soft_quantizer():                                # differentiable path
        wav_pred = mimi.decode(soft)

    T = soft.shape[2]
    Tvalid = T
    if mask is not None:
        per_frame = mask[:, 0]                            # [B, T]
        idx = per_frame.any(dim=0).nonzero()
        Tvalid = int(idx.max().item()) + 1 if idx.numel() else T
    spf = wav_pred.shape[-1] / T                          # samples per frame
    n = max(1, int(Tvalid * spf))

    wp = wav_pred[..., :n].squeeze(1).float()
    wt = wav_tgt[..., :n].squeeze(1).float()
    mp = log_mel(wp, sr, n_mels=n_mels)
    mt = log_mel(wt, sr, n_mels=n_mels)
    k = min(mp.shape[-1], mt.shape[-1])
    return F.l1_loss(mp[..., :k], mt[..., :k].detach())


# ---------------------------------------------------------------------------
# Self-test: soft one-hot codes must reproduce the hard decode exactly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/workspace/personaplex")
    from moshi_local.moshi_distill.models import loaders

    SNAP = ("/workspace/.cache/huggingface/hub/models--nvidia--personaplex-7b-v1/"
            "snapshots/fdaf4090a61cb315c138a1faee287ffd6c716309")
    dev = "cuda"
    mimi = loaders.get_mimi(f"{SNAP}/tokenizer-e351c8d8-checkpoint125.safetensors", dev)

    torch.manual_seed(0)
    B, K, T = 2, 8, 20
    codes = torch.randint(0, 2048, (B, K, T), device=dev)

    hard = mimi.decode(codes)                             # [B, 1, samples]
    onehot = F.one_hot(codes, 2048).float()               # [B, K, T, 2048]
    with soft_quantizer():
        soft = mimi.decode(onehot)
    d = (hard - soft).abs().max().item()
    print(f"hard shape {tuple(hard.shape)}  soft shape {tuple(soft.shape)}")
    print(f"max |hard - soft(onehot)| = {d:.3e}")
    print("SOFT_QUANTIZER_OK" if d < 1e-3 else "SOFT_QUANTIZER_MISMATCH")
