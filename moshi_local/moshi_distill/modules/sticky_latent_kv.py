"""Sticky latent KV: extra attention memory that does not occupy stream time.

Latents are projected with the *same* K/V slices of each layer's `in_proj_weight`
and concatenated in front of the sliding-window cache. RingKVCache offset is
unchanged. Slot count `n_latents` is fixed so CUDA graphs keep a constant shape;
inactive / padded slots are masked out.
"""

from __future__ import annotations

import typing as tp

from einops import rearrange
import torch
from torch import nn
from torch.nn import functional as F


class StickyLatentMemory(nn.Module):
    """Shared [B, K, dim] latent bank + pooled vector for AdaLN / residual.

    Inference writes into registered buffers (in-place, graph-safe).
    Training sets `_train_*` tensors so gradients flow through concat.
    """

    def __init__(
        self,
        n_latents: int,
        dim: int,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.n_latents = n_latents
        self.dim = dim
        factory = {"device": device, "dtype": dtype}
        self.register_buffer("latents", torch.zeros(1, n_latents, dim, **factory), persistent=False)
        self.register_buffer("slot_mask", torch.zeros(1, n_latents, **factory), persistent=False)
        self.register_buffer("pooled", torch.zeros(1, dim, **factory), persistent=False)
        self.register_buffer("active", torch.zeros(1, 1, **factory), persistent=False)
        self._train_latents: torch.Tensor | None = None
        self._train_slot_mask: torch.Tensor | None = None
        self._train_pooled: torch.Tensor | None = None
        self._train_active: torch.Tensor | None = None

    def set_training_context(
        self,
        latents: torch.Tensor,
        slot_mask: torch.Tensor,
        pooled: torch.Tensor,
        active: torch.Tensor,
    ) -> None:
        self._train_latents = latents
        self._train_slot_mask = slot_mask
        self._train_pooled = pooled
        self._train_active = active

    def clear_training_context(self) -> None:
        self._train_latents = None
        self._train_slot_mask = None
        self._train_pooled = None
        self._train_active = None

    def set_inference_context(self, latents: torch.Tensor) -> None:
        """latents: [K, dim], [1, K, dim], or [B, K, dim] (B must be 1 for buffers)."""
        if latents.dim() == 2:
            latents = latents.unsqueeze(0)
        B, K, D = latents.shape
        if B != 1:
            latents = latents[:1]
        padded = torch.zeros_like(self.latents)
        mask = torch.zeros_like(self.slot_mask)
        n = min(K, self.n_latents)
        padded[:, :n] = latents[:, :n].to(dtype=padded.dtype, device=padded.device)
        mask[:, :n] = 1.0
        self.latents.copy_(padded)
        self.slot_mask.copy_(mask)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (padded * mask.unsqueeze(-1)).sum(dim=1) / denom
        self.pooled.copy_(pooled)
        self.active.fill_(1.0)

    def clear(self) -> None:
        self.slot_mask.zero_()
        self.latents.zero_()
        self.pooled.zero_()
        self.active.zero_()
        self.clear_training_context()

    def current(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._train_latents is not None:
            return (
                self._train_latents,
                self._train_slot_mask,
                self._train_pooled,
                self._train_active,
            )
        latents = self.latents.to(device=device, dtype=dtype)
        slot_mask = self.slot_mask.to(device=device, dtype=dtype)
        pooled = self.pooled.to(device=device, dtype=dtype)
        active = self.active.to(device=device, dtype=dtype)
        if latents.shape[0] == 1 and batch_size != 1:
            latents = latents.expand(batch_size, -1, -1)
            slot_mask = slot_mask.expand(batch_size, -1)
            pooled = pooled.expand(batch_size, -1)
            active = active.expand(batch_size, -1)
        return latents, slot_mask, pooled, active


def attach_sticky_kv(transformer: nn.Module, memory: StickyLatentMemory) -> None:
    """Point each main-transformer MHA at `memory` without double-registering it."""
    from .transformer import StreamingMultiheadAttention

    for module in transformer.modules():
        if isinstance(module, StreamingMultiheadAttention) and not module.weights_per_step:
            module._sticky_holder = [memory]


def project_latent_kv(
    latents: torch.Tensor,
    in_proj_weight: torch.Tensor,
    num_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map [B, K, dim] latents through the K/V slices of packed in_proj."""
    embed_dim = latents.shape[-1]
    w_k = in_proj_weight[embed_dim : 2 * embed_dim]
    w_v = in_proj_weight[2 * embed_dim : 3 * embed_dim]
    k_flat = F.linear(latents, w_k)
    v_flat = F.linear(latents, w_v)
    k = rearrange(k_flat, "b t (h d) -> b h t d", h=num_heads)
    v = rearrange(v_flat, "b t (h d) -> b h t d", h=num_heads)
    return k, v


def prepend_sticky_kv(
    attn: nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_bias: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Concat sticky K/V in front of window K/V. Does not write into RingKVCache."""
    holder: list | None = getattr(attn, "_sticky_holder", None)
    if not holder:
        return k, v, attn_bias
    memory: StickyLatentMemory = holder[0]
    B, H, Tq, _ = q.shape
    Tk = k.shape[2]
    latents, slot_mask, _pooled, active = memory.current(B, q.device, q.dtype)
    if latents.shape[-1] != attn.embed_dim or attn.weights_per_step:
        return k, v, attn_bias
    k_ctx, v_ctx = project_latent_kv(latents, attn.in_proj_weight, attn.num_heads)
    k_out = torch.cat([k_ctx, k], dim=2)
    v_out = torch.cat([v_ctx, v], dim=2)
    Kslots = latents.shape[1]
    visible = (slot_mask * active) > 0
    ctx_bias = visible[:, None, None, :].expand(B, 1, Tq, Kslots)
    if attn_bias is None:
        win_bias = torch.ones(B, 1, Tq, Tk, dtype=torch.bool, device=q.device)
    else:
        win_bias = attn_bias
        if win_bias.dim() == 2:
            win_bias = win_bias.view(1, 1, Tq, Tk).expand(B, 1, Tq, Tk)
        elif win_bias.dim() == 3:
            win_bias = win_bias.unsqueeze(1)
    attn_bias_out = torch.cat([ctx_bias, win_bias], dim=-1)
    return k_out, v_out, attn_bias_out
