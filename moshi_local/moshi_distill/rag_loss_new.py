"""
Extended loss computation for RAG fine-tuning — literal <ref> text approach
and Perceiver-resampler approach both use this module identically.

Builds on the base text/audio CE by adding:
  - per-segment text weights (lead / ref-token-frame / filler / ref-context / body)
  - NaN-safe handling of undelayed logits (masked positions come back as
    real NaN from _undelay_sequence's fill_value, not zero — must be
    sanitized BEFORE cross_entropy, not after, or backward can propagate
    NaN gradients even though the forward-masked value looks fine)

No contrastive loss in this version — dropped along with the streaming_sum
adapter path. If you reintroduce a contrastive objective later, it should
compare against a forward pass with a genuinely wrong <ref> BLOCK spliced
into codes (literal approach) or a wrong reference fed to the resampler
(latent approach) — not the old streaming_sum-vector mechanism.
"""

from __future__ import annotations
from dataclasses import dataclass
import random
import torch
import torch.nn.functional as F


from dataclasses import dataclass, field

def make_audio_synthetic_mask(B, T, seg_metas, device):
    keep = torch.ones((B, T), dtype=torch.float32, device=device)
    for meta in seg_metas:
        b = meta.batch_idx
        for span in meta.turns:
            if span.kind != "ref":   # "lookup" removed — no longer synthetic
                continue
            s, e = max(0, span.start_frame), min(T, span.end_frame)
            if s < e:
                keep[b, s:e] = 0.0
    return keep
    

@dataclass
class TurnSpan:
    kind: str          # "regular" | "lead" | "lookup" | "filler" | "ref" | "body"
    start_frame: int
    end_frame: int      # exclusive


@dataclass
class SegmentMeta:
    turns: list[TurnSpan] = field(default_factory=list)
    n_ref_tokens: int = 0
    n_lookup_tokens: int = 0
    batch_idx: int = 0

    def spans_of(self, kind: str) -> list[TurnSpan]:
        return [t for t in self.turns if t.kind == kind]


def make_segment_weight_tensor(
    B: int, T: int, seg_metas: list[SegmentMeta],
    device: torch.device,
    ref_weight: float = 0.0,
    lookup_weight: float = 5.0,
    lead_weight: float = 1.0,
    filler_weight: float = 1.0,
    ref_context_weight: float = 0.0,
    ref_slot_weight: float = 0.0,
    body_weight: float = 2.0,
    no_rag_weight: float = 1.0,
) -> torch.Tensor:
    """
    Build a [B, T] float tensor of per-frame text loss weights.

    ``ref`` / ``ref_slot`` default to **0**: the reference block is force-fed
    by the server, the model must never learn to emit it. Set them > 0 only
    for the literal-`<ref>`-text variant, never for the latent-slot design.
    """
    weights = torch.full((B, T), no_rag_weight, dtype=torch.float32, device=device)

    kind_weight = {
        "regular": no_rag_weight,
        "lead":    lead_weight,
        "lookup":  lookup_weight,
        "filler":  filler_weight,
        "ref":     ref_context_weight,
        "ref_slot": ref_slot_weight,
        "body":    body_weight,
    }

    for meta in seg_metas:
        b = meta.batch_idx

        for span in meta.turns:
            s = max(0, span.start_frame)
            e = min(T, span.end_frame)
            if s >= e:
                continue
            w = kind_weight.get(span.kind, no_rag_weight)
            weights[b, s:e] = w

        

    return weights


# ---------------------------------------------------------------------------
# Text loss with segment weights
# ---------------------------------------------------------------------------

def compute_text_loss_rag(
    text_logits: torch.Tensor,   # [B, 1, T, text_card] — contains real NaN at masked positions
    text_codes:  torch.Tensor,   # [B, 1, T]
    text_mask:   torch.Tensor,   # [B, 1, T]
    seg_weights: torch.Tensor,   # [B, T]
    text_padding_weight: float = 1.0,
    text_padding_ids: set[int] | None = None,
) -> torch.Tensor:
    """
    Extended text CE with per-frame segment weights.

    IMPORTANT: text_logits comes from _undelay_sequence(..., fill_value=NaN).
    Masked-out positions contain ACTUAL NaN, not zero. NaN must be
    sanitized BEFORE cross_entropy — NaN * 0 == NaN in IEEE754, so
    multiplying by a zero weight AFTER the fact does not clean it, and
    can leak NaN gradients into the backward graph even though the
    forward value looks correct via torch.where.
    """
    mask_2d = text_mask[:, 0, :].float()
    weights = seg_weights * mask_2d   # [B, T]

    if text_padding_ids is not None:
        codes_2d = text_codes[:, 0, :]
        for pad_id in text_padding_ids:
            weights[codes_2d == pad_id] *= text_padding_weight

    logits_flat = text_logits[:, 0].reshape(-1, text_logits.size(-1)).float()
    logits_flat = torch.nan_to_num(logits_flat, nan=0.0, posinf=0.0, neginf=0.0)  # sanitize BEFORE CE

    target_flat = text_codes[:, 0].reshape(-1)
    weights_flat = weights.reshape(-1)

    target_safe = torch.where(
        weights_flat > 0, target_flat, torch.zeros_like(target_flat),
    )

    loss_per_tok = F.cross_entropy(logits_flat, target_safe, reduction="none")
    loss_per_tok = loss_per_tok * weights_flat   # safe now: no NaN operand anywhere
    return loss_per_tok.sum() / weights_flat.sum().clamp(min=1e-8)

# ---------------------------------------------------------------------------
# Lookup Token Loss
# ---------------------------------------------------------------------------
def compute_rag_trigger_loss(rag_logits: torch.Tensor, seg_metas, pos_weight: float = 20.0) -> torch.Tensor:
    """Binary decision, dedicated head — not competing in the ~32k-way text
    vocab softmax. pos_weight compensates for lookup frames being rare
    without diluting the signal the way folding it into a full-sequence
    weighted mean did."""
    B, T, _ = rag_logits.shape
    targets = torch.zeros(B, T, device=rag_logits.device)
    for meta in seg_metas:
        for span in meta.spans_of("lookup"):
            if span.start_frame < T:
                targets[meta.batch_idx, span.start_frame] = 1.0
    logits_flat = rag_logits.squeeze(-1).reshape(-1)
    targets_flat = targets.reshape(-1)
    return F.binary_cross_entropy_with_logits(
        logits_flat, targets_flat,
        pos_weight=torch.tensor(pos_weight, device=rag_logits.device),
    )
    
# ---------------------------------------------------------------------------
# Audio loss
# ---------------------------------------------------------------------------

def compute_audio_loss(logits, audio_codes, mask, first_codebook_weight_multiplier=1.0,
                        synthetic_keep=None, num_agent_codebooks=None):
    target = torch.where(mask, audio_codes, torch.zeros_like(audio_codes))
    weights = mask.float()

    if num_agent_codebooks is not None:

        agent_mask = torch.zeros(weights.shape[1], device=weights.device, dtype=weights.dtype)
        agent_mask[:num_agent_codebooks] = 1.0
        weights = weights * agent_mask[None, :, None]

    weights[:, 0] *= first_codebook_weight_multiplier
    if synthetic_keep is not None:
        weights = weights * synthetic_keep[:, None, :]

    logits_flat = logits.reshape(-1, logits.size(-1)).float()
    logits_flat = torch.nan_to_num(logits_flat, nan=0.0, posinf=0.0, neginf=0.0)
    target_flat = target.reshape(-1)
    weights_flat = weights.reshape(-1)
    loss_per_tok = F.cross_entropy(logits_flat, target_flat, reduction="none")
    loss_per_tok = loss_per_tok * weights_flat
    return loss_per_tok.sum() / weights_flat.sum().clamp(min=1e-8)


# ---------------------------------------------------------------------------
# Combined RAG loss — 2-tuple return, matches both train_rag.py and
# train_rag_resampler.py's `text_loss, audio_loss = compute_rag_losses(...)`
# ---------------------------------------------------------------------------

def compute_rag_losses(
    model,
    output,
    codes: torch.Tensor,
    seg_metas: list[SegmentMeta],
    args,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, K, T = codes.shape
    device = codes.device
    text_padding_ids = {model.text_padding_token_id, model.end_of_text_padding_id}

    seg_weights = make_segment_weight_tensor(
        B=B, T=T, seg_metas=seg_metas, device=device,
        ref_weight=getattr(args, "ref_token_weight", 0.0),
        lookup_weight=getattr(args, "lookup_weight", 5.0),
        lead_weight=getattr(args, "lead_weight", 1.0),
        filler_weight=getattr(args, "filler_weight", 1.0),
        ref_context_weight=getattr(args, "ref_context_weight", 0.0),
        ref_slot_weight=getattr(args, "ref_slot_weight", 0.0),
        body_weight=getattr(args, "body_weight", 2.0),
        no_rag_weight=getattr(args, "no_rag_weight", 1.0),
    )

    text_loss = compute_text_loss_rag(
        text_logits=output.text_logits,
        text_codes=codes[:, :model.audio_offset],
        text_mask=output.text_mask,
        seg_weights=seg_weights,
        text_padding_weight=getattr(args, "text_padding_weight", 1.0),
        text_padding_ids=text_padding_ids,
    )

    synthetic_keep = make_audio_synthetic_mask(B=B, T=T, seg_metas=seg_metas, device=device)
    audio_loss = compute_audio_loss(
        logits=output.logits,
        audio_codes=codes[:, model.audio_offset : model.audio_offset + model.dep_q],
        mask=output.mask,
        first_codebook_weight_multiplier=getattr(args, "first_codebook_weight_multiplier", 1.0),
        synthetic_keep=synthetic_keep,
        # Default to ALL codebooks: the duplex user stream (codebooks 8..15)
        # must be predicted and accounted for, never masked out. Callers that
        # genuinely train a single-stream (mono) model can opt in explicitly.
        num_agent_codebooks=getattr(args, "num_agent_codebooks", None),
    )
    return text_loss, audio_loss