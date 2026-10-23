"""
interleaver.py  (modified for RAG fine-tuning — literal <ref> text approach)
=============================================================================
<ref> is treated exactly like <system>: plain text through the ordinary
tokenizer, wrapped as "<ref> ... <ref>", tokenized normally, and spliced
into the text codebook at filler_end+1 (right after the model's filler
segment, before the body/grounding segment). No vocabulary surgery, no
new embedding row — the model learns the convention purely from training
exposure, same as it already learned <system>.

Changes from original
----------------------
1. `Sample` carries an optional `segment_meta: SegmentMeta | None` field.
2. `InterleavedTokenizer.__call__` reads segment_meta + reference text from
   the sidecar JSON, splices a <ref>...<ref> block at filler_end+1, and
   shifts body_start by the spliced block's length.
3. `Batch` carries `segment_metas: list[SegmentMeta | None]`.
4. Reference length is capped via `max_ref_tokens` at construction time —
   bounds worst-case sequence growth regardless of source reference length.
"""

from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import math
from collections import deque
from dataclasses import dataclass, field
from functools import reduce
from typing import Optional

import numpy as np
import sentencepiece
import torch
from .conditioners import ConditionAttributes

from .models.lm import SILENCE_TOKENS, SINE_TOKENS, AUDIO_TOKENS_PER_STREAM

import logging
logger = logging.getLogger(__name__)

class _DiagCounter:
    """Rate-limited diagnostic logging: prints full detail for the first
    `verbose_n` samples of each category, then just counts silently so
    training isn't spammed once you've confirmed the pipeline is sane."""
    def __init__(self, verbose_n: int = 5):
        self.verbose_n = verbose_n
        self.counts: dict[str, int] = {}

    def hit(self, category: str) -> bool:
        n = self.counts.get(category, 0)
        self.counts[category] = n + 1
        return n < self.verbose_n

    def summary(self) -> str:
        return ", ".join(f"{k}={v}" for k, v in sorted(self.counts.items()))


_diag = _DiagCounter()

def wrap_with_system_tags(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


def ref_start_id(tokenizer) -> int:
    return tokenizer.piece_to_id("<ref_start>")

def ref_end_id(tokenizer) -> int:
    return tokenizer.piece_to_id("<ref_end>")

def lookup_id(tokenizer) -> int:
    return tokenizer.piece_to_id("<lookup>")
# ---------------------------------------------------------------------------
# Helpers for Duplex
# ---------------------------------------------------------------------------
def _build_speaker_masked_waveform(wav: np.ndarray, chunk_relative_alignments: list,
                                    speaker_label: str, sample_rate: int,
                                    fade_samples: int = 32) -> np.ndarray:
    """Returns a copy of wav with all samples OUTSIDE speaker_label's word
    spans zeroed out. chunk_relative_alignments must already be offset so
    that word times are seconds-from-start-of-wav (same convention already
    used for text_tokens via the `a[1][0] - start_sec` adjustment).
    fade_samples applies a short linear ramp at each span edge so hard
    zeroing doesn't inject audible clicks that could distort Mimi's
    codebook assignment right at the boundary."""
    masked = np.zeros_like(wav)
    T = wav.shape[-1]
    for word, (start_sec, end_sec), speaker in chunk_relative_alignments:
        if speaker != speaker_label:
            continue
        s = max(0, int(start_sec * sample_rate))
        e = min(T, int(end_sec * sample_rate))
        if s >= e:
            continue
        masked[..., s:e] = wav[..., s:e]
        if fade_samples > 0:
            fs = min(fade_samples, e - s)
            ramp = np.linspace(0.0, 1.0, fs)
            masked[..., s:s+fs] *= ramp
            masked[..., e-fs:e] *= ramp[::-1]
    return masked

def _pad_to_target_n_q(audio_tokens: torch.Tensor, target_n_q: int, device) -> torch.Tensor:
    """audio_tokens: [1, n_q, T]. If n_q < target_n_q, pad the missing
    codebooks with SINE_TOKENS (the existing convention for 'no real
    user audio' used elsewhere in this file) rather than silently
    truncating or leaving the sample malformed."""
    n_q = audio_tokens.shape[1]
    if n_q == target_n_q:
        return audio_tokens
    if n_q > target_n_q:
        raise ValueError(f"sample has MORE codebooks ({n_q}) than model expects ({target_n_q}) — investigate, don't truncate silently")
    missing = target_n_q - n_q
    T = audio_tokens.shape[-1]
    sine = torch.as_tensor(SINE_TOKENS, dtype=torch.long, device=device)
    pad = sine[:missing].view(1, missing, 1).expand(-1, -1, T) if missing <= len(SINE_TOKENS) \
        else sine.repeat((missing + len(SINE_TOKENS) - 1) // len(SINE_TOKENS))[:missing].view(1, missing, 1).expand(-1, -1, T)
    return torch.cat([audio_tokens, pad], dim=1)

def compute_ref_slot_count(token_count: int, max_slots: int,
                           tokens_per_slot: int = 4, min_slots: int = 1) -> int:
    """Number of active ref slots for a reference of `token_count` tokens."""
    if token_count <= 0:
        return 0
    return min(max_slots,
               max(min_slots, math.ceil(token_count / max(tokens_per_slot, 1))))


# ---------------------------------------------------------------------------
# SegmentMeta
# ---------------------------------------------------------------------------

@dataclass
class TurnSpan:
    kind: str          # "regular" | "lead" | "lookup" | "filler" | "ref" | "body"
    start_frame: int
    end_frame: int


@dataclass
class SegmentMeta:
    turns: list[TurnSpan] = field(default_factory=list)
    batch_idx: int = 0

    def spans_of(self, kind: str) -> list[TurnSpan]:
        return [t for t in self.turns if t.kind == kind]

    @property
    def lookup_start(self) -> int:
        """Frame where the RAG-trigger head's positive label lives, or -1
        if this sample has no RAG turn. No splicing needed — this is just
        wherever the 'lookup' span already starts in the raw alignment."""
        spans = self.spans_of("lookup")
        return spans[0].start_frame if spans else -1

    @property
    def body_start(self) -> int:
        """Frame from which the gated context sum becomes active — the
        start of the grounded response, wherever it naturally occurs."""
        spans = self.spans_of("body")
        return spans[0].start_frame if spans else -1
    @property
    def body_end(self) -> int:
        spans = self.spans_of("body")
        return spans[0].end_frame if spans else -1


# ---------------------------------------------------------------------------
# Sample and Batch
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    codes: torch.Tensor                           # [1, K, T]
    condition_attributes: ConditionAttributes | None = None
    segment_meta: SegmentMeta | None = None       # None for non-RAG turns
    reference_text: str = ""


@dataclass
class Batch:
    codes: torch.Tensor
    condition_attributes: list[ConditionAttributes] | None = None
    segment_metas: list[SegmentMeta | None] = field(default_factory=list)
    reference_texts: list[str] = field(default_factory=list)  

    @classmethod
    def collate(cls, batch: list[Sample], pad_value: int) -> "Batch":
        """Samples can have different T (ref/system-prompt-bearing samples
        are longer), so shorter ones are right-padded to the batch max
        before concatenation. pad_value is the sentinel compute_audio_loss /
        compute_text_loss_rag treat as 'ignore this frame'.

        The batch max T is rounded UP to an even number: the distillation
        student's 2-frame compressor requires an even frame count, and the
        extra -1 frame is masked out by every loss."""
        max_T = max(b.codes.shape[-1] for b in batch)
        max_T += max_T % 2
        padded_codes = []
        for b in batch:
            T = b.codes.shape[-1]
            if T < max_T:
                pad = torch.full(
                    (1, b.codes.shape[1], max_T - T),
                    pad_value, dtype=b.codes.dtype, device=b.codes.device,
                )
                padded_codes.append(torch.cat([b.codes, pad], dim=-1))
            else:
                padded_codes.append(b.codes)
        codes = torch.cat(padded_codes, dim=0)

        if any(b.condition_attributes is not None for b in batch):
            condition_attributes = [b.condition_attributes for b in batch]
        else:
            condition_attributes = None

        segment_metas: list[SegmentMeta | None] = []
        reference_texts: list[str] = []
        for i, sample in enumerate(batch):
            if sample.segment_meta is not None:
                sample.segment_meta.batch_idx = i
            segment_metas.append(sample.segment_meta)
            reference_texts.append(sample.reference_text)

        return Batch(codes, condition_attributes, segment_metas, reference_texts)

    @property
    def rag_segment_metas(self) -> list[SegmentMeta]:
        return [m for m in self.segment_metas if m is not None]


# ---------------------------------------------------------------------------
# Tokenisation utilities (unchanged)
# ---------------------------------------------------------------------------

Alignment = tuple[str, tuple[float, float], str]
TokenizedAlignment = tuple[list[int], tuple[float, float], str]


def tokenize(
    tokenizer: sentencepiece.SentencePieceProcessor,
    text: str,
    bos: bool = True,
    alpha: float | None = None,
):
    nl_piece = tokenizer.encode("\n")[-1]
    if alpha is not None:
        tokens = tokenizer.encode(
            text.split("\n"), enable_sampling=True, alpha=alpha, nbest_size=-1
        )
    else:
        tokens = tokenizer.encode(text.split("\n"))
    tokens = reduce(lambda a, b: [*a, nl_piece, *b], tokens)
    if bos:
        tokens = [tokenizer.bos_id(), *tokens]
    return tokens


# ---------------------------------------------------------------------------
# Interleaver (unchanged)
# ---------------------------------------------------------------------------

class Interleaver:
    def __init__(
        self,
        tokenizer: sentencepiece.SentencePieceProcessor,
        audio_frame_rate: float,
        text_padding: int,
        end_of_text_padding: int,
        zero_padding: int,
        in_word_padding: int | None = None,
        keep_main_only: bool = False,
        main_speaker_label: str = "USER",
        use_bos_eos: bool = False,
        keep_and_shift: bool = False,
        audio_delay: float = 0.0,
        proba: float = 1.0,
        device: str | torch.device = "cuda",
    ):
        self.tokenizer = tokenizer
        self.audio_frame_rate = audio_frame_rate
        self.text_padding = text_padding
        self.end_of_text_padding = end_of_text_padding
        self.zero_padding = zero_padding
        self.in_word_padding = (
            self.text_padding if in_word_padding is None else in_word_padding
        )
        self.keep_main_only = keep_main_only
        self.main_speaker_label = main_speaker_label
        self.use_bos_eos = use_bos_eos
        self.keep_and_shift = keep_and_shift
        self.audio_delay = audio_delay
        self.proba = proba
        self.device = device

    @property
    def special_tokens(self) -> set[int]:
        return {
            self.text_padding, self.end_of_text_padding,
            self.tokenizer.bos_id(), self.tokenizer.eos_id(),
            self.zero_padding, self.in_word_padding,
        }

    def _tokenize(self, alignments: list[Alignment]) -> list[TokenizedAlignment]:
        out = []
        for word, ts, speaker in alignments:
            toks = tokenize(self.tokenizer, word.strip(), bos=False)
            out.append((toks, ts, speaker))
        return out

    def _keep_main_only(self, alignments, main_speaker):
        return [a for a in alignments if a[2] == main_speaker]

    def _keep_those_with_duration(self, alignments):
        return [a for a in alignments if a[1][0] < a[1][1]]

    def _add_delay(self, alignments):
        return [
            (a[0], (a[1][0] - self.audio_delay, a[1][1] - self.audio_delay), a[2])
            for a in alignments if a[1][1] > self.audio_delay
        ]

    def _insert_bos_eos(self, alignments, main_speaker):
        out: list[TokenizedAlignment] = []
        last_speaker = None
        for toks, ts, speaker in alignments:
            toks = list(toks)
            if speaker == last_speaker:
                pass
            elif speaker == main_speaker:
                toks.insert(0, self.tokenizer.bos_id())
            elif last_speaker == main_speaker:
                assert out
                toks.insert(0, self.tokenizer.eos_id())
            last_speaker = speaker
            out.append((toks, ts, speaker))
        return out

    def build_token_stream(self, alignments, segment_duration: float) -> torch.Tensor:
        T = math.ceil(segment_duration * self.audio_frame_rate)
        if alignments is None:
            text_tokens = [self.zero_padding] * T
        else:
            text_tokens = [self.text_padding] * T
            i = 0
            to_append_stack: deque = deque()
            last_word_end = -1
            for t in range(T):
                while (
                    i < len(alignments)
                    and alignments[i][1][0] * self.audio_frame_rate < t + 1
                ):
                    tokenized = alignments[i][0]
                    last_word_end = int(alignments[i][1][1] * self.audio_frame_rate)
                    if self.keep_and_shift:
                        to_append_stack.extend(tokenized)
                    else:
                        to_append_stack = deque(tokenized)
                    i += 1
                if to_append_stack:
                    if t > 0 and text_tokens[t - 1] in [self.text_padding, self.in_word_padding]:
                        text_tokens[t - 1] = self.end_of_text_padding
                    next_token = to_append_stack.popleft()
                    text_tokens[t] = next_token
                elif t <= last_word_end:
                    text_tokens[t] = self.in_word_padding
        if self.audio_delay < 0:
            prefix_length = int(self.audio_frame_rate * -self.audio_delay)
            text_tokens[:prefix_length] = [self.zero_padding] * prefix_length
        return torch.tensor(text_tokens, device=self.device).view(1, 1, -1)

    def prepare_item(self, alignments, segment_duration: float, main_speaker: str | None = None):
        if alignments is None:
            tokenized = None
        else:
            tokenized = self._tokenize(sorted(alignments, key=lambda x: x[1][0]))
            if self.keep_main_only:
                main_speaker = main_speaker or self.main_speaker_label
                tokenized = self._keep_main_only(tokenized, main_speaker)
            elif self.use_bos_eos:
                main_speaker = main_speaker or self.main_speaker_label
                tokenized = self._insert_bos_eos(tokenized, main_speaker)
            tokenized = self._keep_those_with_duration(tokenized)
            if self.audio_delay != 0:
                tokenized = self._add_delay(tokenized)
        return self.build_token_stream(tokenized, segment_duration)


def dicho(alignment, val, i=0, j=None):
    if j is None:
        j = len(alignment)
    if i == j:
        return i
    k = (i + j) // 2
    if alignment[k][1][0] < val:
        return dicho(alignment, val, k + 1, j)
    else:
        return dicho(alignment, val, i, k)
        
def _build_forced_audio_channels(n_q: int, n_frames: int, zero_padding: int, device) -> torch.Tensor:
    """Builds the audio-codebook portion for synthetic spliced frames
    (lookup / ref-slot / system-prompt / ref-block), respecting the
    ACTUAL n_q of this sample rather than assuming full duplex width.
    Mono/single-stream samples (n_q <= AUDIO_TOKENS_PER_STREAM) have no
    separate user channel to fill with sine — only agent silence applies."""
    APS = AUDIO_TOKENS_PER_STREAM
    silence = torch.as_tensor(SILENCE_TOKENS, dtype=torch.long, device=device)
    sine    = torch.as_tensor(SINE_TOKENS,    dtype=torch.long, device=device)

    if n_q <= APS:
        # mono/single-stream: only agent codebooks exist
        agent = silence[:n_q].view(1, n_q, 1).expand(-1, -1, n_frames)
        return agent

    agent_frames = silence.view(1, APS, 1).expand(-1, -1, n_frames)
    rest = n_q - APS
    if rest <= APS:
        # partial user stream (APS < n_q <= 2*APS)
        user_frames = sine[:rest].view(1, rest, 1).expand(-1, -1, n_frames)
        return torch.cat([agent_frames, user_frames], dim=1)

    # full duplex + possible extra padding codebooks
    user_frames = sine.view(1, APS, 1).expand(-1, -1, n_frames)
    remaining = n_q - 2 * APS
    if remaining > 0:
        pad = torch.full((1, remaining, n_frames), zero_padding, dtype=torch.long, device=device)
        return torch.cat([agent_frames, user_frames, pad], dim=1)
    return torch.cat([agent_frames, user_frames], dim=1)


# ---------------------------------------------------------------------------
# Shared RAG-splice helpers (used by both the full-rate and distillation
# tokenizers so the train/serve convention cannot diverge)
# ---------------------------------------------------------------------------

def parse_segment_meta(raw_meta, start_sec: float, duration_sec: float,
                       frame_rate: float) -> "SegmentMeta | None":
    """Window a sidecar `segment_meta` to this chunk and convert to frames."""
    turns_raw = (raw_meta or {}).get("turns")
    if not turns_raw:
        return None
    chunk_end_sec = start_sec + duration_sec

    def to_frame(sec: float) -> int:
        return int(max(0.0, min(sec, chunk_end_sec) - start_sec) * frame_rate)

    spans: list[TurnSpan] = []
    for t in turns_raw:
        t_start, t_end = t["start_sec"], t["end_sec"]
        if t_end <= start_sec or t_start >= chunk_end_sec:
            continue
        spans.append(TurnSpan(
            kind=t["kind"],
            start_frame=to_frame(max(t_start, start_sec)),
            end_frame=to_frame(min(t_end, chunk_end_sec)),
        ))
    return SegmentMeta(turns=spans) if spans else None


def shift_spans_from(meta: "SegmentMeta", insert_at: int, n_inserted: int) -> "SegmentMeta":
    """Shift every span boundary at/after insert_at right; a span ENDING exactly
    at insert_at is before the insertion and must not absorb it."""
    new_turns = []
    for t in meta.turns:
        start = t.start_frame + n_inserted if t.start_frame >= insert_at else t.start_frame
        end = t.end_frame + n_inserted if t.end_frame > insert_at else t.end_frame
        new_turns.append(TurnSpan(kind=t.kind, start_frame=start, end_frame=end))
    meta.turns = new_turns
    return meta


def build_lookup_frame(n_q: int, interleaver):
    device = interleaver.device
    ids = [lookup_id(interleaver.tokenizer)]
    text = torch.tensor(ids, device=device).view(1, 1, 1)
    audio = _build_forced_audio_channels(n_q, 1, interleaver.zero_padding, device)
    return text, audio


def build_ref_slot_frames(n_q: int, n_slots: int, interleaver):
    """`<ref_start> [K x zero_token] <ref_end>`; inner frames in user cb 8."""
    device = interleaver.device
    tok = interleaver.tokenizer
    ids = [ref_start_id(tok)] + [interleaver.zero_padding] * n_slots + [ref_end_id(tok)]
    n = len(ids)
    text = torch.tensor(ids, device=device).view(1, 1, n)
    audio = _build_forced_audio_channels(n_q, n, interleaver.zero_padding, device)
    audio[:, 8, 1:1 + n_slots] = interleaver.zero_padding
    return text, audio


def splice_rag_frames(text_tokens, audio_tokens, segment_meta, reference_text,
                      interleaver, *, n_ref_slots: int = 16,
                      max_ref_source_tokens: int = 400, ref_tokens_per_slot: int = 4,
                      min_ref_slots: int = 1, ref_slots_after_query: bool = True):
    """Insert `<lookup>` at the lookup boundary and `<ref_start> [K slots]
    <ref_end>` right after it. Returns (text_tokens, audio_tokens, segment_meta).

    Prefill-only: the K slot-frame embeddings are filled later by `RefAdapter`;
    this only allocates and locates them (and records the `ref`/`ref_slot`
    spans). `no_rag_qa` / `regular_chat` sidecars have no lookup span, so they
    pass through untouched."""
    if segment_meta is None:
        return text_tokens, audio_tokens, segment_meta

    has_lookup = segment_meta.lookup_start >= 0
    has_body = segment_meta.body_start >= 0

    if has_lookup:
        lookup_pos = segment_meta.lookup_start
        if 0 <= lookup_pos <= text_tokens.shape[-1]:
            lk_text, lk_audio = build_lookup_frame(audio_tokens.shape[1], interleaver)
            text_tokens = torch.cat(
                [text_tokens[..., :lookup_pos], lk_text, text_tokens[..., lookup_pos:]], dim=-1)
            audio_tokens = torch.cat(
                [audio_tokens[..., :lookup_pos], lk_audio, audio_tokens[..., lookup_pos:]], dim=-1)
            shift_spans_from(segment_meta, lookup_pos, 1)
            for t in segment_meta.turns:
                if t.kind == "lookup" and t.start_frame == lookup_pos + 1:
                    t.start_frame, t.end_frame = lookup_pos, lookup_pos + 1
                    break
            has_lookup = segment_meta.lookup_start >= 0
            has_body = segment_meta.body_start >= 0

    if reference_text and has_lookup and has_body:
        if ref_slots_after_query:
            insert_at = segment_meta.lookup_start + 1
        else:
            filler_spans = segment_meta.spans_of("filler")
            insert_at = (filler_spans[0].end_frame + 1) if filler_spans \
                else segment_meta.lookup_start + 1
        if 0 <= insert_at <= text_tokens.shape[-1]:
            n_q = audio_tokens.shape[1]
            ref_token_count = len(
                interleaver.tokenizer.encode(reference_text)[:max_ref_source_tokens])
            n_slots = compute_ref_slot_count(
                ref_token_count, n_ref_slots, ref_tokens_per_slot, min_ref_slots)
            ref_text_ids, ref_audio = build_ref_slot_frames(n_q, n_slots, interleaver)
            n_ref = ref_text_ids.shape[-1]
            text_tokens = torch.cat(
                [text_tokens[..., :insert_at], ref_text_ids, text_tokens[..., insert_at:]], dim=-1)
            audio_tokens = torch.cat(
                [audio_tokens[..., :insert_at], ref_audio, audio_tokens[..., insert_at:]], dim=-1)
            shift_spans_from(segment_meta, insert_at, n_ref)
            segment_meta.turns.append(TurnSpan(
                kind="ref", start_frame=insert_at, end_frame=insert_at + n_ref))
            segment_meta.turns.append(TurnSpan(
                kind="ref_slot", start_frame=insert_at + 1,
                end_frame=insert_at + 1 + n_slots))
    return text_tokens, audio_tokens, segment_meta


# ---------------------------------------------------------------------------
# InterleavedTokenizer
# ---------------------------------------------------------------------------

class InterleavedTokenizer:
    def __init__(self, mimi, interleaver, duration_sec, n_ref_slots=16,
                 max_ref_source_tokens=400,
                 agent_speaker_label: str = "ASSISTANT",
                 user_speaker_label: str = "USER",
                 target_n_audio_codebooks: int | None = 16,
                 ref_mode: str = "slots",
                 ref_slots_after_query: bool = True,
                 ref_tokens_per_slot: int = 4,
                 min_ref_slots: int = 1):
        self.mimi = mimi
        self.interleaver = interleaver
        self.duration_sec = duration_sec
        self.n_ref_slots = n_ref_slots
        self.max_ref_source_tokens = max_ref_source_tokens
        self.num_audio_frames = math.ceil(duration_sec * mimi.frame_rate)
        self.agent_speaker_label = agent_speaker_label
        self.user_speaker_label = user_speaker_label
        self.target_n_audio_codebooks = target_n_audio_codebooks
        assert ref_mode in ("slots", "off"), ref_mode
        self.ref_mode = ref_mode
        self.ref_slots_after_query = ref_slots_after_query
        self.ref_tokens_per_slot = ref_tokens_per_slot
        self.min_ref_slots = min_ref_slots

    def _build_ref_slot_frames(self, n_q: int, n_slots: int):
        """`<ref_start> [K x zero_token] <ref_end>` block.  The K inner frames
        live in the USER AUDIO channel (cb_index=8): with keep_main_only=True
        the text codebook is dedicated to the model's own speech, whereas the
        user-audio channel is architecturally the "listen to external content"
        pathway.  The adapter/RefAdapter writes the K context vectors into the
        summed embedding of exactly these inner frames (see `ref_adapter.py`)."""
        device = self.interleaver.device
        tok = self.interleaver.tokenizer
        K = n_slots
        ids = [ref_start_id(tok)] + [self.interleaver.zero_padding] * K + [ref_end_id(tok)]
        n = len(ids)
        text = torch.tensor(ids, device=device).view(1, 1, n)
        audio = _build_forced_audio_channels(n_q, n, self.interleaver.zero_padding, device)
        # Only the inner slot frames get their injection codebook zeroed; the
        # <ref_start>/<ref_end> markers keep ordinary silence/sine audio.
        audio[:, 8, 1:1 + K] = self.interleaver.zero_padding
        return text, audio

    def _build_lookup_frame(self, n_q: int):
        device = self.interleaver.device
        ids = [lookup_id(self.interleaver.tokenizer)]
        text = torch.tensor(ids, device=device).view(1, 1, 1)
        audio = _build_forced_audio_channels(n_q, 1, self.interleaver.zero_padding, device)
        return text, audio
    
    def _sec_to_frame(self, sec: float) -> int:
        return int(sec * self.mimi.frame_rate)

    def _build_forced_text_frames(self, wrapped_text: str, n_q: int) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.interleaver.tokenizer.encode(wrapped_text)
        n = len(tokens)
        device = self.interleaver.device
        text = torch.tensor(tokens, device=device).view(1, 1, n)
        audio = _build_forced_audio_channels(n_q, n, self.interleaver.zero_padding, device)
        return text, audio
    def _speaker_channel_encode(self, wav: np.ndarray, agent_channel: int, user_channel: int,
                             chunk_alignments: list, sample_rate: int) -> torch.Tensor:
        assert wav.ndim == 2 and wav.shape[0] >= 2, f"expected stereo (C>=2,T), got {wav.shape}"
        agent_wav = _build_speaker_masked_waveform(
            wav[agent_channel:agent_channel+1], chunk_alignments,
            self.agent_speaker_label, sample_rate,
        )
        user_wav = _build_speaker_masked_waveform(
            wav[user_channel:user_channel+1], chunk_alignments,
            self.user_speaker_label, sample_rate,
        )
        agent_tensor = torch.from_numpy(agent_wav).float().cuda().unsqueeze(0)
        user_tensor  = torch.from_numpy(user_wav).float().cuda().unsqueeze(0)
        with torch.no_grad():
            agent_tokens = self.mimi.encode(agent_tensor)
            user_tokens  = self.mimi.encode(user_tensor)
        return torch.cat([agent_tokens, user_tokens], dim=1)

    def _build_system_prompt_frames(self, system_prompt: str, n_q: int):
        return self._build_forced_text_frames(wrap_with_system_tags(system_prompt), n_q)

    def _build_ref_block_frames(self, reference: str, n_q: int):
        """Caps reference length at max_ref_tokens BEFORE wrapping+encoding,
        so the closing <ref> tag is never truncated away, and the encoded
        wrapped block always matches exactly what inference will inject."""
        content = reference.strip()
        if self.max_ref_tokens is not None:
            ids = self.interleaver.tokenizer.encode(content)
            if len(ids) > self.max_ref_tokens:
                ids = ids[: self.max_ref_tokens]
                content = self.interleaver.tokenizer.decode(ids)
        return self._build_forced_text_frames(wrap_with_ref_tags(content), n_q)

    def _parse_segment_meta(self, raw_meta: dict, chunk_start_sec: float) -> SegmentMeta | None:
        """Consumes the new {'turns': [{kind, start_sec, end_sec}, ...]} format.
        Turns are clipped to the current [chunk_start_sec, chunk_end_sec) window;
        turns entirely outside it are dropped. Frame indices are relative to the
        chunk, matching build_token_stream's own indexing."""
        turns_raw = raw_meta.get("turns")
        if not turns_raw:
            return None
    
        chunk_end_sec = chunk_start_sec + self.duration_sec
    
        def to_frame(sec: float) -> int:
            return self._sec_to_frame(max(0.0, min(sec, chunk_end_sec) - chunk_start_sec))
    
        spans: list[TurnSpan] = []
        for t in turns_raw:
            t_start, t_end = t["start_sec"], t["end_sec"]
            if t_end <= chunk_start_sec or t_start >= chunk_end_sec:
                continue  # entirely outside this chunk
            spans.append(TurnSpan(
                kind=t["kind"],
                start_frame=to_frame(max(t_start, chunk_start_sec)),
                end_frame=to_frame(min(t_end, chunk_end_sec)),
            ))
    
        if not spans:
            return None
    
        # sanity: RAG-turn sub-kinds must be internally ordered lead < lookup 
        # filler < ref < body if present at all, since splicing logic below
        # depends on it.
        order = {"lookup": 0, "lead": 1, "filler": 2, "ref": 3, "body": 4}
        rag_spans = sorted(
            (s for s in spans if s.kind in order), key=lambda s: order[s.kind]
        )
        for a, b in zip(rag_spans, rag_spans[1:]):
            if a.end_frame > b.start_frame:
                logger.warning(
                    f"segment_meta turn ordering looks off: {a.kind} ends at "
                    f"{a.end_frame} but {b.kind} starts at {b.start_frame}"
                )
                return None 
    
        return SegmentMeta(turns=spans)

    def __call__(self, wav: np.ndarray, start_sec: float, path: str) -> Sample:
        with torch.no_grad():
            info_file = os.path.splitext(path)[0] + ".sidecar.json"
            with open(info_file) as f:
                data = json.load(f)
    
            alignments = data["alignments"]
            start_alignment = dicho(alignments, start_sec)
            end_alignment   = dicho(alignments, start_sec + self.duration_sec)
            chunk_alignments = [
                (a[0], (a[1][0] - start_sec, a[1][1] - start_sec), a[2])
                for a in alignments[start_alignment:end_alignment]
            ]
    
            is_stereo = wav.ndim == 2 and wav.shape[0] >= 2
            if is_stereo:
                agent_ch = data.get("metadata", {}).get("agent_channel", 0)
                user_ch  = data.get("metadata", {}).get("user_channel", 1)
                audio_tokens = self._speaker_channel_encode(
                    wav, agent_ch, user_ch,
                    chunk_alignments, self.mimi.sample_rate,
                )
            else:
                if wav.ndim == 1: wav = wav[None, :]
                audio_tokens = self.mimi.encode(torch.from_numpy(wav).float().cuda().unsqueeze(0))
                if _diag.hit("mono_fallback_no_stereo"):
                    logger.warning(f"[diag] {path}: mono input, no channel separation")
    
            this_num_audio_frames = audio_tokens.shape[-1]
            audio_tokens = torch.nn.functional.pad(
                audio_tokens[..., : self.num_audio_frames],
                (0, self.num_audio_frames - this_num_audio_frames),
                value=self.interleaver.zero_padding,
            )
            audio_tokens = audio_tokens.view(1, -1, self.num_audio_frames)
            if audio_tokens.shape[1] != self.target_n_audio_codebooks:
                audio_tokens = _pad_to_target_n_q(audio_tokens, self.target_n_audio_codebooks, audio_tokens.device)
    
            text_tokens = self.interleaver.prepare_item(chunk_alignments, self.duration_sec)
            text_tokens = torch.nn.functional.pad(
                text_tokens, (0, self.num_audio_frames - text_tokens.shape[-1]),
                value=self.interleaver.zero_padding,
            )
    
            condition_attributes = data.get("text_conditions", None)
            reference_text = ""
            if isinstance(condition_attributes, dict):
                reference_text = condition_attributes.get("reference", "") or ""
    
            raw_meta = data.get("segment_meta")
            segment_meta: SegmentMeta | None = None
            if raw_meta is not None:
                segment_meta = self._parse_segment_meta(raw_meta, start_sec)
    
            has_lookup = segment_meta is not None and segment_meta.lookup_start >= 0
            has_body   = segment_meta is not None and segment_meta.body_start >= 0

            # ── <lookup> + <ref_start> [K slots] <ref_end> splice ──────────
            if self.ref_mode == "slots":
                text_tokens, audio_tokens, segment_meta = splice_rag_frames(
                    text_tokens, audio_tokens, segment_meta, reference_text,
                    self.interleaver,
                    n_ref_slots=self.n_ref_slots,
                    max_ref_source_tokens=self.max_ref_source_tokens,
                    ref_tokens_per_slot=self.ref_tokens_per_slot,
                    min_ref_slots=self.min_ref_slots,
                    ref_slots_after_query=self.ref_slots_after_query,
                )
                has_lookup = segment_meta is not None and segment_meta.lookup_start >= 0
                has_body = segment_meta is not None and segment_meta.body_start >= 0

            # Diagnostics — no injection to report anymore, just confirm the
            # turn structure and reference text line up as expected.
            if segment_meta is None:
                if _diag.hit("no_segment_meta"):
                    logger.info(f"[diag] {path}@{start_sec:.1f}s: no segment_meta")
            elif has_lookup and has_body and not reference_text:
                if _diag.hit("rag_spans_no_reference"):
                    logger.warning(f"[diag] {path}@{start_sec:.1f}s: has lookup/body spans "
                                    f"but NO reference_text — RAG loss will fire with no ground truth")
            elif has_lookup and has_body and reference_text:
                if _diag.hit("rag_turn_ok"):
                    logger.info(f"[diag] {path}@{start_sec:.1f}s: RAG turn — "
                                f"lookup@{segment_meta.lookup_start} body@{segment_meta.body_start} "
                                f"ref_len={len(reference_text)}chars")
    
            system_prompt = data.get("metadata", {}).get("system_prompt")
            n_prompt = 0
            if system_prompt and start_sec == 0.0:
                sys_text, sys_audio = self._build_system_prompt_frames(system_prompt, n_q=audio_tokens.shape[1])
                n_prompt = sys_text.shape[-1]
                text_tokens  = torch.cat([sys_text,  text_tokens],  dim=-1)
                audio_tokens = torch.cat([sys_audio, audio_tokens], dim=-1)
                if segment_meta is not None and n_prompt:
                    segment_meta = self._shift_spans_from(segment_meta, 0, n_prompt)
    
            return Sample(
                codes=torch.cat([text_tokens, audio_tokens], dim=1),
                condition_attributes=condition_attributes,
                segment_meta=segment_meta,
                reference_text=reference_text if has_body else "",
            )

    @staticmethod
    def _shift_spans_from(meta: SegmentMeta, insert_at: int, n_inserted: int) -> SegmentMeta:
        """Every span boundary at or after insert_at shifts right by
        n_inserted frames. Spans that straddle insert_at get their end
        pushed but not their start (they contain the insertion). A span that
        ENDS exactly at insert_at is before the insertion and must not absorb
        it (notably the single-frame <lookup> span)."""
        new_turns = []
        for t in meta.turns:
            start = t.start_frame + n_inserted if t.start_frame >= insert_at else t.start_frame
            end   = t.end_frame   + n_inserted if t.end_frame   >  insert_at else t.end_frame
            new_turns.append(TurnSpan(kind=t.kind, start_frame=start, end_frame=end))
        meta.turns = new_turns
        return meta