"""
server_student_rag.py — serve the RAG student (S21) with latent-slot prefill
==========================================================================

Same student serving loop as `server_student.py` (one Mimi, pruned backbone),
plus the pieces a grounded turn needs:

    user audio -> Mimi codes -> LM (listening)
    user stops -> STT transcript -> dense retrieval -> top-1 passage
    model emits <lookup> -> force <ref_start> [K latent slots] <ref_end>
    -> model answers, attending to the slots through its ordinary KV cache

The reference block is **prefill only**: those K+2 frames produce silence on
the wire, the text tokens `<ref_start>` / `-1` / `<ref_end>` are written into
the streaming cache, and the K slot embeddings are the adapter vectors
(`RefAdapter`, per-chunk pooling over the LM's own `text_emb`) added to the slot
frames' embeddings.  This mirrors `forward_train_with_context` exactly,
including the +1 embedding offset; `rag_stream_check.py --tiny` proves the
streaming path reproduces the training forward.

Frame accounting (matches the training span layout):

    frame      ref_slot span [rs, re), ref_end at re, ref_start at rs-1
    emb[i]     = codes[i-1]           (initial token at index 0)
    injection  emb[rs+1+j] += vec[j]  for j in [0, K),  K = re - rs

so the forced call sequence after `<lookup>` is

    write ref_start   (no vec)     <- input = <lookup>
    write slot 0      (no vec)     <- input = ref_start
    write slot j+1    (vec[j])     <- input = slot j       j = 0..K-2
    write ref_end     (vec[K-1])   <- input = slot K-1
    then normal generation resumes (first body token <- input = ref_end)

``ref_mode=add`` (MoshiRAG eq. 2) removes the block entirely: on the frame
after ``<lookup>`` the server queues one adapter vector per fact token and adds
it to the embedding of each following frame, which the model speaks anyway.
Zero extra frames, zero silence -- the reference rides along the lead instead
of pausing for it.

STT / VAD / barge-in are ported from `moshi/server_s.py`; retrieval is either
the numpy+MiniLM hybrid index built by `eval_retrieval_dense.py`
(``--retriever local``), a live web search (``web``: Tavily / Brave /
DuckDuckGo), or both (``hybrid``, web first and the local passage as the
fallback).  In text-pause mode the served block is the contiguous passage
window that matches the question (``rag_ref_select.select_window_ids``), the
same shape the S28 checkpoint was trained on.  A web search runs in a
background thread and is collected at ``<lookup>``; it never blocks a frame.

Usage
-----
    python -m moshi_distill.server_student_rag \
        --student-checkpoint /workspace/personaplex/ckpt/rag_S21/step_3300 \
        --moshi-weight .../model.safetensors --mimi-weight .../tokenizer-...safetensors \
        --tokenizer /workspace/personaplex/moshi_local/tokenizer.model \
        --rag-index /workspace/personaplex/rag_index_dense \
        --default-text-prompt-file /workspace/personaplex/persona.txt \
        --voice-prompt-dir /workspace/personaplex/voices --port 8999

    # validate checkpoint + adapter + index without serving:
    python -m moshi_distill.server_student_rag ... --dry-run

    # live web search instead of (or on top of) the local index:
    export TAVILY_API_KEY=tvly-...
    python -m moshi_distill.server_student_rag ... --retriever hybrid \
        --web-provider tavily --web-wait-frames 8
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import html as html_lib
import json
import os
import random
from pathlib import Path
import re
import sys
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import sentencepiece
import sphn
import torch

import aiohttp
from aiohttp import web

from .interleaver import compute_ref_slot_count
from .models import loaders, MimiModel, LMGen
from .models.lm import SILENCE_TOKENS, SINE_TOKENS, LMModel
from .rag_ref_select import select_ids as select_ref_ids
from .rag_ref_select import select_window_ids
from .ref_adapter import RefAdapter, RefAddAdapter, RefBagAdapter
from .server_student import (torch_auto_device, seed_all, wrap_with_system_tags,
                             _get_voice_prompt_dir, _get_static_path)
from .utils.connection import create_ssl_context, get_lan_ip
from .utils.logging import setup_logger, ColorizedLog

logger = setup_logger(__name__)

REF_START_ID = 32000
REF_END_ID = 32001
LOOKUP_ID = 32002
APS = 8                       # audio codebooks per stream

ST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Built-in web-search key, carried over from moshi/server_w.py
# (`--web-search-api-key`).  Override with --web-api-key or TAVILY_API_KEY.
TAVILY_KEY = "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"


# ---------------------------------------------------------------------------
# Retrieval (dense MiniLM + lexical hybrid + cross-encoder rerank)
# ---------------------------------------------------------------------------

class DenseRetriever:
    """Serving-side retrieval over the grounded_real KB.

    The KB is `passages.jsonl` + `emb.npy` written by
    `eval_retrieval_dense.py`.  Lexical candidates come from the same
    hashed n-gram index used during corpus analysis; the union is reranked
    by a MiniLM cross-encoder.
    """

    def __init__(self, index_dir: str, device: str = "cuda",
                 use_rerank: bool = True, dense_k: int = 50,
                 lexical_k: int = 20):
        from sentence_transformers import CrossEncoder, SentenceTransformer
        self.device = device
        self.dense_k = dense_k
        self.lexical_k = lexical_k
        with open(os.path.join(index_dir, "passages.jsonl")) as f:
            self.docs = [json.loads(line)["text"] for line in f]
        self.emb = np.load(os.path.join(index_dir, "emb.npy"))
        assert self.emb.shape[0] == len(self.docs), (self.emb.shape, len(self.docs))
        self.model = SentenceTransformer(ST_MODEL, device=device)
        self.ce = CrossEncoder(RERANK_MODEL, device=device) if use_rerank else None

        from .eval_retrieval import build as lex_build
        self._lidf, self._lpost, self._lnd = lex_build([{"ref": t} for t in self.docs])
        logger.info("retriever: %d passages | dense=%d lexical=%d rerank=%s",
                    len(self.docs), self.dense_k, self.lexical_k,
                    self.ce is not None)

    @torch.no_grad()
    def retrieve(self, query: str, top_k: int = 1) -> list[dict]:
        if not query.strip() or not self.docs:
            return []
        q = self.model.encode([query], normalize_embeddings=True,
                              convert_to_numpy=True, show_progress_bar=False)
        sims = self.emb @ q[0].astype(np.float32)
        cand = np.argsort(-sims)[: self.dense_k].tolist()

        if self.lexical_k:
            from .eval_retrieval import score as lex_score
            from .rag_index_v3 import features as lex_features
            ls = lex_score(self._lidf, self._lpost, self._lnd,
                           lex_features(query))
            seen = set(cand)
            cand.extend(int(i) for i in np.argsort(-ls)[: self.lexical_k]
                        if int(i) not in seen)

        if self.ce is not None and cand:
            scores = self.ce.predict([(query, self.docs[i]) for i in cand],
                                     batch_size=64, show_progress_bar=False)
            order = np.argsort(-np.asarray(scores))
            cand = [cand[i] for i in order]

        return [{"text": self.docs[i], "index": int(i),
                 "score": float(sims[i])} for i in cand[:top_k]]


class WebRetriever:
    """Query -> retrieved passages from a web search API.

    Returns the same shape as :class:`DenseRetriever`
    (``{"text", "score", "url"}``) so the reference path does not care where
    the passage came from.  Standard library only (``urllib``), no new
    dependency, and it is always called from a background thread: a search
    takes 0.3-2 s, the frame loop has 80 ms.

    Providers:

    * ``tavily``  requires ``TAVILY_API_KEY`` (or ``--web-api-key``); returns a
      written ``content`` snippet per result -- the best passage for the card.
    * ``brave``   requires ``BRAVE_API_KEY``; returns ``description`` snippets.
    * ``ddg``     no key, scrapes the DuckDuckGo HTML endpoint; snippets only,
      rate limited, use it to smoke-test the pipeline.
    """

    ENV = {"tavily": "TAVILY_API_KEY", "brave": "BRAVE_API_KEY"}

    def __init__(self, provider: str = "tavily", api_key: str | None = None,
                 timeout: float = 4.0, max_results: int = 3,
                 cache_size: int = 32):
        self.provider = provider
        self.timeout = float(timeout)
        self.max_results = max(1, int(max_results))
        self.api_key = (api_key or os.environ.get(self.ENV.get(provider, ""), "")
                        or (TAVILY_KEY if provider == "tavily" else ""))
        if provider in self.ENV and not self.api_key:
            raise SystemExit(
                f"web search provider {provider!r} needs an API key: pass "
                f"--web-api-key or set {self.ENV[provider]}")
        self._cache: dict[str, list[dict]] = {}
        self._cache_size = int(cache_size)

    # -- http ---------------------------------------------------------------
    def _open(self, url: str, headers: dict | None = None,
              data: bytes | None = None) -> bytes:
        req = urllib.request.Request(url, data=data, headers=headers or {})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return r.read()

    def _tavily(self, query: str) -> list[dict]:
        body = json.dumps({
            "api_key": self.api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": self.max_results,
            "include_answer": False,
            "include_raw_content": False,
        }).encode()
        raw = self._open("https://api.tavily.com/search",
                         headers={"Content-Type": "application/json",
                                  "Authorization": f"Bearer {self.api_key}"},
                         data=body)
        out = []
        for r in json.loads(raw).get("results", []):
            text = (r.get("content") or r.get("title") or "").strip()
            if text:
                out.append({"text": text, "score": float(r.get("score") or 0.0),
                            "url": r.get("url", "")})
        return out

    def _brave(self, query: str) -> list[dict]:
        url = ("https://api.search.brave.com/res/v1/web/search?"
               + urllib.parse.urlencode({"q": query,
                                         "count": self.max_results}))
        raw = self._open(url, headers={"Accept": "application/json",
                                       "X-Subscription-Token": self.api_key})
        out = []
        for r in json.loads(raw).get("web", {}).get("results", []):
            text = (r.get("description") or "").strip()
            if text:
                out.append({"text": text, "score": 0.0, "url": r.get("url", "")})
        return out

    def _ddg(self, query: str) -> list[dict]:
        url = ("https://html.duckduckgo.com/html/?"
               + urllib.parse.urlencode({"q": query}))
        raw = self._open(url, headers={
            "User-Agent": ("Mozilla/5.0 (X11; Linux x86_64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/124.0 Safari/537.36"),
            "Accept-Language": "en-US,en;q=0.9",
        }).decode("utf-8", "replace")
        snips = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', raw, re.S)
        out = []
        for s in snips[: self.max_results]:
            s = html_lib.unescape(re.sub(r"<[^>]+>", " ", s))
            s = re.sub(r"\s+", " ", s).strip()
            if s:
                out.append({"text": s, "score": 0.0, "url": ""})
        if not out:
            logger.warning("duckduckgo returned no results for %r (bot "
                           "challenge / rate limit -- use --web-provider "
                           "tavily or brave for real use)", query[:60])
        return out

    # -- public -------------------------------------------------------------
    def retrieve(self, query: str, top_k: int = 1) -> list[dict]:
        query = (query or "").strip()
        if not query:
            return []
        if query in self._cache:
            return self._cache[query][:top_k]
        fn = {"tavily": self._tavily, "brave": self._brave,
              "ddg": self._ddg}.get(self.provider)
        if fn is None:
            raise SystemExit(f"unknown web provider {self.provider!r}")
        try:
            hits = fn(query)
        except Exception as exc:                      # network, quota, parsing
            logger.warning("web search failed (%s): %s", self.provider, exc)
            return []
        if len(self._cache) >= self._cache_size:
            self._cache.pop(next(iter(self._cache)))
        self._cache[query] = hits
        logger.info("web search (%s): %d hits for %r", self.provider,
                    len(hits), query[:80])
        return hits[:top_k]



# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def decode_stt_tokens(text_tokens: list[torch.Tensor], tokenizer,
                      padding_token_id: int) -> str:
    if not text_tokens:
        return ""
    all_tokens = torch.cat(text_tokens, dim=-1).cpu().view(-1)
    valid = all_tokens[all_tokens > padding_token_id]
    if valid.numel() == 0:
        return ""
    return tokenizer.decode(valid.tolist())


_FADE_SAMPLES = 240


def _fade_from_silence(pcm: np.ndarray) -> np.ndarray:
    out = pcm.copy()
    n = min(_FADE_SAMPLES, out.shape[-1])
    out[:n] *= np.linspace(0.0, 1.0, n, dtype=np.float32)
    return out


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_rag_student(path: str | Path, device, dtype):
    ckpt = Path(path)
    if ckpt.is_dir():
        ckpt = ckpt / "student.pt"
    pkg = torch.load(ckpt, map_location="cpu", weights_only=False)
    kw = dict(pkg["base_kwargs"])
    kw["num_layers"] = len(pkg["kept_layers"])
    model = LMModel(device=device, dtype=dtype, **kw)
    model = model.to(device=device, dtype=dtype)
    model.load_state_dict(pkg["state_dict"])
    model.eval()
    ref_mode = pkg.get("ref_mode", "latent")
    kind = pkg.get("adapter_kind")
    if kind is None:
        kind = "latent" if pkg.get("adapter") is not None else None
    adapter = None
    if pkg.get("adapter") is not None:
        if kind == "bag":
            cls, extra = RefBagAdapter, {"decay": float(
                pkg.get("ref_decay") or 0.85)}
        elif kind == "add":
            cls, extra = RefAddAdapter, {}
        else:
            cls, extra = RefAdapter, {}
        adapter = cls(model.dim, model.text_emb,
                      n_slots=int(pkg["n_slots"]), **extra)
        adapter = adapter.to(device=device, dtype=dtype)
        adapter.load_state_dict(pkg["adapter"])
        adapter.eval()
    logger.info("loaded RAG student %s | text_card=%d | ref_mode=%s | "
                "adapter=%s | n_slots=%d | step=%s", ckpt, model.text_card,
                ref_mode, kind, pkg["n_slots"], pkg.get("step"))
    return model, adapter, pkg


# ---------------------------------------------------------------------------
# Server state
# ---------------------------------------------------------------------------

@dataclass
class RAGServerState:
    mimi: MimiModel
    text_tokenizer: sentencepiece.SentencePieceProcessor
    lm_gen: LMGen
    adapter: RefAdapter
    retriever: "DenseRetriever | None"
    stt_mimi: MimiModel
    stt_lm_gen: object
    stt_tokenizer: object
    device: torch.device
    voice_prompt_dir: str | None = None
    default_text_prompt: str | None = None
    vad_threshold: float = 0.5
    n_slots: int = 16
    ref_mode: str = "latent"
    ref_pos: str = "pre_body"
    max_ref_tokens: int = 256
    ref_tokens_per_slot: int = 4
    lookup_id: int = LOOKUP_ID
    stt_padding_token_id: int = 3
    web_retriever: "WebRetriever | None" = None
    web_wait_frames: int = 8
    web_trigger_below: float = 1.1

    def __post_init__(self):
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        # Additive mode: vectors still to be added, one per upcoming frame.
        self._ref_queue: list[torch.Tensor] = []
        # Text-pause mode: card ids still to be forced, one per upcoming frame.
        self._text_queue: list[int] = []
        # Async web search for the current turn (dict: query/done/hits).
        self._ref_job: dict | None = None
        self._ref_wait = 0
        self._ref_triggered = False
        self._speech_run = 0
        self.lock = asyncio.Lock()
        self.gpu_executor = ThreadPoolExecutor(max_workers=1)
        self._step_times: list[float] = []
        self._last_report = time.time()
        self.report_every_sec = 2.0
        self.mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)
        self.stt_mimi.streaming_forever(1)
        self.stt_lm_gen.streaming_forever(1)
        self.voice_prompt_fallback = None
        if self.voice_prompt_dir and os.path.isdir(self.voice_prompt_dir):
            cands = sorted(f for f in os.listdir(self.voice_prompt_dir)
                           if f.lower().endswith((".pt", ".wav", ".mp3", ".opus")))
            if cands:
                self.voice_prompt_fallback = os.path.join(self.voice_prompt_dir,
                                                          cands[0])

    # -- timing -------------------------------------------------------------
    def _time_step(self, codes: torch.Tensor, ref_vec: torch.Tensor | None = None,
                   text_token: int | None = None):
        sync = self.device.type == "cuda"
        if sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        if ref_vec is not None:
            tokens = self.lm_gen.step_with_ref(codes, ref_vec=ref_vec)
        elif text_token is not None:
            # Force the text stream for this frame (the RAG card riding the
            # pause).  Audio is untouched: the model speaks its own frame.
            tokens = self.lm_gen.step(codes, text_token=self._text_tok(text_token))
        else:
            tokens = self.lm_gen.step(codes)
        if sync:
            torch.cuda.synchronize()
        if tokens is not None:
            self._step_times.append((time.perf_counter() - t0) * 1e3)
        return tokens

    def step_stats_and_reset(self) -> str | None:
        if not self._step_times:
            return None
        ts = sorted(self._step_times)
        n = len(ts)
        mean = sum(ts) / n
        self._step_times = []
        self._last_report = time.time()
        return (f"step {mean:6.1f} ms/frame mean | p50 {ts[n // 2]:6.1f} | "
                f"p95 {ts[min(n - 1, int(0.95 * n))]:6.1f} | max {ts[-1]:6.1f} | n={n}")

    # -- token builders -----------------------------------------------------
    def _silence(self) -> torch.Tensor:
        return torch.as_tensor(SILENCE_TOKENS, dtype=torch.long,
                               device=self.device).view(1, APS, 1)

    def _sine(self, slot: bool = False) -> torch.Tensor:
        t = torch.as_tensor(SINE_TOKENS, dtype=torch.long,
                            device=self.device).view(1, APS, 1).clone()
        if slot:
            # Inner slot frames zero the first *user* codebook (LM cb 9),
            # exactly as `_build_ref_slot_frames` does in training.
            t[:, 0] = -1
        return t

    def _frame_tokens(self, text_id: int, slot: bool = False) -> torch.Tensor:
        """One [1, 17, 1] token frame: text + agent silence + user sine."""
        text = torch.full((1, 1, 1), int(text_id), dtype=torch.long,
                          device=self.device)
        return torch.cat([text, self._silence(), self._sine(slot)], dim=1)

    def _text_tok(self, value: int) -> torch.Tensor:
        return torch.tensor([int(value)], dtype=torch.long, device=self.device)

    # -- reference block injection -----------------------------------------
    @torch.no_grad()
    def inject_ref_block(self, vecs: torch.Tensor) -> int:
        """Force `<ref_start> [K slots] <ref_end>` into the live stream.

        `vecs` is [1, K, dim], already sliced to the K produced by the same
        `compute_ref_slot_count` the cache used.  Returns the number of LM
        frames consumed (K + 2) so the caller can emit that much silence.
        """
        lm = self.lm_gen.lm_model
        K = int(vecs.shape[1])
        assert vecs.shape[0] == 1
        if K <= 0:
            return 0

        silence = self._silence()
        sine = self._sine(slot=False)
        slot_user = self._sine(slot=True)
        slot_frame = self._frame_tokens(-1, slot=True)      # emb input for slot j
        emb_dtype = lm.text_emb.weight.dtype
        vecs = vecs.to(dtype=emb_dtype)

        # 1. write <ref_start>; input is the <lookup> frame already in cache
        self.lm_gen.step(input_tokens=sine, moshi_tokens=silence,
                         text_token=self._text_tok(REF_START_ID))
        # 2. write slot 0; input is <ref_start> (no adapter vector)
        self.lm_gen.step(input_tokens=slot_user, moshi_tokens=silence,
                         text_token=self._text_tok(-1))

        # 3. write slot j+1 while feeding slot j + vec[j]   (j = 0 .. K-2)
        for j in range(K - 1):
            emb = lm.embed_codes(slot_frame) + vecs[:, j:j + 1]
            self.lm_gen.step_embeddings_with_tokens(
                emb, input_tokens=slot_user, moshi_tokens=silence,
                text_token=self._text_tok(-1))

        # 4. write <ref_end> while feeding slot K-1 + vec[K-1]
        emb = lm.embed_codes(slot_frame) + vecs[:, K - 1:K]
        self.lm_gen.step_embeddings_with_tokens(
            emb, input_tokens=sine, moshi_tokens=silence,
            text_token=self._text_tok(REF_END_ID))
        return K + 2

    @torch.no_grad()
    def inject_ref_text(self, tokens: list[int]) -> int:
        """Force `<ref_start> tok0 .. tok(K-1) <ref_end>` into the live stream.

        Identical frame budget to :meth:`inject_ref_block` (K + 2 frames, K + 2
        model steps): each text token *is* one frame, so the LM reads the
        passage with the text embedding it already has.  Returns the number of
        frames consumed so the caller emits that much silence.
        """
        K = len(tokens)
        if K <= 0:
            return 0
        silence = self._silence()
        sine = self._sine(slot=False)
        slot_user = self._sine(slot=True)

        # write <ref_start>; input is the <lookup> frame already in the cache
        self.lm_gen.step(input_tokens=sine, moshi_tokens=silence,
                         text_token=self._text_tok(REF_START_ID))
        # write token j; input is token j-1 (or <ref_start> for j = 0)
        for t in tokens:
            self.lm_gen.step(input_tokens=slot_user, moshi_tokens=silence,
                             text_token=self._text_tok(int(t)))
        # write <ref_end>; input is the last reference token
        self.lm_gen.step(input_tokens=sine, moshi_tokens=silence,
                         text_token=self._text_tok(REF_END_ID))
        return K + 2

    @torch.no_grad()
    def start_ref_text_pause(self, tokens: list[int]) -> int:
        """Queue the fact card for the next frames; the model keeps speaking.

        Unlike :meth:`inject_ref_text` this consumes **no frame**: the text
        stream of the frames the model is about to produce is forced to the
        card ids, exactly as training writes them into the pad-text run that
        already sits before the answer.  Their audio is the model's own pause,
        so no silence is added and nothing is delayed.
        """
        self._text_queue = [int(t) for t in tokens if int(t) > 0]
        return len(self._text_queue)

    @torch.no_grad()
    def start_ref_add(self, tokens: list[int]) -> int:
        """Queue additive reference vectors for the next ``len(tokens)`` frames.

        Nothing is written into the cache and nothing is emitted here: the
        vectors are added to the input embeddings of the frames the model is
        about to speak anyway (its lead), frame ``j`` getting ``vec[j]``.  The
        cost is **zero frames**, so the caller emits no silence.  Returns the
        number of vectors queued (for logging only).
        """
        if not tokens:
            self._ref_queue = []
            return 0
        vecs = self.adapter.encode_ids(list(tokens), self.device)   # [K, dim]
        self._ref_queue = [v.view(1, 1, -1) for v in vecs]
        return int(len(self._ref_queue))

    # -- reference preparation ---------------------------------------------
    def reference_from_hit(self, query: str, hit):
        """Passage hit -> ``(k, block, hit)`` for this checkpoint.

        ``block`` is the kind of thing :meth:`inject_ref_text` /
        :meth:`start_ref_add` / :meth:`start_ref_text_pause` expect: card ids
        in text mode, reference vectors otherwise.
        """
        ref = hit["text"] if hit else ""
        ids = (self.text_tokenizer.encode(ref)[: self.max_ref_tokens]
               if ref else [])
        if self.ref_mode == "text":
            # S28 was trained on a *contiguous sentence window* of the passage
            # (4-gram Jaccard against the answer), not on the passage head and
            # not on a keyword bag.  Serve the same shape: the window that
            # matches the user's question, `n_slots` tokens (the trained card
            # length).  `select_ref_ids` here would reintroduce the bag and a
            # train/serve mismatch.
            k = min(max(int(self.n_slots), 1), len(ids)) if ids else 0
            if k <= 0:
                return 0, [], hit
            query_ids = self.text_tokenizer.encode(query) if query else []
            return k, select_window_ids(self.text_tokenizer, ids, query_ids, k), hit
        k = max(compute_ref_slot_count(len(ids), self.n_slots,
                                       self.ref_tokens_per_slot), 1)
        if self.ref_mode == "add":
            # No block at all: one vector per fact token over the frames that
            # follow <lookup>, so the "budget" is the card length itself.
            sel = select_ref_ids(self.text_tokenizer, ids,
                                 max(1, int(self.n_slots)))
            return len(sel), sel, hit
        if ids:
            id_t = torch.tensor([ids], dtype=torch.long, device=self.device)
            mask = torch.ones_like(id_t)
            vecs = self.adapter(id_t, mask)[:, :k]
        else:
            vecs = torch.zeros(1, k, self.adapter.dim, device=self.device,
                               dtype=self.adapter.fc1.weight.dtype)
        return k, vecs, hit

    def prepare_reference(self, transcript: str):
        """Retrieve for this turn; web search runs in the background.

        The local index is synchronous (tens of ms, it already ran inside the
        frame loop); the web search takes 0.3-2 s and would starve the 80 ms
        audio loop, so it is started here and collected at ``<lookup>`` --
        the lead + filler the model speaks in between is 2-4 s, which is
        usually enough.  The local passage is the fallback if it is not.
        """
        self._ref_job = None
        self._ref_wait = 0
        if self._is_small_talk(transcript):
            logger.info("small talk (%r): no retrieval, no card", transcript)
            return 0, [], None
        local = []
        if self.retriever is not None:
            local = self.retriever.retrieve(transcript, top_k=1)
        local_score = float(local[0].get("score", 0.0)) if local else 0.0
        # server_w.py behaviour: the web is a supplement for weak local hits
        # (--web-trigger-below 0.45).  The default (1.1) always searches.
        if (self.web_retriever is not None and (transcript or "").strip()
                and local_score < self.web_trigger_below):
            job: dict = {"query": transcript, "done": False, "hits": None}

            def _run():
                try:
                    job["hits"] = self.web_retriever.retrieve(transcript, top_k=1)
                except Exception:
                    logger.exception("web retrieval crashed")
                finally:
                    job["done"] = True

            threading.Thread(target=_run, daemon=True,
                             name="web-retrieve").start()
            self._ref_job = job
        return self.reference_from_hit(transcript, local[0] if local else None)

    # Words that carry no retrieval signal: greetings, back-channel, wh-words.
    # A transcript made only of these is chit-chat; injecting a passage there
    # produced things like answering "hello, how are you?" with a movie quote.
    _SMALL_TALK = frozenset("""
        a ah all alright alrighty am an and any are around as at
        back bad be been being bye can cant care cheers clear cool could
        day did do does doing done dont evening ever fine for from
        good goodbye great had has have having hear hearing hello hey hi
        how hows hmm i im in is it its just kind know later let like listen
        listening lot me meet mic microphone mm morning much my name neat
        nice night no nope not now of oh ok okay old on one or our out please
        pretty right see shall should so sound sounds speaking sure talk
        telling test testing thanks thank that thats the then there these they
        thing this those three time to today tomorrow too two four five six
        seven eight nine ten up us very was
        we well were what whats when where which who whom whose why will with
        would yeah yep yes yesterday you your youre
    """.split())

    @classmethod
    def _is_small_talk(cls, text: str) -> bool:
        """True if the transcript has no content word to retrieve on."""
        q = (text or "").strip().lower()
        if not q:
            return True
        if re.search(r"\d", q):
            return False
        words = re.findall(r"[a-z']+", q.replace("'", ""))
        if not words:
            return True
        return not any(w not in cls._SMALL_TALK for w in words)

    @staticmethod
    def _in_pause(pcm) -> bool:
        """True while the model's own frame carries (near) silence."""
        if pcm is None or getattr(pcm, "size", 0) == 0:
            return True
        return float(np.sqrt(np.mean(np.square(pcm)))) < 0.02

    # -- warmup -------------------------------------------------------------
    def warmup(self):
        for _ in range(4):
            chunk = torch.zeros(1, 1, self.frame_size, dtype=torch.float32,
                                device=self.device)
            codes = self.mimi.encode(chunk)
            for c in range(codes.shape[-1]):
                tokens = self.lm_gen.step(codes[:, :, c:c + 1])
                if tokens is not None:
                    _ = self.mimi.decode(tokens[:, 1:9])
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    # ------------------------------------------------------------------
    # Websocket session
    # ------------------------------------------------------------------
    async def handle_chat(self, request):
        ws = web.WebSocketResponse(heartbeat=10)
        await ws.prepare(request)
        clog = ColorizedLog.randomize()
        peer = request.remote
        peer_port = request.transport.get_extra_info("peername")[1]
        clog.log("info", f"Incoming connection from {peer}:{peer_port}")

        # ---- voice prompt ------------------------------------------------
        requested = request.query.get("voice_prompt", "")
        voice_prompt_path = None
        if self.voice_prompt_dir is not None:
            if requested:
                p = os.path.join(self.voice_prompt_dir, requested)
                voice_prompt_path = p if os.path.exists(p) else self.voice_prompt_fallback
                if voice_prompt_path != p:
                    clog.log("warning", f"voice prompt {requested!r} missing; "
                                        f"fallback {voice_prompt_path}")
            else:
                voice_prompt_path = self.voice_prompt_fallback

        if self.lm_gen.voice_prompt != voice_prompt_path:
            if voice_prompt_path is not None:
                if voice_prompt_path.endswith(".pt"):
                    self.lm_gen.load_voice_prompt_embeddings(voice_prompt_path)
                else:
                    self.lm_gen.load_voice_prompt(voice_prompt_path)
            else:
                clog.log("warning", "clearing previously loaded voice prompt")
                self.lm_gen.voice_prompt = None
                self.lm_gen.voice_prompt_audio = None
                self.lm_gen.voice_prompt_embeddings = None
                self.lm_gen.voice_prompt_cache = None

        text_prompt = (request.query.get("text_prompt", "")
                       or self.default_text_prompt or "")
        self.lm_gen.text_prompt_tokens = (
            self.text_tokenizer.encode(wrap_with_system_tags(text_prompt))
            if text_prompt else None)
        seed = int(request.query["seed"]) if "seed" in request.query else None

        # ---- per-session state ------------------------------------------
        listening = True
        generating = False
        lookup_done = False
        ref_vecs = None
        ref_info = None
        current_transcript = ""
        model_consecutive_pads = 0
        IDLE_PAD_LIMIT = 25                      # 2 s of pads ends the turn
        barge_in_frame_count = 0
        BARGE_IN_FRAMES_REQUIRED = 4
        stt_token_buffer: list[torch.Tensor] = []
        stt_in_utterance = False
        stt_silence_frame_count = 0
        stt_last_vad_end = False
        VAD_SILENCE_FRAMES_REQUIRED = 12
        last_frame_was_silence = True
        close = False
        opus_writer = sphn.OpusStreamWriter(self.mimi.sample_rate)
        opus_reader = sphn.OpusStreamReader(self.mimi.sample_rate)

        async def recv_loop():
            nonlocal close
            try:
                async for message in ws:
                    if message.type == aiohttp.WSMsgType.ERROR:
                        clog.log("error", f"{ws.exception()}")
                        break
                    if message.type in (aiohttp.WSMsgType.CLOSED,
                                        aiohttp.WSMsgType.CLOSE):
                        break
                    if message.type != aiohttp.WSMsgType.BINARY:
                        continue
                    data = message.data
                    if not isinstance(data, bytes) or not data:
                        continue
                    if data[0] == 1:
                        opus_reader.append_bytes(data[1:])
            finally:
                close = True
                clog.log("info", "connection closed")

        async def send_loop():
            while True:
                if close:
                    return
                await asyncio.sleep(0.001)
                msg = opus_writer.read_bytes()
                if len(msg) > 0:
                    await ws.send_bytes(b"\x01" + msg)

        async def opus_loop():
            nonlocal listening, generating, lookup_done, ref_vecs, ref_info
            nonlocal current_transcript, stt_token_buffer, stt_in_utterance
            nonlocal stt_silence_frame_count, stt_last_vad_end
            nonlocal model_consecutive_pads, barge_in_frame_count
            nonlocal last_frame_was_silence
            # A barge-in wipes the KV cache, and with it the system prompt.
            # Re-prefill it (between frames, on the event loop) so the model
            # keeps its persona and its <lookup> behaviour for the next turn.
            needs_prompt_replay = False

            async def _session_alive():
                return not close and not ws.closed

            all_pcm = None
            loop = asyncio.get_event_loop()

            def reset_streams():
                self.lm_gen.reset_streaming()
                self.mimi.reset_streaming()
                self.stt_lm_gen.reset_streaming()
                self.stt_mimi.reset_streaming()
                self._ref_queue = []
                self._text_queue = []
                self._ref_job = None
                self._ref_wait = 0
                self._ref_triggered = False
                self._speech_run = 0

            def run_frame(chunk_tensor):
                """One 80 ms frame, FULL DUPLEX.

                The model generates every frame and keeps its own turn timing
                (exactly as in training and in server_student.py); the STT/VAD
                only *triggers retrieval* on end of speech — it never forces
                the model silent.  Returns (pcm, kind, meta).
                """
                nonlocal lookup_done, ref_vecs, ref_info
                nonlocal current_transcript, stt_token_buffer, stt_in_utterance
                nonlocal stt_silence_frame_count, stt_last_vad_end
                nonlocal barge_in_frame_count, needs_prompt_replay

                input_codes = self.mimi.encode(chunk_tensor)
                stt_codes = self.stt_mimi.encode(chunk_tensor)
                stt_tokens, vad_heads = self.stt_lm_gen.step_with_extra_heads(stt_codes)
                vad_score = (float(vad_heads[2][0, 0, 0].cpu().item())
                             if vad_heads and len(vad_heads) > 2 else 0.0)

                # ---- STT transcript -> retrieval trigger (never mutes the model)
                fired = None
                if stt_tokens is not None:
                    stt_token_buffer.append(stt_tokens[:, :1, :].cpu())
                    if stt_tokens[0, 0, 0].item() not in (0, 3):
                        stt_in_utterance = True
                        stt_last_vad_end = False
                if vad_score > self.vad_threshold:
                    stt_silence_frame_count += 1
                else:
                    stt_silence_frame_count = 0
                vad_fired = (stt_silence_frame_count >= VAD_SILENCE_FRAMES_REQUIRED
                             and not stt_last_vad_end)
                stt_last_vad_end = (stt_silence_frame_count
                                    >= VAD_SILENCE_FRAMES_REQUIRED)
                if vad_fired and stt_in_utterance and stt_token_buffer:
                    transcript = decode_stt_tokens(
                        stt_token_buffer, self.stt_tokenizer,
                        self.stt_padding_token_id)
                    stt_token_buffer = []
                    stt_in_utterance = False
                    stt_silence_frame_count = 0
                    if transcript.strip():
                        k, vecs, info = self.prepare_reference(transcript)
                        ref_vecs = vecs
                        ref_info = info
                        current_transcript = transcript
                        lookup_done = False
                        self._ref_triggered = False
                        self._ref_queue = []
                        self._text_queue = []
                        self._speech_run = 0
                        fired = {"transcript": transcript, "k": k, "hit": info}

                # ---- free generation (full duplex) -------------------------
                ref_vec = self._ref_queue.pop(0) if self._ref_queue else None
                forced_text = self._text_queue.pop(0) if self._text_queue else None
                tokens = self._time_step(input_codes, ref_vec, forced_text)
                if tokens is None:
                    return (np.zeros(self.frame_size, np.float32), "silence", None)

                text_token = int(tokens[0, 0, 0].item())
                with torch.no_grad():
                    main_pcm = self.mimi.decode(tokens[:, 1:9]).cpu()
                pcm = np.clip(main_pcm[0, 0].float().numpy(), -1.0, 1.0)

                # ---- barge-in: the user talks over the model's speech ------
                speech_this_frame = (stt_tokens is not None
                                     and stt_tokens[0, 0, 0].item() not in (0, 3)
                                     and vad_score < self.vad_threshold)
                model_speaking = not self._in_pause(pcm)
                barge_in_frame_count = (barge_in_frame_count + 1
                                        if (speech_this_frame and model_speaking) else 0)
                if barge_in_frame_count >= BARGE_IN_FRAMES_REQUIRED:
                    barge_in_frame_count = 0
                    reset_streams()
                    lookup_done = False
                    ref_vecs = ref_info = None
                    stt_token_buffer = []
                    stt_in_utterance = True
                    stt_last_vad_end = False
                    stt_silence_frame_count = 0
                    needs_prompt_replay = True
                    return (np.zeros(self.frame_size, np.float32), "barge_in", None)

                # ---- never let card tokens spill into the answer -----------
                if self._text_queue:
                    if self._in_pause(pcm):
                        self._speech_run = 0
                    else:
                        self._speech_run += 1
                        if self._speech_run >= 2:
                            logger.info("card truncated at speech: %d tokens dropped",
                                        len(self._text_queue))
                            self._text_queue = []
                else:
                    self._speech_run = 0

                # ---- <lookup> -> card injection ----------------------------
                if text_token == self.lookup_id and not lookup_done:
                    self._ref_triggered = True
                if self._ref_triggered and not lookup_done:
                    job = self._ref_job
                    if job is not None and not job["done"]:
                        if (self._ref_wait < self.web_wait_frames
                                and self._in_pause(pcm)):
                            if self._ref_wait == 0:
                                logger.info("web search still running: "
                                            "spending pause frames")
                            self._ref_wait += 1
                            return (pcm, "ref_wait", {"waited": self._ref_wait})
                        self._ref_job = None          # timed out -> local card
                    elif job is not None:
                        self._ref_job = None
                        if job.get("hits") and job["hits"][0].get("text"):
                            _, ref_vecs, ref_info = self.reference_from_hit(
                                job["query"], job["hits"][0])
                            logger.info("using web passage %r",
                                        job["hits"][0]["text"][:90])
                    if self._ref_wait > 0 and not self._in_pause(pcm):
                        logger.info("ref dropped: pause ended before injection")
                        ref_vecs = None
                    n_ref = (int(ref_vecs.shape[1])
                             if torch.is_tensor(ref_vecs)
                             else len(ref_vecs or []))
                    self._ref_triggered = False
                    lookup_done = True
                    if n_ref > 0:
                        if self.ref_mode == "add":
                            k = self.start_ref_add(ref_vecs)
                            return (np.zeros(self.frame_size, np.float32),
                                    "ref_inject",
                                    {"frames": 0, "k": k, "hit": ref_info})
                        if self.ref_mode == "text" and self.ref_pos == "pause":
                            k = self.start_ref_text_pause(
                                ref_vecs if not torch.is_tensor(ref_vecs) else [])
                            self._speech_run = 0
                            return (pcm, "ref_inject",
                                    {"frames": 0, "k": k, "hit": ref_info})
                        n_frames = (self.inject_ref_text(ref_vecs)
                                    if self.ref_mode == "text"
                                    else self.inject_ref_block(ref_vecs))
                        return (np.zeros(self.frame_size * n_frames, np.float32),
                                "ref_inject",
                                {"frames": n_frames, "k": n_ref, "hit": ref_info})
                if text_token == self.lookup_id:
                    # duplicate / no-reference lookup: never leak the token
                    pcm = np.zeros(self.frame_size, np.float32)

                if fired is not None:
                    return (pcm, "turn_start", fired)
                return (pcm, "audio",
                        {"text": text_token if forced_text is None else 0})

            try:
                while True:
                    if close:
                        return
                    if needs_prompt_replay:
                        needs_prompt_replay = False
                        clog.log("info", "re-prefilling system prompt after barge-in")
                        await self.lm_gen.step_system_prompts_async(
                            self.mimi, is_alive=_session_alive)
                        self.mimi.reset_streaming()
                        clog.log("info", "system prompt replayed")
                    await asyncio.sleep(0.001)
                    pcm = opus_reader.read_pcm()
                    if pcm is None or pcm.shape[-1] == 0:
                        continue
                    all_pcm = pcm if all_pcm is None else np.concatenate(
                        (all_pcm, pcm))
                    while all_pcm.shape[-1] >= self.frame_size:
                        chunk = all_pcm[: self.frame_size]
                        all_pcm = all_pcm[self.frame_size:]
                        chunk_t = torch.from_numpy(chunk).to(
                            device=self.device)[None, None]
                        frame, kind, meta = await loop.run_in_executor(
                            self.gpu_executor, run_frame, chunk_t)

                        if kind == "turn_start":
                            clog.log("info", f"transcript={meta['transcript']!r} "
                                             f"k={meta['k']}")
                            if meta["hit"]:
                                clog.log("info", f"  ref score={meta['hit']['score']:.3f} "
                                                 f"{meta['hit']['text'][:90]!r}")
                        elif kind == "ref_inject":
                            clog.log("info", f"<lookup> -> ref block "
                                             f"({meta['frames']} frames, K={meta['k']})")
                        elif kind == "barge_in":
                            clog.log("info", "barge-in: reset to listening")
                        elif kind == "turn_end":
                            clog.log("info", "turn ended (pads) -> listening")

                        if kind == "audio":
                            if last_frame_was_silence:
                                frame = _fade_from_silence(frame)
                                last_frame_was_silence = False
                            t = int(meta["text"])
                            if t not in (0, 3) and t < REF_START_ID:
                                piece = self.text_tokenizer.id_to_piece(t)
                                await ws.send_bytes(b"\x02" + bytes(
                                    piece.replace("\u2581", " "), "utf8"))
                        else:
                            last_frame_was_silence = True

                        # `frame` may hold several silence frames (ref block).
                        for i in range(0, frame.shape[-1], self.frame_size):
                            opus_writer.append_pcm(
                                frame[i:i + self.frame_size])

                        if time.time() - self._last_report >= self.report_every_sec:
                            rep = self.step_stats_and_reset()
                            if rep:
                                clog.log("info", rep)
            except Exception as e:  # noqa: BLE001
                clog.log("error", f"opus_loop crashed: {e!r}")
                raise

        async with self.lock:
            if seed is not None and seed != -1:
                seed_all(seed)
            self.mimi.reset_streaming()
            self.lm_gen.reset_streaming()
            self.stt_mimi.reset_streaming()
            self.stt_lm_gen.reset_streaming()

            async def is_alive():
                if close or ws.closed:
                    return False
                return True

            await self.lm_gen.step_system_prompts_async(self.mimi, is_alive=is_alive)
            self.mimi.reset_streaming()
            clog.log("info", "system prompts done")

            if not ws.closed:
                await ws.send_bytes(b"\x00")
                tasks = [asyncio.create_task(recv_loop()),
                         asyncio.create_task(opus_loop()),
                         asyncio.create_task(send_loop())]
                done, pending = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED)
                for t in pending:
                    t.cancel()
                    try:
                        await t
                    except asyncio.CancelledError:
                        pass
                await ws.close()
        return ws


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8999)
    ap.add_argument("--student-checkpoint", required=True)
    ap.add_argument("--moshi-weight", type=str)
    ap.add_argument("--mimi-weight", type=str)
    ap.add_argument("--tokenizer", type=str,
                    default="/workspace/personaplex/moshi_local/tokenizer.model")
    ap.add_argument("--rag-index", type=str,
                    default="/workspace/personaplex/rag_index_dense")
    ap.add_argument("--retriever", choices=["local", "web", "hybrid"],
                    default="local",
                    help="passage source: local KB, live web search, or both "
                         "(web first, local as fallback)")
    ap.add_argument("--web-provider", choices=["tavily", "brave", "ddg"],
                    default="tavily")
    ap.add_argument("--web-api-key", default=None,
                    help="or set TAVILY_API_KEY / BRAVE_API_KEY")
    ap.add_argument("--web-timeout", type=float, default=4.0)
    ap.add_argument("--web-max-results", type=int, default=3)
    ap.add_argument("--web-wait-frames", type=int, default=8,
                    help="frames of the model's own pause spent waiting for "
                         "an in-flight web search (80 ms each)")
    ap.add_argument("--web-trigger-below", type=float, default=1.1,
                    help="search the web only when the local top score is "
                         "below this (server_w.py used 0.45; 1.1 = always)")
    ap.add_argument("--stt-hf-repo", type=str,
                    default="kyutai/stt-1b-en_fr-candle")
    ap.add_argument("--voice-prompt-dir", type=str)
    ap.add_argument("--default-text-prompt", type=str, default=None)
    ap.add_argument("--default-text-prompt-file", type=str, default=None)
    ap.add_argument("--vad-threshold", type=float, default=0.5)
    ap.add_argument("--max-ref-tokens", type=int, default=256)
    ap.add_argument("--ref-tokens-per-slot", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--static", type=str)
    ap.add_argument("--ssl", type=str)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--bench-steps", type=int, default=0)
    args = ap.parse_args()

    if args.default_text_prompt_file:
        with open(args.default_text_prompt_file, encoding="utf8") as f:
            args.default_text_prompt = f.read().strip()

    device = torch_auto_device(args.device)
    seed_all(42424242)
    args.voice_prompt_dir = _get_voice_prompt_dir(args.voice_prompt_dir,
                                                  loaders.DEFAULT_REPO)
    static_path = _get_static_path(args.static, loaders.DEFAULT_REPO)

    logger.info("loading mimi")
    if args.mimi_weight is None:
        from huggingface_hub import hf_hub_download
        args.mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(args.mimi_weight, device)

    if args.tokenizer is None:
        from huggingface_hub import hf_hub_download
        args.tokenizer = hf_hub_download(loaders.DEFAULT_REPO,
                                         loaders.TEXT_TOKENIZER_NAME)
    text_tokenizer = sentencepiece.SentencePieceProcessor(args.tokenizer)
    assert text_tokenizer.vocab_size() == 32003, text_tokenizer.vocab_size()

    logger.info("loading RAG student")
    student, adapter, pkg = load_rag_student(args.student_checkpoint, device,
                                             torch.bfloat16)
    lm_gen = LMGen(student, device=device,
                   audio_silence_frame_cnt=int(0.5 * mimi.frame_rate),
                   sample_rate=mimi.sample_rate, frame_rate=mimi.frame_rate,
                   save_voice_prompt_embeddings=False,
                   fast_provided_tail=True)

    logger.info("loading STT")
    # Importing `moshi_distill` prepends .../moshi_local to sys.path, so a bare
    # `import moshi` would load the old in-repo fork (no CheckpointInfo).  The
    # STT checkpoint needs the installed 0.2.x package.
    local_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path[:] = [p for p in sys.path
                   if os.path.abspath(p or os.getcwd()) != local_root]
    from moshi.models import loaders as stt_loaders
    from moshi.models import LMGen as STTLMGen
    logger.info("STT loaders: %s", stt_loaders.__file__)
    stt_info = stt_loaders.CheckpointInfo.from_hf_repo(args.stt_hf_repo)
    stt_mimi = stt_info.get_mimi(device=device)
    stt_lm = stt_info.get_moshi(device=device, dtype=torch.bfloat16)
    stt_lm.eval()
    stt_lm_gen = STTLMGen(stt_lm, temp=0, temp_text=0.0)
    stt_tokenizer = stt_info.get_text_tokenizer()
    stt_padding = stt_info.raw_config.get("text_padding_token_id", 3)

    retriever = None
    if args.retriever in ("local", "hybrid"):
        logger.info("loading retrieval index")
        retriever = DenseRetriever(args.rag_index, device=str(device))
    web_retriever = None
    if args.retriever in ("web", "hybrid"):
        web_retriever = WebRetriever(args.web_provider, args.web_api_key,
                                     timeout=args.web_timeout,
                                     max_results=args.web_max_results)
        logger.info("web search enabled: provider=%s timeout=%.1fs",
                    web_retriever.provider, web_retriever.timeout)

    state = RAGServerState(
        mimi=mimi, text_tokenizer=text_tokenizer, lm_gen=lm_gen,
        adapter=adapter, retriever=retriever, stt_mimi=stt_mimi,
        stt_lm_gen=stt_lm_gen, stt_tokenizer=stt_tokenizer, device=device,
        voice_prompt_dir=args.voice_prompt_dir,
        default_text_prompt=args.default_text_prompt,
        vad_threshold=args.vad_threshold,
        n_slots=int(pkg["n_slots"]),
        ref_mode=str(pkg.get("ref_mode", "latent")),
        ref_pos=str(pkg.get("ref_pos", "pre_body")),
        max_ref_tokens=args.max_ref_tokens,
        ref_tokens_per_slot=args.ref_tokens_per_slot,
        stt_padding_token_id=stt_padding,
        web_retriever=web_retriever,
        web_wait_frames=max(0, args.web_wait_frames),
        web_trigger_below=args.web_trigger_below)
    if args.default_text_prompt:
        lm_gen.text_prompt_tokens = text_tokenizer.encode(
            wrap_with_system_tags(args.default_text_prompt))

    logger.info("warming up")
    state.warmup()
    if torch.cuda.is_available():
        logger.info("GPU memory: allocated=%.2f GB reserved=%.2f GB",
                    torch.cuda.memory_allocated() / 1e9,
                    torch.cuda.memory_reserved() / 1e9)

    if args.dry_run:
        logger.info("dry run OK: student+adapter+index+STT load and step")
        return

    if args.bench_steps > 0:
        for _ in range(args.bench_steps):
            chunk = torch.zeros(1, 1, state.frame_size, dtype=torch.float32,
                                device=device)
            codes = state.mimi.encode(chunk)
            for c in range(codes.shape[-1]):
                tokens = state._time_step(codes[:, :, c:c + 1])
                if tokens is not None:
                    _ = state.mimi.decode(tokens[:, 1:9])
        logger.info("BENCH: %s", state.step_stats_and_reset())
        return

    app = web.Application()
    app.router.add_get("/api/chat", state.handle_chat)
    if static_path is not None:
        async def root(_):
            return web.FileResponse(os.path.join(static_path, "index.html"))
        app.router.add_get("/", root)
        app.router.add_static("/", path=static_path, follow_symlinks=True)
    ssl_context = None
    if args.ssl is not None:
        ssl_context, _ = create_ssl_context(args.ssl)
    host_ip = args.host if args.host not in ("0.0.0.0", "::") else get_lan_ip()
    logger.info("serving RAG student at http://%s:%d", host_ip, args.port)
    web.run_app(app, port=args.port, ssl_context=ssl_context)


if __name__ == "__main__":
    with torch.no_grad():
        main()
