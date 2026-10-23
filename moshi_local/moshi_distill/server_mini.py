import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import random
import os
from pathlib import Path
import tarfile
import time
import secrets
import sys
from typing import Literal, Optional
import json

import aiohttp
from aiohttp import web
from huggingface_hub import hf_hub_download
import numpy as np
import sentencepiece
import sphn
import torch
import faiss
import re
from transformers import AutoModel, AutoTokenizer
_SYMBOL_RE = re.compile(r'[*_#`~]+')

from .client_utils import make_log, colorize
from .models import loaders, MimiModel, LMModel, LMGen
from .utils.connection import create_ssl_context, get_lan_ip
from .utils.logging import setup_logger, ColorizedLog
import moshi.models

from .models.lm import SILENCE_TOKENS, SINE_TOKENS, AUDIO_TOKENS_PER_STREAM
from .train_latent import RefResampler, add_special_tokens_and_resize


logger = setup_logger(__name__)
DeviceString = Literal["cuda"] | Literal["cpu"]


def torch_auto_device(requested: Optional[DeviceString] = None) -> torch.device:
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def wrap_with_system_tags(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


def lookup_id(tokenizer) -> int:
    return tokenizer.piece_to_id("<lookup>")


# ── RAG helpers ──────────────────────────────────────────────────────────

from transformers import StoppingCriteria, StoppingCriteriaList

class _SentenceEndCriteria(StoppingCriteria):
    def __init__(self, tokenizer, prompt_len, min_new=6):
        self.prompt_len = prompt_len
        self.min_new = min_new
        end_marks = tuple(".!?\n")
        self.end_token_ids = set()
        for piece, idx in tokenizer.get_vocab().items():
            if piece.rstrip("▁").endswith(end_marks):
                self.end_token_ids.add(idx)

    def __call__(self, input_ids, scores, **kwargs):
        new_len = input_ids.shape[1] - self.prompt_len
        if new_len < self.min_new:
            return False
        return input_ids[0, -1].item() in self.end_token_ids

        
def load_rag_index(index_dir: str):
    manifest_path = os.path.join(index_dir, "manifest.json")
    chunks_path = os.path.join(index_dir, "chunks.npz")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest.get("mode") == "text", (
        f"Index at {index_dir} has mode='{manifest.get('mode')}'. "
        f"Rebuild with --mode text."
    )
    arrays = np.load(chunks_path)
    embeddings = arrays["text_embeddings"].astype(np.float32)
    chunks = manifest["chunks"]
    embedding_model_name = manifest.get("embedding_model", "all-MiniLM-L6-v2")
    return chunks, embeddings, embedding_model_name


def retrieve_chunks_fast(
    transcript: str, chunks: list, faiss_index, embedding_model, top_k: int, min_score: float = 0.45
) -> list[dict]:
    """Fast retrieval using FAISS."""
    if not transcript.strip():
        return []
    query_vec = embedding_model.encode(transcript, normalize_embeddings=True).astype(np.float32)
    query_vec = query_vec.reshape(1, -1)
    scores, indices = faiss_index.search(query_vec, top_k)
    results = []
    for idx, score in zip(indices[0], scores[0]):
        if score < min_score:
            continue
        chunk = dict(chunks[idx])
        chunk["similarity_score"] = float(score)
        results.append(chunk)
    return results

def retrieve_chunks_hierarchical_fast(
    transcript: str, chunks: list, faiss_index, embeddings_t: torch.Tensor,
    embedding_model, top_k: int, min_score: float = 0.40,
    sibling_min_score: Optional[float] = None,
) -> list[dict]:
    """
    Hierarchical retrieval using FAISS for base hits, then sibling expansion
    with GPU dot product.
    """
    base_hits = retrieve_chunks_fast(transcript, chunks, faiss_index, embedding_model, top_k, min_score)
    if not base_hits:
        return base_hits

    sibling_min_score = min_score if sibling_min_score is None else sibling_min_score

    # Encode query once
    query_vec = embedding_model.encode(transcript, normalize_embeddings=True).astype(np.float32)
    query_vec_t = torch.from_numpy(query_vec).to(embeddings_t.device)

    # Build source -> list of (chunk_index, chunk_dict)
    by_source: dict[str, list] = {}
    id_to_idx: dict[str, int] = {}
    for i, c in enumerate(chunks):
        by_source.setdefault(c["source"], []).append((i, c))
        id_to_idx[c["id"]] = i

    expanded = []
    seen_ids = set()

    for hit in base_hits:
        if hit["id"] not in seen_ids:
            expanded.append(hit)
            seen_ids.add(hit["id"])

        src = hit["source"]
        siblings = by_source[src]
        pos = next(j for j, (_, c) in enumerate(siblings) if c["id"] == hit["id"])

        # Only consider immediate next sibling (pos+1) – original logic
        sib_pos = pos + 1
        if sib_pos >= len(siblings):
            continue
        sib_idx, sib_chunk = siblings[sib_pos]
        if sib_chunk["id"] in seen_ids:
            continue

        # Compute score on GPU
        sib_emb = embeddings_t[sib_idx]          # shape (dim,)
        sib_score = float((sib_emb @ query_vec_t).cpu().item())
        if sib_score < sibling_min_score:
            continue

        merged = dict(sib_chunk)
        merged["similarity_score"] = sib_score
        expanded.append(merged)
        seen_ids.add(sib_chunk["id"])

    expanded.sort(key=lambda c: c["similarity_score"], reverse=True)
    return expanded


def summarize_context(
    transcript: str, retrieved_chunks: list[dict], embedding_model,
    max_sentences: int = 3, max_chars: int = 400,
) -> str:
    """Extractive fallback summarizer — used only if the ContextCompressor
    LLM isn't loaded or fails."""
    import re
    full_text = " ".join(c["text"] for c in retrieved_chunks)
    sentences = re.split(r'(?<=[.!?])\s+', full_text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
    if not sentences:
        return full_text[:max_chars]

    query_vec = embedding_model.encode(transcript, normalize_embeddings=True).astype(np.float32)
    sent_vecs = embedding_model.encode(sentences, normalize_embeddings=True).astype(np.float32)
    scores = sent_vecs @ query_vec

    top_indices = sorted(np.argsort(scores)[::-1][:max_sentences])
    summary = " ".join(sentences[i] for i in top_indices)
    return summary[:max_chars]


# ── Web search helper ────────────────────────────────────────────────────

async def web_search_query(
    query: str,
    api_key: Optional[str],
    provider: str,
    max_results: int,
    timeout: float,
) -> list[dict]:
    """Fire an async web search and normalize results into the same
    chunk-dict shape used by local RAG retrieval, so they can be merged
    and fed straight into ContextCompressor.compress() unchanged.

    A fresh ClientSession is used per call (rather than a session held on
    ServerState) to sidestep event-loop lifetime issues — ServerState is
    constructed in main() before web.run_app() creates/owns its loop, so
    binding a long-lived session at __init__ time would risk attaching it
    to the wrong (or no) running loop.
    """
    if not api_key:
        logger.warning("web search requested but no API key configured — skipping")
        return []

    try:
        timeout_cfg = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
            if provider == "tavily":
                async with session.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": api_key,
                        "query": query,
                        "max_results": max_results,
                        "search_depth": "basic",
                    },
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                raw_results = data.get("results", [])[:max_results]
                return [
                    {
                        "id": f"web-{i}",
                        "source": r.get("url", "web"),
                        "text": r.get("content", "")[:1000],
                        "similarity_score": float(r.get("score", 0.5)),
                    }
                    for i, r in enumerate(raw_results)
                ]

            elif provider == "serper":
                async with session.post(
                    "https://google.serper.dev/search",
                    headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
                    json={"q": query, "num": max_results},
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                raw_results = data.get("organic", [])[:max_results]
                return [
                    {
                        "id": f"web-{i}",
                        "source": r.get("link", "web"),
                        "text": r.get("snippet", "")[:1000],
                        "similarity_score": max(0.5, 0.9 - 0.05 * i),
                    }
                    for i, r in enumerate(raw_results)
                ]

            elif provider == "bing":
                async with session.get(
                    "https://api.bing.microsoft.com/v7.0/search",
                    headers={"Ocp-Apim-Subscription-Key": api_key},
                    params={"q": query, "count": max_results},
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                raw_results = data.get("webPages", {}).get("value", [])[:max_results]
                return [
                    {
                        "id": f"web-{i}",
                        "source": r.get("url", "web"),
                        "text": r.get("snippet", "")[:1000],
                        "similarity_score": max(0.5, 0.9 - 0.05 * i),
                    }
                    for i, r in enumerate(raw_results)
                ]

            else:
                logger.error(f"unknown web search provider '{provider}'")
                return []

    except asyncio.TimeoutError:
        logger.warning(f"web search timed out after {timeout}s for query: {query!r}")
        return []
    except aiohttp.ClientError as e:
        logger.warning(f"web search request failed: {e}")
        return []
    except Exception as e:
        logger.warning(f"web search unexpected error: {e}")
        return []


# ── STT helpers (unchanged) ─────────────────────────────────────────────

def decode_stt_tokens(text_tokens: list[torch.Tensor], tokenizer, padding_token_id: int) -> str:
    if not text_tokens:
        return ""
    all_tokens = torch.cat(text_tokens, dim=-1)
    all_tokens = all_tokens.cpu().view(-1)
    valid = all_tokens[all_tokens > padding_token_id]
    if valid.numel() == 0:
        return ""
    return tokenizer.decode(valid.tolist())

_FADE_SAMPLES = 240  

def _fade_to_silence(pcm: np.ndarray) -> np.ndarray:
    """Ramp the tail of a real audio frame down to zero so the following
    hard-silence frame doesn't create a sample discontinuity."""
    out = pcm.copy()
    n = min(_FADE_SAMPLES, out.shape[-1])
    ramp = np.linspace(1.0, 0.0, n, dtype=np.float32)
    out[-n:] *= ramp
    return out

def _fade_from_silence(pcm: np.ndarray) -> np.ndarray:
    """Ramp the head of a real audio frame up from zero so it doesn't pop
    in after a run of hard-silence frames."""
    out = pcm.copy()
    n = min(_FADE_SAMPLES, out.shape[-1])
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    out[:n] *= ramp
    return out


# ── Context compressor (small LLM: query + top-k hits -> 1-2 sentence grounding) ──
_URL_RE = re.compile(r'https?://\S+|www\.\S+')
_DIRECTIVE_RE = re.compile(
    r'\b(visit|check out|click here|for more information|see here|refer to|go to)\b',
    re.IGNORECASE
)
_NO_INFO_PATTERNS = re.compile(
    r'\b(does not (provide|contain|mention)|no (specific )?(information|context)( provided)?|'
    r'not (mentioned|addressed|found))\b',
    re.IGNORECASE
)

class ContextCompressor:
    COMPRESS_PROMPT = (
        "Answer in 1 short spoken sentence, plain text, no markdown, no lead-in. "
        "Speak the fact directly, as if you already know it — never mention "
        "URLs, links, websites, or phrases like 'visit', 'check out', 'you can "
        "find', 'according to', 'for more information'. Reasonable inference "
        "from the passages is fine; invented facts, numbers, or specifics not "
        "stated in the passages are not. "
        "If the passages don't answer the question, reply NO_CONTEXT.\n\n"
        "Conversation so far:\n{history}\n\n"
        "Q: {question}\nPassages:\n{passages}\nA:"
    )

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str = "cuda",
        max_new_tokens: int = 24,
        quantize_4bit: bool = True,
        max_passages: int = 2,
    ):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        logger.info(f"[compressor] loading {model_name} on {device} (4bit={quantize_4bit}) ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.getenv("HF_TOKEN"))
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        common_kwargs = dict(
            token=os.getenv("HF_TOKEN"),
            attn_implementation="sdpa",
        )
        if quantize_4bit and device != "cpu":
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, quantization_config=bnb_config, device_map=device, **common_kwargs,
            )
        else:
            if quantize_4bit and device == "cpu":
                logger.warning("[compressor] quantize_4bit requested but device=cpu — falling back to fp32 on CPU.")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.float16 if device != "cpu" else torch.float32,   # `dtype`, not `torch_dtype` — see below
                device_map=device, **common_kwargs,
            )
        self.model.eval()
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.max_passages = max_passages
        self.eos_ids = [self.tokenizer.eos_token_id]
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="compressor")
        logger.info(f"[compressor] ready — {sum(p.numel() for p in self.model.parameters())/1e9:.2f}B params")
        # No prefix-cache setup — reverted, see reasoning above.

    def compress(self, question: str, history: list[tuple[str, str]], chunks: list[dict]) -> str:
        if not chunks:
            return ""
        passages = "\n".join(
            f"[{i+1}] {c['text'][:200]}"
            for i, c in enumerate(chunks[: min(self.max_passages, 2)])
        )
        history_text = "\n".join(f"User: {u}\nMoshi: {m}" for u, m in history[-2:]) if history else "None"
        user_content = self.COMPRESS_PROMPT.format(history=history_text, question=question, passages=passages)

        messages = [{"role": "user", "content": user_content}]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        prompt_len = inputs["input_ids"].shape[1]

        stopping = StoppingCriteriaList([_SentenceEndCriteria(self.tokenizer, prompt_len)])
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.eos_ids,
                stopping_criteria=stopping,
            )
        new_tokens = out[0, prompt_len:]
        result = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        result = _SYMBOL_RE.sub('', result).strip()
        result = re.sub(r'^[\-•\d\.\)]+\s*', '', result)

        if "NO_CONTEXT" in result or not result or _NO_INFO_PATTERNS.search(result):
            return ""
        if _URL_RE.search(result) or _DIRECTIVE_RE.search(result):
            return ""

        passage_words = set(w.lower() for c in chunks[: self.max_passages] for w in c["text"].split())
        answer_words = set(w.lower().strip(".,!?") for w in result.split())
        overlap = len(answer_words & passage_words) / max(1, len(answer_words))
        if overlap < 0.15:
            return ""

        return result

    async def compress_async(self, question, history, chunks, loop):
        return await loop.run_in_executor(self._executor, self.compress, question, history, chunks)

@torch.no_grad()
def merge_lora_for_inference(lm_base) -> None:
    """Bakes each attached LoRAWeightDelta into the underlying weight and
    removes the parametrization — matching doc33's merge_and_unload()
    intent. Training needs the live parametrization (gradients flow
    through lora_A/lora_B); inference needs zero per-step recompute
    overhead, especially since forward_depformer re-triggers this
    dep_q times per generation step."""
    from torch.nn.utils.parametrize import remove_parametrizations
    n = 0
    for module in lm_base.modules():
        if hasattr(module, "parametrizations") and "in_proj_weight" in module.parametrizations:
            remove_parametrizations(module, "in_proj_weight", leave_parametrized=True)
            n += 1
        if hasattr(module, "parametrizations") and "weight" in getattr(module, "parametrizations", {}):
            remove_parametrizations(module, "weight", leave_parametrized=True)
    logger.info(f"merged LoRA into base weights for zero-overhead inference ({n} attention modules)")

# ── ServerState ──────────────────────────────────────────────────────────

@dataclass
class ServerState:
    mimi: MimiModel
    other_mimi: MimiModel
    text_tokenizer: sentencepiece.SentencePieceProcessor
    lm_gen: LMGen
    lock: asyncio.Lock

    def __init__(
        self,
        mimi: MimiModel,
        other_mimi: MimiModel,
        text_tokenizer: sentencepiece.SentencePieceProcessor,
        lm: LMModel,
        mimi_weight,
        device: str | torch.device,
        voice_prompt_dir: str | None = None,
        save_voice_prompt_embeddings: bool = False,
        rag_index_dir: Optional[str] = None,
        rag_top_k: int = 2,
        rag_min_score: float = 0.40,
        stt_hf_repo: str = "kyutai/stt-1b-en_fr-candle",
        vad_threshold: float = 0.5,
        compressor_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        compressor_device: str = "cuda",
        compressor_4bit: bool = True,
        n_ref_slots: int = 16,
        max_ref_source_tokens: int = 400,
        ref_text_slots: int = 32,
        ref_encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        manual_lookup_fallback_seconds: float = 1.5,
        rag_trigger_score: float = 0.40,
        web_search_enabled: bool = False,
        web_search_api_key: Optional[str] = None,
        web_search_provider: str = "tavily",
        web_search_max_results: int = 3,
        web_search_trigger_below: float = 0.45,
        web_search_timeout: float = 3.0,
        compressor_max_passages: int = 2,
    ):
        self.mimi = mimi
        self.other_mimi = other_mimi
        self.text_tokenizer = text_tokenizer
        self.device = device
        self.voice_prompt_dir = voice_prompt_dir
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        self.lm_gen = LMGen(
            lm,
            audio_silence_frame_cnt=int(0.5 * self.mimi.frame_rate),
            sample_rate=self.mimi.sample_rate,
            device=device,
            frame_rate=self.mimi.frame_rate,
            save_voice_prompt_embeddings=save_voice_prompt_embeddings,
        )
        self.lock = asyncio.Lock()
        self.mimi.streaming_forever(1)
        self.other_mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)

        self.vad_threshold = vad_threshold

        self.gpu_executor = ThreadPoolExecutor(max_workers=1)

        # RAG + STT state
        self.rag_enabled = rag_index_dir is not None
        self.rag_top_k = rag_top_k
        self.rag_min_score = rag_min_score
        self.rag_chunks = []
        self.rag_embeddings = None
        self.rag_embedding_model = None
        self.stt_lm_gen = None
        self.stt_tokenizer = None
        self.stt_padding_token_id = 3
        self.context_compressor = None
        self.n_ref_slots = n_ref_slots                       
        self.max_ref_source_tokens = max_ref_source_tokens
        # NOTE: ref_text_slots is kept only for CLI backward-compatibility.
        # The current training pipeline (train_latent.py) injects a fixed
        # number of *continuous* resampler vectors (n_ref_slots) at ref-slot
        # frames rather than a fixed number of discrete grounding tokens, so
        # this value no longer drives the injected block width — n_ref_slots
        # does (see _inject_ref_block_sync).
        self.ref_text_slots = ref_text_slots
        self.ref_encoder_name = ref_encoder_name
        self.manual_lookup_fallback_seconds = manual_lookup_fallback_seconds
        self.resampler: Optional[RefResampler] = None
        self.ref_tokenizer = None
        self.rag_trigger_score = rag_trigger_score

        if self.rag_enabled and self.rag_min_score > self.rag_trigger_score:
            logger.warning(
                f"--rag-min-score ({self.rag_min_score}) is greater than "
                f"--rag-trigger-score ({self.rag_trigger_score}) — hits will be "
                f"filtered out by min_score before the trigger check ever sees "
                f"them, effectively disabling injection. Lower rag_min_score or "
                f"raise rag_trigger_score."
            )

        # Web search state
        self.web_search_enabled = web_search_enabled
        self.web_search_api_key = web_search_api_key
        self.web_search_provider = web_search_provider
        self.web_search_max_results = web_search_max_results
        self.web_search_trigger_below = web_search_trigger_below
        self.web_search_timeout = web_search_timeout
        if self.web_search_enabled and not self.web_search_api_key:
            logger.warning(
                "--web-search-enabled set but no API key found "
                "(--web-search-api-key / SEARCH_API_KEY env var) — "
                "web search will no-op at request time."
            )

        self.pre_ref_silence_frames = max(2, int(0.1 * self.mimi.frame_rate))
        self.post_ref_silence_frames = max(2, int(0.15 * self.mimi.frame_rate))

        if self.rag_enabled:
            try:
                logger.info(f"loading RAG index from {rag_index_dir}...")
                chunks, embeddings, emb_model_name = load_rag_index(rag_index_dir)
                self.rag_chunks = chunks
                self.rag_embeddings = embeddings
                self.rag_embedding_model_name = emb_model_name
                logger.info(f"RAG index loaded: {len(self.rag_chunks)} chunks")
            except Exception as e:
                logger.error(f"RAG disabled — index load failed: {e}")
                self.rag_enabled = False

        if self.rag_enabled:
            try:
                from sentence_transformers import SentenceTransformer
                device_for_emb = "cuda" if torch.cuda.is_available() else "cpu"
                self.rag_embedding_model = SentenceTransformer(
                    self.rag_embedding_model_name, device=device_for_emb
                )
                self.rag_embeddings_t = torch.from_numpy(self.rag_embeddings).to(device_for_emb)
                dim = self.rag_embeddings.shape[1]
                index = faiss.IndexFlatIP(dim)
                index.add(self.rag_embeddings.astype(np.float32))
                self.rag_index = index
                self.rag_device = device_for_emb
                logger.info(f"RAG index built with {len(self.rag_chunks)} chunks on {device_for_emb}")
            except Exception as e:
                logger.error(f"RAG disabled – index/embedding setup failed: {e}")
                self.rag_enabled = False

        if self.rag_enabled:
            try:
                logger.info(f"loading STT model from {stt_hf_repo}...")
                stt_info = moshi.models.loaders.CheckpointInfo.from_hf_repo(stt_hf_repo)
                self.stt_mimi = stt_info.get_mimi(device=device)
                stt_lm = stt_info.get_moshi(device=device, dtype=torch.bfloat16)
                stt_lm.eval()
                self.stt_lm_gen = moshi.models.LMGen(stt_lm, temp=0, temp_text=0.0)
                self.stt_lm_gen.streaming_forever(1)
                self.stt_mimi.streaming_forever(1)
                self.stt_tokenizer = stt_info.get_text_tokenizer()
                self.stt_padding_token_id = stt_info.raw_config.get("text_padding_token_id", 3)
                logger.info(
                    f"STT model loaded: "
                    f"params={sum(p.numel() for p in stt_lm.parameters()) / 1e9:.1f}B "
                    f"padding_id={self.stt_padding_token_id}"
                )
            except Exception as e:
                logger.error(f"RAG disabled — STT model load failed: {e}")
                self.rag_enabled = False

        if self.rag_enabled:
            try:
                self.context_compressor = ContextCompressor(
                    model_name=compressor_model_name,
                    device=compressor_device,
                    quantize_4bit=compressor_4bit,
                    max_passages=compressor_max_passages,
                )
            except Exception as e:
                logger.error(f"context compressor disabled — load failed: {e} "
                              f"(falling back to extractive summarize_context)")
                self.context_compressor = None
        if self.context_compressor is not None:
            try:
                t0 = time.monotonic()
                _ = self.context_compressor.compress(
                    question="What is the deferred tax liability?",
                    history=[],
                    chunks=[{"id": "warmup", "source": "warmup", "text": "Deferred tax liability is 1.01.", "similarity_score": 1.0}],
                )
                logger.info(f"[compressor] warmup call took {time.monotonic()-t0:.2f}s")
            except Exception as e:
                logger.warning(f"[compressor] warmup call failed (non-fatal): {e}")
    

    def load_lora_and_adapter(self, checkpoint_dir: str, merge_lora: bool = True,
                           lora_rank: int = 128, lora_scaling: float = 2.0) -> None:
        """merge_lora kept for CLI back-compat only — no-op now, see prior note.
        lora_rank/lora_scaling MUST match the values training was run with
        (--lora-rank/--lora-scaling), since they determine the shape of
        lora_A/lora_B — a mismatch here causes either a shape error on load or,
        worse, silent key-mismatch same as the bug this fixes."""
        from pathlib import Path
        import logging
        logger = logging.getLogger(__name__)
        ckpt = Path(checkpoint_dir)
        lm = self.lm_gen.lm_model
    
        # 1. Resize vocab FIRST (unchanged — <lookup> only).
        add_special_tokens_and_resize(lm, self.text_tokenizer, new_tokens=("<lookup>",))
        expected_vocab = lm.text_card  # new_text_card after resize
        actual_vocab = self.text_tokenizer.get_piece_size()
        if expected_vocab != actual_vocab:
            logger.error(
                f"VOCAB MISMATCH: model resized to text_card={expected_vocab} but "
                f"tokenizer has {actual_vocab} pieces. This WILL cause IndexError on "
                f"sampled tokens at inference. Check that --tokenizer points at the "
                f"exact .model file produced by add_ref_tokens_to_spm.py for this "
                f"checkpoint, not the base tokenizer or a different variant."
            )
    
        # 2. THEN attach LoRA parametrization — THIS IS THE FIX. Must happen
        # before load_state_dict, or the checkpoint's parametrizations.* keys
        # have nothing to bind to and get silently dropped (strict=False),
        # leaving the model running as the untrained base checkpoint.
        from .train_latent import apply_lora_to_streaming_attention
        apply_lora_to_streaming_attention(lm, rank=lora_rank, alpha=lora_rank * lora_scaling)
    
        lm_state_path = ckpt / "lm_state.pt"
        if lm_state_path.exists():
            logger.info(f"Loading full model state (LoRA + grounding + resized embeddings) from {lm_state_path}")
            state_dict = torch.load(lm_state_path, map_location=self.device)
            missing, unexpected = lm.load_state_dict(state_dict, strict=False)
            merge_lora_for_inference(lm)
    
            # Sanity check: after attaching parametrization, missing/unexpected
            # should be near-zero for lora_A/lora_B/cross_attn keys specifically.
            # If they're not, rank/alpha or architecture still don't match.
            lora_missing = [k for k in missing if "lora_A" in k or "lora_B" in k]
            lora_unexpected = [k for k in unexpected if "lora_A" in k or "lora_B" in k]
            if lora_missing or lora_unexpected:
                logger.error(f"LoRA weights STILL not binding correctly: "
                             f"{len(lora_missing)} lora keys missing, {len(lora_unexpected)} unexpected. "
                             f"Check --lora-rank/--lora-scaling match training exactly.")
            else:
                logger.info(f"LoRA weights bound successfully "
                           f"({sum(1 for k in state_dict if 'lora_A' in k or 'lora_B' in k)} lora tensors loaded)")
    
            if missing:
                logger.warning(f"{len(missing)} other keys missing: {missing[:5]}{'...' if len(missing) > 5 else ''}")
            if unexpected:
                logger.warning(f"{len(unexpected)} other unexpected keys ignored: {unexpected[:5]}")
        else:
            logger.error(f"No lm_state.pt at {lm_state_path} — running with untrained LoRA/grounding (random init).")
    
        base = lm
    
        d_model = base.text_emb.embedding_dim
        model_dtype = next(base.parameters()).dtype
        self.ref_tokenizer = AutoTokenizer.from_pretrained(self.ref_encoder_name)
        self.resampler = RefResampler(
            encoder_name=self.ref_encoder_name, d_model=d_model,
            n_queries=self.n_ref_slots, device=self.device, dtype=model_dtype,
        )
        resampler_path = ckpt / "resampler.pt"
        if resampler_path.exists():
            self.resampler.load_state_dict(torch.load(resampler_path, map_location=self.device))
            self.resampler.eval()
            logger.info(f"loaded resampler from {resampler_path}")
        else:
            logger.error(f"No resampler.pt at {resampler_path} — RAG grounding disabled.")
            self.resampler = None
            self.ref_tokenizer = None

    def warmup(self):
        for _ in range(4):
            chunk = torch.zeros(1, 1, self.frame_size, dtype=torch.float32, device=self.device)
            codes = self.mimi.encode(chunk)
            _ = self.other_mimi.encode(chunk)

            stt_codes = self.stt_mimi.encode(chunk)
            _ = self.stt_lm_gen.step_with_extra_heads(stt_codes)

            for c in range(codes.shape[-1]):
                tokens = self.lm_gen.step(codes[:, :, c: c + 1])
                if tokens is None:
                    continue
                _ = self.mimi.decode(tokens[:, 1:9])
                _ = self.other_mimi.decode(tokens[:, 1:9])

        if self.device.type == 'cuda':
            torch.cuda.synchronize()

    def _step_silence_frames_sync(self, n_frames: int):
        """Shared turn-boundary stepping helper: n forced steps of
        zero agent-audio / zero text / sine user-audio. Used both as the
        pre/post framing around an injected <ref> block, and as the
        *entire* transition cue on turns where retrieval never triggers
        at all. Must run on gpu_executor."""
        for _ in range(n_frames):
            self.lm_gen.step(
                moshi_tokens=self.lm_gen._encode_zero_frame(),
                text_token=self.lm_gen.zero_text_code,
                input_tokens=self.lm_gen._encode_sine_frame(),
            )
    @torch.no_grad()
    def _prime_conditioning_batch_sync(self, ref_content: str) -> int:
        """User-audio-channel (cb_index=8) embedding-overwrite injection,
        matching training exactly. Batched into one transformer forward, no
        depformer (frames are fully forced, never sampled). Explicitly
        replicates process_transformer_output's provided-clear + offset
        advance for LMGen's small lookahead ring buffer (_LMGenState.cache,
        size max_delay+3 — NOT the attention KV cache) — omitting this leaves
        stale provided=True/cache=zero_token_id entries that corrupt the next
        real generation step, which is what previously caused -1 to leak into
        the sampled text channel and crash the session."""
        lm_gen = self.lm_gen
        lm_model = lm_gen.lm_model
        state = lm_gen._streaming_state
        if state is None:
            raise RuntimeError("must be called within lm_gen.streaming(...)")
    
        INJECT_CB = 8  # n_q-indexed user audio channel
    
        vecs = self._encode_reference_for_resampler(ref_content)  # [n_ref_slots, dim]
        n = vecs.shape[0]
    
        input_frames = []
        for _ in range(n):
            prepared = None
            while prepared is None:
                prepared = lm_gen.prepare_step_input(
                    input_tokens=lm_gen._encode_sine_frame(),   # user channel content overwritten below anyway
                    moshi_tokens=lm_gen._encode_zero_frame(),
                    text_token=lm_model.zero_text_code if hasattr(lm_model, "zero_text_code") else 3,  # ordinary PAD — text untouched now
                )
            input_, provided_, target_, model_input_position, target_position = prepared
            input_frames.append(input_)
            state.provided[:, :, model_input_position] = False
            state.offset += 1
    
        batch_codes = torch.cat(input_frames, dim=2)   # [1, K, n]
        audio_sum = None
        for cb_index in range(lm_model.num_audio_codebooks):
            audio_emb = lm_model.emb[cb_index](batch_codes[:, cb_index + lm_model.audio_offset])
            if cb_index == INJECT_CB:
                audio_emb = audio_emb + vecs.unsqueeze(0)   # overwrite user-audio channel
            audio_sum = audio_emb if audio_sum is None else audio_sum + audio_emb
        text_emb = lm_model.text_emb(batch_codes[:, 0])
        embedded = audio_sum + text_emb
    
        lm_model.transformer(embedded)
    
        return n

    def _encode_reference_for_resampler(self, ref_content: str) -> torch.Tensor:
        """Raw reference text -> [n_ref_slots, dim] resampler output.
        Mirrors train_latent._build_ref_injection_tensors exactly: tokenize
        with the encoder's own HF tokenizer (NOT the SentencePiece
        text_tokenizer used for the main sequence), padding="max_length" to
        max_ref_source_tokens. max_ref_source_tokens is a compute-only cap
        on the resampler's *input* — it does NOT affect how many frames get
        injected into the main sequence (always n_ref_slots, fixed)."""
        enc = self.ref_tokenizer(
            ref_content, padding="max_length", truncation=True,
            max_length=self.max_ref_source_tokens, return_tensors="pt",
        )
        tok_ids = enc["input_ids"].to(self.device)
        mask = enc["attention_mask"].to(self.device).bool()
        with torch.no_grad():
            vecs = self.resampler(tok_ids, mask)   # [1, n_ref_slots, dim]
        return vecs[0]

    

    async def handle_chat(self, request):
        ws = web.WebSocketResponse(heartbeat=10)
        await ws.prepare(request)
        clog = ColorizedLog.randomize()
        peer = request.remote
        peer_port = request.transport.get_extra_info("peername")[1]
        clog.log("info", f"Incoming connection from {peer}:{peer_port}")
        clog.log("info",
            f"RAG enabled={self.rag_enabled} chunks={len(self.rag_chunks)} "
            f"vad_threshold={self.vad_threshold} web_search={self.web_search_enabled}"
        )

        requested_voice_prompt_path = None
        voice_prompt_path = None
        if self.voice_prompt_dir is not None:
            voice_prompt_filename = request.query.get("voice_prompt")
            if not voice_prompt_filename:
                raise web.HTTPBadRequest(
                    reason="Missing required 'voice_prompt' query parameter"
                )
            requested_voice_prompt_path = os.path.join(self.voice_prompt_dir, voice_prompt_filename)
            if not os.path.exists(requested_voice_prompt_path):
                raise FileNotFoundError(
                    f"Voice prompt '{voice_prompt_filename}' not found in '{self.voice_prompt_dir}'"
                )
            voice_prompt_path = requested_voice_prompt_path

        if voice_prompt_path is not None and self.lm_gen.voice_prompt != voice_prompt_path:
            if voice_prompt_path.endswith('.pt'):
                self.lm_gen.load_voice_prompt_embeddings(voice_prompt_path)
            else:
                self.lm_gen.load_voice_prompt(voice_prompt_path)

        base_text_prompt = request.query.get("text_prompt", "")
        seed = int(request.query["seed"]) if "seed" in request.query else None

        # ── Per-session state ────────────────────────────────────────────
        listening = True
        generating = False
        injecting = False
        model_consecutive_pads = 0
        MODEL_IDLE_PAD_THRESHOLD = 25

        stt_token_buffer = []
        stt_in_utterance = False
        stt_silence_frame_count = 0
        stt_last_vad_end = False
        VAD_SILENCE_FRAMES_REQUIRED = 12

        BARGE_IN_FRAMES_REQUIRED = 4
        barge_in_frame_count = 0

        turn_epoch = 0
        ref_committed_this_turn = False

        awaiting_ref = False
        pending_ref_content = None      
        filler_frame_count = 0
        MAX_FILLER_FRAMES = int(2.0 * self.mimi.frame_rate)
        last_frame_was_silence = True

        # ── <lookup>-driven RAG state ───────────────────────────────────
        # Instead of the server forcing a <lookup> token whenever a cheap
        # heuristic thinks a turn needs grounding, the model — trained on
        # <lookup> as a real vocabulary token — is left to emit it itself
        # during normal generation. The heuristic below still runs (cheap,
        # local-only) but now only decides whether to (a) speculatively
        # warm up background retrieval so it's ready the instant the model
        # asks, and (b) arm a manual fallback that force-injects <lookup>
        # if the model stays confidently silent about it for too long.
        LOOKUP_ID = lookup_id(self.text_tokenizer)
        MANUAL_LOOKUP_FALLBACK_FRAMES = max(1, int(self.manual_lookup_fallback_seconds * self.mimi.frame_rate))
        rag_quick_score = 0.0
        rag_lexical_hit = False
        frames_since_turn_start = 0
        lookup_prep_started = False

        # Conversation history for the compressor + response-text accumulation
        session_history: list[tuple[str, str]] = []
        current_transcript = ""
        response_text_buffer = ""
        last_real_pcm: Optional[np.ndarray] = None

        gpu_executor = self.gpu_executor

        async def send_context_window(retrieved: list[dict], transcript: str):
            clog.log("info", "─" * 60)
            clog.log("info", "RAG CONTEXT WINDOW")
            clog.log("info", f"transcript: \"{transcript}\"")
            for i, c in enumerate(retrieved):
                clog.log("info", f"  [{i+1}] score={c['similarity_score']:.3f} source={c['source']}")
                clog.log("info", f"       \"{c['text'][:100]}\"")
            clog.log("info", "─" * 60)
            payload = {
                "transcript": transcript,
                "chunks": [
                    {"id": c["id"], "text": c["text"], "source": c["source"],
                     "score": round(c["similarity_score"], 3)}
                    for c in retrieved
                ],
            }
            msg = b"\x03" + json.dumps(payload).encode("utf-8")
            try:
                await ws.send_bytes(msg)
            except Exception:
                pass

        async def start_generation_turn(transcript: str):
            """Called when STT+VAD detects the user has finished speaking.
            If RAG confidence is high enough, we immediately hold generation and
            prepare a grounded reference block. The model will not produce any
            response until the reference is injected."""
            nonlocal listening, generating, injecting, current_transcript
            nonlocal awaiting_ref, pending_ref_content, filler_frame_count
            nonlocal turn_epoch, ref_committed_this_turn
        
            if not listening:
                return
        
            listening = False
            generating = True
            awaiting_ref = False
            pending_ref_content = None
            ref_committed_this_turn = False
            filler_frame_count = 0
            current_transcript = transcript
            turn_epoch += 1
            my_epoch = turn_epoch
            loop = asyncio.get_event_loop()
        
            def _quick_score():
                if not self.rag_chunks:
                    return 0.0, False
                q = self.rag_embedding_model.encode(transcript, normalize_embeddings=True).astype(np.float32)
                k = min(3, len(self.rag_chunks))
                scores, indices = self.rag_index.search(q.reshape(1, -1), k)
                valid = [s for s in scores[0] if s > -1e9]
                top = float(max(valid)) if valid else 0.0
        
                query_words = {w.lower() for w in transcript.split() if len(w) > 3}
                lexical_hit = False
                for idx in indices[0][:k]:
                    if idx < 0:
                        continue
                    chunk_words = {w.lower() for w in self.rag_chunks[idx]["text"].split() if len(w) > 3}
                    if len(query_words & chunk_words) >= 2:
                        lexical_hit = True
                        break
                return top, lexical_hit
        
            QUICK_GATE_MARGIN = 0.03
            top_score, lexical_hit = await loop.run_in_executor(None, _quick_score)
            clog.log("info", f"quick_score={top_score:.3f} lexical_hit={lexical_hit} "
                              f"gate={self.rag_min_score - QUICK_GATE_MARGIN:.3f} transcript={transcript!r}")
        
            if my_epoch != turn_epoch:
                return
        
            # Always play turn‑boundary silence framing
            def _turn_boundary_sync():
                self._step_silence_frames_sync(self.pre_ref_silence_frames)
            await loop.run_in_executor(gpu_executor, _turn_boundary_sync)
            if my_epoch != turn_epoch:
                return
        
            # Decide whether to use grounding
            use_rag = (top_score >= (self.rag_min_score - QUICK_GATE_MARGIN)) or lexical_hit
        
            if not use_rag:
                # Low confidence – let the model generate freely without grounding
                clog.log("info", "low retrieval confidence – skipping grounding")
                return
        
            # Force grounding: hold generation and start retrieval
            clog.log("info", f"high retrieval confidence – grounding will be injected before response")
            awaiting_ref = True
            ref_committed_this_turn = False
            filler_frame_count = 0
            # Launch the retrieval + compression task asynchronously
            asyncio.create_task(_prepare_ref_block(transcript, loop, my_epoch))

        async def _prepare_ref_block(transcript: str, loop, my_epoch: int):
            nonlocal pending_ref_content
        
            def _retrieve_local():
                return retrieve_chunks_hierarchical_fast(
                    transcript, self.rag_chunks, self.rag_index, self.rag_embeddings_t,
                    self.rag_embedding_model, self.rag_top_k, min_score=self.rag_min_score,
                )
            t_retrieve0 = time.monotonic()
            hits = await loop.run_in_executor(None, _retrieve_local)
            clog.log("info", f"[epoch {my_epoch}] TIMING local_retrieval={time.monotonic()-t_retrieve0:.3f}s hits={len(hits)}")
            for h in hits:
                clog.log("info", f"[epoch {my_epoch}]   local hit score={h['similarity_score']:.3f} source={h['source']} text={h['text'][:80]!r}")
        
            top_score = max((h["similarity_score"] for h in hits), default=0.0)
        
            used_web = False
            if self.web_search_enabled and top_score < self.web_search_trigger_below:
                t_web0 = time.monotonic()
                web_hits = await web_search_query(
                    query=transcript, api_key=self.web_search_api_key,
                    provider=self.web_search_provider, max_results=self.web_search_max_results,
                    timeout=self.web_search_timeout,
                )
                clog.log("info", f"[epoch {my_epoch}] TIMING web_search={time.monotonic()-t_web0:.3f}s hits={len(web_hits)}")
                if web_hits:
                    hits = sorted(web_hits, key=lambda c: c["similarity_score"], reverse=True)[: self.web_search_max_results]
                    used_web = True
                    for h in hits:
                        clog.log("info", f"[epoch {my_epoch}]   web hit score={h['similarity_score']:.3f} source={h['source']} text={h['text'][:80]!r}")
        
            if my_epoch != turn_epoch:
                clog.log("info", f"[epoch {my_epoch}] discarding stale grounding (current={turn_epoch})")
                return
        
            grounding = ""
            if hits and self.context_compressor is not None:
                t_compress0 = time.monotonic()
                grounding = await self.context_compressor.compress_async(
                    question=transcript, history=session_history[-2:], chunks=hits, loop=loop,
                )
                clog.log("info",
                    f"[epoch {my_epoch}] TIMING compress={time.monotonic()-t_compress0:.3f}s "
                    f"source={'web' if used_web else 'local'} result={grounding[:150]!r}")
            if not grounding and hits:
                t_extract0 = time.monotonic()
                grounding = await loop.run_in_executor(
                    None, lambda: summarize_context(transcript, hits, self.rag_embedding_model,
                                                      max_sentences=2, max_chars=200)
                )
                clog.log("info", f"[epoch {my_epoch}] TIMING extractive_fallback={time.monotonic()-t_extract0:.3f}s result={grounding[:150]!r}")
        
            if my_epoch != turn_epoch or ref_committed_this_turn:
                clog.log("info", f"[epoch {my_epoch}] discarding late/superseded grounding")
                return
        
            ref_content = grounding.strip() if grounding else (
                "There's no specific information available on this, so answer from general knowledge."
            )
            pending_ref_content = ref_content
            clog.log("info", f"[epoch {my_epoch}] prepared reference ({len(ref_content)} chars): {ref_content[:150]!r}")
            
        async def recv_loop():
            nonlocal close
            try:
                async for message in ws:
                    if message.type == aiohttp.WSMsgType.ERROR:
                        clog.log("error", f"{ws.exception()}")
                        break
                    elif message.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSE):
                        break
                    elif message.type != aiohttp.WSMsgType.BINARY:
                        continue
                    message = message.data
                    if not isinstance(message, bytes) or len(message) == 0:
                        continue
                    kind = message[0]
                    if kind == 1:
                        opus_reader.append_bytes(message[1:])
                    else:
                        clog.log("warning", f"unknown message kind {kind}")
            finally:
                close = True
                clog.log("info", "connection closed")

        async def opus_loop():
            nonlocal listening, generating, injecting
            nonlocal stt_token_buffer, stt_in_utterance, stt_silence_frame_count, stt_last_vad_end
            nonlocal model_consecutive_pads, barge_in_frame_count
            nonlocal response_text_buffer, current_transcript
            nonlocal awaiting_ref, pending_ref_content, filler_frame_count
            nonlocal turn_epoch, ref_committed_this_turn
            nonlocal last_real_pcm, last_frame_was_silence
            nonlocal rag_quick_score, rag_lexical_hit, frames_since_turn_start, lookup_prep_started

            all_pcm_data = None
            loop = asyncio.get_event_loop()
            frame_budget_s = self.frame_size / self.mimi.sample_rate

            try:
                while True:
                    if close:
                        return
                    await asyncio.sleep(0.001)

                    if injecting:
                        pcm = opus_reader.read_pcm()
                        if pcm is not None and pcm.shape[-1] > 0:
                            if all_pcm_data is None:
                                all_pcm_data = pcm
                            else:
                                all_pcm_data = np.concatenate((all_pcm_data, pcm))

                            while all_pcm_data.shape[-1] >= self.frame_size:
                                chunk_pcm = all_pcm_data[:self.frame_size]
                                all_pcm_data = all_pcm_data[self.frame_size:]
                                chunk_tensor = torch.from_numpy(chunk_pcm).to(device=self.device)[None, None]

                                def _injecting_step():
                                    input_codes = self.mimi.encode(chunk_tensor)
                                    _ = self.other_mimi.encode(chunk_tensor)

                                    _ = self.lm_gen.step(
                                        input_tokens=input_codes,
                                        moshi_tokens=self.lm_gen._encode_zero_frame(),
                                        text_token=self.lm_gen.zero_text_code,
                                    )
                                await loop.run_in_executor(gpu_executor, _injecting_step)

                        silent_pcm = np.zeros(self.frame_size, dtype=np.float32)
                        opus_writer.append_pcm(silent_pcm)
                        continue

                    pcm = opus_reader.read_pcm()
                    if pcm is None or pcm.shape[-1] == 0:
                        if generating:
                            pcm = np.zeros(self.frame_size, dtype=np.float32)
                        else:
                            continue
                    if all_pcm_data is None:
                        all_pcm_data = pcm
                    else:
                        all_pcm_data = np.concatenate((all_pcm_data, pcm))

                    while all_pcm_data.shape[-1] >= self.frame_size:
                        if injecting:
                            break

                        chunk_pcm = all_pcm_data[:self.frame_size]
                        all_pcm_data = all_pcm_data[self.frame_size:]
                        chunk_tensor = torch.from_numpy(chunk_pcm).to(device=self.device)[None, None]

                        if listening:
                            def _listen_step():
                                input_codes = self.mimi.encode(chunk_tensor)
                                _ = self.other_mimi.encode(chunk_tensor)
                                stt_res = None
                                if self.rag_enabled and self.stt_lm_gen is not None:
                                    stt_codes = self.stt_mimi.encode(chunk_tensor)
                                    stt_res = self.stt_lm_gen.step_with_extra_heads(stt_codes)

                                _ = self.lm_gen.step(
                                    input_tokens=input_codes,
                                    moshi_tokens=self.lm_gen._encode_zero_frame(),
                                    text_token=self.lm_gen.zero_text_code,
                                )

                                return input_codes, stt_res

                            t_step0 = time.monotonic()
                            input_codes, stt_result = await loop.run_in_executor(gpu_executor, _listen_step)
                            t_step_dt = time.monotonic() - t_step0
                            if t_step_dt > frame_budget_s:
                                clog.log("warning",
                                    f"STEP OVERRUN (listen): {t_step_dt*1000:.1f}ms (budget {frame_budget_s*1000:.1f}ms)")

                            if stt_result is not None:
                                stt_tokens, vad_heads = stt_result
                                vad_score = 0.0
                                if vad_heads and len(vad_heads) > 2:
                                    vad_score = float(vad_heads[2][0, 0, 0].cpu().item())
                                if stt_tokens is not None:
                                    stt_token_buffer.append(stt_tokens[:, :1, :].cpu())
                                    text_token = stt_tokens[0, 0, 0].item()
                                    if text_token not in (0, 3):
                                        stt_in_utterance = True
                                        stt_last_vad_end = False
                                if vad_score > self.vad_threshold:
                                    stt_silence_frame_count += 1
                                else:
                                    stt_silence_frame_count = 0
                                vad_fired = (
                                    stt_silence_frame_count >= VAD_SILENCE_FRAMES_REQUIRED
                                    and not stt_last_vad_end
                                )
                                stt_last_vad_end = stt_silence_frame_count >= VAD_SILENCE_FRAMES_REQUIRED
                                if vad_fired and stt_in_utterance and stt_token_buffer:
                                    transcript = decode_stt_tokens(
                                        stt_token_buffer, self.stt_tokenizer, self.stt_padding_token_id
                                    )
                                    stt_token_buffer = []
                                    stt_in_utterance = False
                                    stt_silence_frame_count = 0
                                    if transcript.strip():
                                        if self.stt_lm_gen is not None:
                                            self.stt_lm_gen.reset_streaming()
                                            self.stt_mimi.reset_streaming()
                                        asyncio.create_task(start_generation_turn(transcript))

                            silent_pcm = np.zeros(self.frame_size, dtype=np.float32)
                            opus_writer.append_pcm(silent_pcm)

                            if injecting:
                                break

                        elif generating:
                            if awaiting_ref:
                                # ---------- HOLD PHASE: model is waiting for grounded reference ----------
                                def _hold_step():
                                    input_codes = self.mimi.encode(chunk_tensor)
                                    _ = self.other_mimi.encode(chunk_tensor)
                                    stt_res = None
                                    if self.rag_enabled and self.stt_lm_gen is not None:
                                        stt_codes = self.stt_mimi.encode(chunk_tensor)
                                        stt_res = self.stt_lm_gen.step_with_extra_heads(stt_codes)
                                    _ = self.lm_gen.step(
                                        input_tokens=input_codes,
                                        moshi_tokens=self.lm_gen._encode_zero_frame(),
                                        text_token=self.lm_gen.zero_text_code,
                                    )
                                    return stt_res
                        
                                t_step0 = time.monotonic()
                                stt_result = await loop.run_in_executor(gpu_executor, _hold_step)
                                t_step_dt = time.monotonic() - t_step0
                                if t_step_dt > frame_budget_s:
                                    clog.log("warning", f"STEP OVERRUN (hold): {t_step_dt*1000:.1f}ms (budget {frame_budget_s*1000:.1f}ms)")
                        
                                opus_writer.append_pcm(np.zeros(self.frame_size, dtype=np.float32))
                                last_frame_was_silence = True
                        
                                # Barge‑in detection (same as before)
                                speech_this_frame = False
                                if stt_result is not None:
                                    stt_tokens, _vad_heads = stt_result
                                    vad_score_bi = 0.0
                                    if _vad_heads and len(_vad_heads) > 2:
                                        vad_score_bi = float(_vad_heads[2][0, 0, 0].cpu().item())
                                    if stt_tokens is not None:
                                        stt_text_token = stt_tokens[0, 0, 0].item()
                                        if stt_text_token not in (0, 3) and vad_score_bi < self.vad_threshold:
                                            speech_this_frame = True
                                barge_in_frame_count = barge_in_frame_count + 1 if speech_this_frame else 0
                                barge_in = barge_in_frame_count >= BARGE_IN_FRAMES_REQUIRED
                                if barge_in:
                                    clog.log("info", "Barge‑in detected during hold – interrupting")
                                    # Reset state and go back to listening
                                    barge_in_frame_count = 0
                                    generating = False
                                    listening = True
                                    model_consecutive_pads = 0
                                    stt_token_buffer = []
                                    stt_in_utterance = True
                                    stt_last_vad_end = False
                                    stt_silence_frame_count = 0
                                    response_text_buffer = ""
                                    turn_epoch += 1
                                    awaiting_ref = False
                                    frames_since_turn_start = 0
                                    self.lm_gen.reset_streaming()
                                    self.mimi.reset_streaming()
                                    self.other_mimi.reset_streaming()
                                    if self.stt_lm_gen is not None:
                                        self.stt_lm_gen.reset_streaming()
                                        self.stt_mimi.reset_streaming()
                                    last_real_pcm = None
                                    last_frame_was_silence = True
                                    opus_writer.append_pcm(np.zeros(self.frame_size, dtype=np.float32))
                                    continue
                        
                                filler_frame_count += 1
                        
                                # If the reference is ready, inject it
                                if pending_ref_content is not None:
                                    ref_content = pending_ref_content
                                    pending_ref_content = None
                                    awaiting_ref = False
                                    ref_committed_this_turn = True
                                    injecting = True
                                    t0 = time.monotonic()
                                    n_frames = await loop.run_in_executor(
                                        gpu_executor, self._prime_conditioning_batch_sync, ref_content
                                    )
                                    clog.log("info", f"TIMING conditioning_prime={time.monotonic()-t0:.3f}s ({n_frames} frames, "
                                                      f"after {filler_frame_count} filler frames)")
                                    silent_frame = np.zeros(self.frame_size, dtype=np.float32)
                                    if last_real_pcm is not None:
                                        opus_writer.append_pcm(_fade_to_silence(last_real_pcm))
                                        n_frames -= 1
                                        last_real_pcm = None
                                    for _ in range(max(0, n_frames)):
                                        opus_writer.append_pcm(silent_frame)
                                    injecting = False
                                    filler_frame_count = 0
                                    frames_since_turn_start = 0
                                    last_frame_was_silence = True
                        
                                elif filler_frame_count >= MAX_FILLER_FRAMES:
                                    clog.log("warning", f"filler exceeded {MAX_FILLER_FRAMES} frames with no grounding ready – forcing fallback")
                                    fallback = "There's no specific information available on this, so answer from general knowledge."
                                    awaiting_ref = False
                                    ref_committed_this_turn = True
                                    injecting = True
                                    t0 = time.monotonic()
                                    n_frames = await loop.run_in_executor(
                                        gpu_executor, self._prime_conditioning_batch_sync, fallback   # was: ref_content (undefined here)
                                    )
                                    clog.log("info", f"TIMING conditioning_prime={time.monotonic()-t0:.3f}s (fallback, {n_frames} frames)")
                                    silent_frame = np.zeros(self.frame_size, dtype=np.float32)
                                    if last_real_pcm is not None:
                                        opus_writer.append_pcm(_fade_to_silence(last_real_pcm))
                                        n_frames -= 1
                                        last_real_pcm = None
                                    for _ in range(max(0, n_frames)):
                                        opus_writer.append_pcm(silent_frame)
                                    injecting = False
                                    filler_frame_count = 0
                                    frames_since_turn_start = 0
                                    last_frame_was_silence = True
                        
                                if injecting:
                                    break
                        
                            else:
                                # ---------- NORMAL GENERATION (no longer waits for <lookup>) ----------
                                def _step_and_decode():
                                    input_codes = self.mimi.encode(chunk_tensor)
                                    _ = self.other_mimi.encode(chunk_tensor)
                                    stt_res = None
                                    if self.rag_enabled and self.stt_lm_gen is not None:
                                        stt_codes = self.stt_mimi.encode(chunk_tensor)
                                        stt_res = self.stt_lm_gen.step_with_extra_heads(stt_codes)
                                    tokens = self.lm_gen.step(input_codes)
                                    if tokens is not None:
                                        main_pcm = self.mimi.decode(tokens[:, 1:9])
                                        _ = self.other_mimi.decode(tokens[:, 1:9])
                                        return tokens, main_pcm.cpu(), input_codes, stt_res
                                    return None, None, input_codes, stt_res
                        
                                t_step0 = time.monotonic()
                                tokens, main_pcm, input_codes, stt_result = await loop.run_in_executor(gpu_executor, _step_and_decode)
                                t_step_dt = time.monotonic() - t_step0
                                if t_step_dt > frame_budget_s:
                                    clog.log("warning", f"STEP OVERRUN (generate): {t_step_dt*1000:.1f}ms (budget {frame_budget_s*1000:.1f}ms)")
                        
                                # Barge‑in detection (unchanged)
                                speech_this_frame = False
                                if stt_result is not None:
                                    stt_tokens, _vad_heads = stt_result
                                    vad_score_bi = 0.0
                                    if _vad_heads and len(_vad_heads) > 2:
                                        vad_score_bi = float(_vad_heads[2][0, 0, 0].cpu().item())
                                    if stt_tokens is not None:
                                        stt_text_token = stt_tokens[0, 0, 0].item()
                                        if stt_text_token not in (0, 3) and vad_score_bi < self.vad_threshold:
                                            speech_this_frame = True
                        
                                barge_in_frame_count = barge_in_frame_count + 1 if speech_this_frame else 0
                                barge_in = barge_in_frame_count >= BARGE_IN_FRAMES_REQUIRED
                        
                                if barge_in:
                                    clog.log("info", f"Barge‑in confirmed – interrupting response")
                                    barge_in_frame_count = 0
                                    generating = False
                                    listening = True
                                    model_consecutive_pads = 0
                                    stt_token_buffer = []
                                    stt_in_utterance = True
                                    stt_last_vad_end = False
                                    stt_silence_frame_count = 0
                                    response_text_buffer = ""
                                    turn_epoch += 1
                                    frames_since_turn_start = 0
                                    self.lm_gen.reset_streaming()
                                    self.mimi.reset_streaming()
                                    self.other_mimi.reset_streaming()
                                    if self.stt_lm_gen is not None:
                                        self.stt_lm_gen.reset_streaming()
                                        self.stt_mimi.reset_streaming()
                                    last_real_pcm = None
                                    last_frame_was_silence = True
                                    opus_writer.append_pcm(np.zeros(self.frame_size, dtype=np.float32))
                                    continue
                        
                                if tokens is not None:
                                    frame = np.clip(main_pcm[0, 0].numpy(), -1.0, 1.0)
                                    if last_frame_was_silence:
                                        frame = _fade_from_silence(frame)
                                        last_frame_was_silence = False
                                    opus_writer.append_pcm(frame)
                                    last_real_pcm = frame
                        
                                    text_token = tokens[0, 0, 0].item()
                                    original_vocab_size = self.text_tokenizer.get_piece_size()
                        
                                    # Treat any non‑pad token as response text (including <lookup> if it appears)
                                    if text_token not in (0, 3):
                                        model_consecutive_pads = 0
                                        frames_since_turn_start += 1
                                    
                                        if text_token == LOOKUP_ID:
                                            if not lookup_prep_started and pending_ref_content is None:
                                                lookup_prep_started = True
                                                awaiting_ref = True
                                                filler_frame_count = 0
                                                print("injecting")
                                                asyncio.create_task(_prepare_ref_block(current_transcript, loop, turn_epoch))
                                        else:
                                            try:
                                                print(text_token)
                                                _text = self.text_tokenizer.id_to_piece(text_token)
                                            except IndexError:
                                                clog.log("error", f"token id {text_token} out of tokenizer vocab range "
                                                                  f"(vocab size {self.text_tokenizer.get_piece_size()}) — "
                                                                  f"this indicates a real vocab/checkpoint mismatch, not "
                                                                  f"normal behavior")
                                                _text = ""
                                            _text = _text.replace("▁", " ")
                                            response_text_buffer += _text
                                            msg = b"\x02" + bytes(_text, encoding="utf8")
                                            await ws.send_bytes(msg)
        
                                    else:
                                        model_consecutive_pads += 1
                                        frames_since_turn_start += 1
                                        if model_consecutive_pads >= MODEL_IDLE_PAD_THRESHOLD:
                                            # End of response
                                            generating = False
                                            listening = True
                                            model_consecutive_pads = 0
                                            barge_in_frame_count = 0
                                            stt_token_buffer = []
                                            stt_in_utterance = False
                                            stt_silence_frame_count = 0
                                            frames_since_turn_start = 0
                                            if response_text_buffer.strip():
                                                session_history.append((current_transcript, response_text_buffer.strip()))
                                                if len(session_history) > 6:
                                                    session_history[:] = session_history[-6:]
                                            response_text_buffer = ""
                                            if self.stt_lm_gen is not None:
                                                self.stt_lm_gen.reset_streaming()
                                                self.stt_mimi.reset_streaming()
                                            if last_real_pcm is not None:
                                                opus_writer.append_pcm(_fade_to_silence(last_real_pcm))
                                                last_real_pcm = None
                                            last_frame_was_silence = True
                                            clog.log("info", "Response finished – back to listening")
                        
                            if injecting:
                                break
            except Exception:
                import traceback
                clog.log("error", "opus_loop crashed")
                clog.log("error", traceback.format_exc())
                raise

        async def send_loop():
            while True:
                if close:
                    return
                await asyncio.sleep(0.001)
                msg = opus_writer.read_bytes()
                if len(msg) > 0:
                    await ws.send_bytes(b"\x01" + msg)

        clog.log("info", "accepted connection")
        if base_text_prompt:
            clog.log("info", f"text prompt: {base_text_prompt}")
        if voice_prompt_path:
            clog.log("info", f"voice prompt: {voice_prompt_path}")

        self.lm_gen.text_prompt_tokens = (
            self.text_tokenizer.encode(wrap_with_system_tags(base_text_prompt))
            if base_text_prompt else None
        )

        close = False
        async with self.lock:
            if seed is not None and seed != -1:
                seed_all(seed)

            opus_writer = sphn.OpusStreamWriter(self.mimi.sample_rate)
            opus_reader = sphn.OpusStreamReader(self.mimi.sample_rate)

            self.mimi.reset_streaming()
            self.other_mimi.reset_streaming()
            self.lm_gen.reset_streaming()

            if self.stt_lm_gen is not None:
                self.stt_lm_gen.reset_streaming()
                self.stt_mimi.reset_streaming()

            async def is_alive():
                if close or ws.closed:
                    return False
                try:
                    msg = await asyncio.wait_for(ws.receive(), timeout=0.01)
                    if msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        return False
                except asyncio.TimeoutError:
                    return True
                except aiohttp.ClientConnectionError:
                    return False
                return True

            loop = asyncio.get_event_loop()

            async def _run_initial_system_prompts():
                def _sync_system_prompts():
                    self.lm_gen.step_system_prompts(self.mimi)
                await loop.run_in_executor(self.gpu_executor, _sync_system_prompts)

            await _run_initial_system_prompts()
            self.mimi.reset_streaming()
            clog.log("info", "done with system prompts")

            if await is_alive():
                await ws.send_bytes(b"\x00")
                clog.log("info", "sent handshake bytes")

                recv_task = asyncio.create_task(recv_loop())
                send_task = asyncio.create_task(send_loop())

                GREETING_MAX_FRAMES = int(8 * self.mimi.frame_rate)
                GREETING_PAD_THRESHOLD = 15

                silence_chunk = torch.zeros(1, 1, self.frame_size, dtype=torch.float32, device=self.device)
                greeting_pads = 0
                clog.log("info", "generating greeting...")

                for _ in range(GREETING_MAX_FRAMES):
                    if close or ws.closed:
                        break

                    def _greeting_step():
                        inp = self.mimi.encode(silence_chunk)
                        _ = self.other_mimi.encode(silence_chunk)
                        tok = self.lm_gen.step(inp)
                        if tok is not None:
                            pcm = self.mimi.decode(tok[:, 1:9])
                            _ = self.other_mimi.decode(tok[:, 1:9])
                            return tok, pcm.cpu()
                        return None, None

                    g_tokens, g_pcm = await loop.run_in_executor(self.gpu_executor, _greeting_step)
                    if g_tokens is None:
                        continue

                    opus_writer.append_pcm(g_pcm[0, 0].numpy())
                    g_text_token = g_tokens[0, 0, 0].item()
                    if g_text_token not in (0, 3):
                        greeting_pads = 0
                        _t = self.text_tokenizer.id_to_piece(g_text_token)
                        _t = _t.replace("▁", " ")
                        await ws.send_bytes(b"\x02" + bytes(_t, encoding="utf8"))
                    else:
                        greeting_pads += 1
                        if greeting_pads >= GREETING_PAD_THRESHOLD:
                            break

                clog.log("info", "greeting complete — entering listening mode")

                opus_task = asyncio.create_task(opus_loop())
                tasks = [recv_task, send_task, opus_task]
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                await ws.close()
                clog.log("info", "session closed")

        clog.log("info", "done with connection")
        return ws


def _get_voice_prompt_dir(voice_prompt_dir: Optional[str], hf_repo: str) -> Optional[str]:
    if voice_prompt_dir is not None:
        return voice_prompt_dir
    logger.info("retrieving voice prompts")
    voices_tgz = hf_hub_download(hf_repo, "voices.tgz")
    voices_tgz = Path(voices_tgz)
    voices_dir = voices_tgz.parent / "voices"
    if not voices_dir.exists():
        logger.info(f"extracting {voices_tgz} to {voices_dir}")
        with tarfile.open(voices_tgz, "r:gz") as tar:
            tar.extractall(path=voices_tgz.parent)
    if not voices_dir.exists():
        raise RuntimeError("voices.tgz did not contain a 'voices/' directory")
    return str(voices_dir)


def _get_static_path(static: Optional[str]) -> Optional[str]:
    if static is None:
        logger.info("retrieving the static content")
        dist_tgz = hf_hub_download("nvidia/personaplex-7b-v1", "dist.tgz")
        dist_tgz = Path(dist_tgz)
        dist = dist_tgz.parent / "dist"
        if not dist.exists():
            with tarfile.open(dist_tgz, "r:gz") as tar:
                tar.extractall(path=dist_tgz.parent)
        return str(dist)
    elif static != "none":
        return static
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost", type=str)
    parser.add_argument("--port", default=8998, type=int)
    parser.add_argument("--static", type=str)
    parser.add_argument("--gradio-tunnel", action='store_true')
    parser.add_argument("--gradio-tunnel-token", type=str)
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--moshi-weight", type=str)
    parser.add_argument("--mimi-weight", type=str)
    parser.add_argument("--hf-repo", type=str, default=loaders.DEFAULT_REPO)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--voice-prompt-dir", type=str)
    parser.add_argument("--ssl", type=str)
    parser.add_argument(
    "--checkpoint-dir", type=str, default=None,
    help="Path to training checkpoint (e.g. checkpoints/rag_lora/step_2000). "
         "Must contain lm_state.pt and resampler.pt. Omit to run base model."
    )
    parser.add_argument("--n-ref-slots", type=int, default=16,
    help="Fixed resampler injection width — number of consecutive frames "
         "with zero_token_id in the text codebook whose embedding gets "
         "overwritten by the resampler's output. MUST match training's "
         "--n-ref-slots.")
    parser.add_argument("--ref-text-slots", type=int, default=32,
    help="DEPRECATED / unused by injection now: current training injects "
         "n_ref_slots continuous resampler vectors, not discrete grounding "
         "tokens, so this no longer controls the injected block width. "
         "Kept only so old launch commands don't break.")
    parser.add_argument("--ref-encoder-name", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
        help="Frozen pretrained text encoder wrapped by RefResampler. MUST "
             "match training's --ref-encoder-name for the checkpoint to load "
             "and behave correctly.")
    parser.add_argument("--manual-lookup-fallback-seconds", type=float, default=1.0,
        help="If local retrieval confidence was high for this turn but the "
             "model generates this many seconds of response without ever "
             "emitting <lookup> itself, force a grounded lookup anyway.")
    parser.add_argument("--max-ref-source-tokens", type=int, default=400,
        help="Cap on raw reference tokens the resampler reads. Compute-only — "
             "does not affect main sequence length. Should roughly match training.")
    parser.add_argument(
        "--no-merge-lora", action="store_true",
        help="Keep peft wrapper instead of merging LoRA into base weights."
    )
    parser.add_argument("--rag-index-dir", type=str, default=None,
        help="Path to RAG index (build_index.py --mode text). Omit to disable RAG.")
    parser.add_argument("--rag-top-k", type=int, default=2)
    parser.add_argument("--rag-min-score", type=float, default=0.25)
    parser.add_argument("--rag-trigger-score", type=float, default=0.30,
    help="Stricter threshold than --rag-min-score: only fire retrieval+injection "
         "if the top hit clears this bar. Below it, the turn proceeds as a plain "
         "conversational turn with no <ref> injection at all.")
    parser.add_argument("--stt-hf-repo", type=str, default="kyutai/stt-1b-en_fr-candle")
    parser.add_argument("--vad-threshold", type=float, default=0.5)
    parser.add_argument("--compressor-model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--compressor-device", type=str, default="cuda",
        help="Device for the context-compressor LLM. Use a CUDA device separate "
             "from the main model for real throughput with 4-bit quantization.")
    parser.add_argument("--compressor-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-ref-tokens", type=int, default=250,
        help="Cap on injected <ref> block length, in tokens. Must match the "
             "training-time max_ref_tokens for train/inference consistency.")
    parser.add_argument("--compressor-max-passages", type=int, default=2,
        help="Max retrieved passages fed to the compressor per call — fewer is faster.")
    parser.add_argument("--lora-rank", type=int, default=128,
    help="MUST match training's --lora-rank for this checkpoint, or LoRA weights won't bind on load.")
    parser.add_argument("--lora-scaling", type=float, default=2.0,
        help="MUST match training's --lora-scaling for this checkpoint.")

    # ── Web search ────────────────────────────────────────────────────────
    parser.add_argument("--web-search-enabled", action="store_true",
        help="Enable live web search as a fallback/supplement to local RAG "
             "retrieval when local hits are weak.")
    parser.add_argument("--web-search-api-key", type=str, default="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",
        help="API key for the web search provider. Defaults to SEARCH_API_KEY env var.")
    parser.add_argument("--web-search-provider", type=str, default="tavily",
        choices=["tavily", "serper", "bing"])
    parser.add_argument("--web-search-max-results", type=int, default=3)
    parser.add_argument("--web-search-trigger-below", type=float, default=0.45,
        help="If local top_score is below this, supplement with web search results.")
    parser.add_argument("--web-search-timeout", type=float, default=3.0,
        help="Max seconds to wait for web search before falling back to local-only.")

    args = parser.parse_args()

    if args.rag_index_dir is not None and args.rag_min_score > args.rag_trigger_score:
        logger.warning(
            f"--rag-min-score ({args.rag_min_score}) > --rag-trigger-score "
            f"({args.rag_trigger_score}) — injection will effectively never "
            f"trigger since hits are filtered before the trigger check. "
            f"Consider setting --rag-min-score <= --rag-trigger-score."
        )

    args.voice_prompt_dir = _get_voice_prompt_dir(args.voice_prompt_dir, args.hf_repo)
    if args.voice_prompt_dir is not None:
        assert os.path.exists(args.voice_prompt_dir), f"Directory missing: {args.voice_prompt_dir}"
    logger.info(f"voice_prompt_dir = {args.voice_prompt_dir}")

    static_path = _get_static_path(args.static)
    assert static_path is None or os.path.exists(static_path), f"Static path does not exist: {static_path}."
    logger.info(f"static_path = {static_path}")
    args.device = torch_auto_device(args.device)

    seed_all(42424242)

    setup_tunnel = None
    tunnel_token = ''
    if args.gradio_tunnel:
        try:
            from gradio import networking
        except ImportError:
            logger.error("Cannot find gradio. pip install gradio")
            sys.exit(1)
        setup_tunnel = networking.setup_tunnel
        tunnel_token = args.gradio_tunnel_token if args.gradio_tunnel_token else secrets.token_urlsafe(32)

    hf_hub_download(args.hf_repo, "config.json")

    logger.info("loading mimi")
    if args.mimi_weight is None:
        args.mimi_weight = hf_hub_download(args.hf_repo, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(args.mimi_weight, args.device)
    other_mimi = loaders.get_mimi(args.mimi_weight, args.device)
    logger.info("mimi loaded")

    if args.tokenizer is None:
        args.tokenizer = hf_hub_download(args.hf_repo, loaders.TEXT_TOKENIZER_NAME)
    text_tokenizer = sentencepiece.SentencePieceProcessor(args.tokenizer)

    logger.info("loading moshi")
    if args.moshi_weight is None:
        args.moshi_weight = hf_hub_download(args.hf_repo, loaders.MOSHI_NAME)

    torch.cuda.empty_cache()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    lm = loaders.get_moshi_lm(args.moshi_weight, device=args.device, cpu_offload=args.cpu_offload)
    lm.eval()
    logger.info("moshi loaded")

    state = ServerState(
        mimi=mimi,
        other_mimi=other_mimi,
        text_tokenizer=text_tokenizer,
        lm=lm,
        mimi_weight=args.mimi_weight,
        device=args.device,
        voice_prompt_dir=args.voice_prompt_dir,
        save_voice_prompt_embeddings=False,
        rag_index_dir=args.rag_index_dir,
        rag_top_k=args.rag_top_k,
        rag_min_score=args.rag_min_score,
        stt_hf_repo=args.stt_hf_repo,
        vad_threshold=args.vad_threshold,
        compressor_model_name=args.compressor_model,
        compressor_device=args.compressor_device,
        compressor_4bit=args.compressor_4bit,
        n_ref_slots=args.n_ref_slots,
        max_ref_source_tokens=args.max_ref_source_tokens,
        ref_text_slots=args.ref_text_slots,
        ref_encoder_name=args.ref_encoder_name,
        manual_lookup_fallback_seconds=args.manual_lookup_fallback_seconds,
        rag_trigger_score=args.rag_trigger_score,
        web_search_enabled=args.web_search_enabled,
        web_search_api_key=args.web_search_api_key,
        web_search_provider=args.web_search_provider,
        web_search_max_results=args.web_search_max_results,
        web_search_trigger_below=args.web_search_trigger_below,
        web_search_timeout=args.web_search_timeout,
        compressor_max_passages=args.compressor_max_passages
    )

    if args.checkpoint_dir:
        state.load_lora_and_adapter(
            args.checkpoint_dir,
            merge_lora=not args.no_merge_lora,
            lora_rank=args.lora_rank,
            lora_scaling=args.lora_scaling,
        )
    else:
        logger.info("no checkpoint — running base model (RAG injection disabled)")

    logger.info("warming up the model")
    state.gpu_executor.submit(state.warmup).result()

    app = web.Application()
    app.router.add_get("/api/chat", state.handle_chat)

    if static_path is not None:
        async def handle_root(_):
            return web.FileResponse(os.path.join(static_path, "index.html"))
        logger.info(f"serving static content from {static_path}")
        app.router.add_get("/", handle_root)
        app.router.add_static("/", path=static_path, follow_symlinks=True, name="static")

    protocol = "http"
    ssl_context = None
    if args.ssl is not None:
        ssl_context, protocol = create_ssl_context(args.ssl)

    host_ip = args.host if args.host not in ("0.0.0.0", "::", "localhost") else get_lan_ip()
    logger.info(f"Access the Web UI at {protocol}://{host_ip}:{args.port}")
    if setup_tunnel is not None:
        tunnel = setup_tunnel('localhost', args.port, tunnel_token, None)
        logger.info(f"Tunnel: {tunnel}")

    web.run_app(app, port=args.port, ssl_context=ssl_context)


with torch.no_grad():
    main()