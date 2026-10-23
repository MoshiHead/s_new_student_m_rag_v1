"""
server_student.py — dedicated server for distilled STUDENT backbones
====================================================================

Serves **either** distilled student type, auto-detected from the checkpoint:

1. **Pruned backbone** (the current setup, `backbone_distill.py`):
   a full `LMModel` with fewer transformer layers (e.g. 16/32).  It keeps
   Mimi, the embeddings, `out_norm`, `text_linear` and the frozen DepFormer
   unchanged, so it is served exactly like the teacher — a plain `LMGen`
   at 12.5 Hz.  Checkpoint: ``{"state_dict", "kept_layers", "save_dtype"}``
   (bf16 by default), from ``backbone_distill.py fit``.

2. **6.25 Hz temporal student** (legacy, `train_distill.py` /
   `models/student.py`): compressor (12.5 -> 6.25 Hz) -> small Temporal
   Transformer -> upsampler (6.25 -> 12.5 Hz) -> the frozen teacher heads.
   Checkpoint: ``{"state_dict", "cfg"}`` -> served via `StudentLMGen`.

Why a separate file (vs server.py):
- server.py keeps every branch (teacher/legacy-student, two Mimi, RAG); this
  file is student-ONLY, so it is smaller and cheaper on-device: one Mimi, no
  teacher-mode branches, no second Mimi.
- A checkpoint + this file is enough to deploy.

Student-specific behaviour:
- the serving loop provides the user stream's codebooks every frame, so
  `fast_provided_tail=True` lets the frozen DepFormer skip those codebook
  steps — verified BIT-IDENTICAL, halves its serial chain per frame;
- pre-saved `.pt` voice-prompt embeddings are NOT usable for the 6.25 Hz
  student (captured with the teacher pipeline): such requests fall back to a
  `.wav` voice prompt from --voice-prompt-dir, or continue without one.  For
  the pruned backbone they ARE usable.

Usage
-----
    # pruned backbone (current setup)
    python -m moshi_distill.server_student \
        --student-checkpoint ckpt/backbone_S16/step_1000 \
        --moshi-weight /path/model.safetensors \
        --mimi-weight  /path/mimi.safetensors \
        --tokenizer    /path/tokenizer.model \
        --voice-prompt-dir /path/voices \
        --port 8998

    # validate a checkpoint without serving:
    python -m moshi_distill.server_student --student-checkpoint .../step_1000 \
        --moshi-weight ... --mimi-weight ... --tokenizer ... --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import os
from pathlib import Path
import random
import secrets
import sys
import tarfile
import time
from typing import Literal, Optional

import aiohttp
from aiohttp import web
import numpy as np
import sentencepiece
import sphn
import torch

from .models import loaders, MimiModel, LMModel, LMGen
from .models.student import StudentConfig, StudentLM
from .student_gen import _StudentLMGenState  # noqa: F401  (streaming state)
from .backbone_distill import detect_student_kind, load_student_backbone, _resolve_ckpt_path
from .utils.connection import create_ssl_context, get_lan_ip
from .utils.logging import setup_logger, ColorizedLog

logger = setup_logger(__name__)
DeviceString = Literal["cuda"] | Literal["cpu"]


# ---------------------------------------------------------------------------
# Small helpers (kept local so this file is self-contained)
# ---------------------------------------------------------------------------

def torch_auto_device(requested: Optional[DeviceString] = None) -> torch.device:
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_all(seed: int) -> None:
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


# ---------------------------------------------------------------------------
# Checkpoint loading (both student types)
# ---------------------------------------------------------------------------

LEGACY_KIND = "legacy_6p25hz"
PRUNED_KIND = "pruned_backbone"


def load_student_legacy(checkpoint: str | Path, lm: LMModel, device, dtype):
    """Load a `train_distill.py` 6.25 Hz student checkpoint."""
    ckpt = _resolve_ckpt_path(checkpoint)
    pkg = torch.load(ckpt, map_location="cpu", weights_only=False)
    if "cfg" not in pkg:
        raise ValueError(
            f"{ckpt} has no 'cfg': not a 6.25 Hz StudentLM checkpoint.")
    cfg = StudentConfig(**pkg["cfg"])
    logger.info(f"loading 6.25 Hz student from {ckpt}")
    student = StudentLM(cfg, teacher_lm=lm, device=device, dtype=dtype)
    sd = {k: (v.to(dtype) if v.is_floating_point() else v)
          for k, v in pkg["state_dict"].items()}
    student.load_state_dict(sd, strict=True)
    student.eval()
    n = sum(p.numel() for p in student.parameters())
    logger.info(f"6.25 Hz student loaded: {n:,} params | dim={cfg.dim} "
                f"layers={cfg.num_layers} heads={cfg.num_heads} "
                f"context={cfg.context}")
    return student


def load_student_auto(checkpoint: str | Path, lm: LMModel, device, dtype):
    """Detect the checkpoint kind and load the right student.

    Returns ``(kind, student, extra)`` where ``extra`` is the kept-layer list
    for the pruned backbone, or ``None`` for the 6.25 Hz student.
    """
    kind = detect_student_kind(checkpoint)
    if kind == PRUNED_KIND:
        student, kept = load_student_backbone(checkpoint, lm, device, dtype)
        n = sum(p.numel() for p in student.parameters())
        logger.info(f"pruned backbone loaded: {n:,} params | "
                    f"layers {len(kept)}/{len(lm.transformer.layers)}")
        return PRUNED_KIND, student, kept
    student = load_student_legacy(checkpoint, lm, device, dtype)
    return LEGACY_KIND, student, None


# ---------------------------------------------------------------------------
# Server state (student only; single Mimi)
# ---------------------------------------------------------------------------

@dataclass
class ServerState:
    mimi: MimiModel
    text_tokenizer: sentencepiece.SentencePieceProcessor
    lm_gen: LMGen
    lock: asyncio.Lock

    def __init__(self, mimi: MimiModel,
                 text_tokenizer: sentencepiece.SentencePieceProcessor,
                 lm: LMModel, student, device: str | torch.device,
                 kind: str = PRUNED_KIND,
                 voice_prompt_dir: str | None = None,
                 fast_provided_tail: bool = True,
                 default_text_prompt: str | None = None):
        assert student is not None, "server_student requires a student checkpoint"
        self.mimi = mimi
        self.text_tokenizer = text_tokenizer
        self.device = device
        self.kind = kind
        self.default_text_prompt = default_text_prompt
        self.voice_prompt_dir = voice_prompt_dir
        # Fallback voice prompt: first .pt/.wav/.mp3/.opus in --voice-prompt-dir.
        # (.pt = pre-saved voice embeddings; valid for the pruned backbone only.)
        self.voice_prompt_fallback = None
        if voice_prompt_dir is not None and os.path.isdir(voice_prompt_dir):
            candidates = sorted(
                f for f in os.listdir(voice_prompt_dir)
                if f.lower().endswith(('.pt', '.wav', '.mp3', '.opus')))
            if kind == LEGACY_KIND:
                candidates = [f for f in candidates if not f.lower().endswith('.pt')]
            if candidates:
                self.voice_prompt_fallback = os.path.join(voice_prompt_dir, candidates[0])
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        # The serving loop provides the second audio stream's codebooks every
        # frame, so the DepFormer skips those steps: bit-identical, halved chain.
        lm_gen_kwargs = dict(
            audio_silence_frame_cnt=int(0.5 * self.mimi.frame_rate),
            sample_rate=self.mimi.sample_rate,
            device=device,
            frame_rate=self.mimi.frame_rate,
            save_voice_prompt_embeddings=False,
            fast_provided_tail=fast_provided_tail,
        )
        if kind == LEGACY_KIND:
            # 6.25 Hz student: its own LMGen subclass drives the student
            # backbone and the frozen teacher heads. The teacher must stay
            # resident (its frozen heads run every frame).
            from .student_gen import StudentLMGen
            self.lm_gen = StudentLMGen(student, lm, **lm_gen_kwargs)
        else:
            # Pruned backbone is a full LMModel served exactly like the teacher:
            # plain LMGen over the student model.
            self.lm_gen = LMGen(student, **lm_gen_kwargs)
        self.lock = asyncio.Lock()
        self.mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)
        # per-step timing (ms/frame), reported periodically to the client log
        self._step_times: list[float] = []
        self._last_report = time.time()
        self.report_every_sec = 2.0

    def _time_step(self, codes: torch.Tensor):
        """One LMGen step, timed (with a CUDA sync so the number is real)."""
        sync = self.device.type == 'cuda'
        if sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        tokens = self.lm_gen.step(codes)
        if sync:
            torch.cuda.synchronize()
        if tokens is not None:
            self._step_times.append((time.perf_counter() - t0) * 1e3)
        return tokens

    def step_stats_and_reset(self) -> str | None:
        """One-line ms/frame summary over the recent window, then reset it."""
        if not self._step_times:
            return None
        ts = sorted(self._step_times)
        n = len(ts)
        mean = sum(ts) / n
        p50 = ts[n // 2]
        p95 = ts[min(n - 1, int(0.95 * n))]
        self._step_times = []
        self._last_report = time.time()
        return (f"step {mean:6.1f} ms/frame mean | p50 {p50:6.1f} | "
                f"p95 {p95:6.1f} | max {ts[-1]:6.1f} | n={n}")

    def warmup(self):
        for _ in range(4):
            chunk = torch.zeros(1, 1, self.frame_size, dtype=torch.float32,
                                device=self.device)
            codes = self.mimi.encode(chunk)
            for c in range(codes.shape[-1]):
                tokens = self.lm_gen.step(codes[:, :, c: c + 1])
                if tokens is None:
                    continue
                _ = self.mimi.decode(tokens[:, 1:9])
        if self.device.type == 'cuda':
            torch.cuda.synchronize()

    async def handle_chat(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        clog = ColorizedLog.randomize()
        peer = request.remote
        peer_port = request.transport.get_extra_info("peername")[1]
        clog.log("info", f"Incoming connection from {peer}:{peer_port}")

        # ---- voice prompt resolution (with graceful fallback) --------------
        requested_voice_prompt_path = None
        voice_prompt_path = None
        voice_prompt_filename = request.query.get("voice_prompt", "")
        if self.voice_prompt_dir is not None and voice_prompt_filename:
            requested_voice_prompt_path = os.path.join(
                self.voice_prompt_dir, voice_prompt_filename)
            if (requested_voice_prompt_path.endswith('.pt')
                    and self.kind == LEGACY_KIND):
                clog.log(
                    "warning",
                    f"voice prompt '{voice_prompt_filename}' is a pre-saved .pt "
                    "(teacher embeddings); not supported for the 6.25 Hz student")
                requested_voice_prompt_path = self.voice_prompt_fallback
                if requested_voice_prompt_path is not None:
                    clog.log("info", f"falling back to voice prompt: "
                                     f"{requested_voice_prompt_path}")
            if requested_voice_prompt_path is None or not os.path.exists(requested_voice_prompt_path):
                clog.log("warning",
                         f"Requested voice prompt '{voice_prompt_filename}' not "
                         f"found in '{self.voice_prompt_dir}'")
                requested_voice_prompt_path = self.voice_prompt_fallback
                if requested_voice_prompt_path is not None and os.path.exists(requested_voice_prompt_path):
                    clog.log("info", f"falling back to voice prompt: "
                                     f"{requested_voice_prompt_path}")
                else:
                    requested_voice_prompt_path = None
                    clog.log("warning", "continuing WITHOUT a voice prompt")
        if (requested_voice_prompt_path is None
                and self.voice_prompt_fallback is not None
                and not (self.voice_prompt_fallback.endswith('.pt')
                         and self.kind == LEGACY_KIND)):
            requested_voice_prompt_path = self.voice_prompt_fallback
            clog.log("info", f"no voice requested; using fallback voice: "
                             f"{requested_voice_prompt_path}")
        voice_prompt_path = requested_voice_prompt_path
        assert voice_prompt_path is None or not (
            voice_prompt_path.endswith('.pt') and self.kind == LEGACY_KIND), \
            "6.25 Hz student mode cannot use .pt voice prompts (see fallback above)"

        if self.lm_gen.voice_prompt != voice_prompt_path:
            if voice_prompt_path is not None:
                if voice_prompt_path.endswith('.pt'):
                    self.lm_gen.load_voice_prompt_embeddings(voice_prompt_path)
                else:
                    self.lm_gen.load_voice_prompt(voice_prompt_path)
            elif self.lm_gen.voice_prompt is not None:
                clog.log("warning", "clearing previously loaded voice prompt")
                self.lm_gen.voice_prompt = None
                self.lm_gen.voice_prompt_audio = None
                self.lm_gen.voice_prompt_embeddings = None
                self.lm_gen.voice_prompt_cache = None

        text_prompt = (request.query.get("text_prompt", "")
                       or self.default_text_prompt or "")
        self.lm_gen.text_prompt_tokens = (
            self.text_tokenizer.encode(wrap_with_system_tags(text_prompt))
            if len(text_prompt) > 0 else None)
        seed = int(request.query["seed"]) if "seed" in request.query else None

        async def recv_loop():
            nonlocal close
            try:
                async for message in ws:
                    if message.type == aiohttp.WSMsgType.ERROR:
                        clog.log("error", f"{ws.exception()}")
                        break
                    elif message.type in (aiohttp.WSMsgType.CLOSED,
                                          aiohttp.WSMsgType.CLOSE):
                        break
                    elif message.type != aiohttp.WSMsgType.BINARY:
                        clog.log("error", f"unexpected message type {message.type}")
                        continue
                    message = message.data
                    if not isinstance(message, bytes):
                        clog.log("error", f"unsupported message type {type(message)}")
                        continue
                    if len(message) == 0:
                        clog.log("warning", "empty message")
                        continue
                    if message[0] == 1:  # audio
                        opus_reader.append_bytes(message[1:])
                    else:
                        clog.log("warning", f"unknown message kind {message[0]}")
            finally:
                close = True
                clog.log("info", "connection closed")

        async def opus_loop():
            all_pcm_data = None
            while True:
                if close:
                    return
                await asyncio.sleep(0.001)
                pcm = opus_reader.read_pcm()
                if pcm.shape[-1] == 0:
                    continue
                if all_pcm_data is None:
                    all_pcm_data = pcm
                else:
                    all_pcm_data = np.concatenate((all_pcm_data, pcm))
                while all_pcm_data.shape[-1] >= self.frame_size:
                    chunk = all_pcm_data[: self.frame_size]
                    all_pcm_data = all_pcm_data[self.frame_size:]
                    chunk = torch.from_numpy(chunk).to(device=self.device)[None, None]
                    codes = self.mimi.encode(chunk)
                    for c in range(codes.shape[-1]):
                        tokens = self._time_step(codes[:, :, c: c + 1])
                        if tokens is None:
                            continue
                        assert tokens.shape[1] == self.lm_gen.lm_model.dep_q + 1
                        main_pcm = self.mimi.decode(tokens[:, 1:9]).cpu()
                        opus_writer.append_pcm(main_pcm[0, 0].numpy())
                        text_token = tokens[0, 0, 0].item()
                        if text_token not in (0, 3):
                            _text = self.text_tokenizer.id_to_piece(text_token)  # type: ignore
                            _text = _text.replace("▁", " ")
                            await ws.send_bytes(b"\x02" + bytes(_text, encoding="utf8"))
                if time.time() - self._last_report >= self.report_every_sec:
                    report = self.step_stats_and_reset()
                    if report is not None:
                        clog.log("info", report)

        async def send_loop():
            while True:
                if close:
                    return
                await asyncio.sleep(0.001)
                msg = opus_writer.read_bytes()
                if len(msg) > 0:
                    await ws.send_bytes(b"\x01" + msg)

        clog.log("info", "accepted connection")
        if len(request.query.get("text_prompt", "")) > 0:
            clog.log("info", f"text prompt: {request.query['text_prompt']}")
        if voice_prompt_path is not None:
            clog.log("info", f"voice prompt: {voice_prompt_path} "
                             f"(requested: {voice_prompt_filename})")
        elif voice_prompt_filename:
            clog.log("info", f"voice prompt: <skipped> "
                             f"(requested: {voice_prompt_filename})")

        close = False
        async with self.lock:
            if seed is not None and seed != -1:
                seed_all(seed)

            opus_writer = sphn.OpusStreamWriter(self.mimi.sample_rate)
            opus_reader = sphn.OpusStreamReader(self.mimi.sample_rate)
            self.mimi.reset_streaming()
            self.lm_gen.reset_streaming()

            async def is_alive():
                if close or ws.closed:
                    return False
                try:
                    msg = await asyncio.wait_for(ws.receive(), timeout=0.01)
                    if msg.type in (aiohttp.WSMsgType.CLOSE,
                                    aiohttp.WSMsgType.CLOSED,
                                    aiohttp.WSMsgType.ERROR):
                        return False
                except asyncio.TimeoutError:
                    return True
                except aiohttp.ClientConnectionError:
                    return False
                return True

            # Voice prompt -> silence -> text prompt -> silence, then talk.
            await self.lm_gen.step_system_prompts_async(self.mimi, is_alive=is_alive)
            self.mimi.reset_streaming()
            clog.log("info", "done with system prompts")

            if await is_alive():
                await ws.send_bytes(b"\x00")
                clog.log("info", "sent handshake bytes")
                tasks = [
                    asyncio.create_task(recv_loop()),
                    asyncio.create_task(opus_loop()),
                    asyncio.create_task(send_loop()),
                ]
                done, pending = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED)
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


# ---------------------------------------------------------------------------
# Assets
# ---------------------------------------------------------------------------

def _get_voice_prompt_dir(voice_prompt_dir: Optional[str], hf_repo: str) -> Optional[str]:
    if voice_prompt_dir is not None:
        return voice_prompt_dir
    logger.info("retrieving voice prompts")
    from huggingface_hub import hf_hub_download
    voices_tgz = Path(hf_hub_download(hf_repo, "voices.tgz"))
    voices_dir = voices_tgz.parent / "voices"
    if not voices_dir.exists():
        logger.info(f"extracting {voices_tgz} to {voices_dir}")
        with tarfile.open(voices_tgz, "r:gz") as tar:
            tar.extractall(path=voices_tgz.parent)
    if not voices_dir.exists():
        raise RuntimeError("voices.tgz did not contain a 'voices/' directory")
    return str(voices_dir)


def _get_static_path(static: Optional[str], hf_repo: str) -> Optional[str]:
    if static is None:
        logger.info("retrieving the static content")
        from huggingface_hub import hf_hub_download
        dist_tgz = Path(hf_hub_download(hf_repo, "dist.tgz"))
        dist = dist_tgz.parent / "dist"
        if not dist.exists():
            with tarfile.open(dist_tgz, "r:gz") as tar:
                tar.extractall(path=dist_tgz.parent)
        return str(dist)
    elif static != "none":
        return static
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Serve a distilled student backbone (see module docstring).")
    parser.add_argument("--host", default="localhost", type=str)
    parser.add_argument("--port", default=8998, type=int)
    parser.add_argument("--static", type=str)
    parser.add_argument("--gradio-tunnel", action='store_true',
                        help='Activate a gradio tunnel.')
    parser.add_argument("--gradio-tunnel-token", type=str)

    parser.add_argument("--student-checkpoint", type=str, required=True,
                        help="'student.pt' file or step_<N> directory from "
                             "backbone_distill.py (pruned backbone) or "
                             "train_distill.py (6.25 Hz student). REQUIRED.")
    parser.add_argument("--moshi-weight", type=str,
                        help="Broadcast/teacher checkpoint the student's frozen "
                             "parts are built from (the pruned backbone also "
                             "needs its embeddings/DepFormer; the 6.25 Hz "
                             "student needs the frozen heads).")
    parser.add_argument("--mimi-weight", type=str)
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--hf-repo", type=str, default=loaders.DEFAULT_REPO,
                        help="HF repo for weights when local paths are omitted.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--student-dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float32"],
                        help="student parameter dtype (teacher stays bf16).")
    parser.add_argument("--cpu-offload", action="store_true",
                        help="Offload teacher layers to CPU (needs accelerate).")
    parser.add_argument("--voice-prompt-dir", type=str,
                        help="Directory of .wav prompts. If omitted, voices.tgz "
                             "is downloaded from --hf-repo and extracted.")
    parser.add_argument("--default-text-prompt", type=str, default=None,
                        help="persona/system prompt used when the client does "
                             "not send a text_prompt query param (training used "
                             "per-persona system prompts)")
    parser.add_argument("--ssl", type=str,
                        help="Directory containing key.pem and cert.pem for https")
    parser.add_argument("--no-fast-tail", dest="fast_provided_tail",
                        action="store_false", default=True,
                        help="Disable the bit-identical DepFormer fast path "
                             "(debug only).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load everything, run the warmup, print a summary "
                             "and exit without serving (checkpoint validation).")
    parser.add_argument("--bench-steps", type=int, default=0,
                        help="If >0, run that many silent frames, print "
                             "ms/frame p50/p95/max, and exit (no websocket).")
    args = parser.parse_args()

    args.voice_prompt_dir = _get_voice_prompt_dir(args.voice_prompt_dir, args.hf_repo)
    if args.voice_prompt_dir is not None:
        assert os.path.exists(args.voice_prompt_dir), \
            f"Directory missing: {args.voice_prompt_dir}"
    logger.info(f"voice_prompt_dir = {args.voice_prompt_dir}")
    static_path = _get_static_path(args.static, args.hf_repo)
    assert static_path is None or os.path.exists(static_path), \
        f"Static path does not exist: {static_path}"
    args.device = torch_auto_device(args.device)
    student_dtype = getattr(torch, args.student_dtype)
    seed_all(42424242)

    setup_tunnel = None
    tunnel_token = ''
    if args.gradio_tunnel:
        try:
            from gradio import networking  # type: ignore
        except ImportError:
            logger.error("Cannot find gradio which is required to activate a "
                         "tunnel. Please install with `pip install gradio`.")
            sys.exit(1)
        setup_tunnel = networking.setup_tunnel
        tunnel_token = (args.gradio_tunnel_token or secrets.token_urlsafe(32))

    logger.info("loading mimi")
    if args.mimi_weight is None:
        from huggingface_hub import hf_hub_download
        args.mimi_weight = hf_hub_download(args.hf_repo, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(args.mimi_weight, args.device)
    logger.info("mimi loaded")

    if args.tokenizer is None:
        from huggingface_hub import hf_hub_download
        args.tokenizer = hf_hub_download(args.hf_repo, loaders.TEXT_TOKENIZER_NAME)
    text_tokenizer = sentencepiece.SentencePieceProcessor(args.tokenizer)  # type: ignore

    logger.info("loading broadcast/teacher checkpoint")
    if args.moshi_weight is None:
        from huggingface_hub import hf_hub_download
        args.moshi_weight = hf_hub_download(args.hf_repo, loaders.MOSHI_NAME)

    # Detect the student kind from the checkpoint BEFORE loading the teacher, so
    # the pruned-backbone path can keep the 7B teacher on CPU: it is only used to
    # copy weights into the student, and a 4090 (24 GB) cannot hold both the 7B
    # teacher (~17 GB bf16) and the student (~10 GB) at once.
    kind_hint = detect_student_kind(args.student_checkpoint)
    teacher_device = "cpu" if kind_hint == PRUNED_KIND else args.device
    if teacher_device == "cpu":
        logger.info("pruned backbone detected: loading the teacher on CPU to "
                    "avoid holding both models on the GPU")

    lm = loaders.get_moshi_lm(args.moshi_weight, device=teacher_device,
                              cpu_offload=args.cpu_offload)
    lm.eval()

    if kind_hint == PRUNED_KIND:
        # Build the student on CPU (teacher is CPU), then move only the student
        # to the GPU and drop the teacher entirely.
        student, _ = load_student_backbone(args.student_checkpoint, lm,
                                           device="cpu", dtype=student_dtype)
        student = student.to(args.device)
        import gc
        del lm
        gc.collect()
        logger.info("student built on CPU and moved to %s; teacher released "
                    "(GPU allocated=%.2f GB, reserved=%.2f GB)",
                    args.device, torch.cuda.memory_allocated() / 1e9,
                    torch.cuda.memory_reserved() / 1e9)
        kind = PRUNED_KIND
        lm_for_state = None
    else:
        kind, student, _ = load_student_auto(args.student_checkpoint, lm,
                                             args.device, student_dtype)
        lm_for_state = lm

    state = ServerState(
        mimi=mimi, text_tokenizer=text_tokenizer, lm=lm_for_state, student=student,
        device=args.device, kind=kind,
        voice_prompt_dir=args.voice_prompt_dir,
        fast_provided_tail=args.fast_provided_tail,
        default_text_prompt=args.default_text_prompt)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU memory after student build: allocated=%.2f GB "
                    "reserved=%.2f GB", torch.cuda.memory_allocated() / 1e9,
                    torch.cuda.memory_reserved() / 1e9)

    logger.info("warming up the student")
    state.warmup()
    logger.info("serving the distilled student (%s; fast_provided_tail=%s)",
                kind, args.fast_provided_tail)
    if torch.cuda.is_available():
        logger.info("GPU memory after load+warmup: allocated=%.2f GB "
                    "reserved=%.2f GB", torch.cuda.memory_allocated() / 1e9,
                    torch.cuda.memory_reserved() / 1e9)

    if args.dry_run:
        logger.info("dry run complete: checkpoint loads and steps correctly")
        return

    if args.bench_steps > 0:
        # End-to-end ms/frame without a websocket: Mimi encode -> LMGen step ->
        # Mimi decode, exactly like the serving loop.
        for _ in range(args.bench_steps):
            chunk = torch.zeros(1, 1, state.frame_size, dtype=torch.float32,
                                device=args.device)
            codes = state.mimi.encode(chunk)
            for c in range(codes.shape[-1]):
                tokens = state._time_step(codes[:, :, c: c + 1])
                if tokens is not None:
                    _ = state.mimi.decode(tokens[:, 1:9])
        logger.info("BENCH (%s, %s): %s", args.device, kind,
                    state.step_stats_and_reset())
        return

    app = web.Application()
    app.router.add_get("/api/chat", state.handle_chat)
    if static_path is not None:
        async def handle_root(_):
            return web.FileResponse(os.path.join(static_path, "index.html"))
        logger.info(f"serving static content from {static_path}")
        app.router.add_get("/", handle_root)
        app.router.add_static("/", path=static_path, follow_symlinks=True,
                              name="static")

    protocol = "http"
    ssl_context = None
    if args.ssl is not None:
        ssl_context, protocol = create_ssl_context(args.ssl)
    host_ip = args.host if args.host not in ("0.0.0.0", "::", "localhost") else get_lan_ip()
    logger.info(f"Access the Web UI directly at {protocol}://{host_ip}:{args.port}")
    if setup_tunnel is not None:
        tunnel = setup_tunnel('localhost', args.port, tunnel_token, None)
        logger.info(f"Tunnel started, if executing on a remote GPU, you can use {tunnel}.")
    web.run_app(app, port=args.port, ssl_context=ssl_context)


if __name__ == "__main__":
    with torch.no_grad():
        main()
