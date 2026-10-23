#!/bin/bash
# Serve the S28 RAG student (12.5 Hz Moshi, fact card in the pause) with live
# web search (Tavily by default) and the local dense index as fallback.
#
# Layout expected next to this script (that is what the zip / HF repo give):
#   moshi_local/moshi_distill/   server code
#   tokenizer.model              text tokenizer
#   rag_index_dense/             local fallback index
#   ckpt/rag_S28/step_1800/      student.pt
#
# Usage: bash serve_s28_web.sh [CHECKPOINT_DIR] [HOST] [PORT] [PROVIDER]
#   PROVIDER  = tavily (default) | brave | ddg
#   RETRIEVER = hybrid (default) | web | local
#   ENV_DIR   = python env (default <root>/menv, then /workspace/menv)
#   SNAP      = personaplex snapshot dir (auto-detected / downloaded)
#
#   DRY=1 bash serve_s28_web.sh     # load everything, serve nothing
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="${ROOT:-$HERE}"

# The package is imported as moshi_local.moshi_distill, so the *parent* of
# moshi_local must be on sys.path.
if [ -d "$ROOT/moshi_local/moshi_distill" ]; then
  PY_ROOT="$ROOT"
else
  echo "ERROR: $ROOT/moshi_local/moshi_distill not found" >&2
  exit 1
fi

if [ -n "${ENV_DIR:-}" ]; then :
elif [ -x "$ROOT/menv/bin/python" ]; then ENV_DIR="$ROOT/menv"
elif [ -x /workspace/menv/bin/python ]; then ENV_DIR=/workspace/menv
else
  echo "ERROR: no python env; run 'bash setup_env.sh'" >&2
  exit 1
fi
PY="$ENV_DIR/bin/python"

CKPT="${1:-$ROOT/ckpt/rag_S28/step_1800}"
HOST="${2:-0.0.0.0}"
PORT="${3:-8998}"
PROVIDER="${4:-${WEB_PROVIDER:-tavily}}"
RETRIEVER="${RETRIEVER:-hybrid}"

if [ ! -f "$CKPT/student.pt" ]; then
  echo "ERROR: $CKPT/student.pt missing; run 'bash download_assets.sh'" >&2
  exit 1
fi
if [ ! -f "$ROOT/tokenizer.model" ]; then
  echo "ERROR: $ROOT/tokenizer.model missing; run 'bash download_assets.sh'" >&2
  exit 1
fi

# --- base Moshi / Mimi / voices / web UI (gated nvidia repo) ---------------
# Prefer the persistent workspace cache when it exists (runpod volumes).
if [ -z "${HF_HOME:-}" ] && [ -d /workspace/.cache/huggingface ]; then
  export HF_HOME=/workspace/.cache/huggingface
fi
# SNAP can be set explicitly; otherwise use the HF cache or download it.
if [ -z "${SNAP:-}" ]; then
  SNAP="$("$PY" - <<'PY'
import os
from huggingface_hub import snapshot_download
pats = ["model.safetensors", "tokenizer-e351c8d8-checkpoint125.safetensors",
        "voices/*", "dist/*"]
try:
    p = snapshot_download("nvidia/personaplex-7b-v1",
                          allow_patterns=pats, local_files_only=True)
except Exception:
    print("  (base weights not cached, downloading...)", flush=True)
    p = snapshot_download("nvidia/personaplex-7b-v1",
                          allow_patterns=pats,
                          token=os.environ.get("HF_TOKEN"))
print(p)
PY
)"
fi
[ -f "$SNAP/model.safetensors" ] || {
  echo "ERROR: $SNAP/model.safetensors missing." >&2
  echo "Accept the licence at https://huggingface.co/nvidia/personaplex-7b-v1" >&2
  echo "with the account behind HF_TOKEN, then rerun." >&2
  exit 1
}

DEFAULT_PROMPT="You are Drew Kimura, a Knowledge Researcher at Verity Intelligence Hub. You specialise in cross-domain factual queries and encyclopedic reference. Communication style: analytical and reliable, citing the most specific facts available. When a conversation starts, greet the user warmly and introduce yourself briefly. In casual chat, be personable but keep it short -- one or two sentences at most -- before steering back to how you can help. When answering factual questions, be accurate and concise; lead with the direct answer. Keep spoken answers under 40 words unless the question clearly requires more detail."

EXTRA=""
if [ -n "${DRY:-}" ]; then EXTRA="--dry-run"; fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# torch>=2.6 loads with weights_only=True by default; our checkpoints carry
# plain python metadata, so keep the old behaviour (we only load our own files).
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
cd "$PY_ROOT"

exec "$PY" -u -m moshi_local.moshi_distill.server_student_rag \
  --student-checkpoint "$CKPT" \
  --moshi-weight "$SNAP/model.safetensors" \
  --mimi-weight  "$SNAP/tokenizer-e351c8d8-checkpoint125.safetensors" \
  --tokenizer    "$ROOT/tokenizer.model" \
  --rag-index    "$ROOT/rag_index_dense" \
  --retriever    "$RETRIEVER" \
  --web-provider "$PROVIDER" \
  --web-wait-frames 8 \
  --voice-prompt-dir "$SNAP/voices" \
  --default-text-prompt "$DEFAULT_PROMPT" \
  --static "$SNAP/dist" \
  --max-ref-tokens 256 --ref-tokens-per-slot 4 \
  --host "$HOST" --port "$PORT" $EXTRA
