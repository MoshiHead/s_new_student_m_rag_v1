#!/bin/bash
# Download the student checkpoint + tokenizer + local index from Hugging Face,
# and the gated base Moshi weights used by the server.
#
#   HF_TOKEN=hf_... bash download_assets.sh
#
# The base weights live in nvidia/personaplex-7b-v1 (gated): the account behind
# HF_TOKEN must have accepted the licence on its model page once.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="${ROOT:-$HERE}"
REPO="${HF_REPO:-sadmN/moshi-s28-rag}"
if [ -n "${ENV_DIR:-}" ]; then :
elif [ -x "$ROOT/menv/bin/python" ]; then ENV_DIR="$ROOT/menv"
elif [ -x /workspace/menv/bin/python ]; then ENV_DIR=/workspace/menv
else
  echo "ERROR: no python env; run 'bash setup_env.sh'" >&2
  exit 1
fi
PY="$ENV_DIR/bin/python"

"$PY" - "$ROOT" "$REPO" <<'PY'
import os
import sys

from huggingface_hub import snapshot_download

root, repo = sys.argv[1], sys.argv[2]
token = os.environ.get("HF_TOKEN")

print(f"== {repo} -> {root}")
snapshot_download(
    repo, local_dir=root,
    allow_patterns=["ckpt/**", "tokenizer.model",
                    "rag_index_dense/*", "*.sh", "*.txt", "README.md"],
    token=token)

print("== nvidia/personaplex-7b-v1 (gated)")
p = snapshot_download(
    "nvidia/personaplex-7b-v1",
    allow_patterns=["model.safetensors",
                    "tokenizer-e351c8d8-checkpoint125.safetensors",
                    "voices/*", "dist/*"],
    token=token)
print("base:", p)
PY

echo
echo "OK. Start the server with: bash serve_s28_web.sh"
