#!/bin/bash
# Recreate the inference environment for the Moshi RAG student.
#
#   bash setup_env.sh            # -> /workspace/menv (or ./menv if no /workspace)
#   bash setup_env.sh /opt/menv  # explicit location
#
# Needs python3.11 + a CUDA 12.8 driver.  cu128 wheels cover sm_80/86/89 (A100,
# A10, 3090, 4090) and sm_120 (RTX 5090 / B200), so one env runs everywhere.
# Override with TORCH_VER=... TORCH_CUDA=cu121 for an older driver.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
if [ -n "${1:-}" ]; then
  ENV_DIR="$1"
elif [ -n "${ENV_DIR:-}" ]; then
  ENV_DIR="$ENV_DIR"
elif [ -d /workspace ]; then
  ENV_DIR=/workspace/menv
else
  ENV_DIR="$HERE/menv"
fi
PYTHON="${PYTHON:-python3.11}"
TORCH_VER="${TORCH_VER:-2.8.0}"
TORCH_CUDA="${TORCH_CUDA:-cu128}"

command -v "$PYTHON" >/dev/null 2>&1 || {
  echo "ERROR: $PYTHON not found. Install Python 3.11 first." >&2
  exit 1
}

echo "== venv: $ENV_DIR"
"$PYTHON" -m venv "$ENV_DIR"
"$ENV_DIR/bin/pip" install --upgrade pip wheel

echo "== torch ${TORCH_VER} / ${TORCH_CUDA}"
"$ENV_DIR/bin/pip" install torch=="${TORCH_VER}" torchaudio=="${TORCH_VER}" \
  --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"

echo "== server requirements"
"$ENV_DIR/bin/pip" install -r "$HERE/requirements-server.txt"

echo "== moshi (no-deps: it would pull sphn 0.2 whose opus API the server does not use)"
"$ENV_DIR/bin/pip" install --no-deps moshi==0.2.13

echo "== check"
"$ENV_DIR/bin/python" - <<'PY'
import aiohttp
import numpy as np
import sentence_transformers
import sentencepiece
import sphn
import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "| devices", torch.cuda.device_count())
print("sentencepiece", sentencepiece.__version__)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0),
          torch.cuda.get_device_capability(0))
    a = torch.randn(512, 512, device="cuda")
    print("kernel ok:", (a @ a).sum().item())

# audio path smoke test (the server streams opus both ways)
w = sphn.OpusStreamWriter(24000)
w.append_pcm(np.zeros(1920, dtype=np.float32))
blob = w.read_bytes()
r = sphn.OpusStreamReader(24000)
r.append_bytes(blob)
print("opus ok:", len(blob), "bytes ->", r.read_pcm().shape)
PY
echo
echo "OK. Environment ready: $ENV_DIR"
echo "Next: HF_TOKEN=... bash download_assets.sh   (model + index)"
echo "Then: bash serve_s28_web.sh"
