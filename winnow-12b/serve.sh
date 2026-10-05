#!/usr/bin/env bash
# Start the Winnow-12B (NVFP4) decision server on http://127.0.0.1:8091. Leave it running in its own
# terminal; Ctrl-C stops it. Run winnow-12b/setup.sh once first.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/src"
PY="${PYTHON:-python}"
[ -x "$SRC/.build/bin/winnow-server" ] || { echo "Server not built; run winnow-12b/setup.sh first"; exit 1; }

NV="$("$PY" -c 'import nvidia; print(list(nvidia.__path__)[0])' 2>/dev/null || true)"
[ -n "$NV" ] && export LD_LIBRARY_PATH="$NV/cu13/lib:$NV/nccl/lib:${LD_LIBRARY_PATH:-}"

# The directory holding gguf/Winnow-12B-NVFP4.gguf: the Hugging Face cache snapshot by default.
HUB="${HF_HOME:-$HOME/.cache/huggingface}/hub"
MODEL_DIR="${MODEL_DIR:-$(ls -d "$HUB"/models--EldanRing--Winnow-12B/snapshots/*/ | head -1)}"
[ -f "$MODEL_DIR/gguf/Winnow-12B-NVFP4.gguf" ] || { echo "Missing $MODEL_DIR/gguf/Winnow-12B-NVFP4.gguf"; exit 1; }

# 32K context holds the benchmark's longest documents (80,000 characters is about 20k tokens).
cd "$SRC"
exec "$PY" scripts/winnow.py serve --model nv4 --vision off --reasoning off --mtp off \
  --context "${CONTEXT:-32k}" --model-dir "$MODEL_DIR" --gpu "${GPU:-0}" --port "${PORT:-8091}"
