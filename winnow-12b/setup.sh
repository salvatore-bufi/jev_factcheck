#!/usr/bin/env bash
# One-time setup: build the Winnow inference server (a patched llama.cpp) from source for THIS machine.
# Idempotent: finished steps are skipped. Needs: git-free tarball download, g++, OpenSSL headers, a CUDA
# toolkit (system nvcc, or the pip "nvidia-cuda-nvcc" packages), and Python with pip. Run inside the
# conda env that has CUDA 13 libraries (the "factcheck" env).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/src"
PY="${PYTHON:-python}"
URL="${WINNOW_SOURCE_URL:-https://codeload.github.com/EldanRing/winnow-inference/tar.gz/refs/heads/main}"

if [ -x "$SRC/.build/bin/winnow-server" ]; then
  echo "Already built: $SRC/.build/bin/winnow-server"; exit 0
fi

# 1. Source
if [ ! -f "$SRC/scripts/build.py" ]; then
  mkdir -p "$SRC"
  curl -fsSL --retry 4 "$URL" | tar xz -C "$SRC" --strip-components=1
fi

# 2. cmake
command -v cmake >/dev/null || "$PY" -m pip install cmake

# 3. CUDA toolkit: prefer a system nvcc, else the pip packages
NV="$("$PY" -c 'import nvidia; print(list(nvidia.__path__)[0])' 2>/dev/null || true)"
if command -v nvcc >/dev/null; then
  NVCC="$(command -v nvcc)"
elif [ -x "$NV/cu13/bin/nvcc" ]; then
  NVCC="$NV/cu13/bin/nvcc"
  # The pip packages ship only versioned libraries (libcudart.so.13); CMake needs unversioned names.
  L="$HERE/cuda-libs"; mkdir -p "$L"
  for f in cudart cublas cublasLt; do ln -sf "$NV/cu13/lib/lib$f.so.13" "$L/lib$f.so"; done
  ln -sf "$NV/nccl/lib/libnccl.so.2" "$L/libnccl.so"
  ln -sf /lib/x86_64-linux-gnu/libcuda.so.1 "$L/libcuda.so"
  export CMAKE_LIBRARY_PATH="$L" LIBRARY_PATH="$L:$NV/cu13/lib"
  # nvcc and the runtime headers can differ in minor version (13.3 vs 13.0); CUDA 13.x is binary compatible.
  export CUDAFLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK" CXXFLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
else
  echo "No CUDA compiler found (need nvcc on PATH, or pip packages nvidia-cuda-nvcc)"; exit 1
fi

# 4. GPU architecture of the first GPU (e.g. 12.0 -> 120, 8.9 -> 89)
ARCH="${CUDA_ARCH:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')}"
echo "Building with nvcc=$NVCC, CUDA architecture $ARCH (about 6 minutes on 12 cores)"

cd "$SRC"
"$PY" scripts/build.py --backend cuda --cuda-arch "$ARCH" --cuda-compiler "$NVCC" --jobs "${JOBS:-12}"
echo "Built: $SRC/.build/bin/winnow-server"
