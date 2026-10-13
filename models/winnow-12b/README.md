# Winnow-12B (NVFP4, on the authors' server)

[EldanRing/Winnow-12B](https://huggingface.co/EldanRing/Winnow-12B): a Gemma 4 12B fine-tune for typed decisions,
published only as GGUF files. This folder benchmarks **`gguf/Winnow-12B-NVFP4.gguf`** (8.16 GB) on Winnow's own
llama.cpp-based server; [../winnow-12b-bf16](../winnow-12b-bf16/README.md) benchmarks the unquantized weights with vLLM.
The authors report NVFP4 a little below their Q8_0 file (83.55% vs 85.71% on the JevBench public subset). Licence: Apache 2.0.

## How the benchmark runs it

The NVFP4 GGUF runs only on Winnow's server, which exposes `/v1/systemone` and renders the prompts and reads the option
logits itself. `adapter.py` starts the server (`serve.sh`) on the benchmark GPU, sends one request per row with all its
questions (the server reads the state once and answers the questions as parallel branches), asks for `diagnostics` so the
raw option logits are saved, and stops the server at the end. Latency therefore includes the local HTTP round trip.
Server output goes to `models/winnow-12b/server.log`.

## Run

```sh
hf download EldanRing/Winnow-12B --include "gguf/Winnow-12B-NVFP4.gguf"
models/winnow-12b/setup.sh                                 # once: builds the server for this machine (about 6 minutes)
python benchmark.py --model winnow-12b --gpu 0 --limit 32
python benchmark.py --model winnow-12b --gpu 0
```

Options, in [config/models/winnow-12b.json](../../config/models/winnow-12b.json) (or `--opt winnow-12b:KEY=VALUE`): `port` (8091), `context` (32k: holds the longest documents, 80,000 characters being about 20k tokens),
`workers` (4 concurrent requests), `timeout` (300 s), `startup_timeout` (900 s), `url` (use an already running server
instead of starting one), `batch` (64). About 9.4 GB of GPU memory.

## Building the server (`setup.sh`)

The authors' prebuilt runtime contains AVX-512 instructions and crashes with `SIGILL` on CPUs without AVX-512, so
`setup.sh` builds their pinned llama.cpp fork from source (`src/`, git-ignored, about 575 MB). It needs `cmake` (installed
with pip if missing), a C++ compiler, OpenSSL headers and a CUDA compiler: a system `nvcc`, or the pip `nvidia-cuda-nvcc`
packages, for which it creates unversioned library symlinks in `cuda-libs/` and disables CCCL's header-version check
(`nvcc` 13.3 with 13.0 headers; CUDA 13.x is binary compatible). The CUDA architecture comes from `nvidia-smi` (12.0 → 120
on an RTX 5090, 8.9 → 89 on an L40S); override with `CUDA_ARCH`. On a machine with AVX-512 the prebuilt
`local-review-runtime-v9.tar.gz` of the [2026.10.04 release](https://github.com/EldanRing/winnow-inference/releases/tag/v2026.10.04)
may work and saves the build.

## Verified

On test row `test:5075` P(yes) was 0.994 / 0.005 / 0.015, and the server's expected model hash matched the download.
Thinking, MTP, vision and the experimental reasoning mode are off.
