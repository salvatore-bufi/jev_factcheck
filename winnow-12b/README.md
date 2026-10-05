# Winnow-12B (NVFP4) on LLM-AggreFact

Evaluates [EldanRing/Winnow-12B](https://huggingface.co/EldanRing/Winnow-12B), a Gemma 4 12B fine-tune for typed decisions, on the 29,320-row LLM-AggreFact test split and produces a row for the results table in the top-level README. The quantization tested is **`Winnow-12B-NVFP4.gguf`** (8.16 GB), the file in `~/.cache/huggingface/hub/models--EldanRing--Winnow-12B`. Its authors report that NVFP4 scores a little below their Q8_0 file on their own benchmarks (for example 83.55% vs 85.71% on the JevBench public subset), so this row is not a Q8 or BF16 result.

## How it works

Winnow ships only as GGUF and runs on its own llama.cpp-based server, which exposes the same `/v1/systemone` API as Jev. vLLM cannot load this file, so unlike the other models in this repository `benchmark.py` is an HTTP client: the server renders the prompts and reads the answer logits itself, as its authors intend (decision temperature 1, no calibration map).

For each (document, claim) pair `benchmark.py` sends the three questions of the frozen Jev checker ([../packs/claim_support.json](../packs/claim_support.json)) and applies the frozen rule:

```
score = min(p_support, 1 - p_unsupported_detail, 1 - p_contradicted_detail)
supported = score > 0.30
```

This is a **fixed transfer evaluation**, like the CLM, JPT-9B and Rune runs: same questions, rule, threshold and 80,000-character document cap (75% head, 25% tail). Nothing is tuned for Winnow, and the 0.30 threshold was fitted to Jev's probabilities. The cache stores the raw probabilities, so `--threshold` only rescales the report; do not pick it from test results.

## Run

From the repository root, in the `factcheck` conda environment.

```sh
winnow-12b/setup.sh                          # once: builds the server for this machine (about 6 min)
winnow-12b/serve.sh                          # terminal 1: leave running (Ctrl-C stops it)
python winnow-12b/benchmark.py --limit 1     # terminal 2: one claim vs its document (smoke test)
python winnow-12b/benchmark.py               # full test split; resumable
python winnow-12b/benchmark.py --offline     # rebuild the report from cached answers (no server)
python winnow-12b/test_benchmark.py          # offline unit checks
```

Measured here (RTX 5090 Laptop, 24 GB): about **4.3 rows/s**, so the full 29,320 rows take roughly **2 hours**. The server uses about 9.4 GB of GPU memory with a 32K context.

If interrupted (Ctrl-C), rerun the same command: finished rows are skipped. The cache is tied to the model, document cap, pack, data hash and `--limit`; a changed configuration is refused instead of mixed in. A row the server rejects (for example a prompt longer than the 32K context) is printed as a failure and excluded; the report is then marked incomplete. It is never shortened silently or given an invented score.

- **Data:** `data/test.parquet` if present, else `./lytang___llm-aggre_fact`; override with `--data`.
- **Model:** `serve.sh` reads the GGUF from the Hugging Face cache (`MODEL_DIR=... ./serve.sh` to use another folder containing `gguf/Winnow-12B-NVFP4.gguf`).
- **Options:** `serve.sh` honours `GPU` (default 0), `PORT` (8091), `CONTEXT` (32k). `benchmark.py` has `--url`, `--workers` (4), `--window`, `--timeout`.

## Output

In `results/runs/winnow-12b/` (git-ignored): `answers.jsonl` (the cache), `report.json` and `table_row.md`, a ready-made row such as `| Winnow-12B-NVFP4 (ours; fixed Jev rule) | 12B | <avg> | <11 sources> |`. Smoke runs go to `results/runs/winnow-12b-limit<N>/` and are labelled as non-benchmark.

## Why the server is built from source

The authors publish a prebuilt 48 MB Linux runtime, but it contains about 10,000 AVX-512 instructions and crashes with `SIGILL` (illegal instruction, exit 132) on CPUs without AVX-512, such as the Intel laptop CPU used here. `setup.sh` therefore builds the pinned llama.cpp fork (downloaded by the source checkout) for the local CPU. On a server CPU with AVX-512 the prebuilt `local-review-runtime-v9.tar.gz` from the [2026.10.04 release](https://github.com/EldanRing/winnow-inference/releases/tag/v2026.10.04) may work and saves the build.

Build details worth knowing (all handled by `setup.sh`):

- It needs `cmake`, a C++ compiler, OpenSSL headers and a CUDA compiler. Without a system CUDA toolkit it uses the pip `nvidia-cuda-nvcc` packages and creates unversioned library symlinks in `winnow-12b/cuda-libs/`, since the pip packages ship only `libcudart.so.13` and similar.
- In this conda environment `nvcc` is 13.3 but the runtime headers are 13.0, which trips a CCCL compatibility check; `setup.sh` disables that check with `-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK`. Minor versions within CUDA 13 are binary compatible, and the server ran normally.
- The CUDA architecture is read from `nvidia-smi` (12.0 → `120` for an RTX 5090, 8.9 → `89` for an L40S); override with `CUDA_ARCH`.
- The source is the `main` branch of `EldanRing/winnow-inference` as of 2026-10-05 (checkout and build output, about 575 MB, are git-ignored).

## Notes

- Verified on one test row (`test:5075`, label supported): P(yes) is 0.994 / 0.005 / 0.015 for support / unsupported detail / contradicted detail, so the claim is accepted. The server's expected model hash matched the downloaded file's.
- Thinking, MTP, vision and the experimental reasoning mode are off.
- License of the weights: Apache 2.0.
