# Rune 26B-A4B v3 on LLM-AggreFact

Evaluates [surogate/rune-26b-a4b-GGUF](https://huggingface.co/surogate/rune-26b-a4b-GGUF) (Rune v3, a Gemma 4 mixture-of-experts decision model; the repository name is historical, v3 holds bf16 safetensors) on the 29,320-row LLM-AggreFact test split and produces a row for the results table in the top-level README.

## What it does

`benchmark.py` runs the model in-process with **vLLM**, no server. It reproduces the protocol of surogate's *decisions v1* endpoint (its documentation and golden-test generator define it byte for byte): a fixed system prompt, a user turn `SHARED STATE (JSON string): <json> / QUESTION: ... / OPTIONS: A: <false description> / B: <true description> / Answer with one option letter only.`, the model's chat template with thinking off, and a softmax over the two option-letter logits divided by a decision temperature. A yes/no answer is P(option B).

For each (document, claim) pair it asks the three questions of the frozen Jev checker ([../packs/claim_support.json](../packs/claim_support.json)) and applies the frozen rule:

```
score = min(p_support, 1 - p_unsupported_detail, 1 - p_contradicted_detail)
supported = score > 0.30
```

This is a **fixed transfer evaluation**, like the CLM and JPT-9B runs: same questions, rule, threshold and 80,000-character document cap (75% head, 25% tail). Nothing is tuned for Rune; the 0.30 threshold was fitted to Jev's probabilities. The decision temperature defaults to **2**, the model card's recommendation for calibrated probabilities (its default of 1 is described as overconfident). The cache stores the raw option logits, so `--temperature` and `--threshold` only rescore it for free. Do not pick either value from test results.

## Hardware: read this first

The weights are **51.6 GB in bf16**; the GPU here has 24 GB. `--cpu-offload-gb` (default 36) keeps part of the weights in system RAM and streams them to the GPU at every step. This is exact (no quantization) but slow, and it needs about 40 GB of free RAM.

Measured on this machine (RTX 5090 Laptop, 24 GB): about **0.4 rows/s**, so the full 29,320 rows take roughly **20 hours** (model loading adds about 5 minutes per start). A GPU that holds the weights (about 60 GB or more) would be far faster. Because the run is resumable, you can stop and continue at any time.

## Run

From the repository root, in the `factcheck` conda environment:

```sh
python rune-26b/benchmark.py --limit 1       # one claim vs its document (smoke test; separate output folder)
python rune-26b/benchmark.py                 # full test split; resumable
python rune-26b/benchmark.py --offline       # rebuild the report from cached answers (no GPU)
python rune-26b/test_benchmark.py            # offline unit checks, no GPU
```

Rerun the same command to resume after an interruption: finished rows are skipped. The cache is tied to the model, document cap, system prompt, pack, data hash and `--limit`; a changed configuration is refused instead of mixed in (the temperature and threshold are not part of it, since they are applied at report time).

- **Data:** `data/test.parquet` if present, else `./lytang___llm-aggre_fact`; override with `--data`.
- **Model:** read from `~/.cache/huggingface/hub` (`HF_HUB_OFFLINE=1`).
- **Memory options:** `--cpu-offload-gb` (36), `--max-model-len` (24576), `--max-num-seqs` (32), `--max-batched-tokens` (16384), `--gpu-util` (0.92), `--window` (128 rows per batch), `--gpu ID`. A prompt longer than `--max-model-len` is printed as a failure and excluded, and the report is then marked incomplete; it is never shortened silently or given an invented score. These settings are the ones that worked: a 32768-token context or larger batches left too little GPU memory for the KV cache.

## Development threshold

The 0.30 threshold was fitted to Jev. To give this model its own threshold the way Jev got one, score Jev's 4,520 development rows (the tune and selection pools, rebuilt from the local dev split and verified against the hashes in `results/frozen.json` by [`dev_split.py`](../dev_split.py)), then fit and apply with [`calibrate_transfer.py`](../calibrate_transfer.py):

```sh
python rune-26b/benchmark.py --split dev            # development rows only -> results/runs/rune-26b-dev/
python calibrate_transfer.py fit                    # threshold from the dev cache; reads no test data
python calibrate_transfer.py apply                  # once: fitted threshold on the existing test cache
```

`apply` scores the existing test cache (no new inference) and writes `results/runs/rune-26b/report_dev_threshold.json` and `results/runs/dev_threshold_summary.md`; `report.json` keeps the frozen-0.30 result. Only the threshold is fitted; the temperature stays at the default.

On two 45 GB GPUs (for example the L40S pair) the bf16 weights fit without offload: `--gpu 0,1 --tensor-parallel 2 --cpu-offload-gb 0`.

## Output

In `results/runs/rune-26b/` (git-ignored): `answers.jsonl` (the cache: option logits per question and prompt tokens), `report.json` and `table_row.md`, a ready-made row such as `| Rune-26B-A4B-v3 (ours; fixed Jev rule) | 26B-A4B | <avg> | <11 sources> |`. Smoke runs go to `results/runs/rune-26b-limit<N>/` and are labelled as non-benchmark.

## Notes from setup

- `VLLM_USE_FLASHINFER_SAMPLER=0` is set by the script: there is no `nvcc` here, and FlashInfer's sampler would try to compile CUDA at start-up. The script never samples (one greedy token, label logprobs only).
- Verified on one test row (`test:5075`, label supported): P(yes) is 1.000 / 0.042 / 0.002 for support / unsupported detail / contradicted detail at temperature 1 (1.000 / 0.173 / 0.042 at temperature 2), so the claim is accepted. Letters `A` and `B` are single tokens after the prompt, checked in context as surogate does.
- Not checked: agreement with surogate's own engine. surogate was not run; this is a re-implementation of its documented protocol on vLLM, so tiny numeric differences are possible.
- Thinking mode and images are not used.
- License: Apache 2.0.
