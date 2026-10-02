# JPT-9B on LLM-AggreFact

Evaluates [kirp/jpt-9b](https://huggingface.co/kirp/jpt-9b), an open model that implements the Jev typed-decision interface (one forward pass returns a probability for every option), on the 29,320-row LLM-AggreFact test split, and produces a row for the results table in the top-level README.

## What it does

`benchmark.py` runs the model in-process with **vLLM** (no server). For each (document, claim) pair it asks the three questions of the frozen Jev checker ([../packs/claim_support.json](../packs/claim_support.json)): direct support, any unsupported detail, any contradicted detail. Prompts, label handling and probabilities come from the [`llm2jev`](https://github.com/tic-top/llm2jev) reference implementation named on the model card (chat template with thinking off, `Answer:` prefill, softmax at the card's fixed temperature 1.087). The decision is the frozen rule:

```
score = min(p_support, 1 - p_unsupported_detail, 1 - p_contradicted_detail)
supported = score > 0.30
```

This is a **fixed transfer evaluation**, like the CLM run in the top-level README: same questions, rule, threshold and 80,000-character document cap (75% head, 25% tail). Nothing is tuned for JPT-9B, and the 0.30 threshold was fitted to Jev's probabilities, not this model's, so treat the result as "the Jev checker run with a different model". Raw answers are cached, so `--threshold` only rescales the report. Do not pick a threshold from test results.

## Run

From the repository root, in the `factcheck` conda environment (see `requirements.txt`; a CUDA GPU with about 24 GB is enough):

```sh
python jpt-9b/benchmark.py --limit 1       # one claim vs its document (smoke test; separate output folder)
python jpt-9b/benchmark.py                 # full test split; resumable
python jpt-9b/benchmark.py --offline       # rebuild the report from cached answers
python jpt-9b/test_benchmark.py            # offline unit checks, no GPU
```

If interrupted (Ctrl-C), rerun the same command: finished rows are skipped. The cache is tied to the model, temperature, pack, document cap, data hash and `--limit`, so a changed configuration is refused instead of mixed in.

- **Data:** `data/test.parquet` if present, else `./lytang___llm-aggre_fact` (the Hugging Face `datasets` cache); override with `--data`. The dataset revision must be `981dfd0bd8e58e7238a9ab92b2e6ea44bce918e4`.
- **Model:** read from `~/.cache/huggingface/hub` (`HF_HUB_OFFLINE=1` is set).
- **Memory options:** `--max-model-len` (32768), `--max-num-seqs` (32), `--max-batched-tokens` (8192), `--gpu-util` (0.92), `--window` (64 rows per batch), `--gpu ID`. A prompt longer than `--max-model-len` is recorded as a failure, printed and excluded; the report is then marked incomplete. It is never shortened silently or given an invented score.

## Output

In `results/runs/jpt-9b/` (git-ignored): `answers.jsonl` (the cache: per-question yes-probabilities and prompt tokens), `report.json` (per-source and pooled metrics) and `table_row.md`, a ready-made row such as `| JPT-9B (ours; fixed Jev rule) | 9B | <avg> | <11 sources> |` in the table's format. Smoke runs go to `results/runs/jpt-9b-limit1/` and are labelled as non-benchmark.

## Notes from setup

- `VLLM_USE_FLASHINFER_SAMPLER=0` is set by the script: this machine has no `nvcc`, and FlashInfer's sampler would try to compile CUDA at start-up. The script never samples (one greedy token, label logprobs only), so nothing is lost.
- Qwen3.5's DeltaNet layers need one state block per concurrent sequence, which is why `--max-num-seqs` defaults to a small 32 on a 24 GB GPU.
- Verified on one test row (`test:5075`): this script and `llm2jev`'s transformers reference backend give the same three probabilities to four decimals (0.9900 / 0.0344 / 0.0157), and the claim is correctly accepted.
- License of the weights: CC BY-NC 4.0 (non-commercial).
