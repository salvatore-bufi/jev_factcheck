# d1-3B on LLM-AggreFact

Evaluates [LiquidAI/d1-3B](https://huggingface.co/LiquidAI/d1-3B), a 3B decision model built on LFM2.5-VL-3B, on the 29,320-row LLM-AggreFact test split and produces a row for the results table in the top-level README.

## How it works

d1-3B ships its own code (loaded with `trust_remote_code=True`) and answers typed questions with `model.system_one(state, questions)` in a **single forward pass with no generated tokens**. `benchmark.py` calls that official API through plain transformers, so the prompt, the readout and the batching are the authors' own:

- the state (`{"document": ..., "claim": ...}`) is written as indented JSON, followed by `QUESTION:` and the question; a yes/no question ends with "Reply with yes or no only." and, when it has descriptions, lists them as `Yes: ...` / `No: ...`;
- the answer is a softmax over the yes/no tokens at the answer position, and the yes-probability is the result;
- a request with several questions reads the state once, so each row costs one pass for all three questions.

It runs with transformers, not vLLM: for a 3B model there is no need, and this is the model's reference path. I read the shipped Python files (`prompt.py`, `runner.py`, `api.py`, `modeling_d1.py`) and scanned the others for network calls, subprocesses, `eval` and file writes; there are none.

For each (document, claim) pair it asks the three questions of the frozen Jev checker ([../packs/claim_support.json](../packs/claim_support.json)) and applies the frozen rule:

```
score = min(p_support, 1 - p_unsupported_detail, 1 - p_contradicted_detail)
supported = score > 0.30
```

This is a **fixed transfer evaluation**, like the JPT-9B, Rune, Winnow and CLM runs: same questions, rule, threshold and 80,000-character document cap (75% head, 25% tail). Nothing is tuned for d1-3B, and the 0.30 threshold was fitted to Jev's probabilities. The cache stores the raw yes-probabilities, so `--threshold` only rescales the report for free; do not pick it from test results. The model reports no calibration here: `calibration=None` is the authors' default in the shipped engine.

## Run

From the repository root, in the `factcheck` conda environment (needs `transformers>=5.14`). The Hugging Face cache must hold `LiquidAI/d1-3B` (6 GB, with its code files).

```sh
python d1-3b/benchmark.py --limit 1       # one claim vs its document (smoke test; separate output folder)
python d1-3b/benchmark.py                 # full test split; resumable
python d1-3b/benchmark.py --offline       # rebuild the report from cached answers (no GPU)
python d1-3b/test_benchmark.py            # offline unit checks
```

If interrupted (Ctrl-C), rerun the same command: finished rows are skipped. The cache is tied to the model snapshot, document cap, pack, data hash and `--limit`; a changed configuration is refused instead of mixed in. A prompt longer than the model's 32,768-token context, or a row that fails (for example out of memory), is printed and excluded, and the report is then marked incomplete; it is never shortened silently or given an invented score.

- **GPU:** one card, GPU 0 unless you pass `--gpu ID`. bf16 on CUDA (float32 on CPU, which is far too slow for the full run).
- **Data:** `data/test.parquet` if present, else `./lytang___llm-aggre_fact`; override with `--data`.
- **Cache location:** the run is offline (`HF_HUB_OFFLINE=1`); `HF_HOME` / `HF_HUB_CACHE` are honoured, and a missing model gives a clear message.
- **Speed:** about 12 rows/s on an RTX 5090 Laptop GPU (measured on 400 development rows), so the full split should take roughly 40 minutes; this is an extrapolation, not a measured full run.

## Output

In `results/runs/d1-3b/` (git-ignored): `answers.jsonl` (the cache), `report.json` and `table_row.md`, a ready-made row such as `| d1-3B (ours; fixed Jev rule) | 3B | <avg> | <11 sources> |`. Smoke runs go to `results/runs/d1-3b-limit<N>/` and are labelled as non-benchmark.

## What was verified

- **Smoke test** on `test:5075` (label supported): P(yes) is 0.977 / 0.047 / 0.042 for support / unsupported detail / contradicted detail, so the score is 0.95 and the claim is accepted.
- **Signal check on 400 development rows** (not test): AUC against the labels is 0.90 for `support_simple`, 0.90 for `unsupported_detail` (inverted) and 0.77 for `contradicted_detail` (inverted); the combined rule has AUC 0.89 and, pooled over those rows and at the frozen 0.30, balanced accuracy 78.6% (supported recall 0.93, unsupported recall 0.65). This is a pooled figure on a small sample and says little about the per-source macro average; it only shows that, unlike CLM, the three questions carry usable signal with the frozen rule.
- Offline unit checks cover the rule, the report, the cache identity and the missing-model message.
- **Not checked:** the full benchmark, and agreement with a second implementation (there is none; this uses the authors' own code).

## Notes

- The model's logits are read in bf16, so the yes/no probabilities take quantized values (for example exactly 0.0474 = sigmoid(-3.0)); that is how the shipped engine behaves.
- Images, audio and the model's compile option are not used.
- License: LFM Open License v1.0 (see the model's LICENSE; check it before any commercial use).
