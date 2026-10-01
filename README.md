# Jev as a source-grounded claim checker

An experiment that evaluates `jev-1.13.0` (the [System One API](https://docs.typesafe.ai/api)) on the [`lytang/LLM-AggreFact`](https://huggingface.co/datasets/lytang/LLM-AggreFact) benchmark. The task is **document-grounded support**: given a document and a claim, does the document support the whole claim? It does not check truth against the open web.

## Headline result

**79.09% mean per-source balanced accuracy on all 29,320 test examples** (11 sources, equal weight). This is numerically above the highest published leaderboard score (77.4, Bespoke-Minicheck-7B) and below the 80% target set for the experiment.

Caveats, all of which apply to every number in this repository:

- The result is **development-tuned**: questions, score rule and threshold were chosen on development data. Official leaderboard entries are described as [zero-shot](https://llm-aggrefact.github.io/blog). This is a numerical comparison, not an official submission or a like-for-like zero-shot claim.
- The test split was observed partway through the run (an interim check), so it is not a pristine one-shot evaluation. See [results/INTERIM_AMENDMENT.md](results/INTERIM_AMENDMENT.md).
- Document-cluster bootstrap 95% interval: 77.66-80.44%. It conditions on the frozen checker and ignores prompt-search uncertainty.
- Full details, source-level results, controls and cost: [results/report.md](results/report.md).

## The frozen checker

One request per (document, claim) pair carries three yes/no ("Noul") questions: direct support, any unsupported detail, any contradicted detail. The exact wording is in [results/frozen.json](results/frozen.json) and [packs/claim_support.json](packs/claim_support.json).

```python
score = min(p_support, 1 - p_unsupported_detail, 1 - p_contradicted_detail)
supported = score > 0.30          # strict inequality
```

The score is a decision score, not a calibrated probability. The single global threshold optimizes balanced accuracy. Only the document and claim are sent to the API: never labels, source names or contamination identifiers. Documents over 80,000 characters keep their first 60,000 and last 20,000 characters, joined by an omission marker (8 of 29,320 test documents were shortened).

## Protocol in brief

Full text: [PROTOCOL.md](PROTOCOL.md). Benchmark semantics and metric sources: [benchmark_notes.md](benchmark_notes.md).

1. Development data only is split by document hash into a tuning pool (60%) and a selection pool (40%); no document crosses pools.
2. Seven initial questions ([question_candidates.json](question_candidates.json), rationale in [question_design.md](question_design.md)) are tuned on the tuning pool. One revision round adds three questions ([question_round2.json](question_round2.json), [results/round2_rationale.md](results/round2_rationale.md)). [question_all.json](question_all.json) combines both.
3. Three candidates are locked ([results/shortlist.json](results/shortlist.json)) before selection-pool inference. The winner is chosen on source-macro balanced accuracy of the selection pool; its threshold is refitted on all observed development rows.
4. The configuration and code hashes are frozen ([results/frozen.json](results/frozen.json)) before the test split is downloaded ([results/test_access.json](results/test_access.json)).
5. The full test split is scored once with the frozen checker. Nothing is changed in response to test results.

The headline metric is the unweighted mean over sources of per-source balanced accuracy, `0.5 * (TPR + TNR)`. Pooled metrics are secondary.

Cost: input tokens are priced at $0.042 per million, output is free. The full test cost about $1.48 and all development plus test work about $1.97 (estimates from response usage, not invoices).

## LLM-AggreFact results

Balanced accuracy (%) per source; **Average gives equal weight to all 11 sources**. The Jev row is our full 29,320-example test result (two decimals). The other 39 rows reproduce every model in the [official leaderboard](https://llm-aggrefact.github.io/) snapshot audited on 2026-09-28 (one decimal, as published); its default visible rows were rechecked on 2026-09-30. Evaluation settings differ: Jev is dev-tuned, official entries are described as zero-shot.

| Model | Size | Average | CNN | XSum | MediaS | MeetB | WiCE | REVEAL | ClaimVerify | FactCheck | ExpertQA | LFQA | RAGTruth |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **jev-1.13.0 (ours; dev-tuned)** | — | **79.09** | 67.12 | 77.46 | 75.65 | 81.85 | 85.75 | 91.41 | 77.27 | 78.18 | 59.36 | 88.95 | 86.94 |
| Bespoke-Minicheck-7B | 7B | 77.4 | 65.5 | 77.8 | 76.0 | 78.3 | 83.0 | 88.0 | 75.3 | 77.7 | 59.2 | 86.7 | 84.0 |
| Claude-3.5 Sonnet | - | 77.2 | 67.6 | 75.1 | 73.4 | 84.6 | 77.7 | 89.1 | 71.4 | 77.8 | 60.9 | 85.6 | 86.1 |
| Granite Guardian 3.3 | 8B | 76.5 | 67.0 | 74.9 | 74.0 | 78.6 | 76.6 | 89.6 | 75.9 | 76.1 | 59.6 | 86.9 | 82.2 |
| Mistral-Large 2 | 123B | 76.5 | 64.8 | 74.7 | 69.6 | 84.2 | 80.3 | 87.7 | 71.8 | 74.5 | 60.8 | 87.0 | 85.9 |
| gpt-4-turbo-preview | - | 76.2 | 66.7 | 76.5 | 71.4 | 79.9 | 80.4 | 87.8 | 67.6 | 79.9 | 59.2 | 83.1 | 85.3 |
| gpt-4o-2024-05-13 | - | 75.9 | 68.1 | 76.8 | 71.4 | 79.8 | 78.5 | 86.5 | 69.0 | 77.5 | 59.6 | 83.6 | 84.3 |
| FactCG-DeBERTa-L | 0.4B | 75.6 | 70.1 | 73.9 | 72.3 | 74.3 | 74.2 | 88.4 | 78.5 | 72.1 | 59.1 | 86.7 | 82.3 |
| Qwen2.5-72B-Instruct | 72B | 75.6 | 63.6 | 73.0 | 71.9 | 80.4 | 80.2 | 88.9 | 70.0 | 77.0 | 60.1 | 84.3 | 81.9 |
| Llama-3.1-70B-Instruct | 70B | 75.1 | 65.7 | 72.5 | 72.9 | 81.0 | 73.9 | 86.4 | 70.3 | 78.6 | 58.5 | 83.8 | 83.0 |
| MiniCheck-Flan-T5-L | 0.8B | 75.0 | 69.9 | 74.3 | 73.6 | 77.3 | 72.2 | 86.2 | 74.6 | 74.7 | 59.0 | 85.2 | 78.0 |
| Claude-3 Opus | - | 74.8 | 65.2 | 72.4 | 74.1 | 82.4 | 75.0 | 83.8 | 69.3 | 78.8 | 58.8 | 81.6 | 81.8 |
| Llama-3.3-70B-Instruct | 70B | 74.5 | 68.7 | 74.7 | 69.5 | 78.4 | 76.6 | 85.5 | 67.4 | 78.5 | 58.3 | 79.8 | 82.6 |
| Llama-3.1-405B-Instruct | 405B | 74.4 | 64.8 | 75.1 | 68.6 | 81.2 | 71.8 | 86.4 | 67.5 | 79.4 | 58.5 | 81.9 | 82.9 |
| Mistral-Large | - | 74.2 | 58.4 | 76.3 | 67.3 | 78.9 | 76.6 | 88.4 | 67.6 | 79.0 | 60.0 | 81.7 | 81.7 |
| gpt-4o-mini-2024-07-18 | - | 74.0 | 61.8 | 73.6 | 71.3 | 79.7 | 76.3 | 85.8 | 69.8 | 76.0 | 58.3 | 80.3 | 81.6 |
| Llama-3-70B-Instruct | 70B | 73.7 | 63.7 | 70.2 | 71.5 | 80.6 | 74.4 | 85.9 | 67.8 | 76.2 | 57.8 | 82.4 | 80.6 |
| MiniCheck-RoBERTa-L | 0.4B | 73.5 | 63.7 | 70.8 | 71.9 | 75.9 | 67.6 | 88.8 | 77.4 | 73.3 | 57.4 | 84.4 | 77.2 |
| MiniCheck-DeBERTa-L | 0.4B | 73.1 | 64.2 | 71.0 | 69.3 | 72.7 | 69.4 | 87.3 | 75.6 | 73.0 | 58.9 | 83.9 | 78.8 |
| Qwen2.5-7B-Instruct | 7B | 72.8 | 54.6 | 69.9 | 71.5 | 75.2 | 76.3 | 86.9 | 70.1 | 74.9 | 61.0 | 84.2 | 76.5 |
| QwQ-32B-Preview | 32B | 71.8 | 57.0 | 71.6 | 69.3 | 78.5 | 72.3 | 86.2 | 67.7 | 75.6 | 60.0 | 78.9 | 72.4 |
| HHEM-2.1-open | 0.1B | 71.8 | 62.8 | 67.8 | 67.2 | 71.0 | 75.9 | 86.6 | 73.9 | 71.8 | 58.4 | 84.1 | 69.9 |
| Mixtral-8x22B | 176B | 71.5 | 57.3 | 70.3 | 69.0 | 78.5 | 69.5 | 85.8 | 63.8 | 79.5 | 57.4 | 76.5 | 78.8 |
| Claude-2.1 | - | 71.1 | 59.9 | 66.4 | 69.2 | 72.3 | 64.3 | 88.2 | 69.7 | 79.3 | 59.8 | 78.2 | 75.0 |
| AlignScore | 0.4B | 70.5 | 52.4 | 71.4 | 69.2 | 72.6 | 66.0 | 85.3 | 69.6 | 74.3 | 58.3 | 84.5 | 71.7 |
| Llama-3.1-8B-Instruct | 8B | 70.3 | 54.7 | 68.5 | 71.1 | 75.5 | 72.0 | 83.5 | 66.5 | 72.3 | 57.8 | 77.5 | 73.6 |
| GPT-3.5-Turbo | - | 70.1 | 63.2 | 72.4 | 66.8 | 73.4 | 68.5 | 84.7 | 65.2 | 70.8 | 57.2 | 73.8 | 75.6 |
| SummaC-ZS | 0.1B | 67.7 | 51.1 | 61.5 | 69.5 | 71.0 | 62.8 | 85.3 | 69.7 | 75.2 | 55.2 | 77.6 | 65.6 |
| Tülu-3-70B | 70B | 67.6 | 58.3 | 66.4 | 62.7 | 70.6 | 64.6 | 82.5 | 62.2 | 76.5 | 55.7 | 70.8 | 73.8 |
| Mistral-8x7B | 56B | 67.4 | 55.0 | 65.5 | 68.5 | 73.3 | 63.8 | 80.8 | 64.3 | 75.1 | 56.3 | 70.8 | 68.1 |
| PaLM2-Bison | - | 66.1 | 52.4 | 59.0 | 68.3 | 73.6 | 63.4 | 84.2 | 60.5 | 76.4 | 56.6 | 71.4 | 61.6 |
| Gemini-Pro | - | 65.4 | 49.4 | 60.6 | 63.8 | 65.8 | 65.8 | 85.5 | 61.8 | 76.8 | 56.8 | 75.9 | 57.6 |
| DAE | 0.1B | 65.0 | 50.8 | 59.1 | 65.1 | 69.5 | 58.5 | 81.3 | 64.0 | 72.5 | 56.2 | 72.2 | 65.3 |
| Qwen2.5-0.5B-Instruct | 0.5B | 64.2 | 55.5 | 62.9 | 60.0 | 64.7 | 65.9 | 85.8 | 61.4 | 68.9 | 56.8 | 72.4 | 51.6 |
| InternLM2.5-7B-chat | 7B | 63.6 | 58.4 | 57.9 | 62.5 | 68.0 | 63.7 | 77.3 | 59.8 | 67.5 | 55.2 | 64.9 | 64.0 |
| Tülu-3-8B | 8B | 63.2 | 51.0 | 58.5 | 63.4 | 67.7 | 58.7 | 81.8 | 63.2 | 68.6 | 56.2 | 68.5 | 57.8 |
| T5-NLI-Mixed | 11B | 61.1 | 54.6 | 52.3 | 59.1 | 55.3 | 55.3 | 87.2 | 59.5 | 69.0 | 55.6 | 61.8 | 62.5 |
| SummaC-CV | 0.1B | 61.0 | 65.2 | 54.5 | 63.7 | 62.8 | 54.3 | 67.7 | 70.9 | 53.4 | 54.9 | 62.1 | 61.7 |
| Llama-3.2-3B-Instruct | 3B | 60.0 | 51.5 | 60.5 | 53.1 | 52.5 | 58.5 | 81.9 | 62.3 | 62.6 | 55.8 | 58.5 | 63.0 |
| Llama-3.2-1B-Instruct | 1B | 50.3 | 50.1 | 50.9 | 50.0 | 50.2 | 49.7 | 50.4 | 50.5 | 50.2 | 49.9 | 50.1 | 50.9 |

[Score provenance and all-model audit](references/leaderboard_audit.json) · [Full Jev report](results/report.md) · [Validation error analysis](results/validation_error_analysis/report.md)

## Repository layout

| Path | Purpose |
|---|---|
| `bench.py` | Frozen experiment runner: `prepare` development pools, `run` a phase (`tune`, `selection`, `test`). Also holds shared helpers and the API client. |
| `analyze.py` | Frozen analysis: `tune`, `shortlist`, `freeze`, `test-report`. |
| `fetch_data.py` | Frozen downloader for one pinned split (`dev` or `test`; `test` requires a verified freeze). |
| `metrics.py` | Frozen metric code: per-source balanced accuracy, threshold tuning, document-cluster bootstrap. |
| `pilot.py` | Fixed diagnostic sample (LFQA and RAGTruth) run with the unchanged frozen checker. |
| `iterate_cached.py` | Offline, development-only logistic calibration study on cached predictions. Rejected; see the report. |
| `analyze_validation_errors.py` | Offline audit of selection-pool errors. Output: [results/validation_error_analysis/](results/validation_error_analysis/report.md). |
| `check_claim.py` | Check one document/claim with the frozen checker. |
| `FactCG/` | Self-hosted FactCG-DeBERTa-v3-Large classifier. `FactCG/benchmark.py` runs it on the test split (see below); `test_benchmark.py` is its offline test. The Italian-language docs and `main.py` belong to the module's original repository and are not used here. |
| `evaluate.py` | Reusable runner for any System One model, data file or question pack (see below). |
| `test_bench.py`, `test_metrics.py`, `test_evaluate.py` | Offline tests (synthetic data, mocked HTTP). |
| `packs/` | Question packs for `evaluate.py`: `claim_support.json` (frozen three-question rule), `binary_support.json` (one choice question). |
| `data/hub_metadata.json` | Pinned Hugging Face dataset metadata (revision `981dfd0bd8e58e7238a9ab92b2e6ea44bce918e4`). Downloaded splits are also stored in `data/`, but not distributed. |
| `references/` | Saved API/model documentation and the [39-model leaderboard audit](references/leaderboard_audit.md) (data: `leaderboard_audit.json`). |
| `results/` | Reports, metrics, frozen configuration, audits and the 29,320 cached test answers. See below. |

**The four files `bench.py`, `analyze.py`, `fetch_data.py` and `metrics.py` are hash-locked** by `results/frozen.json`. Modifying any of them makes `verify_freeze()` fail, so test access and test reporting stop working. Make changes in new files instead.

### Key files in `results/`

- `report.md`: full test report. `INTERIM_AMENDMENT.md`: disclosure of the interim test check and later decisions.
- `frozen.json`: frozen questions, threshold, selection results, versions and code hashes. `test_pack.json`: the frozen pack used for the test.
- `test_metrics.json`: aggregate test metrics. `interim_test_metrics.json`: the partial interim observation (not a full score).
- `test_predictions.jsonl` (about 20 MB): one line per test row with numeric answers, scores, token usage and latency. It holds no document text, labels or credentials, so the Jev metrics can be recomputed without new API calls.
- `final_audit.json`: independent audit (37 checks). `provenance.json`, `test_access.json`, `test_download.json`: dataset fingerprints and access timing.
- `tune_v1_metrics.json`, `tune_all_metrics.json`, `shortlist.json`, `selection_pack.json`, `selection_uncertainty.json`, `development_summary.json`: development comparisons and selection.
- `iteration_cached.json`, `iteration_cached_model.json`, `source_calibration_development.json`, `fresh_dev_availability.json`: development-only calibration study and data availability.
- `pilot_plan.json`, `pilot_metrics.json`, `usage_summary.json`: the LFQA/RAGTruth pilot and token usage.

## Setup

Requires Linux or macOS (`evaluate.py` uses `fcntl`) and Python 3; developed with Python 3.12.

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env      # then set JEV_API_KEY
```

`requirements.txt` pins `requests`, `numpy`, `pyarrow` and `scikit-learn`. `iterate_cached.py` also needs `scipy`, which scikit-learn installs.

Credentials:

- `JEV_API_KEY` (or `TYPESAFE_API_KEY` for `evaluate.py`) in the environment, else in a local `.env`. The frozen client also checks `../uncertainty/.env` as a legacy fallback. Keys are never written to result files.
- `HF_TOKEN`: a Hugging Face token with access to the gated dataset, used by `fetch_data.py`.

Offline checks (no network, no data):

```sh
python test_bench.py
python test_metrics.py
python test_evaluate.py
```

## Check your own claim

```sh
python check_claim.py --document /path/to/source.txt --claim "The claim to check."
python check_claim.py --document /path/to/source.txt --claim-file /path/to/claim.txt
```

Makes one request with the frozen question pack, threshold and preprocessing, and prints JSON: decision, score, component scores, usage and truncation flag. Requires the frozen files to be unmodified.

## Reproduce the experiment

Needs `JEV_API_KEY` and `HF_TOKEN`, and spends API credit (about $2). Run from the repository root:

```sh
python fetch_data.py dev
python bench.py prepare
python bench.py run
python analyze.py tune
python bench.py run --pack question_round2.json --output results/tune_v2_predictions.jsonl
python analyze.py tune --pack question_all.json --predictions results/tune_v1_predictions.jsonl results/tune_v2_predictions.jsonl --output results/tune_all_metrics.json
python analyze.py shortlist --pack question_all.json --output results/tune_all_metrics.json
python bench.py run --phase selection --pack results/selection_pack.json --output results/selection_predictions.jsonl --workers 32 --rps 18
python analyze.py freeze --predictions results/tune_v1_predictions.jsonl results/tune_v2_predictions.jsonl
python fetch_data.py test
python bench.py run --phase test --pack results/test_pack.json --output results/test_predictions.jsonl --workers 32 --rps 18
python analyze.py test-report
```

Notes: the second question round was written after inspecting tuning errors, so replaying it reproduces that fixed experiment, not a fresh prompt search. Caches are append-only and checked against input, pack and model identity, so interrupted runs resume. The `freeze` and `test-report` steps refuse to run if a frozen file has changed. The checked-in `results/frozen.json` and `results/test_access.json` already record the original run, so a full replay in this directory needs a fresh copy of the repository.

## Try another System One model (`evaluate.py`)

A reusable runner for any model ID supported by the System One API ([models](https://docs.typesafe.ai/models), currently `jev-1.13.0`, `jev-latest`, `jev-preview`; no hard-coded whitelist). It defaults to a **100-example development pilot** and never selects a threshold automatically. It does not touch the frozen experiment.

```sh
python evaluate.py models                                   # list visible models; no inference
python evaluate.py run --model jev-1.13.0 --dry-run         # preview; no credentials, writes or API calls
python evaluate.py run --model jev-1.13.0 --limit 100       # pilot
python evaluate.py run --model jev-1.13.0 --limit 500       # extend the same cached run
python evaluate.py run --model jev-latest jev-preview --limit 100   # compare model names
python evaluate.py run --model jev-1.13.0 --pack packs/binary_support.json --limit 100
python evaluate.py run --model jev-1.13.0 --limit 500 --offline --threshold 0.30   # rescore cache only
python evaluate.py run --model jev-1.13.0 --limit 0         # entire development selection sample
```

- **Aliases** resolve on the first response; later requests and resumptions pin the returned version. Use a new `--run-name` to refresh an alias. Names resolving to the same version are not independent comparisons.
- **Own data**: `--data file.jsonl` or `.parquet`. Required: `doc` (or `document`), `claim`, binary `label` (1 = supported). Optional: unique `id`, `dataset` (source group). Example row: `{"id":"example-1","document":"The meeting starts at 09:00.","claim":"The meeting starts at nine in the morning.","label":1,"dataset":"my-validation"}`. Pilot rows are interleaved across source/label strata (`--seed 42`), so smaller limits are prefixes of larger runs.
- **Question packs** hold API `questions`, score `extractions`, optional `combinations` (`min`, `mean`, `product`) and a `decision` (`score`, `threshold`). Copy a file from `packs/`. Older packs work with explicit `--score NAME --threshold VALUE`. Arithmetic is decimal, so `1 - 0.70` does not pass a strict `> 0.30` test by floating-point error. The original full-test result used the original floating-point rule; the decimal runner has no new full benchmark result.
- **Caching**: each model/data/question/preprocessing configuration gets its own directory under `results/runs/` (git-ignored). Raw answers are reused when only the combination or threshold changes. Partial successes survive failures; a lock prevents concurrent writers.
- **Outputs**: `comparison.md`/`.json` (coverage, source-macro balanced accuracy, pooled TPR/FPR); per-run `run.json`, `responses.jsonl`, `report.json` and versioned `report-*.json` (per-source metrics, tokens, timing, truncation; partial runs are marked incomplete); `errors.jsonl` (failed IDs; rerun to retry).
- **Other options**: `--workers`, `--rps`, `--max-doc-chars` (default 80,000, keeping 75% head and 25% tail; 0 = whole document), `--report-every`, `--output`, `--base-url`, `--api-key-env`. Prices are never assumed; for Jev 1.13 add `--input-price 0.042 --output-price 0`.

## Evaluate FactCG-DeBERTa-v3-Large

`FactCG/benchmark.py` runs [yaxili96/FactCG-DeBERTa-v3-Large](https://huggingface.co/yaxili96/FactCG-DeBERTa-v3-Large) locally on all 29,320 rows of `data/test.parquet` and produces a row for the results table. It is a **fixed zero-shot evaluation**: the model's upstream prompt template, document chunking (at most 550 words per chunk), score rule (maximum support probability over the document's chunks) and threshold 0.5 are not tuned on any benchmark data. It reuses `FactCG/factcg_client.py` and `FactCG/chunking.py` and the shared `metrics.py`, and never touches the frozen Jev files.

```sh
pip install -r FactCG/requirements.txt scikit-learn pyarrow   # torch, transformers, nltk, ...
export HF_TOKEN=...                                           # gated dataset access
python fetch_data.py test                                     # downloads data/test.parquet
python FactCG/benchmark.py --limit 200                        # optional smoke test, separate output dir
python FactCG/benchmark.py                                    # full run; resumable
python FactCG/benchmark.py --offline                          # rebuild the report from cached scores
python FactCG/test_benchmark.py                               # offline test, no GPU or network
```

The first run downloads the ~1.7 GB model into `FactCG/cache/`. A GPU is strongly recommended; `--gpu ID`, `--device`, `--batch-size` and `--window` adjust execution. Outputs go to `results/runs/factcg-deberta-v3-large/` (git-ignored): `scores.jsonl` (resumable cache, tied to model, chunking, threshold and data hash), `report.json` (per-source and pooled metrics) and `table_row.md`, a ready-made row in the format of the table above. Subset runs (`--limit`) write to a separate directory and are labelled as non-benchmark. Note that the leaderboard already lists a published FactCG-DeBERTa-L row (75.6); the row produced here is our own reproduction and is labelled as such. **This model has not been run yet in this repository**, so the table contains no row for it.

## Fixed transfer evaluation: CLM-v0.1-8B

[Contrastive-LM/CLM-v0.1-8B](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B) is self-hosted; its [official server](https://github.com/Contrastive-LM/CLM) exposes the same `/v1/systemone` schema on a Qwen3-8B pooling encoder. The plan is to run the existing three-question pack, minimum rule and `> 0.30` threshold **unchanged** on all 29,320 rows of `data/test.parquet`, with no CLM-specific tuning. Do not adjust the configuration using partial or final results. **This has not been run**: no NVIDIA driver was available where it was prepared, so only offline checks and a dry run were done.

On a CUDA machine with enough memory for the 8B encoder:

```sh
# One-time setup, separate environment
python3 -m venv .venv-clm
.venv-clm/bin/pip install -r requirements.txt 'contrastive-lm==0.1.0'
.venv-clm/bin/clm-download --dest data/clm-v0.1-8b
mkdir -p results/runs/clm-v0.1-8b
.venv-clm/bin/pip freeze > results/runs/clm-v0.1-8b/environment.txt
sha256sum data/clm-v0.1-8b/CLM_v0.1-8B.pt > results/runs/clm-v0.1-8b/checkpoint.sha256

# Terminal 1: encoder (pinned revision; 40,960 is its native context limit)
.venv-clm/bin/vllm serve Qwen/Qwen3-8B --revision b968826 \
  --served-model-name qwen3-8b --runner pooling --host 127.0.0.1 --port 8090 \
  --max-model-len 40960 --max-num-seqs 1 --enforce-eager --gpu-memory-utilization 0.90

# Terminal 2: CLM server (CPU runs only the projection heads). --max-tokens 0 disables
# CLM's default 2,048-token truncation; over-long inputs fail rather than being shortened.
.venv-clm/bin/clm-serve --host 127.0.0.1 --port 8700 --no-ui --device cpu \
  --emb-url http://127.0.0.1:8090/v1/embeddings --emb-model qwen3-8b --max-tokens 0 \
  --ckpt data/clm-v0.1-8b/CLM_v0.1-8B.pt \
  --model Contrastive-LM/CLM-v0.1-8B=data/clm-v0.1-8b/CLM_v0.1-8B.pt

# Terminal 3: evaluation (needs data/test.parquet from `python fetch_data.py test`)
sha256sum --check results/runs/clm-v0.1-8b/checkpoint.sha256 && \
.venv-clm/bin/python evaluate.py run \
  --base-url http://127.0.0.1:8700/v1 --model Contrastive-LM/CLM-v0.1-8B \
  --data data/test.parquet --limit 0 \
  --pack packs/claim_support.json --score min_support_checks --threshold 0.30 \
  --max-doc-chars 80000 --workers 1 --rps 8 --timeout 600 \
  --report-every 100 --output results/runs/clm-v0.1-8b
```

Results go to `results/runs/clm-v0.1-8b/comparison.md` and each run's `report.json`. Rerun to resume; add `--offline` to regenerate reports or `--dry-run` to check configuration. Custom endpoints receive no Jev credentials (use `--api-key-env` if you enable CLM authentication) and the endpoint URL is part of the cache identity. Keep checkpoint and server settings unchanged during a run, since CLM hot-reloads weights and its response does not attest a weight hash; use a new `--run-name` after any change.

## Possible next experiments

The [validation error review](results/validation_error_analysis/report.md) suggests these generic changes, in priority order. None is shown to improve accuracy yet.

1. Preserve the original result under its original arithmetic; measure decimal-arithmetic variants separately.
2. Add a small fixed set of attribute questions (entity, speaker, time, units, quantifiers, proposed vs completed events). Avoid an ever-growing minimum, which amplifies false rejections.
3. Check components of compound claims while keeping the original claim as the scored unit; report the cost and errors of any external decomposer.
4. Structure the evidence (speaker turns, titles, dates, table fields); for long references try relevant passages with a full-context fallback, and measure hidden contradictions.
5. Normalize numbers, dates and units with bounded rules, only when the source supports the conversion.
6. Add a conditional second pass on support/detail disagreements with a development-selected trigger; report calls per claim and coverage.

Keep it small: a short candidate list, a fixed budget, document-disjoint development groups, then one locked configuration evaluated on fresh data. The inspected selection examples are now diagnostic material, the existing test result must not guide selection, and rules must be global (no dataset names, example IDs or per-source thresholds). The rejected cached logistic calibration is not evidence that a larger search would generalize.

