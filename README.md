# Decision-model benchmark

A benchmark of **decision ("System One") models**: models that read a state and answer typed questions
(`noul` yes/no, `choice`, `score`) with probabilities from a single forward pass, without generating text.
Every model gets **the same questions** on the same data; the benchmark records each model's answer to every
question for every row, its accuracy, and how fast and costly it is (latency, throughput, memory, energy).

The first task is document-grounded fact checking on [LLM-AggreFact](https://huggingface.co/datasets/lytang/LLM-AggreFact):
does the document support the claim? The results table so far is in [table.md](table.md).

## Models

| Folder | Model | Size | How it runs |
|---|---|---|---|
| [models/jpt-9b](models/jpt-9b/README.md) | JPT-9B (`kirp/jpt-9b`) | 9B | vLLM, llm2jev prompt |
| [models/rune-26b](models/rune-26b/README.md) | Rune 26B-A4B v3 (`surogate/rune-26b-a4b-GGUF`) | 26B-A4B | vLLM, surogate decisions v1 prompt; CPU offload when the GPU is too small |
| [models/winnow-12b](models/winnow-12b/README.md) | Winnow-12B NVFP4 (`EldanRing/Winnow-12B`) | 12B | the authors' llama.cpp server (built once with `setup.sh`, started by the benchmark) |
| [models/winnow-12b-bf16](models/winnow-12b-bf16/README.md) | Winnow-12B BF16 | 12B | vLLM, after converting the BF16 GGUF with `convert.py` |
| [models/clm-8b](models/clm-8b/README.md) | CLM-v0.1-8B (`Contrastive-LM/CLM-v0.1-8B`) | 8B | vLLM embeddings (Qwen3-8B) + the CLM heads |
| [models/d1-3b](models/d1-3b/README.md) | d1-3B (`LiquidAI/d1-3B`) | 3B | transformers, the model's own `system_one` API |
| [models/d3-turbo](models/d3-turbo/README.md) | d3-turbo (`vllm-sr/d3-turbo`) | 26B | transformers, the model's own System One runtime; one GPU with 56 GiB+ |
| [models/d3-flash](models/d3-flash/README.md) | d3-flash (`vllm-sr/d3-flash`) | 8.4B | transformers, the model's own System One runtime |
| [models/elephant-1](models/elephant-1/README.md) | elephant-1 (`baselabs/elephant-1`) | 9B | transformers, its own runtime (OneJev `qev` + `serve.patch`, copied in the folder); block-bidirectional attention |
| [models/knowline-4b](models/knowline-4b/README.md) | KnowLine-4B-Gen5.5 (`PelaAI/KnowLine-4B-Gen5.5`) | 4B | transformers, the model repo's own front end (`knowline_server.py`); reads short requests twice |
| [models/startlux-9b-instant](models/startlux-9b-instant/README.md) | StartLux-Decision-v1.1-9B-Instant | 9B | the authors' server (`startlux-decide` + vLLM 0.31 in the folder's own `.venv`, from its `setup.sh`); bidirectional prompt attention |
| [models/startlux-9b-autothink](models/startlux-9b-autothink/README.md) | StartLux-Decision-v1.1-9B-AutoThink | 9B | the authors' server; thinks when unsure (not downloaded yet) |
| [models/startlux-27b-autothink](models/startlux-27b-autothink/README.md) | StartLux-Decision-v1.1-27B-AutoThink | 27B | the authors' server; thinks when unsure; `quantization=fp8` on 48 GB GPUs |
| [models/factcg](models/factcg/README.md) | FactCG-DeBERTa-v3-Large | 0.4B | **baseline**, not a decision model: ignores the questions |

`python benchmark.py --list` prints them. Each folder's README covers its installation, protocol and caveats.

## Install

Everything runs in one Python environment (the authors used a conda env called `factcheck`, Python 3.12, on Linux
with NVIDIA GPUs):

```sh
pip install -r requirements.txt                 # shared: torch, vllm, transformers, pyarrow, scikit-learn, nvidia-ml-py, ...
pip install -r models/<model>/requirements.txt  # per model (e.g. llm2jev for JPT-9B, contrastive-lm for CLM)
```

Models are read **offline** from the Hugging Face cache (`~/.cache/huggingface/hub`, or `HF_HOME` / `HF_HUB_CACHE`):
download them first (`hf download kirp/jpt-9b`, ...). Two models need a one-time extra step:
`models/winnow-12b/setup.sh` (builds the Winnow server) and `python models/winnow-12b-bf16/convert.py`
(GGUF to safetensors). The dataset lives in `dataset/` or the Hugging Face cache; see [dataset/README.md](dataset/README.md).

On a machine behind a proxy, `hf download` may fail; copying the cache folders (`rsync -a ~/.cache/huggingface/hub/models--...`)
and the `dataset/` folder from another machine works, keeping the `-a` flag so the cache's symlinks stay intact.

## Run

```sh
python benchmark.py --model d1-3b --gpu 2                                   # LLM-AggreFact test split
python benchmark.py --model d1-3b jpt-9b clm-8b --gpu 2                     # several models, one after another
python benchmark.py --model jpt-9b --gpu 0 --split dev-jev4520              # another split
python benchmark.py --dataset prova --split test.json --model d1-3b --gpu 0 # dataset/prova/test.json
python benchmark.py --model jpt-9b --gpu 0 --limit 32                       # quick check on 32 rows
python benchmark.py --model rune-26b --gpu 1 --opt rune-26b:cpu_offload_gb=18
```

- **One model at a time, one GPU.** Several models run strictly one after another, each in its own process on the
  single GPU given by `--gpu` (the `nvidia-smi` index). Running two models at once would distort latency and memory.
  The run refuses to start if another compute process uses that GPU (`--allow-busy-gpu` overrides it, and is recorded).
- **Resumable.** Interrupt with Ctrl-C and rerun the same command: finished rows are skipped. A run folder is tied to
  the model, its options, the questions, the data and `--limit`; if the data changed under the same folder, the run is
  refused rather than mixed in.
- **Run options.** `--batch`, `--max-doc-chars` (default 80,000, keeping 75% head and 25% tail), `--latency-sample` (200),
  `--throughput-sizes` (`1,8,32`), `--throughput-sample` (64), `--no-profile`, `--tag`.

### Full test and dev runs

A run covers the whole split unless `--limit` is given. LLM-AggreFact has three splits in `dataset/lytang--LLM-AggreFact/`:

| `--split` | Rows | Use |
|---|---:|---|
| `test` (default) | 29,320 | the benchmark result (the numbers for `table.md`) |
| `dev` | 30,420 | the full official development split |
| `dev-jev4520` | 4,520 | the development rows Jev's 0.30 threshold was fitted on; use it to fit each model's own threshold |

```sh
# full test split, every decision model one after another on GPU 2 (FactCG is the baseline: add it if wanted)
nohup python benchmark.py --model d1-3b jpt-9b clm-8b winnow-12b winnow-12b-bf16 rune-26b --gpu 2 \
      --split test > test_run.log 2>&1 &

# full official dev split
nohup python benchmark.py --model d1-3b jpt-9b clm-8b winnow-12b winnow-12b-bf16 rune-26b --gpu 2 \
      --split dev > dev_run.log 2>&1 &

# only the 4,520 rows needed to fit thresholds (much shorter), then apply them to the test results
python benchmark.py --model d1-3b jpt-9b --gpu 2 --split dev-jev4520
python report.py --split test --calibrate-on dev-jev4520
```

- **Same command to resume.** If a long run stops (Ctrl-C, reboot, a lost SSH session), run the same command again:
  each model skips the rows it has already scored. The latency probe and throughput sweep are done once per result folder.
- **Calibration needs the same models and options.** `--calibrate-on` matches the dev and test folders by name, which
  includes the params hash: run both splits with the same options.
- **Duration.** One model's full test split takes from well under an hour (d1-3B: about 12 rows/s on an RTX 5090 Laptop
  GPU) to many hours when the weights do not fit the GPU (Rune with CPU offload on a 24 GB card: about 0.4 rows/s, roughly
  20 hours). `tail -f test_run.log` shows the progress; the per-model summary is printed at the end, and the comparison
  tables are written to `results/lytang--LLM-AggreFact/<split>/summary_<hash>.md`.
- **Test results are final.** Do not tune questions, thresholds or options on the test split: choose them on `dev` or
  `dev-jev4520`, then run the test split once.

## Model options

Each model's hyperparameters (temperature, context length, CPU offload, batch size, model and encoder revisions, ...)
live in [config/models/](config/models/)`<model>.json`, initialised with the values used so far and commented in their
`_comment` field. The precedence is: the adapter's built-in defaults < the config file < the command line:

```sh
python benchmark.py --model rune-26b --gpu 0                                # config/models/rune-26b.json
python benchmark.py --model rune-26b --gpu 0 --opt temperature=1            # one value overridden
python benchmark.py --model jpt-9b rune-26b --gpu 0 --opt rune-26b:cpu_offload_gb=18
python benchmark.py --model rune-26b --gpu 0 --model-config my_rune.json    # another options file (one model)
```

The effective options get an 8-character **params hash**, part of the result folder name; the first run with an option set
records it in `results/<dataset>/<split>/hash_<model>.json` (params hash -> options). So the same model with other options
(another temperature, another revision) lands in its own folder, next to the first one, and both can be compared.

## Questions

The question set is a JSON file in `config/questions/`; the default, [claim_support.json](config/questions/claim_support.json),
holds the three yes/no checks of the original Jev experiment (direct support, any unsupported detail, any contradicted
detail) and its decision rule: `min(P(support), 1 - P(unsupported detail), 1 - P(contradiction)) > 0.30`.
Every model gets exactly these questions, rendered in its own prompt format. Use `--questions FILE` for another set.

Results are kept apart by an 8-character **hash of the questions** (the decision rule is not part of it: it is applied
when the report is built, so it can change without new inference); see also the params hash below. The first run with a question set records it in
`results/<dataset>/<split>/hash_questions.json`; later runs with the same questions leave the file unchanged, a new set is
appended.

## Results

```
results/<dataset>/<split>/hash_questions.json           questions hash -> question set
results/<dataset>/<split>/hash_<model>.json             params hash -> the model's options
results/<dataset>/<split>/summary_<questions hash>.md   comparison tables of all models (written by report.py)
results/<dataset>/<split>/<model>_<params hash>_<questions hash>[-limitN][-tag]/
    run.json          identity, effective options and where they came from, model revision, data fingerprint, GPU,
                      driver, package versions, git commit,
                      per-session start/end, load time, first-call time, telemetry summary per phase
    answers.jsonl     one line per row: source, label, claim/document length (characters, words), truncation, prompt
                      tokens per question, and for EVERY question the full option distribution and raw logits
    batches.jsonl     main pass: one line per batch (rows, failures, tokens, seconds)
    latency.jsonl     latency probe: a fixed stratified sample, one row at a time with a cold cache, seconds per row
    throughput.jsonl  rows/s at several batch sizes on a fixed sample
    telemetry.jsonl   every 0.2 s: GPU power, utilisation, memory, temperature, clocks, throttle reasons; process RSS
    failures.jsonl    rows that failed (too long, out of memory, timeout) and why; they never get an invented score
    report.json       everything below, rebuilt from the files above
```

Every file and field is described in [README_RESULTS.md](README_RESULTS.md). `report.json` holds: balanced accuracy (macro over sources, as the LLM-AggreFact leaderboard computes it), pooled
accuracy, ROC AUC and recalls under the decision rule; the same **per source** and **per length bucket** (claim
characters, document characters, prompt tokens); each question's own AUC; latency percentiles (p50/p90/p95/p99) overall,
per source and per length; throughput (rows, questions and prompt tokens per second) and the batch-size sweep; load time,
first-call time, peak GPU and CPU memory, energy per 1,000 rows, mean/max power, utilisation, temperature and minimum
clock (throttling); coverage and failures. Only the small files (`run.json`, `report.json`, summaries, the hash registry)
are committed; the per-row and telemetry files stay local.

## Reports without inference

```sh
python report.py                                     # refresh every report.json + summary_<hash>.md
python report.py --threshold 0.5                     # another threshold
python report.py --rule support_simple               # judge by one question only (QUESTION or QUESTION:no)
python report.py --decision my_rule.json             # any rule over the stored answers (see core/rules.py)
python report.py --temperature 2                     # re-temper the models that store logits
python report.py --calibrate-on dev-jev4520          # each model's threshold fitted on that split's run, applied here
```

## Code layout

```
benchmark.py   report.py               the two entry points
core/          datasets, questions + hash registry, decision rules, adapter interface, runner, profiling, report, metrics
models/<m>/    adapter.py (the model's protocol) + README, requirements and the model's own tools
config/        question sets (config/questions/) and model options (config/models/)
dataset/       local datasets (not committed)
results/       benchmark output
tests/         python tests/test_core.py (no GPU) and python tests/test_adapters.py (offline prompt checks)
legacy/        the original Jev experiment and the per-model caches of the first runs, kept unchanged (legacy/README.md)
```

To add a model, create `models/<name>/adapter.py` with a class `Adapter(core.adapter.BaseAdapter)` that declares its
options in `DEFAULTS` and implements `load()` and `score(states, questions)`, returning per row the answers built with
`core.adapter.make_answer` (see the existing adapters), plus `config/models/<name>.json` with its options. To add a
dataset, put its splits in `dataset/<name>/`.
