# Development thresholds on the L40S machine

Gives JPT-9B, Rune-26B, Winnow-12B (NVFP4) and Winnow-12B-bf16 their own threshold, fitted on Jev's 4,520 development rows, and applies it once to the existing test caches. Run from the repository root in the `factcheck` conda environment.

## Before starting

- The dev split must be in `./lytang___llm-aggre_fact` (the same Hugging Face cache folder as the test split).
- Model weights in `~/.cache/huggingface/hub`: `kirp/jpt-9b` at revision `b447cc7` (pinned in the script), `surogate/rune-26b-a4b-GGUF`, `EldanRing/Winnow-12B`. For Winnow-bf16, run `python winnow-12b-bf16/convert.py` once.

## Steps

```sh
python dev_split.py                       # rebuilds data/tune.jsonl + data/selection.jsonl; stops if the hashes differ from Jev's

# development runs (resumable; each writes results/runs/<model>-dev/)
python jpt-9b/benchmark.py --split dev --gpu 0
python winnow-12b-bf16/benchmark.py --split dev --gpu 1
python rune-26b/benchmark.py --split dev --gpu 0,1 --tensor-parallel 2 --cpu-offload-gb 0

winnow-12b/setup.sh                       # once per machine: builds the Winnow server
GPU=0 winnow-12b/serve.sh                 # other terminal; leave running
python winnow-12b/benchmark.py --split dev

python calibrate_transfer.py fit          # thresholds from the dev caches only
```

Then copy the complete test caches (`answers.jsonl`) into `results/runs/<model>/` for every model, including `rune-26b` and `winnow-12b-bf16`, and run once:

```sh
python calibrate_transfer.py apply        # -> results/runs/dev_threshold_summary.md
```

Smoke test any step first with `--limit 4` (outputs go to `<model>-dev-limit4/` and are never used by `fit`).
