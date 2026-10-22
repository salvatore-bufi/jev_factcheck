# Development thresholds on the L40S machine

Gives JPT-9B, Rune-26B, Winnow-12B (NVFP4) and Winnow-12B-bf16 their own threshold, fitted on Jev's 4,520 development rows, and applies it once to the existing test caches. Run from the repository root in the `factcheck` conda environment.

## Before starting

- Packages (versions used for the test runs; `llm2jev` builds JPT-9B's prompts, so keep it at exactly 0.6.1):

  ```sh
  pip install llm2jev==0.6.1 "vllm==0.28.0" "transformers==5.14.1" gguf==0.19.0 safetensors
  pip install -r requirements.txt
  ```

- The dev split must be in `./lytang___llm-aggre_fact` (the same Hugging Face cache folder as the test split).
- Model weights in `~/.cache/huggingface/hub`. The benchmarks run with `HF_HUB_OFFLINE=1` and never download, so fetch them once (about 18 + 49 + 43 GB). JPT-9B must be fetched at the revision the script pins, or the offline load fails:

  ```sh
  hf download kirp/jpt-9b --revision b447cc7ee105c0a76a22f8fde8ecf05074dc8be0
  hf download surogate/rune-26b-a4b-GGUF
  hf download EldanRing/Winnow-12B
  python winnow-12b-bf16/convert.py         # once: Winnow GGUF -> bf16 safetensors in winnow-12b-bf16/model/
  ```

## Steps

```sh
python dev_split.py                       # rebuilds data/tune.jsonl + data/selection.jsonl; stops if the hashes differ from Jev's

# development runs (resumable; each writes results/runs/<model>-dev/)
# --split dev is required: without it the script scores the full test split again
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
