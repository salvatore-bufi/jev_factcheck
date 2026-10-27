# Datasets

One folder per dataset; `benchmark.py --dataset NAME --split SPLIT` reads `dataset/NAME/SPLIT.{arrow,parquet,jsonl,json}`
(or the exact file given with `--split FILE`). When the folder or file is missing it falls back to the Hugging Face
hub cache (`datasets--ORG--NAME`, honouring `HF_HOME` / `HF_HUB_CACHE`), where the split is read from
`snapshots/*/data/SPLIT-*.parquet`. The data files themselves are not committed (licences, size).

A row needs `doc` (or `document`), `claim` and a binary `label` (1 = the document supports the claim). The optional
`dataset` (or `source`) column names the source group for the per-source metrics; an optional `id` is kept.

## lytang--LLM-AggreFact

[LLM-AggreFact](https://huggingface.co/datasets/lytang/LLM-AggreFact), revision `981dfd0bd8e58e7238a9ab92b2e6ea44bce918e4`
(also written in `REVISION`). Its licence allows evaluation only, not training.

| File | Rows | Content |
|---|---:|---|
| `test.arrow` | 29,320 | official test split (11 sources) |
| `dev.arrow` | 30,420 | official development split |
| `dev-jev4520.jsonl` | 4,520 | the development rows Jev's threshold was fitted on (its tune and selection pools, ids `dev:<row>`); use it to fit per-model thresholds (`report.py --calibrate-on dev-jev4520`) |
| `test4k.jsonl` | 3,998 | AF-4k: 364 test rows per source (all 358 of Wice), class-balanced where possible, seed 2026, test ids kept; built by `python analysis/make_test4k.py` (sha256 `2f0baa0d…`) |
| `dataset_info.json` | | metadata from the Hugging Face `datasets` cache |

`test.arrow` and `dev.arrow` are the Hugging Face `datasets` cache files of that revision, renamed. To recreate them on
another machine either copy this folder, or download the dataset (gated: accept its terms on the Hub first):
`hf download lytang/LLM-AggreFact --repo-type dataset`, after which the hub-cache fallback finds the parquet files.
The test rows were checked against the legacy Jev run: all 29,320 input hashes match.
