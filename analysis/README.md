# analysis

Scripts of the focused 10-TDM study (`Documentation/paper_idea/15_focused_10_tdm_experiment_guide.md`). They read the
benchmark's saved answers and run no inference, except `make_test4k.py`, which only writes a data file. Run them from
the repository root.

| Script | What it does | Output |
|---|---|---|
| `make_test4k.py` | builds AF-4k: 364 rows per LLM-AggreFact test source (all 358 of Wice), as class-balanced as each source allows, seed 2026. Rows keep their test ids (`test:<index>`), so full-test answers can be restricted to AF-4k without new inference | `dataset/lytang--LLM-AggreFact/test4k.jsonl` (3,998 rows, sha256 `2f0baa0d…`) |
| `make_runs.py` | finds the result folders of the study and records which folder holds which (system, question file, split). A folder is recognised by the identity header and the first answer of its `answers.jsonl`, never by its name; `--limit` runs are ignored; two complete folders for one cell stop the script | `analysis/focused_runs.json` |
| `focused_10.py` | computes every table of the study from the runs file: main results, attribution to backbone / readout / recipe, protocol sensitivity, calibration and threshold transfer, Decision Index vs AggreFact. Thresholds are fitted on dev; intervals come from a paired document-level bootstrap | `results/focused_10/table_*.md` |

```sh
python analysis/make_test4k.py
python analysis/make_runs.py
python analysis/focused_10.py --runs analysis/focused_runs.json --out results/focused_10     # --boot 10000 by default
```

`tests/test_focused.py` runs `make_runs.py` and `focused_10.py` end to end on synthetic runs laid out as the harness
writes them.
