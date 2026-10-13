# Legacy: the original Jev experiment

This folder keeps, unchanged, the experiment the repository started with: `jev-1.13.0` (TypeSafe's System One API)
evaluated on LLM-AggreFact, with questions and a threshold tuned on development data and frozen before the test split
was downloaded. Its full-test result, **79.09%** mean per-source balanced accuracy, is the `jev-1.13.0` row of
[../table.md](../table.md). The new benchmark (`../benchmark.py`) reuses its final question set
(`packs/claim_support.json` is `../config/questions/claim_support.json`) and its metric code (`metrics.py` is
`../core/metrics.py`).

| Path | Content |
|---|---|
| `PROTOCOL.md`, `benchmark_notes.md` | the pre-registered protocol and the benchmark's rules |
| `question_candidates.json`, `question_round2.json`, `question_all.json`, `question_design.md` | the candidate questions of the two tuning rounds |
| `packs/` | the frozen three-question pack and a one-question variant |
| `bench.py`, `analyze.py`, `fetch_data.py`, `pilot.py`, `iterate_cached.py`, `analyze_validation_errors.py`, `check_claim.py` | the original pipeline |
| `evaluate.py` | the old generic System One runner (Jev API or any compatible server) |
| `dev_split.py`, `calibrate_transfer.py` | rebuild of Jev's 4,520 development rows and per-model threshold fitting of the first transfer runs |
| `metrics.py`, `tests/` | metric code and the original offline tests |
| `results/` | frozen configuration (`frozen.json`), Jev's test predictions, reports, audits, validation error analysis |
| `data/` | pinned Hub metadata (and, locally only, the tune/selection pools) |
| `references/` | saved API documentation and the 39-model leaderboard audit |
| `results_runs/` | local only (git-ignored): the per-model caches of the first runs (JPT-9B, Rune, Winnow NVFP4 and BF16, CLM, d1-3B, FactCG; test and dev), in their old format |
| `factcg_original/`, `notes/` | the FactCG module's original CLI and notes, and a list of candidate base models |

The scripts expect to run from this folder (`cd legacy && python bench.py ...`); `bench.verify_freeze()` checks the
frozen files' hashes relative to this folder, so they still verify after the move. The original pipeline downloaded data
into `data/`; the LLM-AggreFact files now live in `../dataset/lytang--LLM-AggreFact/`.
