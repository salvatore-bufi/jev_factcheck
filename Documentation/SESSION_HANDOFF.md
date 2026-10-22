# Session handoff (2026-10-07) — resume from here

User request (Italian, verbatim intent):
1. Delete useless files and tidy the repo.
2. In `Documentation/`:
   - `finetuning_strategy.md`: detailed strategy to fine-tune **Winnow-12B** and **JPT-9B** (individually) for
     **claim–document fact checking, no retrieval**, including experiments/ablations.
   - `dataset_finetuning.md`: review the datasets in `Datasets/`, select usable subsets (no retrieval part),
     build something to extract document–claim pairs, do an extended search for other datasets,
     **rank datasets by expected quality after fine-tuning** (user thinks FactKG is good).
- Always use the conda env **`factcheck`** (`conda run -n factcheck python ...`).
- Write docs in Italian (user writes Italian; existing Datasets/*.md are Italian).

## Status (updated 2026-10-08): all requested deliverables DONE

| Step | Status |
|---|---|
| Cleanup + reorganisation | DONE (offline tests pass) |
| `data_prep/build_pairs.py`, `data_prep/make_mix.py` | DONE; full raw build of all 11 sources in `data/finetune/raw/` |
| Mixes | DONE: `mix_pilot` 32,872/4,204, `mix_core` 106,082/4,480, `mix_extended` 108,999/4,590 (train/val) |
| Contamination (full run) | RAGTruth 98% docs overlap -> excluded from all recipes; WiCE 30% (train-only 13%) -> `extended` only |
| `Documentation/dataset_finetuning.md` | DONE |
| `Documentation/finetuning_strategy.md` | DONE |
| New `README.md`, `.gitignore` fix | DONE |

Next steps (not requested yet, see strategy §8–9): `finetune/render.py`, `finetune/train.py` (letter-logit CE/Brier loss),
`finetune/merge.py`, `--model-dir` option in `jpt-9b/benchmark.py` and `winnow-12b-bf16/benchmark.py`, Phase 0 baselines B2–B4.
Docs revised on 2026-10-08 after user feedback (no code changed, by request): goal = general fact checker
(text doc vs text claim), not only AggreFact; FactKG must not be used as triples (only after LLM verbalisation,
experiment D3); FactCG/CG2C was not used, now recommended (data not released, regenerate with MIT code);
SciFact kept out as OOD test; recommended JSON recipes `pilot_text`/`core_text` documented in dataset_finetuning.md §6
(make_mix.py RECIPES still include factkg/scifact). Strategy doc rewritten: real prompts of both models, letter
token ids (Winnow A=236776/B=236799, A=false; JPT " A"=357/" B"=417, A=yes, T=1.087), loss worked examples,
automatic paraphrases/replay with LLM options (local L40S, free APIs, paid ranking) and prompts, replay datasets
(LocalLLaMA/typed-decisions train split; test split + JevBench + Kev = forgetting evals), general eval suite, contrastive §11.
Note: `conda run` does not forward stdin -> run Python scripts from files, not heredocs.

## 1. What the cleanup did

Deleted: `__pycache__/`, `jpt9.log`, `winnowquantizied.log`, `data/test.part` (partial download), `table.md`
(duplicate of ordered_table), `Datasets/.vscode`, the AggreFact `.incomplete_info.lock`, and **already-extracted
archives** (verified file-by-file, same sizes): `Datasets/ClaimDB/data/{train_dbs,test_dbs}.zip` (9.4 GB),
`Datasets/Fever/data/wiki-pages.zip` (1.6 GB), `Datasets/FACTKG/data/factkg.zip`, all `__MACOSX/` dirs.

Moved:
- Jev-API experiment (not needed for fine-tuning) → `legacy/jev_experiment/`: `analyze.py`, `fetch_data.py`, `pilot.py`,
  `iterate_cached.py`, `analyze_validation_errors.py`, `check_claim.py`, `test_bench.py`, `question_*.json`,
  `question_design.md`, `PROTOCOL.md`, old `README.md`. They import root modules and are archived, not runnable in place.
- FactCG original-repo files not used by `FactCG/benchmark.py` → `legacy/FactCG_original/` (`main.py`, `run_identity.py`, Italian docs).
- `ordered_table.md` → `results/model_comparison.md`
- `model_questions.md` → `Documentation/prompt_formats.md` (exact prompt format per model — important for fine-tuning)
- `run_on_newpc.md` → `Documentation/run_dev_thresholds.md`
- `benchmark_notes.md` → `Documentation/aggrefact_protocol_notes.md` (states AggreFact README forbids fine-tuning on it)
- `finetune.md` → `Documentation/notes/finetune_first_notes.md`
- `Documentation/llm2decisionModels.md` → `Documentation/notes/candidate_base_models.md`

Kept at root (dependencies): `bench.py` (imported by `dev_split.py`), `metrics.py`, `evaluate.py` (`state_for` used by
all model benchmarks), `dev_split.py`, `calibrate_transfer.py`, `packs/`, `table_results.py`, tests. Model folders
(`jpt-9b/`, `winnow-12b/`, `winnow-12b-bf16/` incl. 23 GB bf16 safetensors needed for training, `rune-26b/`, `FactCG/`)
unchanged: their scripts use `ROOT = HERE.parent`, so they must stay one level below root.
Not touched: `results/` (incl. `results/runs/*` caches), `references/`, `.claude/`, dataset contents.
Possible further space saving to *propose* to the user (not done): ClaimDB (32 GB dbs) and TSVer are unusable for
this task; `Datasets/FACTKG/data/dbpedia_2015_undirected.pickle` (5.8 GB) is unused (builder uses the light 1.5 GB one).

Tests after the move (all pass): `test_metrics.py`, `test_dev_split.py`, `test_calibrate_transfer.py`, `test_evaluate.py`,
`jpt-9b/test_benchmark.py`, `winnow-12b/test_benchmark.py`, `winnow-12b-bf16/test_benchmark.py`, `FactCG/test_benchmark.py`.

## 2. Data tooling written (`data_prep/`)

`build_pairs.py <sources|all> [--splits train dev] [--limit N]` → `data/finetune/raw/<src>.jsonl` (+ `.stats.json`).
Row schema: `id, source, split, document, claim, label (1=supported), orig_label, doc_id (group key), meta`.
Docs capped at 24,000 chars (flag `meta.truncated`).

| Source | Where | Document construction | Label mapping | Smoke result |
|---|---|---|---|---|
| factkg | local pickles + `dbpedia_2015_undirected_light.pickle` | verbalised triples `s \| rel \| o` on all annotated evidence paths (≤6 tails/rel) + 8 random distractor triples per entity, shuffled; drops `question`-type claims | True→1, False→0 | ok, doc p50 1.3k chars. Note: paths from literal entities (dates) pull noisy inverse edges |
| hover | local json + `wiki_wo_links.db` | intro text of every supporting page (2–4), shuffled; test unlabelled → skipped | SUPPORTED→1, NOT_SUPPORTED→0 | ok, p50 1.1k |
| fever | train/paper_dev/paper_test parquet in HF cache (`datasets--fever--fever/.../v1.0`) + `Datasets/Fever/data/wiki-pages/wiki-pages/*.jsonl` | full pages (≤40 lines + gold lines) of first evidence set; NEI paired with page = longest title n-gram match in claim | SUPPORTS→1, REFUTES/NEI→0 | ok, p50 1.4k; 300/split takes ~35 s (streams 7 GB twice) |
| averitec | local json | annotator Q/A evidence (+boolean explanation, source domain) + speaker/date header; no justification; Conflicting dropped; test unlabelled | Supported→1, Refuted/NEE→0 | ok |
| minicheck | HF `lytang/C2D-and-D2C-MiniCheck` (splits c2d, d2c; MIT) | doc | label | ok |
| anli | HF `facebook/anli` r1–r3 | premise | entailment→1 | ok |
| vitaminc | HF `tals/vitaminc` | `Title: page` + evidence | SUPPORTS→1 | ok |
| summedits | HF `Salesforce/summedits` (train only) | doc / summary | 1=consistent | ok |
| scifact | HF `allenai/scifact_entailment`, `revision="refs/convert/parquet"` (repo has a loading script) | title + abstract | SUPPORT→1 | ok |
| wice | HF `tasksource/wice` | evidence sentences | supported→1 | ok, **contamination risk** |
| ragtruth | HF `wandb/RAGTruth-processed` | context (+question for QA) vs whole response | no hallucination span→1 | ok, **contamination risk** |

HF downloads sometimes fail with `httpx.ProxyError: 500` (sandbox proxy) → just retry.

`make_mix.py --recipe core|extended|file.json [--name X]` → `data/finetune/mix_<name>/{train,val}.jsonl, stats.json, decontamination.json`.
Decontam vs local AggreFact dev+test arrow (`lytang___llm-aggre_fact/...`): exact normalised claim match, or ≥20% of the
document's sampled word-13-grams (1/8 hash sampling) seen in AggreFact docs. Then length filter, dedup, val = source dev
split or 5% doc-hash slice, train rows sharing a doc_id with val dropped, per-source cap + 50/50 balance.
Recipes: `core` = minicheck(all), factkg 20k, hover 15k, fever 20k, anli 15k, vitaminc 15k, summedits, averitec, scifact;
`extended` = core + wice + ragtruth 8k.

**Key finding of the smoke mix** (small samples): RAGTruth-train documents **20/20** flagged as overlapping AggreFact,
WiCE **25/40**, FEVER 1/300, others 0. ⇒ RAGTruth and WiCE must not be used for training (eval-only or excluded).
Pending command to quantify on 600 rows (was interrupted):
```sh
S=<scratch dir>; conda run -n factcheck python data_prep/build_pairs.py wice ragtruth summedits minicheck --limit 600 --splits train --out $S/raw600
# rename *.limit600.jsonl -> <src>.jsonl, then
echo '{"wice":{"cap":0},"ragtruth":{"cap":0},"summedits":{"cap":0},"minicheck":{"cap":0}}' > $S/r.json
conda run -n factcheck python data_prep/make_mix.py --recipe $S/r.json --name contam --raw $S/raw600
```

## 3. Facts gathered for the two documents

Baseline (LLM-AggreFact test, macro BAcc over 11 sources, fixed Jev 3-question rule; `results/runs/dev_threshold_summary.md`):
| Model | @0.30 | @dev threshold | test macro AUC |
|---|---|---|---|
| jev-1.13.0 (dev-tuned reference) | 79.09 | — | 86.89 |
| JPT-9B (T=1.087) | 76.04 | 76.71 (thr 0.72) | 84.57 |
| Winnow-12B-NVFP4 | 77.49 | 77.41 (0.23) | 85.46 |
| Winnow-12B-BF16 | 76.99 | 77.42 (0.22) | 85.02 |
| Rune-26B-A4B | 76.32 | 76.68 | 84.56 |
| Bespoke-MiniCheck-7B (leaderboard) | 77.4 | | |
Weak sources for JPT: CNN 62, MediaS 67.5; for Winnow-BF16: WiCE 77, FactCheck 74, XSum 73. ExpertQA ~59 for all (label noise ceiling).

Models: JPT-9B = Qwen3.5-9B + merged LoRA (r16, all attn+MLP, lr 5e-5, 1 epoch, ~49k typed questions, multi-class
Brier loss on option-letter logits, `llm2jev` prompt, thinking off, `Answer:` prefill, trained with ms-swift; CC BY-NC 4.0;
weights pinned rev `b447cc7`; option A=Yes/true first; T=1.087). Winnow-12B = Gemma-4-12B fine-tune, GGUF only
(converted losslessly to bf16 safetensors in `winnow-12b-bf16/model/`, 22 GB, `gemma4_text`), own prompt
(system + `State:` compact JSON, `Question: "..."`, `A: "false: ..."`, `B: "true: ..."`, `Return the correct letter label.`,
thinking-off block, `Answer:`), A=false, T=1, Apache 2.0. Prompts are reproduced exactly in
`winnow-12b-bf16/benchmark.py` and `jpt-9b/benchmark.py` (via llm2jev 0.6.1) → reuse them for training data rendering.
Hardware here: RTX 5090 Laptop 24 GB, 62 GB RAM; also available an L40S 45 GB ×2 machine.
AggreFact README forbids training on it; dev (30,420 rows) may be used for threshold/model selection only;
`data/tune.jsonl` + `data/selection.jsonl` = Jev's 4,520 dev rows (`dev_split.py`).

Strategy doc planned content (from earlier notes + analysis):
- Phase 0: baselines already in hand; re-run dev with each model; metric = AggreFact macro BAcc + AUC (threshold-free) + Brier/ECE.
- Phase 1 (no training): threshold/temperature/question-pack ablation (single support question vs 3-question min rule).
- Phase 2: LoRA SFT in each model's *native* prompt; loss = CE/Brier on the two option-letter logits only at the answer
  position; train all three questions (support / unsupported_detail / contradicted_detail) with labels derived from the
  data (supported → q1=true,q2=false,q3=false; contradicted (REFUTES/False with contrary evidence) → q1=false,q3=true,q2=true;
  NEI → q1=false,q2=true,q3=false) — gives the min-rule consistent targets; also random option-order/phrasing paraphrases.
- LoRA r16–32 all proj, lr 1e-5–5e-5, 1–2 epochs, seq ≤ 8k (JPT on 24 GB via QLoRA; Winnow bf16 on L40S), grad ckpt.
- Replay 10–20% generic typed-decision data (no original data public) to avoid forgetting; low LR.
- Curriculum/ablations: data recipe (core vs core-minus-X leave-one-source-out, FactKG-only, synthetic-only), size scaling
  (5k/20k/60k), label-balance, doc length, 1 vs 3 questions, Brier vs CE, LoRA rank, control = same recipe on base model
  (Qwen3.5-9B / Gemma-4-12B) to prove the decision-model init helps.
- Selection only on AggreFact dev (tune/selection pools), one-shot test; report per-source, bootstrap CI by document.
- Synthetic data generation (MiniCheck C2D/D2C style, FactCG CG2C graph multi-hop) as phase 3 to target weak sources
  (summaries of news/meetings: CNN, MediaSum, MeetingBank) — use documents from non-AggreFact corpora.

Dataset ranking planned (expected value for AggreFact-style grounded checking, high→low):
1. MiniCheck C2D/D2C (designed for exactly this task; Bespoke-MiniCheck-7B = 21k ANLI + 14k C2D/D2C → 77.4)
2. ANLI (hard, adversarial; part of MiniCheck recipe) 3. SummEdits (summaries, 10 domains — targets CNN/XSum/MediaS/MeetB)
4. FactKG (huge, balanced, multi-hop/negation/existence; but KG-triple documents ≠ prose → format shift, use as part)
5. HoVer (multi-hop prose, gold pages, 18k train) 6. FEVER (large, single-hop, NEI pairing heuristic) 7. VitaminC (contrastive, short evidence)
8. AVeriTeC (real claims, QA evidence, small) 9. SciFact (small, domain) ; Excluded/eval-only: WiCE & RAGTruth (contamination),
LLM-AggreFact (license), ClaimDB (needs SQL/tool use), TSVer (series data not shipped, 304 claims), LIAR/RealFactBench (no
evidence doc / open-world), LoReFact (only PDF, "link unavailable").
Other datasets found in the web search to list as candidates: DocNLI (`tasksource/doc-nli`, document-level NLI, noisy),
FEVER-NLI (`pietrolesci/nli_fever`), FEVEROUS (`fever/feverous`, text+table), HealthVer / COVID-Fact (`dwadden/*_entailment`),
Climate-FEVER, HaluBench (`PatronusAI/HaluBench`, contains RAGTruth → eval only), HalluMix (`quotientai/HalluMix`, mixture,
check overlap), LLM-Oasis (Babelscape, Wikipedia claims, large), AmbiFC (10k claims, 50k passages, soft labels),
FactCG CG2C data (github derenlei/FactCG), HalluLens, ANAH, SciFact-Open, TabFact, Lynx.
Sources used: arxiv 2404.10774 (MiniCheck), 2501.17144 (FactCG), 2505.00506 (HalluMix), 2411.19655 (LLM-Oasis),
2407.08488 (Lynx/HaluBench), 2103.08541 (VitaminC), AmbiFC TACL 2024, Bespoke-MiniCheck-7B model card.
