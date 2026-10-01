# Official LLM-AggreFact leaderboard audit

Fetched 2026-09-28 at 09:52 UTC. All **39 available models** were checked across **all 11 datasets**.

The highest official displayed score is **77.4**, from **Bespoke-Minicheck-7B**. Claude-3.5 Sonnet follows at **77.2**. The public page selects only 11 models by default, but its embedded `scoresData` contains the full 39-model table. [Official leaderboard](https://llm-aggrefact.github.io/)

The metric is the equally weighted mean of 11 per-dataset balanced accuracies, expressed as percentages. The site recalculates Average when dataset columns are selected, so a subset average is not the full benchmark. The mean of the leader's already rounded cells is 77.40909; this is not its original unrounded score. [Official page script](https://llm-aggrefact.github.io/_next/static/chunks/app/page-bcaff6a888eff26e.js)

The official blog characterizes listed results as zero-shot and discusses per-dataset threshold tuning separately. Results chosen or calibrated on this benchmark's development set must disclose that tuning and cannot be treated as a like-for-like zero-shot leaderboard entry. [Official protocol blog](https://llm-aggrefact.github.io/blog)

The official evaluation notebook uses the test split, a default probability threshold of 0.5, per-dataset balanced accuracy, then an arithmetic mean over datasets. It was inspected but not executed. A development-set result or sampled/partial test result cannot establish a full 11-dataset benchmark improvement. [Official evaluation notebook](https://github.com/Liyan06/MiniCheck/blob/main/benchmark_evaluation_demo.ipynb)

`leaderboard_audit.json` contains every model's 11 aggregate scores, recomputed averages, exact source URLs, fetch timestamps, and source SHA-256 hashes. No benchmark examples, labels, credentials, or local `.env` files were accessed.
