# Interim stop and development-only continuation

On 2026-09-28, after the initial checker had been frozen and the test run had begun, the user requested an interim performance check to save tokens and further iteration if performance was unsatisfactory.

The test process was interrupted with 11,214 successful predictions saved. Its full-test run was not completed. At interruption, up to 32 requests could be in flight, so cached token usage is an estimate of charges rather than an invoice.

The original questions, code hashes and `frozen.json` remain unchanged. `interim_test_metrics.json` records the observed partial results. All nine completely covered sources are included, LFQA is partially covered, and RAGTruth is absent. The aggregate across the ten represented sources is **not** the official full eleven-source benchmark score. The executed prefix was not a random or stratified test sample.

The initial model's test metrics have now been observed. The original full test can no longer be described as a fresh one-shot evaluation of any subsequent adaptation. No individual test claims, documents, labels, or errors were displayed for prompt design. The continuation uses cached development predictions to optimize the score combination, with no additional API calls for that step. Its hyperparameters are selected by grouped cross-validation on the tuning pool, followed by an explicitly reused development selection check. Any later test comparison must disclose this adaptive continuation and its evaluation subset.

Preserve all existing records. Do not overwrite the original freeze, resume the full test automatically, or change questions/parameters using individual test examples.

## Subsequent decision to finish the original checker

The user subsequently asked to beat the linked leaderboard and explicitly avoid overfitting. The all-39-model audit found a maximum published eleven-source average of 77.4. An eight-variant, strongly regularized logistic calibration study used cached development predictions and did not improve the existing checker on reused development selection. It was rejected. The source-specific calibration diagnostic was also excluded from the primary checker because it requires benchmark-source identity.

No prompts, thresholds, model version, or preprocessing of the original frozen checker changed. A fixed pilot drew 200 rows per label from each of LFQA and RAGTruth, reusing 45 existing responses and requiring 755 new calls. Combined with the nine completely evaluated sources, its estimated eleven-source score was 79.07%, with document-cluster bootstrap interval 77.58–80.35%. This is a sampling estimate, not a leaderboard result or proof of the earlier 80% target.

Because that diagnostic was competitive with the published leaderboard, the original frozen test run was resumed for an exact full-benchmark comparison, reusing every pilot response. No revised checker was evaluated on test data. This decision follows the interim check requested by the user; the run is no longer described as having had no interim observations. The final checker's tuning still entirely precedes original test access.
