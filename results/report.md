# Jev claim checking: full benchmark result

**Jev reached 79.09% mean per-source balanced accuracy on all 29,320 LLM-AggreFact test examples.** This is numerically about 1.7 points above the linked leaderboard’s published maximum of 77.4, but below the requested 80% target.

This is a **development-tuned** result. The linked leaderboard describes its entries as zero-shot, so this is not a like-for-like zero-shot SOTA claim or a registered leaderboard entry. The final checker’s prompts, threshold and preprocessing were frozen before original test access and never changed. Interim test inspection, the rejected development-only calibration experiment, and the decision to resume the unchanged checker are documented in [INTERIM_AMENDMENT.md](INTERIM_AMENDMENT.md).

## Main comparison

| System | Mean source balanced accuracy | Setting |
|---|---:|---|
| Jev, three-question rule | **79.09%** | Dev-tuned, full test |
| Jev, simple support question | 78.05% | Dev threshold 0.84, full test |
| Bespoke-Minicheck-7B | 77.4% | Published zero-shot leaderboard maximum |
| Jev, simple support question | 76.18% | Default threshold 0.5, full test |

The selected checker’s 95% document-cluster bootstrap interval is **77.66–80.44%**. Its paired gain over the calibrated simple question is **1.03 points**, interval **0.31–1.80**. These intervals condition on the frozen model, use 1,000 replicates, and do not account for prompt-search uncertainty. We do not have paired Bespoke predictions and make no significance claim against its rounded published score.

## Exact checker

Model: `jev-1.13.0`. One request per document/claim pair contains three Noul questions: direct support, any unsupported detail, and any contradicted detail. The exact wording is in [frozen.json](frozen.json).

```python
score = min(p_support, 1 - p_unsupported_detail, 1 - p_contradicted_detail)
supported = score > 0.30
```

The composite is a decision score, not a separately calibrated probability of correctness. Its single global threshold optimizes balanced accuracy. No source identity, gold label, or contamination identifier enters the API request.

## Results by source

| Source | Test rows | Jev | Published Bespoke |
|---|---:|---:|---:|
| CNN | 558 | 67.12% | 65.5% |
| XSum | 558 | 77.46% | 77.8% |
| MediaS | 726 | 75.65% | 76.0% |
| MeetB | 772 | 81.85% | 78.3% |
| WiCE | 358 | 85.75% | 83.0% |
| REVEAL | 1,710 | 91.41% | 88.0% |
| ClaimVerify | 1,088 | 77.27% | 75.3% |
| FactCheck | 1,566 | 78.18% | 77.7% |
| ExpertQA | 3,702 | 59.36% | 59.2% |
| LFQA | 1,911 | 88.95% | 86.7% |
| RAGTruth | 16,371 | 86.94% | 84.0% |

These are per-source balanced accuracies. Sources receive equal weight despite different sizes. Pooled balanced accuracy is 83.20%, pooled accuracy 82.83%, and pooled ROC AUC 0.9043; none replaces the headline benchmark metric.

## Controls against overfitting

- Pinned dataset revision `981dfd0bd8e58e7238a9ab92b2e6ea44bce918e4`; model version and preprocessing fixed.
- Used 2,299 development tuning examples and 2,221 development selection examples. Their normalized document groups are disjoint.
- Compared seven initial questions, three revised questions, fourteen score definitions, and a fixed threshold grid. Only tuning errors informed the one prompt revision.
- Locked three candidates before selection inference. Chose the minimum-score rule and refitted only its global threshold on the combined observed development rows.
- Froze the final configuration at `2026-09-28T09:32:41.253676+00:00` before the recorded test download. The final full test uses that exact configuration and code.
- After the user requested interim inspection, tested eight regularized logistic combinations using cached development predictions and grouped cross-validation. The chosen calibrator scored 78.41% on reused development selection versus 78.62% for the original. It was rejected and never evaluated on test examples.
- Source-specific calibration reached 80.53% on reused development selection but was excluded from the primary checker: it needs benchmark-source identity and does not establish a general checker or a test result.
- No individual test texts or errors were displayed for prompt design. Final test results were not used to change any question, threshold or combination.

## Coverage, cost and latency

All **29,320/29,320** rows have successful predictions. There were **0 recorded retries** and **8 shortened documents**. The predeclared rule keeps the first 60,000 and last 20,000 characters when a document exceeds 80,000 characters, with an omission marker.

Full-test usage: **35,304,971 input tokens**, estimated **$1.48**. All logged development and test work totals **$1.97**. Pilot responses were reused and are counted only once. Output tokens are free at the verified pricing; interrupted in-flight requests may add a small unrecorded charge.

HTTP latency: median **0.322s**, p95 **1.029s**. This is for all three questions in one request, excluding deliberate rate-limit waiting. The request rate was capped at 18/sec.

The official split contains **7 exact document/claim pairs also present in the observed development sample**. Excluding them gives **79.0844%** on 29,313 rows, a change of **−0.0014 percentage points**. The primary official benchmark score retains all rows.

The independent audit passed **37/37 checks**, independently reproducing the scores with scikit-learn and verifying complete prediction coverage, frozen configuration and code hashes, request inputs, and freeze timing before test access. These checks do not establish whether the provider's model training included benchmark material.

## Artifacts

- [Full numeric metrics](test_metrics.json)
- [Frozen questions, threshold and code hashes](frozen.json)
- [Independent audit](final_audit.json)
- [All 39-model leaderboard audit](../references/leaderboard_audit.md)
- [Development-only calibration results](iteration_cached.json)
- [Run and local-checker instructions](../README.md)

Sources: [official leaderboard](https://llm-aggrefact.github.io/), [zero-shot caveats](https://llm-aggrefact.github.io/blog), [Jev API](https://docs.typesafe.ai/api), [Jev models/pricing](https://docs.typesafe.ai/models).

The user accepted this verified result as the final stopping point after reviewing the 79.09% full-test score. No further tuning or API runs are planned.
