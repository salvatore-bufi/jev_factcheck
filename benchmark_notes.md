# LLM-AggreFact protocol notes

Research date: 2026-09-28. This research used public README/model-card text and the original paper; it did not access the Hugging Face dataset viewer or benchmark data files. Published aggregate results are not observations of this Jev experiment.

## Verified benchmark contract

- Dataset: `lytang/LLM-AggreFact`; official split names are `dev` (30,420 examples) and `test` (29,320 examples). There is no training split.
- Fields: `dataset`, `doc`, `claim`, `label`, `contamination_identifier`.
- Label `1` means the claim is supported by the supplied document; `0` means otherwise. The prediction task is document-grounded support, not unrestricted real-world truth.
- The current version has 11 constituent sources: AggreFact-CNN, AggreFact-XSum, TofuEval-MediaSum, TofuEval-MeetingBank, WiCE, REVEAL, ClaimVerify, FactCheck-GPT, ExpertQA, LFQA, and RAGTruth. The old paper evaluates 10 sources; RAGTruth was added later. One README field description still says 10, but its current summary and source list say 11.
- The README explicitly permits evaluation and excludes pretraining/fine-tuning use. Prompt and decision-threshold development here should remain evaluation work.

Source: [dataset README and embedded metadata](https://huggingface.co/datasets/lytang/LLM-AggreFact/blob/main/README.md).

## Scoring and interpretation

For each value of `dataset`, compute balanced accuracy:

`BAcc = 0.5 * (TP / (TP + FN) + TN / (TN + FP))`.

Then take the unweighted arithmetic mean of the 11 source-level scores. Multiplying by 100 gives the benchmark percentage scale. Do not substitute pooled accuracy or pooled balanced accuracy as the headline result: source sizes differ. The official model-card evaluation code directly implements this per-source-then-mean aggregation. Secondary results can include the 11 individual scores, supported/unsupported recalls, pooled accuracy, Brier score if meaningful probabilities exist, failure rate, latency, and cost. Source: [official MiniCheck model-card evaluation code](https://huggingface.co/lytang/MiniCheck-Flan-T5-Large#test-on-our-llm-aggrefact-benchmark).

The original paper uses `score > threshold` for support and a default threshold of 0.5. Its main results use no dataset-specific threshold tuning; Appendix B.1 also reports thresholds selected only on each source's validation data. Existing original validation/test splits were retained where available; other sources were split 50/50 while keeping responses to a query together. Source: [MiniCheck paper, sections 3.3 and 4.1; Appendix B.1/Table 7](https://aclanthology.org/2024.emnlp-main.499.pdf).

The official leaderboard is a zero-shot comparison; its authors explicitly encourage exploring per-source thresholds in practical applications. A prompt/parameter/threshold-tuned Jev result should therefore be labelled **dev-tuned**, and not described as directly equivalent to a zero-shot leaderboard entry. Source: [official benchmark blog, Caveats](https://llm-aggrefact.github.io/blog).

## Proposed Jev experiment (recommendation, not an official protocol)

1. Pin the dataset revision and load only the explicit `data/dev-*` file(s). Do not call an unrestricted dataset loader that might download every split. Record the dev file hash and sampled row identifiers. Keep the test split unopened throughout this tuning run.
2. Split dev into tuning and confirmation portions with a fixed seed. Keep identical normalized documents in one portion; also audit repeated claims/document-claim pairs so repeated evidence does not inflate the independence of confirmation results. Approximately preserve each source and both labels when possible.
3. Screen a small, predefined set of question templates and Jev parameters on a fixed sample of the tuning portion. Balance the sample across sources and labels to align it with macro BAcc, and report the actual sample counts. Expand the best candidates to a larger tuning sample only if the early signal warrants the cost.
4. Ask whether the entire claim is supported by the supplied document. Treat missing evidence and contradiction as unsupported. Do not permit external retrieval or general prior knowledge to override the supplied document. Do not send gold labels, source dataset names, or contamination identifiers to Jev.
5. Include a simple direct support-question baseline. If Jev supplies a numeric support score, compare its fixed 0.5 rule with a single threshold chosen using tuning data only. Consider per-source thresholds only as an explicitly separate, more specialized secondary result.
6. Select the prompt and service parameters using only the tuning portion. Freeze the exact question, preprocessing, missing-output policy, score conversion, threshold, and API version. Evaluate the frozen choice once on the internal dev confirmation portion; do not use confirmation errors to start another selection cycle while still calling it a holdout.
7. Save every candidate's configuration, selection sample, raw responses, scores, failures, and costs to make selection reproducible. Report paired uncertainty for differences; bootstrap by document/group rather than treating all claims as independent. Small samples should be described as pilot estimates.
8. A future one-shot official-test evaluation can use the frozen configuration without human inspection of examples or labels. It should be a separate explicit stage; no official-test access is needed to satisfy the current dev-only tuning stage.

An internal dev confirmation score remains a development estimate. A result on a subset of dev is not a full official benchmark score.
