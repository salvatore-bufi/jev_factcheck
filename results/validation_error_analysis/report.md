# Jev validation error analysis

The recurring difficulties are **preserving qualifiers and relationships, distinguishing missing evidence from reasonable inference, and combining evidence consistently**. Some benchmark disagreements also warrant label or source-context review. The strongest mechanical finding is that the unsupported-detail question alone vetoes **133 of 200 false negatives**, but it also prevents many false positives.

## Scope and method

This analysis reads only the **2,221 cached examples in the internal selection sample of the official development split**, their cached predictions, and the frozen configuration. No test records, predictions or metrics were opened for this analysis, and no Jev requests were made. This is an already-used development selection sample, not a fresh validation set.

At the unchanged frozen rule, `min(support, 1 - unsupported_detail, 1 - contradiction) > 0.30`, there are **439 disagreements with the gold labels**: **239 false positives** (unsupported claims accepted) and **200 false negatives** (supported claims rejected). Source-macro balanced accuracy on this sample is **78.62%**; ordinary pooled accuracy is **80.23%**.

I selected **88 errors deterministically: four per source and error direction**, covering all 11 sources and 82 distinct documents. I inspected the full reference for 87 cases; for the 68,459-character art-history reference in case 24, I inspected its abstract and relevant style-related passages and left the assessment unresolved. This is qualitative manual clustering, with one dominant category per case. Counts below describe this deliberately balanced review sample; they are **not estimates of each category's frequency among all 439 errors**. Some categories overlap conceptually. There was no independent second annotator.

Jev returned scores, not explanations. The explanations below are hypotheses grounded in the supplied documents, claims and score components—not access to the model's internal reasoning. Original labels, prompts, thresholds and benchmark code remain unchanged.

## Clusters

| Main pattern | Reviewed cases | FP | FN |
|---|---:|---:|---:|
| Scope, qualifiers and event status | 23 | 10 | 13 |
| Paraphrase and combining evidence | 19 | 12 | 7 |
| Unstated details or incomplete evidence | 17 | 2 | 15 |
| Entities, attribution and unresolved references | 15 | 14 | 1 |
| Numbers, units and precision | 8 | 4 | 4 |
| Strong apparent label/evidence conflicts | 5 | 1 | 4 |
| Broad interpretive claim, unresolved | 1 | 1 | 0 |

Separately from these topic clusters, the manual assessment marks **24 likely checker errors, 17 evidence gaps, 14 candidates for label review, and 33 ambiguous or unresolved cases**. These provisional judgments are not corrected ground truth and do not establish a dataset-wide label-error rate.

**1. Scope, qualifiers and event status.** A small change can alter the truth conditions while leaving most words intact. In `dev:537`, the reference says England won its first Six Nations title **since 2012**; the claim says it won **for the first time**. The checker accepts it. In `dev:4590`, a budget report appears only as the next agenda item, but the checker accepts that the document discusses it. In `dev:7038`, an argument for equal pay for female athletes becomes higher pay for athletes generally. These suggest sensitivity to topical agreement without consistently preserving quantifiers, population, and event status. Conversely, gold-positive cases sometimes contain similar expansions: a planned writing aid becomes an accomplished aid in `dev:2247`, and a proposed fight becomes an event that happened in `dev:4741`.

**2. Paraphrase and combining evidence.** Some rejections require a reasonable inference rather than finding an identical sentence. In `dev:13140`, dirty filters causing extra black smoke and soot supports the claim that clean filters can reduce emissions, yet the combined score is 0.19. In `dev:3840`, Raintree, two downtown projects and Long Beach occur in different turns of a noisy transcript; the direct support score is 0.90, but the no-unsupported-detail score is 0.28, causing rejection. In `dev:4813`, the publication date, quoted announcement and reference to next June must be combined. This cluster also includes accepted paraphrases with negative labels that are difficult to explain from the document alone—for example, `dev:6115`, whose source describes a third-period hockey goal. Those cases should be reviewed, not automatically treated as demonstrated model failures.

**3. Unstated details or incomplete evidence.** Many gold-positive examples include one detail absent from the actual supplied source. In `dev:1127`, the claim says the driver died after hitting a tree; the reference describes the car leaving the road but mentions neither a death nor a tree. In `dev:2269`, the source describes Bill Sand's age and volunteer devices, but the claim adds that he is a retiree. In `dev:8385`, the reference discusses collecting and reducing datasets but not public data sources. A document-only checker has a defensible reason to reject these. Possible explanations include omitted article context, excerpt selection, or a label standard that allows more background knowledge. The records alone do not establish which explanation applies.

**4. Entities, attribution and unresolved references.** Matching the event or quotation does not establish who it belongs to. In `dev:477`, the reference says Mr Tucker, while the claim supplies **Alex Tucker**; support/no-unsupported-detail/no-contradiction scores are **0.93 / 0.42 / 0.83**, and the claim passes. In `dev:532`, the reference identifies Harris but never establishes the claim's first name **Neil**. In `dev:3790`, fuel prices rising after a Tesoro strike becomes Tesoro itself raising prices. Other cases contain unresolved expressions such as “this modality,” “this method,” or “Fox”; omitted question context can make their gold labels impossible to adjudicate confidently. The likely weakness is checking content similarity without fully binding names, agents, speakers and antecedents.

**5. Numbers, units and precision.** Both excessive strictness and excessive tolerance occur. In `dev:5413`, 148,939,063.133 km² is reasonably rounded to 148,940,000, but the checker rejects the claim. In `dev:1237`, it rejects combining “over budget and overdue” with the later approximate $400 million per-helicopter figure. In the opposite direction, `dev:19824` converts the structured closing time `4:0` to **4 pm**, although the hours use 24-hour-style notation. In `dev:6271`, it accepts a temperature interval that collapses different regions and unit-specific ranges. Exact string comparison and broad numerical similarity are both inadequate; unit, precision and the entity being measured matter.

**6. Strong apparent label/evidence conflicts.** These deserve separate adjudication before changing the checker. In `dev:8447`, the gold-supported claim gives a submarine commissioning date of **December 22, 1914**, while the document explicitly says **August 26, 1914**. In `dev:14582`, a gold-supported generalization says artificial sweeteners are not metabolized, but the document explicitly gives aspartame as a counterexample. In `dev:26171`, the gold-supported claim assigns QC a responsibility that the source assigns to QA. The remaining two cases are `dev:122` (age/caps assigned to the wrong footballer) and `dev:296` (a negative label despite all three statements being directly supported). These are candidate annotation or provenance problems, not relabeled examples. A further nine cases carry weaker label-review flags in other clusters. No scores were recomputed after changing labels.

## What the complete validation cache establishes

The unsupported-detail question participates in **194/200 false-negative decisions** and is the **only blocking component in 133/200 (66.5%)**. Thus, the minimum aggregation makes this question particularly consequential. This establishes the decision mechanism; it does not prove which words triggered that question.

Removing just that check while keeping the other questions and threshold fixed would recover **133 supported claims**, but also incorrectly accept **285 additional unsupported claims**. Source-macro balanced accuracy would fall from **78.62% to 71.39%**. The diagnostic supports improving detail checking, rather than simply deleting it.

Errors are not confined to scores near the threshold: **89/239 false positives score at least 0.70**, and **71/200 false negatives score at most 0.10**. These are score ranges, not calibrated probabilities of correctness. A small threshold adjustment cannot address both extremes.

Only **4/2,221 documents** were shortened by our predeclared character limit, and only **1/439 errors** involved such a document. Input truncation is therefore not the explanation for most observed errors. This does not measure the effects of lengthy, noisy or incomplete source documents.

| Source | Validation rows | False positives | False negatives |
|---|---:|---:|---:|
| CNN | 128 | 9 | 6 |
| XSum | 220 | 13 | 45 |
| ClaimVerify | 201 | 36 | 6 |
| ExpertQA | 220 | 37 | 51 |
| FactCheck-GPT | 213 | 5 | 33 |
| LFQA | 220 | 18 | 10 |
| RAGTruth | 220 | 16 | 13 |
| REVEAL | 220 | 13 | 7 |
| MediaS | 212 | 59 | 4 |
| MeetB | 220 | 23 | 8 |
| WiCE | 147 | 10 | 17 |

These are sample counts, with unequal source/label availability. ExpertQA and XSum account for 96 of the 200 false negatives; MediaS accounts for 59 of the 239 false positives. This describes where failures occur without proposing source-specific rules.

## A separate numerical boundary issue

The frozen implementation computes complements using binary floating point:

```python
1 - 0.70                 # 0.30000000000000004
(1 - 0.70) > 0.30        # True
```

As a result, **15 validation claims at mathematically exactly 0.30 are accepted**. Ten are gold unsupported and five are gold supported. An exact-decimal diagnostic would reject all 15, giving **78.81%** source-macro balanced accuracy on this development sample. This is a numerical-correctness diagnostic, not a newly selected threshold or a replacement official result. The frozen checker remains unchanged; the earlier audit established reproduction of its actual floating-point behavior, not exact-decimal boundary semantics. Any correction should be versioned and validated separately.

## Implications for further development

The useful targets are explicit checks for **entity/attribute binding, qualifiers and event status**, together with a consistent definition of acceptable **paraphrase, rounding and direct inference**. Evidence spans or an identified offending claim fragment would make decisions easier to diagnose, but their benefit has not been measured here. A small independently adjudicated validation set should distinguish model failures from missing context and questionable labels before further prompt work.

This review now makes these 88 examples diagnostic development material. They should not serve as an independent success criterion for future changes. The current task does not tune prompts, select new parameters, or evaluate a new checker.

## Reproducibility and evidence

- [All 88 annotated cases](reviewed_cases.md), including claim, component scores, interpretation and links to the supplied validation reference.
- [Machine-readable counts and provenance](summary.json).
- [Complete 439-error index](error_manifest.jsonl).
- [Manual cluster annotations](review_annotations.jsonl).
- [Reproduction script](../../analyze_validation_errors.py).

Run `python3 analyze_validation_errors.py` from the repository. Its inline checks verify development IDs, complete cache coverage, input and question-pack hashes, frozen model/questions, unchanged frozen code, and independent reconstruction of the three-question scores. The script uses only explicit development paths and the saved configuration, performs no network calls, and does not import the inference client.
