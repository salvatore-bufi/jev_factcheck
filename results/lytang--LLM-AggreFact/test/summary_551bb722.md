# lytang--LLM-AggreFact / test — question set `551bb722` (claim_support)

Default decision rule of the question set.

## Balanced accuracy (%) per source; Average = unweighted mean over sources

| Model | Size | Threshold | Average | Macro AUC | CNN | XSum | MediaS | MeetB | WiCE | REVEAL | ClaimVerify | FactCheck | ExpertQA | LFQA | RAGTruth |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| d1-3B [e3eff318] | 3B | 0.30 | **71.83** | 83.96 | **54.86** | **68.98** | **67.68** | **70.46** | <u>82.29</u> | **84.01** | 67.50 | **76.19** | 59.78 | **79.78** | **78.62** |
| CLM-v0.1-8B | 8B | 0.30 |  | 66.67 |  |  |  |  | 50.00 |  | 50.00 |  | 50.00 |  |  |
| d1-3B [e3eff318] limit32 | 3B | 0.30 |  | <u>95.24</u> |  |  |  |  | 50.00 |  | <u>75.00</u> |  | **92.86** |  |  |
| FactCG-DeBERTa-v3-Large (baseline, ignores the questions) | 0.4B | 0.50 |  | <u>95.24</u> |  |  |  |  | **100.00** |  | **100.00** |  | **92.86** |  |  |
| JPT-9B | 9B | 0.30 |  | <u>95.24</u> |  |  |  |  | 50.00 |  | **100.00** |  | <u>85.71</u> |  |  |
| Rune-26B-A4B-v3 | 26B-A4B | 0.30 |  | **100.00** |  |  |  |  | 50.00 |  | **100.00** |  | **92.86** |  |  |
| Winnow-12B-BF16 | 12B | 0.30 |  | 66.67 |  |  |  |  | 50.00 |  | **100.00** |  | 78.57 |  |  |
| Winnow-12B-NVFP4 | 12B | 0.30 |  | 61.90 |  |  |  |  | 0.00 |  | **100.00** |  | 78.57 |  |  |

## Each question alone: ROC AUC of P(yes) against the label (inverted questions show values below 0.5)

| Model | support_simple | unsupported_detail | contradicted_detail |
|---|---:|---:|---:|
| CLM-v0.1-8B | 0.667 | 0.449 | 0.487 |
| d1-3B [e3eff318] | 0.884 | 0.117 | 0.236 |
| d1-3B [e3eff318] limit32 | 0.913 | 0.154 | 0.253 |
| FactCG-DeBERTa-v3-Large |  |  |  |
| JPT-9B | 0.904 | 0.115 | 0.256 |
| Rune-26B-A4B-v3 | 0.904 | 0.154 | 0.269 |
| Winnow-12B-BF16 | 0.923 | 0.141 | 0.295 |
| Winnow-12B-NVFP4 | 0.955 | 0.096 | 0.276 |

## Speed and resources

Latency: one row (all questions) at a time, cold cache, on a fixed stratified sample. Rows/s: main pass, batched.

| Model | Latency p50 (s) | p95 (s) | p99 (s) | Rows/s | Questions/s | Prompt tokens/s | Load (s) | First call (s) | GPU mem peak (MB) | RAM peak (MB) | Energy / 1k rows (J) | Mean power (W) | Failures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CLM-v0.1-8B | 0.184 | 1.019 | 1.299 | 5.33 | 15.97 | 13490 | 21.4 | 0.51 | 23579 | 11336 | 28712 | 156 | 0 |
| d1-3B [e3eff318] | 0.064 | 0.194 | 0.448 | 13.71 | 41.14 | 32275 | 5.7 | 0.45 | 11308 | 4256 | 12636 | 173 | 0 |
| d1-3B [e3eff318] limit32 | 0.061 | 0.139 | 0.307 | 12.34 | 37.03 | 32619 | 6.0 | 0.41 | 7699 | 6715 | 11656 | 162 | 0 |
| FactCG-DeBERTa-v3-Large | 0.043 | 0.166 | 0.940 | 8.08 | 24.23 | 0 | 4.5 | 0.34 | 15199 | 2621 | 18800 | 157 | 0 |
| JPT-9B | 0.293 | 1.013 | 1.266 | 3.03 | 9.08 | 8681 | 26.6 | 0.59 | 23753 | 7918 | 51444 | 161 | 0 |
| Rune-26B-A4B-v3 | 4.320 | 11.286 | 13.371 | 0.38 | 1.13 | 1099 | 186.3 | 4.74 | 23368 | 49124 | 285412 | 108 | 0 |
| Winnow-12B-BF16 | 13.477 | 51.582 | 65.597 | 0.08 | 0.24 | 231 | 150.0 | 13.17 | 23955 | 25085 | 1463134 | 117 | 0 |
| Winnow-12B-NVFP4 | 0.181 | 0.924 | 1.186 | 4.45 | 13.35 | 5159 | 10.0 | 0.24 | 10481 | 8243 | 34172 | 161 | 0 |

## Throughput (rows/s) against batch size

| Model | bs=1 | bs=4 | bs=8 | bs=32 |
|---|---:|---:|---:|---:|
| CLM-v0.1-8B | 4.40 | 5.22 |  |  |
| d1-3B [e3eff318] | 12.82 |  | 12.80 | 12.75 |
| d1-3B [e3eff318] limit32 | 13.14 | 12.34 |  |  |
| FactCG-DeBERTa-v3-Large | 13.31 | 10.77 |  |  |
| JPT-9B | 3.40 | 3.70 |  |  |
| Rune-26B-A4B-v3 | 0.19 | 0.26 |  |  |
| Winnow-12B-BF16 | 0.08 | 0.12 |  |  |
| Winnow-12B-NVFP4 | 4.55 | 4.59 |  |  |

## Balanced accuracy (%) by document length (characters)

| Model | doc 0-1000 | doc 1000-2000 | doc 2000-4000 | doc 4000-8000 | doc 8000-16000 | doc >=16000 |
|---|---:|---:|---:|---:|---:|---:|
| CLM-v0.1-8B | 50.0 |  |  | 50.0 |  |  |
| d1-3B [e3eff318] | 72.49 | 76.6 | 78.93 | 71.5 | 70.72 | 66.44 |
| d1-3B [e3eff318] limit32 | 93.75 |  |  | 50.0 |  |  |
| FactCG-DeBERTa-v3-Large | 93.75 |  |  | 66.67 |  |  |
| JPT-9B | 87.5 |  |  | 66.67 |  |  |
| Rune-26B-A4B-v3 | 93.75 |  |  | 66.67 |  |  |
| Winnow-12B-BF16 | 81.25 |  |  | 58.33 |  |  |
| Winnow-12B-NVFP4 | 81.25 |  |  | 58.33 |  |  |

## Latency p50 (s) by document length (characters)

| Model | doc 0-1000 | doc 1000-2000 | doc 2000-4000 | doc 4000-8000 | doc 8000-16000 | doc >=16000 |
|---|---:|---:|---:|---:|---:|---:|
| CLM-v0.1-8B | 0.102 | 0.183 | 0.177 | 0.329 |  | 1.369 |
| d1-3B [e3eff318] | 0.035 | 0.053 | 0.070 | 0.108 | 0.182 | 0.398 |
| d1-3B [e3eff318] limit32 | 0.040 | 0.058 | 0.076 | 0.107 |  | 0.380 |
| FactCG-DeBERTa-v3-Large | 0.026 | 0.041 | 0.118 | 0.108 |  | 1.281 |
| JPT-9B | 0.164 | 0.292 | 0.294 | 0.361 |  | 1.329 |
| Rune-26B-A4B-v3 | 2.791 | 4.320 | 4.228 | 6.115 |  | 13.892 |
| Winnow-12B-BF16 | 7.521 | 13.303 | 12.679 | 16.992 |  | 69.100 |
| Winnow-12B-NVFP4 | 0.101 | 0.178 | 0.182 | 0.285 |  | 1.252 |

