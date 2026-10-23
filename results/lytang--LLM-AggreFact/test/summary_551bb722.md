# lytang--LLM-AggreFact / test — question set `551bb722` (claim_support)

Default decision rule of the question set.

## Balanced accuracy (%) per source; Average = unweighted mean over sources

| Model | Size | Threshold | Average | Macro AUC | CNN | XSum | MediaS | MeetB | WiCE | REVEAL | ClaimVerify | FactCheck | ExpertQA | LFQA | RAGTruth |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Winnow-12B-BF16 | 12B | 0.30 | **76.95** | <u>84.99</u> | **72.22** | 72.75 | **73.39** | **80.29** | 77.42 | **90.07** | <u>75.67</u> | 73.90 | 59.02 | <u>84.69</u> | **87.07** |
| Rune-26B-A4B-v3 | 26B-A4B | 0.30 | <u>76.24</u> | 84.64 | <u>66.03</u> | <u>74.22</u> | <u>70.81</u> | 76.50 | 83.70 | 87.12 | 73.10 | <u>78.37</u> | 59.66 | 83.09 | <u>86.04</u> |
| JPT-9B | 9B | 0.30 | 76.04 | 84.48 | 62.44 | **75.84** | 67.61 | <u>76.89</u> | <u>83.77</u> | <u>87.90</u> | 72.78 | **81.08** | 58.76 | **84.85** | 84.58 |
| d1-3B | 3B | 0.30 | 71.84 | 83.89 | 54.86 | 68.79 | 67.97 | 70.87 | 81.64 | 84.34 | 67.21 | 76.36 | <u>59.94</u> | 79.66 | 78.61 |
| CLM-v0.1-8B | 8B | 0.30 | 49.98 | 46.79 | 50.00 | 50.00 | 50.00 | 50.00 | 49.80 | 50.00 | 50.00 | 50.00 | 50.03 | 50.00 | 50.00 |
| d3-flash | 8.4B | 0.30 |  | **95.24** |  |  |  |  | **100.00** |  | **100.00** |  | **85.71** |  |  |

## Each question alone: ROC AUC of P(yes) against the label (inverted questions show values below 0.5)

| Model | support_simple | unsupported_detail | contradicted_detail |
|---|---:|---:|---:|
| CLM-v0.1-8B | 0.759 | 0.488 | 0.444 |
| d1-3B | 0.884 | 0.118 | 0.236 |
| d3-flash | 0.910 | 0.122 | 0.288 |
| JPT-9B | 0.887 | 0.116 | 0.226 |
| Rune-26B-A4B-v3 | 0.904 | 0.110 | 0.283 |
| Winnow-12B-BF16 | 0.893 | 0.099 | 0.377 |

## Speed and resources

Latency: one row (all questions) at a time, cold cache, on a fixed stratified sample. Rows/s: main pass, batched.

| Model | Latency p50 (s) | p95 (s) | p99 (s) | Rows/s | Questions/s | Prompt tokens/s | Load (s) | First call (s) | GPU mem peak (MB) | RAM peak (MB) | Energy / 1k rows (J) | Mean power (W) | Failures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CLM-v0.1-8B | 0.101 | 0.269 | 0.598 | 30.14 | 90.43 | 65478 |  |  | 41564 | 11415 | 9865 | 298 | 0 |
| d1-3B | 0.038 | 0.103 | 0.229 | 24.36 | 73.08 | 57336 | 5.7 | 0.45 | 11616 | 6805 | 12821 | 245 | 0 |
| d3-flash |  |  |  | 1.25 | 3.75 | 3599 | 8.1 | 1.23 | 21540 | 16796 | 130625 | 164 | 0 |
| JPT-9B | 0.172 | 0.336 | 0.630 | 9.41 | 28.22 | 23405 | 148.4 | 0.71 | 43212 | 8235 | 35732 | 336 | 0 |
| Rune-26B-A4B-v3 | 1.450 | 4.620 | 9.929 | 1.67 | 5.00 | 4228 | 257.1 | 2.97 | 43466 | 31403 | 70468 | 117 | 0 |
| Winnow-12B-BF16 | 0.169 | 0.744 | 1.434 | 9.76 | 29.28 | 24637 | 132.8 | 2.05 | 43516 | 7397 | 34916 | 341 | 0 |

## Throughput (rows/s) against batch size

| Model | bs=1 | bs=8 | bs=32 |
|---|---:|---:|---:|
| CLM-v0.1-8B | 8.71 | 12.23 | 13.05 |
| d1-3B | 24.06 | 24.11 | 24.08 |
| d3-flash |  |  |  |
| JPT-9B | 5.49 | 7.24 | 7.41 |
| Rune-26B-A4B-v3 | 0.61 | 0.81 | 0.82 |
| Winnow-12B-BF16 | 4.00 | 4.49 | 4.55 |

## Balanced accuracy (%) by document length (characters)

| Model | doc 0-1000 | doc 1000-2000 | doc 2000-4000 | doc 4000-8000 | doc 8000-16000 | doc >=16000 |
|---|---:|---:|---:|---:|---:|---:|
| CLM-v0.1-8B | 50.06 | 50.0 | 50.0 | 49.95 | 50.0 | 49.93 |
| d1-3B | 72.68 | 76.48 | 78.84 | 71.56 | 70.63 | 66.24 |
| d3-flash | 87.5 |  |  | 83.33 |  |  |
| JPT-9B | 74.07 | 82.33 | 83.41 | 75.46 | 75.46 | 71.13 |
| Rune-26B-A4B-v3 | 73.2 | 82.98 | 84.51 | 77.29 | 75.86 | 68.4 |
| Winnow-12B-BF16 | 73.72 | 85.49 | 86.54 | 78.46 | 78.71 | 70.89 |

## Latency p50 (s) by document length (characters)

| Model | doc 0-1000 | doc 1000-2000 | doc 2000-4000 | doc 4000-8000 | doc 8000-16000 | doc >=16000 |
|---|---:|---:|---:|---:|---:|---:|
| CLM-v0.1-8B | 0.072 | 0.079 | 0.118 | 0.138 | 0.248 | 0.541 |
| d1-3B | 0.022 | 0.030 | 0.040 | 0.055 | 0.096 | 0.206 |
| d3-flash |  |  |  |  |  |  |
| JPT-9B | 0.115 | 0.153 | 0.162 | 0.205 | 0.307 | 0.572 |
| Rune-26B-A4B-v3 | 0.977 | 1.126 | 1.433 | 2.613 | 4.286 | 9.109 |
| Winnow-12B-BF16 | 0.113 | 0.141 | 0.180 | 0.387 | 0.683 | 1.386 |

