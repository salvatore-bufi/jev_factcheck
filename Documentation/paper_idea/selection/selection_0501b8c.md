# Model selection from Decision Index 0.4.1 (snapshot 0501b8c)

Source: `multimodalart/decision-index-leaderboard` at commit `0501b8cae77a94b3c56dc925609d88c6210835f7` (CC BY 4.0). sha256: `leaderboard.parquet` a3b1801d3ac0…, `benchmarks.parquet` 85fc39e97a5e….

Eligible pool: 115 entrants (params ≤ 9.7B, open weights on Hugging Face, public coverage ≥ 0.99, not a training-free technique, 10 declared exclusions). Strong = top quarter of its backbone group by rank(Full) + rank(Public) (at least 2, ties kept). Grounding % = mean percentile rank in the pool on RAGTruth, HoVer, ANLI, ContractNLI, NLI4CT. Full-score differences below 0.9 share a rank on the board.

| Tier | Model | Params (B) | Base (board) | Kind | Full (rank) | Public (rank) | In-group position | Strong | Grounding % | Rule |
|---|---|---:|---|---|---|---|---|---|---:|---|
| A:G4 | KnowLine 4B Gen5.5 | 4.66 | Qwen/Qwen3.5-4B | full fine-tune | 55.99 (64) | 66.76 (18) | 1 of 18 | yes | 93 | S4 top quarter of the group by rank(Full) + rank(Public) |
| A:G4 | StartLux-Decision v1.1 4B | 4.66 | Qwen/Qwen3.5-4B | full fine-tune | 54.80 (70) | 67.81 (14) | 1 of 18 | yes | 96 | S4 top quarter of the group by rank(Full) + rank(Public) |
| A:G4 | Perplexity Decider Flash v1.2 (4B) | 4.54 | Qwen/Qwen3.5-4B | full fine-tune | 54.29 (72) | 60.06 (50) | 4 of 18 | yes | 88 | S4 top quarter of the group by rank(Full) + rank(Public) |
| A:G4 | Jeffan-4B-2010 | 4.21 | Qwen/Qwen3.5-4B | full fine-tune | 52.21 (80) | 64.85 (31) | 3 of 18 | yes | 95 | S4 top quarter of the group by rank(Full) + rank(Public) |
| A:G4 | OneJev1.6-4B | 4.54 | Qwen/Qwen3.5-4B | full fine-tune | 48.88 (87) | 61.95 (41) | 5 of 18 | yes | 90 | S4 top quarter of the group by rank(Full) + rank(Public) |
| A:G4 | CoCo-Decision-4B 2.4 | 4.21 | Qwen/Qwen3.5-4B | LoRA + head | 45.52 (101) | 55.19 (88) | 10 of 18 |  | 85 | S3 training-kind coverage (LoRA + head): best rank-sum |
| A:G4 | ezjev 4B s2 | 4.66 | Qwen/Qwen3.5-4B | LoRA | 43.90 (110) | 50.82 (105) | 11 of 18 |  | 72 | S3 training-kind coverage (LoRA): best rank-sum |
| A:G9 | baselabs elephant-1 (9B) | 9.41 | Qwen/Qwen3.5-9B | full fine-tune | 60.26 (41) | 72.24 (4) | 1 of 12 | yes | 100 | S4 top quarter of the group by rank(Full) + rank(Public) |
| A:G9 | StartLux-Decision v1.1 9B | 9.65 | Qwen/Qwen3.5-9B | full fine-tune | 58.07 (56) | 69.98 (9) | 2 of 12 | yes | 99 | S4 top quarter of the group by rank(Full) + rank(Public) |
| A:G9 | vLLM-SR d3-flash (9B) | 8.39 | Qwen/Qwen3.5-9B | full fine-tune | 54.27 (73) | 57.79 (72) | 3 of 12 | yes | 79 | S4 top quarter of the group by rank(Full) + rank(Public) |
| A:G9 | OneJev1.6-9B | 9.41 | Qwen/Qwen3.5-9B | full fine-tune | 53.05 (77) | 65.30 (27) | 3 of 12 | yes | 94 | S4 top quarter of the group by rank(Full) + rank(Public) |
| A:G9 | Bespoke Nimble 9B v3 | 9.65 | Qwen/Qwen3.5-9B | LoRA | 51.06 (81) | 57.19 (79) | 5 of 12 |  | 79 | S3 training-kind coverage (LoRA): best rank-sum |
| A:G9 | Sieve 9B v2 | 9.65 | Qwen/Qwen3.5-9B | LoRA + head | 47.00 (93) | 53.35 (96) | 7 of 12 |  | 73 | S3 training-kind coverage (LoRA + head): best rank-sum |
| A:G9B | PurpleMIST-Flash-1.0 | 7.94 | Qwen/Qwen3.5-9B-Base | LoRA + head | 47.54 (90) | 46.74 (123) | 2 of 4 | yes | 64 | S4 top quarter of the group by rank(Full) + rank(Public) |
| A:G9B | Cerno 9B v1.1 | 8.95 | Qwen/Qwen3.5-9B-Base | full fine-tune | 47.34 (91) | 58.59 (60) | 1 of 4 | yes | 89 | S4 top quarter of the group by rank(Full) + rank(Public) |
| A:G9B | JPT-9B Qwen3.5-9B LoRA | 9.65 | Qwen/Qwen3.5-9B-Base | LoRA | 44.08 (109) | 46.96 (121) | 3 of 4 |  | 73 | S3 training-kind coverage (LoRA): best rank-sum |
| A:other | Nace.AI Drex v1.5 | 8.95 | XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B | full fine-tune | 45.78 (98) | 58.08 (67) | 1 of 1 | yes | 89 | S5 published LLM-AggreFact entry to reproduce |
| A:other | Cloudflare clef-flash | 9.65 | Qwen/Qwen3.5-9B | full fine-tune | 44.43 (104) | 56.15 (85) | 7 of 12 |  | 60 | S5 published LLM-AggreFact entry to reproduce |
| B:JPT | JPT-4B Qwen3.5-4B LoRA | 4.66 | Qwen/Qwen3.5-4B-Base | LoRA | 39.56 (127) | 42.82 (129) | 3 of 10 | yes | 65 | S3b size ladder of a selected recipe |
| B:JPT | JPT-0.8B Qwen3.5-0.8B LoRA | 0.87 | Qwen/Qwen3.5-0.8B-Base | LoRA | 15.46 (200) | 19.37 (202) | 3 of 5 |  | 31 | S3b size ladder of a selected recipe |
| B:OneJev 1.6 | OneJev1.6-0.8B | 0.85 | Qwen/Qwen3.5-0.8B | full fine-tune | 29.58 (159) | 48.76 (113) | 1 of 11 | yes | 63 | S3b size ladder of a selected recipe |
| B:StartLux v1.1 | StartLux-Decision v1.1 2B | 2.27 | Qwen/Qwen3.5-2B | full fine-tune | 40.11 (124) | 59.23 (56) | 1 of 7 | yes | 80 | S3b size ladder of a selected recipe |
| B:StartLux v1.1 | StartLux-Decision v1.1 0.8B | 0.87 | Qwen/Qwen3.5-0.8B | full fine-tune | 28.88 (162) | 53.79 (92) | 1 of 11 | yes | 70 | S3b size ladder of a selected recipe |
| B:d3 | vLLM-SR d3-mini (4B) | 4.54 | Qwen/Qwen3.5-4B | full fine-tune | 49.97 (84) | 54.90 (90) | 7 of 18 |  | 73 | S3b size ladder of a selected recipe |
| B:d3 | vLLM-SR d3-nano (2B) | 2.21 | Qwen/Qwen3.5-2B | full fine-tune | 34.30 (143) | 42.22 (132) | 4 of 7 |  | 46 | S3b size ladder of a selected recipe |
| B:d3 | vLLM-SR d3-lite (0.8B) | 0.85 | Qwen/Qwen3.5-0.8B | full fine-tune | 26.16 (166) | 36.93 (157) | 4 of 11 |  | 39 | S3b size ladder of a selected recipe |
| B:d3 | vLLM-SR d3-edge (0.6B) | 0.59 | Qwen/Qwen3.5-0.8B | full fine-tune | 17.69 (190) | 28.82 (180) | 8 of 11 |  | 32 | S3b size ladder of a selected recipe |
| B:non-Qwen singleton | Standard One 3B SH (v4) | 3.85 | StandardThinking/StandardOne-3B | LoRA + head | 39.79 (126) | 58.08 (68) | 1 of 1 | yes | 75 | S3e non-Qwen singleton, Full >= 30 |
| B:non-Qwen singleton | Lev1 3.7B | 3.70 | IFM/K2-Horizon-3.7B | LoRA | 35.41 (137) | 42.68 (131) | 1 of 1 | yes | 59 | S3e non-Qwen singleton, Full >= 30 |
| B:non-Qwen singleton | Aurelis-2B | 2.52 | openbmb/MiniCPM5-2B | full fine-tune | 32.55 (148) | 51.29 (103) | 1 of 1 | yes | 62 | S3e non-Qwen singleton, Full >= 30 |
| C | Laya | 0.42 | nan | full fine-tune | 9.21 (217) | 6.19 (223) | 5 of 6 |  | 19 | control: the open TDM most used in prior evaluations (Arize, Red Hat, PremiseCheck): encoder floor |
| C | CLM-v0.1-8B Qwen3-8B heads | 8.19 | Qwen/Qwen3-8B-Base | full fine-tune | 6.49 (222) | 7.41 (221) | 1 of 1 | yes | 14 | control: contrastive readout (cosine between state and option embeddings): readout control |
| N:Gemma-4-E4B | Winnow-E4B | 8.00 | google/gemma-4-E4B | LoRA | 32.72 (147) | 40.86 (141) | 1 of 2 | yes | 51 | S3e non-Qwen family with >= 2 entrants: all members |
| N:Gemma-4-E4B | Gevva E4B | 8.00 | google/gemma-4-E4B-it | full fine-tune | 29.21 (161) | 32.44 (168) | 2 of 2 | yes | 26 | S3e non-Qwen family with >= 2 entrants: all members |
| N:LFM2.5-2.6B | LiquidAI d1-3B | 3.12 | LiquidAI/LFM2.5-VL-3B | full fine-tune | 31.12 (154) | 48.99 (112) | 1 of 2 | yes | 68 | S3e non-Qwen family with >= 2 entrants: all members |
| N:LFM2.5-2.6B | LFM2.5-2.6B-RLCD | 2.70 | LiquidAI/LFM2.5-2.6B-Base | full fine-tune | 8.18 (219) | 6.78 (222) | 2 of 2 | yes | 14 | S3e non-Qwen family with >= 2 entrants: all members |
| N:Ministral-3-8B | Standard One 8B SH (v4) | 8.92 | StandardThinking/StandardOne-8B | LoRA + head | 42.93 (113) | 57.40 (77) | 1 of 2 | yes | 80 | S3e non-Qwen family with >= 2 entrants: all members |
| N:Ministral-3-8B | Jevstral 8B | 8.00 | mistralai/Ministral-3-8B-Base-2512 | LoRA | 32.23 (149) | 34.01 (162) | 2 of 2 | yes | 31 | S3e non-Qwen family with >= 2 entrants: all members |
| N:encoders | GLiClass v4 edge (0.4B) | 0.44 | microsoft/deberta-v3-large | full fine-tune | 31.21 (153) | 47.49 (120) | 1 of 1 | yes | 70 | S3d encoder family DeBERTa-v3: best rank-sum |
| N:encoders | Crivo-1B | 1.03 | cross-encoder/ettin-reranker-1b-v1 | full fine-tune | 29.25 (160) | 47.77 (117) | 1 of 15 | yes | 69 | S3d encoder family Ettin: best rank-sum |
| N:encoders | Amber 0.4B | 0.40 | answerdotai/ModernBERT-large | full fine-tune | 26.00 (167) | 49.74 (110) | 1 of 6 | yes | 60 | S3d encoder family ModernBERT: best rank-sum |

## Motivation rows (training-free readouts and references)

| Model | Params (B) | Kind | Base | Full (rank) | Public (rank) |
|---|---:|---|---|---|---|
| Ryotide Qwen v0.3.1 | 4.00 | inference technique | Qwen/Qwen3.5-4B | 31.99 (151) | 37.28 (155) |
| openvons Qwen3-4B | 4.02 | inference technique | Qwen/Qwen3-4B-Instruct-2507 | 28.78 (163) | 28.66 (181) |
| KnowLine 4B Gen5.5 | 4.66 | full fine-tune | Qwen/Qwen3.5-4B | 55.99 (64) | 66.76 (18) |
| Jobe Qwen3.5-4B | 4.66 | inference technique | Qwen/Qwen3.5-4B-Base | 30.35 (158) | 32.93 (164) |
| SemIf Qwen3.5-4B Base | 4.66 | inference technique | Qwen/Qwen3.5-4B-Base | 26.71 (164) | 26.50 (188) |
| baselabs elephant-1 (9B) | 9.41 | full fine-tune | Qwen/Qwen3.5-9B | 60.26 (41) | 72.24 (4) |
| Winnow-12B | 11.96 | LoRA | google/gemma-4-12B | 46.57 (96) | 50.66 (106) |
| decisio 0.4.0 (Gemma 4 12B) | 11.96 | inference technique | google/gemma-4-12B-it | 48.75 (88) | 49.72 (111) |
| Fastino GLiDE 1.2 | 27.80 | full fine-tune | Qwen/Qwen3.8-27B | 71.52 (1) | 71.99 (5) |
| FlexDM · Gemma-4-31B | 31.27 | inference technique | google/gemma-4-31B-it | 67.56 (9) | 63.63 (34) |
| AnyJev L0 · Gemma-4-31B | 31.27 | inference technique | google/gemma-4-31B-it | 63.65 (30) | 59.79 (51) |
| Jeffan-31b-1710 | 31.27 | full fine-tune | google/gemma-4-31B-it | 69.75 (3) | 73.13 (2) |
| Jev 1.14 |  | hosted API |  | 60.24 (42) | 60.70 (48) |
