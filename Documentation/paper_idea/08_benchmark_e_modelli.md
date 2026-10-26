# Benchmark e modelli: cosa usano i lavori del settore e cosa usare noi

Integra [03_parte_A_valutazione.md](03_parte_A_valutazione.md). Rivisto il 26 ottobre 2026 con i rilasci di ottobre. Valutare solo su LLM-AggreFact resta un punto debole: i paper di riferimento usano più benchmark.

## 1. Su cosa si valutano i lavori di fact checking su documento

| Paper / modello | Benchmark usati | Metrica | Fonte |
|---|---|---|---|
| **TRUE** (NAACL 2022) | 11 dataset (FRANK, SummEval, MNBM, QAGS-CNNDM, QAGS-XSum, BEGIN, Q², DialFact, PAWS, FEVER, VitaminC) | ROC-AUC | [2204.04991](https://arxiv.org/abs/2204.04991) |
| **AlignScore** (ACL 2023) | TRUE + SummaC + altri 6: 22 in tutto, 19 mai visti | AUC, BAcc, correlazioni | [2305.16739](https://arxiv.org/abs/2305.16739) |
| **MiniCheck** (EMNLP 2024) | LLM-AggreFact (introdotto lì) | BAcc media per fonte | [2404.10774](https://arxiv.org/abs/2404.10774) |
| **FactCG** (NAACL 2025) | LLM-AggreFact | BAcc | [2501.17144](https://arxiv.org/abs/2501.17144) |
| **Granite Guardian** (IBM) | TRUE + benchmark di rischio | AUC | [2412.07724](https://arxiv.org/abs/2412.07724) |
| **Lynx** (Patronus) | HaluBench (introdotto lì) | accuratezza | [2407.08488](https://arxiv.org/abs/2407.08488) |
| **LettuceDetect** | RAGTruth | F1 | [2502.17125](https://arxiv.org/abs/2502.17125) |
| **HalluGuard** (Findings ACL 2026) | LLM-AggreFact (75,7) + RAGTruth (84,0) | BAcc | [2510.00880](https://arxiv.org/abs/2510.00880) |
| **FaithBench** (NAACL 2025) | allucinazioni di 10 LLM nei riassunti; i rilevatori migliori vicini al 50% | BAcc, F1 | [2410.13210](https://arxiv.org/abs/2410.13210) |
| **Evaluating Jev** (TDM, 29/9/2026) | 37 dataset, fra cui **LLM-AggreFact intero** (78,6), ANLI, PAWS | BAcc, ECE | [2609.37647](https://arxiv.org/abs/2609.37647) |
| **Jev-as-a-Judge for RAG** (TDM, 30/9/2026) | 495 claim di LLM-AggreFact | BAcc, ammissione dei non supportati, costo | [2610.20634](https://arxiv.org/abs/2610.20634) |
| **PremiseCheck** (TDM, 15/10/2026) | ReviewFact (9.267 commenti di revisione) | AUROC, costo | [2610.21046](https://arxiv.org/abs/2610.21046) |
| **Arize** (8 TDM, ottobre 2026) | RAGTruth (2.675) + valutatori Phoenix + routing | BAcc, errore di calibrazione, polarità | [blog](https://arize.com/blog/decision-model-benchmark/) |
| **Decision Index** (classifica dei TDM) | fra i pubblici: **RAGTruth**, **HoVer**, ANLI, ContractNLI, NLI4CT; dalla 0.4 il punteggio conta solo le parti private | punteggi corretti per il caso | [repo](https://github.com/apolinario/decision-index) |

**La prassi**: almeno un benchmark aggregato (LLM-AggreFact oppure TRUE/SummaC), un benchmark RAG (RAGTruth o HaluBench), e nei lavori recenti un benchmark difficile e moderno (FaithBench). I lavori sui TDM usano quasi sempre un solo benchmark di grounding: è lo spazio in cui la nostra suite fa la differenza.

## 2. La suite proposta

"In cache" = già nella cache HF di questa macchina.

### Livello 1 — indispensabile

| Benchmark | Cosa copre | Dimensione | Metrica | In cache | Note |
|---|---|---|---|---|---|
| **LLM-AggreFact** (11 fonti) | riassunti, dialoghi, RAG, claim di LLM | 29.320 | BAcc media per fonte + AUC; media anche **senza RAGTruth** | sì | CC BY-ND, solo valutazione |
| **HaluBench** (6 fonti) | QA con contesto: finanza, medicina, DROP | ~15.000 | BAcc per fonte | sì | parte RAGTruth riportata a parte |
| **HalluMix** | multi-documento, documenti lunghi, più domini | da verificare | BAcc | sì | sostituisce TRUE se il tempo manca |
| **Verifica di claim con evidenza data**: SciFact, Climate-FEVER, HealthVer, COVID-Fact | scienza, clima, salute | migliaia | BAcc, AUC | sì | fuori dalla Decision Index: test fuori dominio puliti per i TDM |
| **TRUE** (11 dataset) | riassunti, dialogo, parafrasi, verifica | ~100.000 (campionare a 5.000 per dataset) | ROC-AUC per dataset | no | confronto con AlignScore e Granite Guardian; facoltativo se il tempo manca |
| **FaithBench** | allucinazioni difficili nei riassunti | ~800 | BAcc | no | il più duro |

### Livello 2 — consigliato

| Benchmark | Perché | In cache |
|---|---|---|
| **RAGTruth** (test, livello di risposta) | confronto con Arize, LettuceDetect, HalluGuard | sì (`wandb/RAGTruth-processed`) |
| **LLM-Oasis** (claim verification) | testi di Wikipedia | sì |
| SummaC benchmark | standard storico | no |
| WiCE, SummEdits, VitaminC (test) | in distribuzione per B-ctrl; VitaminC anche per le coppie minime | sì |

### Livello 3 — facoltativo

| Benchmark | Perché |
|---|---|
| ContextualJudgeBench | se si parla di LLM-as-judge |
| SEAHORSE o altri multilingue | solo per una sezione multilingue |
| FACTS Grounding | **non adatto così com'è**: non fornisce coppie (documento, claim) etichettate da persone |

### Sempre presenti

- **Set diagnostico ridotto** ([05_set_diagnostico.md](05_set_diagnostico.md)).
- **Generalità del TDM dopo il fine-tuning** (solo B-ctrl, facoltativo): un sottoinsieme della Decision Index pubblica, **escludendo RAGTruth e HoVer**.

## 3. Attenzione alle sovrapposizioni (fra benchmark, con il training, con la Decision Index)

| Sovrapposizione | Problema | Come gestirla |
|---|---|---|
| TRUE, SummaC e LLM-AggreFact condividono dataset di riassunti | coppie contate due volte | deduplicare per documento + claim; riportare ogni aggregato da solo |
| RAGTruth è in LLM-AggreFact, in HaluBench **e nella Decision Index** | per i TDM recenti RAGTruth è probabilmente visto | AggreFact con e senza RAGTruth; blocco RAGTruth riportato una volta, segnalato |
| **HoVer è nella Decision Index** (e nel nostro training di B-ctrl) | non è fuori dominio per nessuno | mai come test fuori dominio |
| ANLI, ContractNLI, NLI4CT nella Decision Index | i TDM sono ottimizzati sull'NLI | non usarli come test; ANLI difficile resta solo nel training di B-ctrl |
| TRUE contiene FEVER e VitaminC, nel training di B-ctrl | in distribuzione per i modelli fine-tunati | riportare TRUE anche senza FEVER e VitaminC (TRUE-9) |

Regola generale: nella tabella principale dichiarare per ogni benchmark se è **mai visto** in training (per B-ctrl) e se è **nella Decision Index** (per i TDM).

## 4. I modelli

### 4.1 Typed decision model

Raggruppati per base, perché è il disegno di RQ2. Stato: **fatto** = run completo sul test di AggreFact; **pronto** = adapter nella repo; **nuovo** = da aggiungere.

| Base | Modello | Lettura | Licenza | Stato |
|---|---|---|---|---|
| — (chiuso) | **Jev 1.13** / **1.14** (TypeSafe) | API | — | 1.13 fatto (dev-tuned, da rifare zero-shot); 1.14 nuovo |
| — (chiuso) | OpenAI Decisions (beta pubblica), Microsoft Decision-1, d1-ultra (Liquid), pplx-decider via API | API | — | facoltativi |
| **Qwen3.5-9B** | JPT-9B | LoRA, lettere | CC BY-NC | fatto (76,04) |
| | elephant-1 (Baseten) | attenzione bidirezionale a blocchi, lettere | MIT | fatto (76,70) |
| | Bespoke-Nimble-9B-v3 | LoRA, 255 codici | CC BY-NC | pronto |
| | d3-flash (vLLM-SR) | attenzione non causale, 255 codici | Apache 2.0 | pronto |
| | StartLux-Decision-v1.1-9B-Instant | attenzione bidirezionale sul prompt (base da verificare) | CC BY-NC | pronto |
| | Kev-9B | LoRA + testa a puntatore | aperto | nuovo |
| | Nimble-v4 9B (base da verificare) | — | CC BY-NC | nuovo (77,8 in leaderboard) |
| **Qwen3.8-27B** | **Clef** (Cloudflare) | testa congiunta sugli stati nascosti | Apache 2.0 | nuovo (pesi in cache; 79,0 in leaderboard) |
| | **Nimble-v4-max** | — | CC BY-NC | nuovo (79,5 in leaderboard; ~55 GB BF16) |
| Gemma-4-12B | Winnow-12B | lettere | Apache 2.0 | fatto (BF16 76,95; NVFP4 77,49) |
| Gemma-4-26B-A4B | Rune-26B-A4B-v3 | lettere | aperto | fatto (76,24) |
| Qwen3.5 26B | d3-turbo | prefisso bidirezionale, 255 codici | Apache 2.0 | pronto (GPU 56 GiB+) |
| Qwen3.5 27B | StartLux-27B-AutoThink | pensa quando è incerto | CC BY-NC | pronto |
| Qwen3.5-4B | KnowLine-4B-Gen5.5 | lettura doppia dei prompt brevi | Apache 2.0 | fatto (75,65) |
| LFM2.5-VL-3B | d1-3B (Liquid) | parole yes/no | LFM Open | fatto (71,84) |
| Qwen3-8B (congelato) | CLM-v0.1-8B | contrastivo | Apache 2.0 | fatto (49,98: la regola a tre domande non è adatta) |
| MiMo-V2.6-Distill-Qwen-9B | Drex v1.5 | — | RAIL | facoltativo (76,5 in leaderboard) |
| ModernBERT | Laya (~0,4B) | encoder + testa | aperto | facoltativo (il più piccolo) |
| — | pplx-decider v1.2 27B / flash 4B | — | pesi aperti (licenza da verificare) | facoltativo (78,7 in leaderboard) |

### 4.2 Checker specializzati

| Modello | Taglia | Perché |
|---|---|---|
| **Bespoke-MiniCheck-7B** | 7B | riferimento storico (77,4); adapter pronto |
| MiniCheck-Flan-T5-L | 0,8B | riferimento di LLM-AggreFact |
| **FactCG-DeBERTa-L** | 0,4B | fatto (75,64) |
| AlignScore-large | 0,4B | riferimento per TRUE |
| **Granite Guardian 3.3** | 8B | forte su AggreFact (76,5) |
| HHEM-2.1-open, Lynx-8B, HalluGuard-4B | 0,1–8B | facoltativi |

### 4.3 Basi generiche lette come un TDM (B1, il cuore di RQ2)

| Base | TDM del gruppo | Letture |
|---|---|---|
| **Qwen3.5-9B** (in cache) | JPT, elephant-1, Nimble-v3, d3-flash, StartLux-9B, Kev-9B | ingenua (prompt di ogni TDM + prompt neutro) e calibrata (AnyJev L0 + temperatura) |
| **Qwen3.8-27B** | Clef, Nimble-v4-max | idem; serve GPU da 80 GB o FP8 |
| Gemma-4-12B | Winnow | idem |
| Qwen3.5-4B | KnowLine | idem |
| LFM2.5-VL-3B | d1-3B | idem |
| Gemma-4-26B-A4B, Qwen3-8B | Rune, CLM | facoltative |

### 4.4 LLM generici che generano (B2, giudici)

| Gruppo | Modelli | Come |
|---|---|---|
| Aperti | un modello piccolo (es. Qwen3.5-9B stesso in modalità generativa) e uno medio (es. gpt-oss-20b o Qwen3.6-35B-A3B) | "Yes/No" generato; con e senza ragionamento |
| Chiusi | un modello economico e uno di punta di un fornitore | **su un sottoinsieme** (es. 200 righe per fonte); API batch; per la cascata con controllo a escalation casuale |

Per i modelli vecchi (GPT-4o, Claude 3.5 Sonnet, Llama-3.1-70B, ...) si citano i numeri della leaderboard.

## 5. Selezione consigliata

### Versione minima (il paper del 4 gennaio)

| Gruppo | Modelli |
|---|---|
| TDM (~13) | Jev 1.13 e 1.14; gruppo Qwen3.5-9B (JPT, elephant-1, Nimble-v3, d3-flash, StartLux-9B, Kev-9B); Winnow-12B; KnowLine-4B; d1-3B; Rune-26B; Clef |
| Basi (4) | Qwen3.5-9B, Gemma-4-12B, Qwen3.5-4B, LFM2.5-VL-3B, con due letture |
| Specializzati (4) | Bespoke-MiniCheck-7B, MiniCheck-Flan-T5-L, FactCG, Granite Guardian 3.3 |
| Generativi (2–3) | un aperto piccolo, un chiuso economico, un chiuso di punta su sottoinsieme |

Benchmark: livello 1 senza TRUE e FaithBench se il tempo manca (tutti gli altri sono già in cache), più il set diagnostico ridotto.

### Versione estesa (se c'è tempo, o nel secondo paper)

Nimble-v4 / v4-max e la base Qwen3.8-27B, d3-turbo, StartLux-27B-AutoThink, pplx-decider, Drex, Laya, CLM con domande adatte; TRUE, FaithBench, SummaC; i chiusi facoltativi.

## 6. Costi stimati

| Voce | Stima |
|---|---|
| Coppie della suite minima (AggreFact + HaluBench + HalluMix + 4 di verifica + diagnostico) | ~60.000–70.000 |
| TDM da ~9B, tre domande per coppia, su L40S | ~2–6 ore per modello sulla suite (riferimento: Winnow BF16 9,8 righe/s, elephant-1 5,9, KnowLine 2,9 sul test di AggreFact) |
| ~15 modelli aperti + 4 basi × 2 letture | ~80–120 ore GPU |
| Jev su tutta la suite | pochi dollari (il test di AggreFact è costato ~1,5 $) |
| LLM chiusi su un sottoinsieme (~5.000 coppie) | da qualche decina a qualche centinaio di dollari; API batch |

## Fonti

- TRUE https://arxiv.org/abs/2204.04991 · AlignScore https://arxiv.org/abs/2305.16739 · MiniCheck https://arxiv.org/abs/2404.10774 · FactCG https://arxiv.org/abs/2501.17144
- Granite Guardian https://arxiv.org/abs/2412.07724 · Lynx / HaluBench https://arxiv.org/abs/2407.08488 · LettuceDetect https://arxiv.org/abs/2502.17125 · HalluGuard https://arxiv.org/abs/2510.00880 · FaithBench https://arxiv.org/abs/2410.13210
- Leaderboard di LLM-AggreFact https://llm-aggrefact.github.io/ · Decision Index https://github.com/apolinario/decision-index
- TDM: Clef https://huggingface.co/Cloudflare/clef · Nimble-v4-max https://huggingface.co/bespokelabs/Nimble-v4-max · Drex https://huggingface.co/nace-ai/drex-v1.5 · Laya https://huggingface.co/convaiinnovations/laya · Kev https://github.com/jaredpalmer/kev · CLM https://huggingface.co/Contrastive-LM/CLM-v0.1-8B · AnyJev https://github.com/nokia-applied-research/AnyJev
- Lavori sui TDM e il grounding: vedi [09_aggiornamento_ottobre_2026.md](09_aggiornamento_ottobre_2026.md)
