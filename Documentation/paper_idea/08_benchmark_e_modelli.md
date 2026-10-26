# Benchmark e modelli: cosa usano i lavori del settore e cosa usare noi

Integra [03_parte_A_valutazione.md](03_parte_A_valutazione.md). Rivisto il 26 ottobre 2026 con i rilasci di ottobre e con il vincolo dei modelli fino a 9B. Valutare solo su LLM-AggreFact resta un punto debole: i paper di riferimento usano più benchmark.

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

**Vincolo**: solo modelli fino a 9B parametri (taglia nominale; la classe "9B" di Qwen3.5 ha ~9,4B parametri con la torre visiva, StartLux-9B ~9,7B). Parametri da Hugging Face dove disponibili.

### 4.1 Typed decision model

Raggruppati per base, perché è il disegno di RQ2. Stato: **fatto** = run completo sul test di AggreFact; **pronto** = adapter nella repo; **nuovo** = da aggiungere.

| Base | Modello | Lettura | Licenza | Stato |
|---|---|---|---|---|
| **Qwen3.5-9B** | JPT-9B | LoRA, lettere | CC BY-NC | fatto (76,04) |
| | elephant-1 (Baseten) | attenzione bidirezionale a blocchi, lettere | MIT | fatto (76,70) |
| | Bespoke-Nimble-9B-v3 | LoRA, 255 codici | CC BY-NC | pronto |
| | d3-flash (vLLM-SR, 8,4B) | attenzione non causale, 255 codici | Apache 2.0 | pronto |
| | StartLux-Decision-v1.1-9B-Instant | attenzione bidirezionale sul prompt (base non dichiarata) | CC BY-NC | pronto |
| | **Clef-flash** (Cloudflare) | testa congiunta sugli stati nascosti | Apache 2.0 | nuovo (76,6 in leaderboard) |
| | **Tacit-9B** (AnyJev) | autodistillazione del ragionamento, lettere | da verificare | nuovo |
| Qwen3.5-9B-Base | Kev-9B | LoRA + testa a puntatore | Apache 2.0 | nuovo |
| **Qwen3.5-4B** | KnowLine-4B-Gen5.5 | lettura doppia dei prompt brevi | Apache 2.0 | fatto (75,65) |
| | **pplx-decider-flash v1.2** (Perplexity) | — | Apache 2.0 | nuovo |
| | SII-Decision-4B, Intern-Decision-4B | autodistillazione; segnaposto per campo | da verificare | facoltativi (base da verificare) |
| Qwen3.5-4B-Base | Kev-4B | LoRA + testa a puntatore | Apache 2.0 | nuovo |
| Qwen3.5-2B | Strands Decider 2B (AWS), Tacit-2B | — | da verificare | nuovi (curva di taglia) |
| Qwen3-8B | CLM-v0.1-8B (encoder congelato) | contrastivo | Apache 2.0 | fatto (49,98: la regola a tre domande non è adatta) |
| | Tacit-8B | autodistillazione, lettere | da verificare | facoltativo |
| LFM2.5-VL-3B | d1-3B (Liquid) | parole yes/no | LFM Open | fatto (71,84) |
| MiMo-V2.6-Distill-Qwen-9B | Drex v1.5 | — | RAIL | facoltativo (76,5 in leaderboard) |
| ModernBERT | Laya (0,42B) | encoder + testa | Apache 2.0 | facoltativo (il più piccolo) |
| — | Intern-Decision 0,8B / 2B | segnaposto per campo | da verificare | facoltativi (curva di taglia) |

**Riferimenti fuori vincolo o non eseguibili in locale** (una riga in fondo alla tabella principale, nessuna analisi):

| Modello | Perché | Costo |
|---|---|---|
| Jev 1.13 / 1.14 (chiuso) | il TDM di riferimento; da riprodurre il 78,6 di [2609.37647](https://arxiv.org/abs/2609.37647) | pochi dollari |
| Winnow-12B (76,95 / 77,49), Rune-26B-A4B (76,24) | già misurati nella repo | zero |
| Nimble-v4 9B (77,8, solo API), Nimble-v4-max (79,5), Clef (79,0), pplx-decider 27B (78,7), d1-ultra (78,5) | numeri della leaderboard | zero |
| d3-turbo, StartLux-27B-AutoThink | adapter pronti nella repo, ma sopra il vincolo: non si eseguono | — |

### 4.2 Checker specializzati (tutti fino a 9B)

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
| **Qwen3.5-9B** (in cache) | JPT, elephant-1, Nimble-v3, d3-flash, StartLux-9B, Clef-flash, Tacit-9B | ingenua (prompt di ogni TDM + prompt neutro) e calibrata (AnyJev L0 + temperatura) |
| Qwen3.5-9B-Base | Kev-9B | idem (il prompt a chat non funziona sulle basi pre-addestrate: [2610.17205](https://arxiv.org/abs/2610.17205); usare il formato di Kev) |
| **Qwen3.5-4B** | KnowLine, pplx-decider-flash | idem |
| Qwen3.5-4B-Base | Kev-4B | idem |
| LFM2.5-VL-3B | d1-3B | idem |
| Qwen3.5-2B, Qwen3-8B | Strands, Tacit-2B; CLM, Tacit-8B | facoltative |

### 4.4 LLM generici che generano (B2, giudici)

| Gruppo | Modelli | Come |
|---|---|---|
| Aperti, fino a 9B | Qwen3.5-9B e Qwen3.5-4B in modalità generativa | "Yes/No" generato; con e senza ragionamento |
| Chiusi (solo riferimento) | un modello economico di un fornitore | **su un sottoinsieme** (es. 200 righe per fonte); API batch; per la cascata con controllo a escalation casuale |

Per i modelli vecchi (GPT-4o, Claude 3.5 Sonnet, Llama-3.1-70B, ...) si citano i numeri della leaderboard.

## 5. Selezione consigliata

### Versione minima (il paper del 4 gennaio)

| Gruppo | Modelli |
|---|---|
| TDM fino a 9B (~13) | gruppo Qwen3.5-9B (JPT, elephant-1, Nimble-v3, d3-flash, StartLux-9B, Clef-flash, Tacit-9B); Kev-9B; gruppo Qwen3.5-4B (KnowLine, pplx-decider-flash); Kev-4B; d1-3B; Laya |
| Basi (4–5) | Qwen3.5-9B, Qwen3.5-9B-Base, Qwen3.5-4B, Qwen3.5-4B-Base, LFM2.5-VL-3B, con due letture |
| Specializzati (4) | Bespoke-MiniCheck-7B, MiniCheck-Flan-T5-L, FactCG, Granite Guardian 3.3 |
| Generativi (2) | Qwen3.5-9B generativo, un chiuso economico su sottoinsieme |
| Riferimenti | Jev 1.13 e 1.14; Winnow e Rune dalla cache; i 27B dalla leaderboard |

Benchmark: livello 1 senza TRUE e FaithBench se il tempo manca (tutti gli altri sono già in cache), più il set diagnostico ridotto.

### Versione estesa (se c'è tempo, o nel secondo paper)

La curva di taglia completa (Strands 2B, Tacit-2B, Intern-Decision 0,8B/2B/4B, SII-Decision-4B), Drex v1.5, Tacit-8B, CLM con domande adatte; TRUE, FaithBench, SummaC.

## 6. Costi stimati

| Voce | Stima |
|---|---|
| Coppie della suite minima (AggreFact + HaluBench + HalluMix + 4 di verifica + diagnostico) | ~60.000–70.000 |
| TDM da ~9B, tre domande per coppia | ~3–6 ore per modello sulla suite (riferimento sul test di AggreFact: elephant-1 5,9 righe/s, KnowLine-4B 2,9 righe/s); i 4B e i più piccoli meno |
| ~15 modelli + 5 basi × 2 letture | ~70–100 ore GPU, su GPU da 24–48 GB (anche in parallelo) |
| Jev su tutta la suite (riferimento) | pochi dollari (il test di AggreFact è costato ~1,5 $) |
| LLM chiuso su un sottoinsieme (~5.000 coppie, riferimento) | qualche decina di dollari; API batch |

## Fonti

- TRUE https://arxiv.org/abs/2204.04991 · AlignScore https://arxiv.org/abs/2305.16739 · MiniCheck https://arxiv.org/abs/2404.10774 · FactCG https://arxiv.org/abs/2501.17144
- Granite Guardian https://arxiv.org/abs/2412.07724 · Lynx / HaluBench https://arxiv.org/abs/2407.08488 · LettuceDetect https://arxiv.org/abs/2502.17125 · HalluGuard https://arxiv.org/abs/2510.00880 · FaithBench https://arxiv.org/abs/2410.13210
- Leaderboard di LLM-AggreFact https://llm-aggrefact.github.io/ · Decision Index https://github.com/apolinario/decision-index
- TDM: Clef-flash https://huggingface.co/Cloudflare/clef-flash · pplx-decider-flash https://huggingface.co/perplexity-ai/pplx-decider-flash-v1.2 · Tacit https://huggingface.co/morriszjm/Tacit-9B · Kev-9B https://huggingface.co/jaredpalmer/kev-9b · Strands Decider https://huggingface.co/amazon/strands-decider-2B-qwen3.5-v1-2610 · Intern-Decision https://huggingface.co/internlm/Intern-Decision-4B · Clef https://huggingface.co/Cloudflare/clef · Nimble-v4-max https://huggingface.co/bespokelabs/Nimble-v4-max · Drex https://huggingface.co/nace-ai/drex-v1.5 · Laya https://huggingface.co/convaiinnovations/laya · Kev https://github.com/jaredpalmer/kev · CLM https://huggingface.co/Contrastive-LM/CLM-v0.1-8B · AnyJev https://github.com/nokia-applied-research/AnyJev
- Lavori sui TDM e il grounding: vedi [09_aggiornamento_ottobre_2026.md](09_aggiornamento_ottobre_2026.md)
