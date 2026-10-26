# Benchmark e modelli: cosa usano i paper del settore e cosa usare noi

Integra [03_parte_A_valutazione.md](03_parte_A_valutazione.md). Valutare solo su LLM-AggreFact sarebbe un punto debole: i paper di riferimento usano **più benchmark**, e i revisori se lo aspettano. Anche i modelli vanno ampliati: oggi la repo ne ha cinque.

## 1. Su cosa si valutano i paper di fact checking su documento

| Paper / modello | Benchmark usati | Metrica | Fonte |
|---|---|---|---|
| **TRUE** (Honovich et al., NAACL 2022) | introduce TRUE: 11 dataset (FRANK, SummEval, MNBM, QAGS-CNNDM, QAGS-XSum, BEGIN, Q², DialFact, PAWS, FEVER, VitaminC) su riassunti, dialogo, parafrasi, verifica di claim | ROC-AUC | [2204.04991](https://arxiv.org/abs/2204.04991) |
| **AlignScore** (ACL 2023) | TRUE (11) + SummaC benchmark (6) + altri 6 dataset: 22 in tutto, 19 mai visti in training | AUC, BAcc, correlazioni | [2305.16739](https://arxiv.org/abs/2305.16739) |
| **MiniCheck** (EMNLP 2024) | LLM-AggreFact (introdotto lì) | BAcc media per fonte | [2404.10774](https://arxiv.org/abs/2404.10774) |
| **FactCG** (NAACL 2025) | LLM-AggreFact (+ analisi sul ragionamento multi-hop) | BAcc | [2501.17144](https://arxiv.org/abs/2501.17144) |
| **Granite Guardian** (IBM) | TRUE (AUC media 0,854 per la versione 3.0-8B) + benchmark di rischio | AUC | [2412.07724](https://arxiv.org/abs/2412.07724) |
| **Lynx** (Patronus) | HaluBench (introdotto lì): DROP, CovidQA, PubMedQA, FinanceBench, HaluEval, RAGTruth | accuratezza | [2407.08488](https://arxiv.org/abs/2407.08488) |
| **LettuceDetect** | RAGTruth (livello di risposta e di span) | F1 | [2502.17125](https://arxiv.org/abs/2502.17125) |
| **HalluGuard** (Findings ACL 2026) | LLM-AggreFact intero (75,7) + focus su RAGTruth (84,0) | BAcc | [2510.00880](https://arxiv.org/abs/2510.00880) |
| **FaithBench** (NAACL 2025) | introduce FaithBench: allucinazioni di 10 LLM moderni nei riassunti, annotate da esperti; i rilevatori migliori sono vicini al 50% | BAcc, F1 | [2410.13210](https://arxiv.org/abs/2410.13210) |
| **JEV-as-a-Judge** (TDM, 2026) | 240 casi di HaluEval con evidenza (fra altri compiti non di fact checking) | accuratezza | [2609.26550](https://arxiv.org/html/2609.26550v1) |

**La prassi**: almeno **un benchmark aggregato** (LLM-AggreFact oppure TRUE/SummaC), più **un benchmark RAG** (RAGTruth o HaluBench), e nei lavori recenti **un benchmark difficile e moderno** (FaithBench). I lavori più solidi (AlignScore) dichiarano quanti dataset non sono mai stati visti in training.

## 2. La suite proposta

### Livello 1 — indispensabile

| Benchmark | Cosa copre | Dimensione | Metrica principale | Note |
|---|---|---|---|---|
| **LLM-AggreFact** (11 fonti) | riassunti, dialoghi, RAG, claim di LLM | 29.320 | BAcc media per fonte (come la leaderboard) + AUC | già valutato per 5 modelli |
| **TRUE** (11 dataset) | riassunti, dialogo, parafrasi, verifica di claim | ~100.000 coppie (dominate da VitaminC e FEVER) | **ROC-AUC** per dataset (come nel paper originale e in AlignScore/Granite Guardian) + BAcc | campionare i dataset più grandi (es. massimo 5.000 per dataset) per contenere i costi |
| **HaluBench** (6 fonti) | QA con contesto: finanza, medicina, DROP | ~15.000 | accuratezza e BAcc per fonte | la parte RAGTruth si riporta a parte (è anche in LLM-AggreFact) |
| **RAGTruth** (test, livello di risposta) | risposte di LLM in RAG: QA, data-to-text, riassunti | ~2.700 risposte (da verificare) | BAcc, F1 | confronto diretto con LettuceDetect e HalluGuard |
| **FaithBench** | allucinazioni difficili di LLM moderni nei riassunti | ~800 (da verificare) | BAcc | il più duro: i rilevatori attuali sono vicini al 50% |

### Livello 2 — consigliato

| Benchmark | Perché |
|---|---|
| **SummaC benchmark** (6 dataset di riassunti) | standard storico; utile per il confronto con AlignScore e SummaC |
| **HalluMix** | multi-documento, documenti lunghi, più domini |
| **Verifica di claim fuori dominio**: SciFact, Climate-FEVER, HealthVer / COVID-Fact | domini scientifico, clima, salute; claim reali; evidenza data |
| **LLM-Oasis** (test annotato a mano) | testi di Wikipedia, se disponibile la variante con evidenza |

### Livello 3 — facoltativo

| Benchmark | Perché |
|---|---|
| ContextualJudgeBench | giudizi contestuali, se si vuole parlare di LLM-as-judge |
| Benchmark multilingue (es. SEAHORSE per i riassunti) | solo se si vuole una sezione multilingue (Laya è multilingue) |
| FACTS Grounding | **non adatto così com'è**: valuta la generazione di risposte giudicata da LLM, non fornisce coppie (documento, claim) etichettate da persone |

### Sempre presenti

- **Set diagnostico** ([05_set_diagnostico.md](05_set_diagnostico.md)).
- **Oblio** (solo per la Parte B): JevBench pubblico, Kev-v9 clean, typed-decisions test.

## 3. Attenzione alle sovrapposizioni fra benchmark (e con il training)

| Sovrapposizione | Problema | Come gestirla |
|---|---|---|
| TRUE, SummaC e LLM-AggreFact condividono dataset di riassunti (FRANK, SummEval, QAGS, XSumFaith compaiono in più aggregati) | la stessa coppia contata due volte, risultati correlati | deduplicare per documento + claim; riportare ogni aggregato da solo, non sommarli |
| RAGTruth è in LLM-AggreFact e in HaluBench | idem | blocco RAGTruth riportato una sola volta |
| TRUE contiene **FEVER e VitaminC**, che sono nel nostro training (Parte B) | per i modelli fine-tunati, quei due dataset diventano "in distribuzione" | per la Parte B riportare TRUE anche **senza FEVER e VitaminC** (TRUE-9); dichiararlo |
| HaluBench contiene HaluEval, già usato in JEV-as-a-Judge | — | non è un problema, ma va citato |

Regola generale: nella tabella principale dichiarare per ogni benchmark se è **mai visto** in training, come fa AlignScore.

## 4. I modelli

### 4.1 Typed decision model / System One

| Modello | Architettura | Accesso | Note |
|---|---|---|---|
| **Jev 1.13** (TypeSafe) | chiuso | API (~0,042 $ per milione di token di input) | riferimento; già valutato |
| **OpenAI Decisions API** (`gpt-6-luna`) | chiuso | **preview limitata**; 0,10 $ per milione di token di input, output gratis | se si ottiene l'accesso, è il secondo TDM chiuso: confronto molto interessante |
| **Winnow-12B** | autoregressivo (Gemma-4-12B fine-tunato), GGUF | aperto, Apache 2.0 | già valutato (BF16, NVFP4); aggiungere Q8_0 |
| **JPT-9B** | autoregressivo (Qwen3.5-9B + LoRA) | aperto, CC BY-NC | già valutato |
| **Rune-26B-A4B** | autoregressivo MoE (Gemma-4) | aperto | già valutato |
| **Kev-9B / Kev-4B** (e taglie minori) | LoRA + "pointer head" su Qwen3.5 | aperto, server compatibile System One | da fare; varie taglie → curva di scala |
| **Laya** (395M) | **encoder** (ModernBERT-large) + testa di decisione, non autoregressivo, multilingue | aperto | da fare; il più piccolo e veloce |
| **djev-0.1** | **diffusione** (DiffusionGemma 26B-A4B) | aperto | da fare |
| **CLM-v0.1-8B** (Stanford / NVIDIA) | **contrastivo**: encoder Qwen3-8B congelato + teste di stato e azione (InfoNCE) | aperto; server con schema System One | runner già pronto nella repo (`evaluate.py`), mai eseguito |

Sono **nove TDM con quattro architetture diverse** (autoregressivo, encoder, diffusione, contrastivo) e taglie da 0,4B a 26B. È un asse di analisi a sé: *quale tipo di decision model funziona come fact checker?*

### 4.2 Checker specializzati

| Modello | Taglia | Perché |
|---|---|---|
| MiniCheck-RoBERTa-L, MiniCheck-DeBERTa-L, MiniCheck-Flan-T5-L | 0,4–0,8B | riferimento di LLM-AggreFact |
| **Bespoke-MiniCheck-7B** | 7B | miglior aperto della leaderboard (77,4); pesi già nella cache HF di questa macchina |
| **FactCG-DeBERTa-L** | 0,4B | già valutato (75,64) |
| AlignScore-large | 0,4B | riferimento per TRUE e SummaC |
| HHEM-2.1-open (Vectara) | 0,1B | molto usato in produzione |
| **Granite Guardian 3.3** | 8B | forte su TRUE e AggreFact (76,5) |
| **Lynx-8B** (70B facoltativo) | 8B / 70B | riferimento di HaluBench |
| **HalluGuard-4B** | 4B | recente (2026), 75,7 su LLM-AggreFact |
| SummaC-ZS | 0,1B | baseline storica (facoltativa) |

### 4.3 LLM generici letti come un TDM (baseline B1 della checklist)

Stesso prompt del TDM corrispondente, si leggono i logit delle lettere, una passata:

| Modello | Perché |
|---|---|
| **Gemma-4-12B** | base di Winnow → confronto diretto |
| **Qwen3.5-9B** | base di JPT e di Kev-9B → confronto diretto |
| Qwen3-8B | base di CLM |
| Gemma-4-26B-A4B | base di Rune (e affine a djev) |
| Una scala della stessa famiglia (es. Qwen3.5 piccoli → 9B, o Gemma-4 → 31B) | effetto della taglia a parità di metodo |

### 4.4 LLM generici che generano (baseline B2, giudici)

| Gruppo | Modelli | Come |
|---|---|---|
| Aperti | gpt-oss-20b, gpt-oss-120b, Qwen3-30B-A3B, Gemma-4-31B, Mistral-Small-3.2-24B, DeepSeek-V4.1-Flash | risposta "Yes/No" generata; con e senza ragionamento |
| Chiusi | un modello di punta e uno economico per fornitore: famiglia GPT-5.x/6 (OpenAI), Claude Sonnet 5.5 e Haiku 4.5 (Anthropic), Gemini 3.x Pro e Flash (Google) | **su un sottoinsieme** (es. 200 righe per fonte) per contenere i costi |

Per i modelli più vecchi (GPT-4o, Claude 3.5 Sonnet, Llama-3.1-70B, ...) si citano i numeri della leaderboard di LLM-AggreFact invece di rifarli.

## 5. Selezione consigliata

### Versione minima (paper solido, ~20 modelli)

| Gruppo | Modelli |
|---|---|
| TDM (8) | Jev, Winnow-12B, JPT-9B, Rune-26B, Kev-9B, Laya, djev-0.1, CLM-8B (+ OpenAI Decisions se accessibile) |
| Specializzati (6) | Bespoke-MiniCheck-7B, MiniCheck-Flan-T5-L, FactCG-DeBERTa-L, AlignScore-L, Granite Guardian 3.3, Lynx-8B |
| LLM letti come TDM (2) | Gemma-4-12B, Qwen3.5-9B |
| LLM generativi (4) | gpt-oss-20b, Qwen3-30B-A3B, un chiuso di punta, un chiuso economico |

Benchmark: livello 1 completo, più i tre di verifica di claim fuori dominio.

### Versione estesa (se c'è tempo)

Aggiungere: Kev-4B e taglie minori (curva di scala), HHEM, HalluGuard-4B, Lynx-70B, Qwen3-8B e Gemma-4-26B letti come TDM, gpt-oss-120b, Gemma-4-31B, DeepSeek-V4.1-Flash; livello 2 completo.

## 6. Costi stimati (da verificare)

| Voce | Stima |
|---|---|
| Coppie della suite di livello 1 (TRUE campionato a 5.000 per dataset) | ~90.000 coppie |
| Modello aperto da ~9–12B, tre domande per coppia, su L40S | ~4–6 ore per modello (riferimento: Winnow NVFP4 ~4,3 coppie/s sul portatile) |
| 15 modelli aperti | ~60–90 ore GPU |
| Modelli piccoli (Laya, DeBERTa, RoBERTa, T5) | minuti o poche ore |
| Jev su tutta la suite | pochi dollari (l'intero test di LLM-AggreFact è costato ~1,5 $) |
| LLM chiusi su un sottoinsieme (~5.000 coppie) | da qualche decina a qualche centinaio di dollari secondo il modello; usare l'API batch |

## 7. Cosa cambia nel paper

- La **tabella principale** diventa: modelli × {LLM-AggreFact, TRUE, HaluBench, RAGTruth, FaithBench, verifica di claim fuori dominio}, più la media.
- Nuova analisi: **architettura del TDM** (autoregressivo, encoder, diffusione, contrastivo) contro qualità e costo.
- Nuova analisi: **taglia** (Kev da piccolo a 9B, Laya 0,4B, Rune 26B).
- RQ2 (TDM contro la base generica) si fa su quattro coppie: Winnow/Gemma-4-12B, JPT e Kev/Qwen3.5-9B, CLM/Qwen3-8B, Rune/Gemma-4-26B.

## Fonti

- TRUE: https://arxiv.org/abs/2204.04991 · AlignScore: https://arxiv.org/abs/2305.16739 · MiniCheck: https://arxiv.org/abs/2404.10774 · FactCG: https://arxiv.org/abs/2501.17144
- Granite Guardian: https://arxiv.org/abs/2412.07724 · Lynx / HaluBench: https://arxiv.org/abs/2407.08488 · LettuceDetect: https://arxiv.org/abs/2502.17125 · HalluGuard: https://arxiv.org/abs/2510.00880 · FaithBench: https://arxiv.org/abs/2410.13210
- TDM: Laya https://huggingface.co/convaiinnovations/laya · Kev https://github.com/jaredpalmer/kev · djev https://github.com/Davipar/djev-dev · CLM https://huggingface.co/Contrastive-LM/CLM-v0.1-8B · OpenAI Decisions API https://www.firecrawl.dev/blog/openai-decisions-api-vs-jev
