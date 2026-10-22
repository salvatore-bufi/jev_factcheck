# Stato dell'arte e novità

## 1. Tre filoni che si incontrano

### 1.1 Typed decision model (TDM)

Modelli che non generano testo: leggono uno stato e domande tipizzate (*choice*, *noul* sì/no, *score*) e restituiscono una probabilità per opzione in una sola passata.

| Modello | Tipo | Note |
|---|---|---|
| Jev 1.13 (TypeSafe) | chiuso, API | rilasciato il 15/09/2026; riferimento del filone |
| Winnow-12B | aperto (Gemma-4-12B), Apache 2.0 | GGUF + server llama.cpp |
| JPT-9B | aperto (Qwen3.5-9B + LoRA), CC BY-NC 4.0 | implementazione `llm2jev` |
| Rune-26B-A4B | aperto (Gemma-4 MoE) | |
| Laya (421M), Kev-9B/4B, djev-0.1 | aperti | citati in JevBench e nei benchmark di Winnow |

Letteratura (settembre–ottobre 2026), tutta molto recente:
- **Audit dei primi 28 paper** ([2609.32160](https://arxiv.org/abs/2609.32160)): rassegna e **checklist di 14 punti** per valutare i TDM (baseline, disegno dei test, riproducibilità, calibrazione e deployment). Nessun paper si concentra su fact checking, allucinazioni, NLI o verifica di claim; il fine-tuning dei modelli aperti non è trattato.
- **JEV-as-a-Judge** ([2609.26550](https://arxiv.org/html/2609.26550v1)): Jev come giudice economico con escalation; fra i compiti, **240 casi di HaluEval** con evidenza; trova rumore nelle etichette di HaluEval. Solo Jev, nessun modello aperto, nessun benchmark di grounding completo.
- **JevBench** ([preprint](https://www.preprints.org/frontend/manuscript/da3d1f1f93bf32fe4a28e70c6168ef87/download_pub)): framework di valutazione generale dei TDM (Jev, djev-0.1, Laya).
- Altri: copertura delle opzioni candidate ([2610.03387](https://arxiv.org/abs/2610.03387)), agent harness ([2610.02267](https://arxiv.org/abs/2610.02267)), penetration testing ([2609.28940](https://arxiv.org/abs/2609.28940)), decisioni scientifiche ([2609.24965](https://arxiv.org/abs/2609.24965)).

### 1.2 Fact checker specializzati su documento

- **MiniCheck** ([2404.10774](https://arxiv.org/abs/2404.10774), EMNLP 2024): introduce LLM-AggreFact e i dati sintetici C2D/D2C; Bespoke-MiniCheck-7B (77,4) è il miglior modello aperto della leaderboard.
- **FactCG** ([2501.17144](https://arxiv.org/abs/2501.17144), NAACL 2025): dati multi-hop da grafi di contesto (CG2C); DeBERTa da 0,4B a 75,6.
- AlignScore, SummaC, HHEM, Granite Guardian, Lynx/HaluBench ([2407.08488](https://arxiv.org/abs/2407.08488)).

Questi modelli sono **addestrati per un solo compito**. I TDM sono generali e vanno adattati al compito solo con la domanda.

### 1.3 LLM come giudici e calibrazione

- LLM generici letti tramite le probabilità di "sì/no" in una passata (classificazione con label log-probability): idea vecchia, che i TDM rendono un prodotto.
- Lavori recenti sostengono che sui modelli proprietari post-2025 la confidenza verbalizzata sia più affidabile delle log-probabilità ([2609.10996](https://arxiv.org/html/2609.10996v1)) e che la calibrazione vada trattata come criterio primario di valutazione ([2609.26489](https://arxiv.org/pdf/2609.26489)). Sono temi che il nostro paper tocca direttamente: i TDM promettono probabilità calibrate per costruzione.

## 2. Il buco

| Domanda | Coperta da qualcuno? |
|---|---|
| Quanto valgono i TDM (aperti e chiusi) come fact checker su documento, su un benchmark completo come LLM-AggreFact? | **No** (solo 240 casi HaluEval con Jev) |
| Battono i checker specializzati di pari taglia? E un LLM generico letto allo stesso modo? | **No** |
| Le loro probabilità sono calibrate su questo compito? La soglia trasferisce fra domini? | **No** (calibrazione dei TDM studiata in generale, non sul grounding) |
| Scomporre la decisione in più domande tipizzate (supporto, dettaglio non stabilito, contraddizione) aiuta? | **No** |
| Sono robusti a formulazione della domanda, nomi dei campi, distrattori, rimozione dell'evidenza? | In parte per altri compiti (invarianza delle decisioni), non per il fact checking |
| Si possono specializzare con fine-tuning senza perdere le capacità generali? | **No** |

## 3. Cosa rende il contributo difendibile

1. **Primo studio sistematico** dei TDM sul fact checking documento–claim, con modelli aperti e chiusi.
2. **Protocollo zero-shot pulito** su LLM-AggreFact (decisioni prese su dati nostri), quindi confrontabile con la leaderboard.
3. **Le baseline che un revisore chiederà**: checker specializzati e LLM generici con lo stesso prompt e la stessa lettura dei logit. Separano l'effetto dell'addestramento "da decision model" dall'effetto del semplice leggere le probabilità.
4. **Calibrazione e robustezza** misurate, non solo l'accuratezza.
5. **Metodo di specializzazione** con misura esplicita dell'oblio.
6. Rispetto della **checklist dell'audit** (§4): è un argomento forte in revisione, perché mostra che il lavoro segue gli standard del filone.

## 4. La checklist dell'audit (da rispettare)

| ID | Requisito | Come lo copriamo |
|---|---|---|
| B1 | baseline con log-probabilità delle etichette | LLM generici (Qwen3.5-9B, Gemma-4-12B, base dei TDM aperti) con lo stesso prompt |
| B2 | cascate generative economiche | LLM piccolo che genera "Yes/No" (più eventuale ragionamento breve) |
| B3 | modelli specifici del compito | MiniCheck, Bespoke-MiniCheck-7B, FactCG, AlignScore |
| T1 | dispersione fra modelli | più TDM e più baseline; intervalli di confidenza |
| T2 | invarianza delle decisioni | test comportamentali (formulazione, distrattori, posizione) |
| T3 | scambio dei nomi delle opzioni | inversione e rinomina delle opzioni e dei campi |
| T4 | provenienza dei dati | tabelle di provenienza e decontaminazione |
| R1 | versioni e date esatte | revisioni dei pesi, versione di Jev, date delle chiamate |
| R2 | intervalli di confidenza | bootstrap per documento |
| R3 | scelte fatte dopo la valutazione | registro degli esperimenti; dichiarazione del controllo intermedio di Jev |
| C1 | calibrazione su dati tenuti da parte | ECE e Brier su test fuori dominio |
| C2 | accuratezza ad alta confidenza | copertura selettiva |
| C3 | soglie congelate prima del test | protocollo dello Step 0 |
| C4 | latenza con e senza cache | misure su vLLM e sul server Winnow |

## 5. Rischi per la novità

- **Scoop**: decine di paper sui TDM in un mese. Un lavoro sul fact checking può uscire in qualsiasi momento. Contromisura: preprint su arXiv della Parte A appena pronta, poi la versione completa.
- **"È solo una valutazione"**: per una conferenza principale serve qualcosa in più del benchmarking. La Parte B (metodo) e le analisi (calibrazione, scomposizione in domande, robustezza) rispondono a questa critica.
- **Modello chiuso (Jev)**: versioni che cambiano, prompt interno non noto. Va riportata la versione esatta, e le conclusioni principali vanno appoggiate ai modelli aperti.

Prima della sottomissione: rifare la ricerca bibliografica (arXiv, ACL Anthology, Semantic Scholar) con le chiavi "typed decision model", "System One", "Jev", "decision model fact", "LLM-AggreFact 2026".
