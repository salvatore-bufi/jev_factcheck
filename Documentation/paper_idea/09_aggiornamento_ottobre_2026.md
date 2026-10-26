# Aggiornamento del 26 ottobre 2026: cosa è uscito e cosa cambia

Ricerca fatta il 26 ottobre 2026, diciotto giorni dopo la prima versione di queste note. Metodo: ricerca web; interrogazione dell'API di arXiv (`abs:Jev`, `abs:"decision model" AND abs:typed`, `abs:"decision models" AND abs:"System One"`, `abs:"typed decision"`, `abs:AggreFact`, `abs:MiniCheck`, `abs:FaithBench`), che dà ~200 paper sui TDM dal 1° settembre, filtrati per titolo e abstract su grounding, fact checking, allucinazioni, NLI, calibrazione e fine-tuning; lettura dei lavori più vicini; metadati e model card su Hugging Face; leaderboard di LLM-AggreFact e Decision Index; siti ufficiali delle conferenze.

## 1. I cambiamenti che toccano direttamente il paper

| # | Cosa è uscito | Cosa cambia per noi |
|---|---|---|
| 1 | **Jev su tutto il test di LLM-AggreFact**: 78,6 [77,6–79,6], una domanda *noul*, soglia 0,5, ECE 0,128; Qwen3.8-27B 76,9 e Gemma-4-E4B 75,6 con la stessa lettura ([2609.37647](https://arxiv.org/abs/2609.37647), 29/9) | "Primo studio dei TDM su AggreFact" non è più vero per Jev. Il numero diventa un controllo da riprodurre |
| 2 | **Jev contro un LLM di frontiera su 495 claim di AggreFact**: pari in BAcc, costo 187 volte minore, ma 23,2% di claim non supportati ammessi contro 14,1%; cascata non migliore di un'escalation casuale a pari budget ([2610.20634](https://arxiv.org/abs/2610.20634), 30/9) | Adottare la metrica "ammissione dei non supportati" e il controllo a escalation casuale per le cascate |
| 3 | **PremiseCheck**: Jev, Kev-27B, Microsoft Decision-1, Perplexity Decider come verificatori di premesse estratte da commenti di revisione; da 50–57 a quasi pari a GPT-5.5 grazie alla scomposizione ([2610.21046](https://arxiv.org/abs/2610.21046), 15/10) | La scomposizione in premesse funziona; la nostra scomposizione "in domande" sembra aiutare poco: è un contrasto da discutere (RQ4) |
| 4 | **La leaderboard di LLM-AggreFact ha i TDM in testa**: nimble-v4-max 79,5, Clef 79,0, Decider v1.2 78,7, Jev 1.13 78,7, d1-ultra 78,5, nimble-v4 77,8, Clef Flash 76,6, Drex v1.5 76,5; Bespoke-MiniCheck-7B (77,4) non è più il miglior aperto | Nasce la domanda RQ3: le voci non dichiarano domanda, soglia, chunking; i primi cinque stanno in 1 punto |
| 5 | **La Decision Index** (classifica dei TDM, 0.4 dal 20/10, >200 sistemi) include RAGTruth e HoVer fra i benchmark pubblici, e ANLI, ContractNLI, NLI4CT | Rischio di contaminazione su RAGTruth per i TDM recenti; HoVer non utilizzabile come fuori dominio |
| 6 | **Audit dei primi 28 paper**: la lettura tipizzata non ha mostrato un vantaggio di accuratezza indipendente rispetto alla lettura delle probabilità delle etichette ([2609.32160](https://arxiv.org/abs/2609.32160)) | Diventa la nostra domanda RQ2, sul grounding |
| 7 | **Letture senza training**: AnyJev (rotazione delle opzioni + divisione per il prior, [2610.00831](https://arxiv.org/abs/2610.00831)); LLM-as-Jev ("gli LLM sono già TDM", [2610.02076](https://arxiv.org/abs/2610.02076)); Gemma-4-31B da 72,9 a 79,6 con la sola temperatura ([2610.17205](https://arxiv.org/abs/2610.17205)) | La baseline B1 deve includere la base **calibrata** senza training, non solo quella ingenua |
| 8 | **Ricette di fine-tuning per TDM**: LoRA sulle lettere + ancoraggio KL (LLM-as-Jev); LoRA + testa di decisione, calibrazione in calo su 3 modelli su 4, nessun calo su BoolQ/MMLU/ARC (Inner Jev); coppie contrastive con certificato di rimozione dell'evidenza (PACT, [2609.35865](https://arxiv.org/abs/2609.35865)); autodistillazione del ragionamento (SII-Decision, [2610.03935](https://arxiv.org/abs/2610.03935); Tacit in AnyJev) | Il metodo della Parte B perde gran parte della novità. Nel primo paper resta B-ctrl (TDM contro base come punto di partenza), che nessuno ha fatto |
| 9 | **Fragilità**: la polarità della domanda inverte o annulla l'AUC di cinque TDM su otto su RAGTruth ([Arize](https://arize.com/blog/decision-model-benchmark/)); altre domande nella stessa richiesta cambiano le risposte ([2610.14631](https://arxiv.org/abs/2610.14631)); un contesto naturale ribalta il 61–73% delle decisioni corrette ([2609.30243](https://arxiv.org/abs/2609.30243)); limiti della singola passata sui fatti aggiornati più volte ([2610.24929](https://arxiv.org/abs/2610.24929)) | Tre famiglie nuove nel set diagnostico (polarità, composizione della richiesta, contesto naturale); la nostra regola a tre domande dentro una richiesta va testata anche domanda per domanda |
| 10 | **"Terzo valore soppresso"**: nelle risposte binarie Jev sembra nascondere un "non so" ([Sys1Cal, 2609.35342](https://arxiv.org/abs/2609.35342)) | Collega la debolezza della domanda sulla contraddizione nei nostri dati alla distinzione "contraddetto / non stabilito" (RQ4) |
| 11 | **Jev 1.14** (14/10): immagini, tipo `box`, contesto lungo come funzione "danger zone"; `jev-latest` passa a 1.14; 1.13 resta con 60 giorni di preavviso prima del ritiro ([blog](https://typesafe.ai/blog/jev-1-14)) | Valutare entrambe le versioni; registrare la versione esatta |
| 12 | **ACL 2027**: ARR il **4 gennaio 2027** (ufficiale); tema speciale "Homogenization and knowledge collapse in LLMs" ([call](https://2027.aclweb.org/calls/main/)) | Il calendario si stringe a dieci settimane |
| 13 | **LLM-AggreFact** è CC BY-ND 4.0, con accesso "come benchmark di valutazione" (metadati del dataset su Hugging Face) | Nessun derivato ridistribuibile: il set diagnostico non può partire da AggreFact |

## 2. Nuovi TDM rilevanti per noi

| Modello | Data | Note |
|---|---|---|
| Clef / Clef-flash (Cloudflare) | 1/10 | Qwen3.8-27B, Apache 2.0, testa congiunta; pesi già nella cache HF di questa macchina |
| Nimble-v4 / v4-max (Bespoke Labs) | v4-max su HF dal 18/10 | Qwen3.8-27B (v4-max), CC BY-NC; il runtime è nel dataset `bespokelabs/nimble-v4-max-decision-index` |
| pplx-decider v1.2 27B / flash 4B (Perplexity) | metà ottobre | pesi aperti secondo la stampa; licenza e base da verificare |
| Drex v1.5 (nace-ai) | 28/9 | base MiMo-V2.6-Distill-Qwen-9B, licenza RAIL |
| mammoth-1 (Baseten) | ottobre | primo su Decision Index 0.4 secondo Baseten (71,24), terzo secondo BenchLM (68,4): fonti in disaccordo |
| Kev-27B | ottobre | usato in PremiseCheck |
| OpenAI Decisions, Microsoft Decision-1, d1-ultra | — | chiusi |

Con quelli già nella repo, il gruppo "Qwen3.5-9B" arriva a sei TDM e il gruppo "Qwen3.8-27B" a due: è ciò che rende possibile RQ2.

## 2bis. Vincolo dei 9B: i modelli che restano (verificati su Hugging Face il 26 ottobre)

Decisione di progetto: lo studio usa solo modelli **fino a 9B parametri** (taglia nominale). Base e licenza lette dai metadati di Hugging Face; numero di parametri dai file safetensors dove disponibile.

| Modello | Base dichiarata | Licenza | Parametri |
|---|---|---|---|
| Cloudflare/clef-flash | Qwen/Qwen3.5-9B | Apache 2.0 | 9,41B |
| baselabs/elephant-1 | Qwen/Qwen3.5-9B | MIT | 9,41B |
| vllm-sr/d3-flash | Qwen/Qwen3.5-9B | Apache 2.0 | 8,39B |
| startlux-models/StartLux-Decision-v1.1-9B-Instant | non dichiarata | CC BY-NC 4.0 | 9,65B |
| jaredpalmer/kev-9b, kev-4b | Qwen3.5-9B-Base, Qwen3.5-4B-Base | Apache 2.0 | — |
| perplexity-ai/pplx-decider-flash-v1.2 | Qwen/Qwen3.5-4B | Apache 2.0 | 4,54B |
| PelaAI/KnowLine-4B-Gen5.5 | Qwen/Qwen3.5-4B | Apache 2.0 | 4,66B |
| LiquidAI/d1-3B | LiquidAI/LFM2.5-VL-3B | LFM Open | 3,12B |
| nace-ai/drex-v1.5 | MiMo-V2.6-Distill-Qwen-9B | RAIL | 8,95B |
| convaiinnovations/laya | — | Apache 2.0 | 0,42B |
| morriszjm/Tacit-9B / 8B / 4B / 2B / 1.7B | Qwen3.5-9B, Qwen3-8B, Qwen3-4B, Qwen3.5-2B, Qwen3-1.7B (dal README di AnyJev) | da verificare | — |
| amazon/strands-decider-2B-qwen3.5-v1-2610 | Qwen3.5-2B (dal nome) | da verificare | — |
| internlm/Intern-Decision-0.8B / 2B / 4B | da verificare | da verificare | — |
| jylin001206/SII-Decision-4B-Full | da verificare | da verificare | — |

Note:
- **Nimble-v4 9B non è su Hugging Face** (c'è solo Nimble-v4-max, 27B): è un'API a pagamento, quindi solo riferimento. Lo stesso per d1-ultra.
- La classe "9B" di Qwen3.5 ha ~9,4B parametri (torre visiva inclusa) e StartLux ~9,7B: il vincolo va inteso come taglia nominale, e va scritto così nel paper.
- Usciti dallo studio, e mantenuti solo come riferimento: Winnow-12B e Rune-26B-A4B (già misurati), d3-turbo e StartLux-27B (adapter pronti, non si eseguono), Clef 27B, Nimble-v4-max, pplx-decider 27B, mammoth-1, Kev-27B.

## 2ter. I dati della Decision Index (Space `multimodalart/jev-decision-index`)

Lo Space pubblica i dati grezzi della classifica in `data/index.json` (generato il 26/10/2026 alle 15:30 UTC; edizione 0.4; 229 sistemi; 110.201 richieste; Jev di riferimento 1.14; una RTX PRO 6000). Per ogni sistema: parametri serviti, base, tecnica (fine-tuning completo, LoRA, testa, "inference technique" senza training), punteggio Full 0.4, ECE e Brier, latenza, e il punteggio di ognuno dei 42 benchmark. Regola dichiarata: "native inference helpers, prompts and readout rules are preserved", la stessa di questa repo.

Cosa ne abbiamo ricavato (dettagli in [10_selezione_modelli.md](10_selezione_modelli.md)):
- **sotto i 9B l'addestramento da decision model vale ~24 punti** rispetto alle letture senza training della stessa taglia; a 31B solo ~2–4;
- **sui cinque benchmark di grounding/NLI** (RAGTruth, HoVer, ANLI, ContractNLI, NLI4CT) elephant-1 (9B) ha la stessa media dei migliori 27–31B (85,4) e 9 punti più di Jev 1.13, ma su AggreFact sta 2–3 punti sotto Jev: da qui RQ8;
- la classifica generale non ordina come AggreFact (JPT-9B 109° ma alla pari con elephant-1, 41°, sul nostro AggreFact);
- possibili inversioni di polarità sulla domanda negativa di RAGTruth (Clef-flash 35,6; d3 nano/lite/edge ~30);
- famiglie a più taglie con lo stesso runtime (StartLux, d3, OneJev, JPT, Kev), utili per la curva di taglia;
- Amber e Crivo dichiarano di essere addestrati sugli split di training del kit della Decision Index (RAGTruth compreso).

## 3. Altri lavori da citare

- Giudici a rubrica: i TDM e gli LLM sbagliano negli stessi punti ([2609.29769](https://arxiv.org/abs/2609.29769)).
- JEV-as-a-Judge ([2609.26550](https://arxiv.org/abs/2609.26550), ora v5): 240 casi di HaluEval, escalation per confidenza.
- JEVal ([2610.03935](https://arxiv.org/abs/2610.03935)): i TDM sovrastimano la probabilità dell'opzione scelta.
- Radiologia ([2609.27607](https://arxiv.org/abs/2609.27607)) e citazioni finanziarie ([2610.08675](https://arxiv.org/abs/2610.08675)): Jev come verificatore di supporto in un dominio, con coppie minime.
- Opzioni di fallback poco affidabili ([2610.19856](https://arxiv.org/abs/2610.19856)); trasferimento delle politiche di rifiuto ([2610.03387](https://arxiv.org/abs/2610.03387)).
- Valutazione per agenti con autoverifica della pipeline ([2610.02267](https://arxiv.org/abs/2610.02267)): un buon modello di come dichiarare i propri errori di analisi.
- Grounding senza TDM: aggregazione di giudici economici ([RAIM, 2609.39229](https://arxiv.org/abs/2609.39229)); le metriche di attribuzione non trasferiscono fra dataset ([2606.23915](https://arxiv.org/abs/2606.23915)); le etichette dei benchmark di allucinazione dipendono dal criterio ([2610.08026](https://arxiv.org/abs/2610.08026)); self-play per rilevatori ([2607.07993](https://arxiv.org/abs/2607.07993)).
- Guardrail: i TDM non battono in modo affidabile classificatori e LLM-as-judge su prompt injection e sicurezza ([Red Hat](https://developers.redhat.com/articles/2026/10/02/benchmarking-ai-decision-models-against-traditional-guardrails)).

## 4. Cosa non ho trovato (e va ricontrollato ogni settimana)

- Nessun lavoro con **molti TDM aperti** sotto un unico protocollo su un benchmark di grounding.
- Nessun confronto **TDM contro la propria base** sul grounding.
- Nessuno studio della **sensibilità al protocollo** delle voci della leaderboard di AggreFact.
- Nessun **fine-tuning di un TDM come checker di grounding** con il controllo sulla base.
- Nessun dettaglio pubblico sul protocollo delle voci TDM della leaderboard (le model card di Nimble-v4-max e Clef non ne parlano; il documento dei benchmark pubblici di Nimble elenca AggreFact fra i dataset scartati per licenza e accesso).

## 5. Incertezze di questa ricerca

- Alcune pagine non erano leggibili (alcuni siti bloccati, PDF compressi): per Nimble-v4 9B, pplx-decider e mammoth-1 i dettagli (base, licenza) vanno verificati sulle model card.
- I punteggi della Decision Index cambiano fra fonti e date (es. mammoth-1: 71,24 secondo Baseten, 68,4 secondo BenchLM al 25/10).
- Le date ARR per EMNLP 2027 non sono ancora annunciate.
