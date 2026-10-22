# Parte A — Valutazione dei typed decision model come fact checker

## 1. Domande di ricerca

| RQ | Domanda | Esperimento principale |
|---|---|---|
| **RQ1** | Quanto valgono i TDM (aperti e chiusi) come fact checker documento–claim, zero-shot, rispetto a checker specializzati e a LLM generici? | tabella principale su LLM-AggreFact + test fuori dominio |
| **RQ2** | Il vantaggio viene dall'addestramento "da decision model" o dal semplice leggere i logit di un LLM? | stesso prompt e stessa lettura su Gemma-4-12B e Qwen3.5-9B base (le basi di Winnow e JPT) |
| **RQ3** | Le probabilità sono calibrate? Una soglia scelta in un dominio trasferisce agli altri? | ECE/Brier; soglia dal nostro dev contro soglia dal dev di AggreFact |
| **RQ4** | Scomporre la decisione in più domande tipizzate aiuta? | una domanda contro la regola a tre domande `min(p1, 1−p2, 1−p3)`; contributo di ciascuna domanda |
| **RQ5** | I TDM guardano davvero il documento? Sono robusti a formulazione, nomi, distrattori? | set diagnostico ([05](05_set_diagnostico.md)) |
| **RQ6** | Qual è il compromesso qualità / costo / latenza? | token/s, costo per 1.000 claim, latenza con e senza cache del prefisso |

## 2. Modelli

| Gruppo | Modelli | Stato nella repo |
|---|---|---|
| TDM chiuso | Jev 1.13 (API) | valutato su tutto il test di AggreFact |
| TDM aperti | Winnow-12B (BF16, NVFP4; Q8_0 da aggiungere), JPT-9B, Rune-26B-A4B | valutati su tutto il test |
| TDM aperti da aggiungere | Laya (421M), Kev-9B / Kev-4B, djev-0.1 | da fare (danno la dimensione: da 0,4B a 26B) |
| **LLM generici, stessa lettura (B1)** | Gemma-4-12B, Qwen3.5-9B (basi di Winnow e JPT); eventualmente Qwen3-8B, Gemma-4-31B | da fare |
| LLM generici, generazione (B2) | un LLM piccolo che risponde "Yes/No" generando; un LLM con breve ragionamento | da fare |
| **Checker specializzati (B3)** | FactCG-DeBERTa-L (fatto: 75,64), MiniCheck-Flan-T5-L, Bespoke-MiniCheck-7B (pesi già nella cache HF), AlignScore | in parte |

Il confronto più informativo per RQ2 è **Winnow contro Gemma-4-12B** e **JPT contro Qwen3.5-9B**: stessa architettura, stesso prompt, stessa lettura dei logit. La differenza misura cosa ha aggiunto l'addestramento a decision model.

## 3. Dati

| Blocco | Dataset | Uso |
|---|---|---|
| Sviluppo | `dev_select` / `dev_confirm` costruiti dalle nostre fonti ([../fine_tuning/00b_split_e_protocollo.md](../fine_tuning/00b_split_e_protocollo.md)) | soglia, temperatura, scelta delle domande |
| Test principale | **LLM-AggreFact test** (11 fonti, 29.320 righe), zero-shot | confronto con la leaderboard |
| Test fuori dominio | HaluBench (senza la parte RAGTruth, riportata a parte), HalluMix, SciFact, Climate-FEVER, HealthVer / COVID-Fact | generalizzazione |
| Test in distribuzione | test di FEVER, ANLI, VitaminC, HoVer, MiniCheck, SummEdits, AVeriTeC | riferimento per la Parte B |
| Diagnostico | [05_set_diagnostico.md](05_set_diagnostico.md) | RQ5 |

Per la Parte A da sola, il test in distribuzione è utile ma non indispensabile; i blocchi chiave sono AggreFact e fuori dominio.

## 4. Protocollo

1. **Prompt**: per ogni TDM il suo formato nativo (già riprodotto per Winnow, JPT, Rune nei benchmark della repo); per gli LLM generici lo stesso formato usato dal TDM corrispondente.
2. **Domande**: il pacchetto `packs/claim_support.json` (tre domande). Va dichiarato che è stato scelto, per Jev, sul dev di AggreFact. Per un'analisi pulita aggiungere anche una domanda "neutra" scritta prima di qualsiasi esperimento (es. la sola `support_simple`).
3. **Soglia e temperatura**: fittate su `dev_select`, confermate su `dev_confirm`, applicate a tutti i test. Come confronto secondario, la soglia fittata sul dev di AggreFact.
4. **Test**: una sola esecuzione per configurazione congelata.
5. **Jev**: riportare versione e date delle chiamate; dichiarare il controllo intermedio sul test fatto nel run originale (`results/INTERIM_AMENDMENT.md`). Rieseguire Jev con il protocollo zero-shot costa pochi dollari.
6. **Documenti lunghi**: stessa regola di troncamento per tutti i modelli (80.000 caratteri, 75% inizio e 25% fine); riportare quanti documenti sono stati troncati per modello e per fonte.

## 5. Metriche

- BAcc per fonte e media (LLM-AggreFact: media sulle 11 fonti, come la leaderboard); AUC.
- Brier, ECE (10 intervalli), diagrammi di affidabilità.
- Copertura selettiva: accuratezza contro percentuale di esempi coperti, al variare della confidenza.
- Stabilità: deviazione standard fra domande parafrasate.
- Costi: token di input, latenza per claim, throughput (con e senza cache del prefisso: le tre domande condividono il documento).
- Intervalli al 95% con bootstrap per documento; test appaiati fra modelli (bootstrap appaiato o test di permutazione).

## 6. Esperimenti e analisi

| ID | Esperimento | Cosa mostra |
|---|---|---|
| A1 | Tabella principale: tutti i modelli, AggreFact (11 fonti) + media fuori dominio | RQ1 |
| A2 | TDM contro la loro base generica, stessa lettura | RQ2: vale l'addestramento da decision model? |
| A3 | Soglia dal nostro dev contro soglia dal dev di AggreFact; ECE per dominio | RQ3: trasferimento della calibrazione |
| A4 | Una domanda contro tre; ablazione di ciascuna domanda nella regola min; regola min contro media | RQ4 |
| A5 | Set diagnostico | RQ5 |
| A6 | Curva qualità–costo (BAcc contro costo per 1.000 claim) | RQ6 |
| A7 | Quantizzazione: Winnow BF16 / Q8_0 / NVFP4 | effetto della compressione su accuratezza e calibrazione |
| A8 | Analisi degli errori: 100 errori per modello, categorizzati (entità, numeri, attribuzione, multi-hop, etichetta dubbia) | dove sbagliano i TDM |
| A9 | Accordo fra modelli: quanto gli errori dei TDM si sovrappongono a quelli dei checker specializzati | utile per ensemble e cascate |

Spunto già visibile dai dati della repo per A2 e A8: Winnow è forte sui riassunti di notizie (CNN 72,1 alla soglia 0,30) e più debole su WiCE (77,4), JPT il contrario (CNN 62,3, WiCE 83,8). Profili di errore diversi a parità di media: argomento interessante per l'analisi.

## 7. Tabelle e figure previste

| N. | Contenuto |
|---|---|
| Tab. 1 | Modelli: tipo, parametri, licenza, formato, costo |
| Tab. 2 | LLM-AggreFact zero-shot per fonte, tutti i modelli (formato leaderboard) |
| Tab. 3 | Fuori dominio + media dei blocchi |
| Tab. 4 | TDM contro la base generica (RQ2) |
| Tab. 5 | Ablazione delle domande (RQ4) |
| Tab. 6 | Set diagnostico (RQ5) |
| Fig. 1 | Diagrammi di affidabilità (RQ3) |
| Fig. 2 | Qualità contro costo (RQ6) |
| Fig. 3 | Copertura selettiva |
| Fig. 4 | Delta soglia trasferita contro soglia in dominio, per fonte |

## 8. Lavoro e costi stimati

| Voce | Stima |
|---|---|
| Split, dev e test fuori dominio | 1 settimana (codice) |
| Inferenza dei modelli aperti su dev + test fuori dominio + diagnostico | ~1–2 giorni su L40S |
| Nuovi modelli (Laya, Kev, basi generiche, Bespoke, MiniCheck) su AggreFact | ~2–3 giorni su L40S |
| Jev con il protocollo zero-shot | pochi dollari |
| Analisi degli errori | 2–3 giorni |
| Scrittura | 2 settimane |
| **Totale Parte A** | **~4–6 settimane** |
