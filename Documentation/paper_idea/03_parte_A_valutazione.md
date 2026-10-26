# Parte A — Il paper di analisi: cosa aggiungono i decision model al fact checking su documento

Rivista il 26 ottobre 2026. Rispetto alla prima versione: Jev su AggreFact è già pubblicato ([2609.37647](https://arxiv.org/abs/2609.37647)), quindi il paper non si regge più su "quanto vale un TDM", ma su **perché** e **quanto è solido il numero**.

## 1. Domande di ricerca

| RQ | Domanda | Esperimento principale |
|---|---|---|
| **RQ1** | Dove stanno i TDM aperti e chiusi, **con un unico protocollo**, rispetto a checker specializzati e LLM generici? Su AggreFact e fuori dominio? | tabella principale (A1) |
| **RQ2** | **Base contro ricetta**: quanto aggiunge l'addestramento da decision model rispetto alla stessa base letta in modo ingenuo e in modo calibrato senza training? Quanto variano le ricette sulla stessa base? | gruppi per base (A2) |
| **RQ3** | **Protocollo contro modello**: quanto spostano il risultato soglia, polarità e parafrasi della domanda, una o tre domande, composizione della richiesta, troncamento, esclusione delle fonti a rischio di contaminazione? | grid search e varianti (A3) |
| **RQ4** | **Che errori**: i TDM ammettono claim non supportati più dei checker? Distinguono "contraddetto" da "non stabilito"? Dove cadono (lunghezza, multi-hop, numeri, entità)? | A4, A8 |
| **RQ5** | **Calibrazione**: le probabilità sono calibrate sul grounding? La soglia scelta in un dominio trasferisce agli altri? | A5 |
| **RQ6** | **Robustezza**: distrattori, posizione dell'evidenza, opzioni invertite o rinominate, richiesta con più domande | set diagnostico ridotto (A6) |
| **RQ7** | **Costo e composizione**: qualità contro costo, latenza, energia; gli errori si sovrappongono? Le cascate battono il controllo a escalation casuale? | A7, A9 |

RQ2 e RQ3 sono il cuore del paper: sono le domande nuove, e qualunque risultato esca è informativo.

## 2. Indizi già visibili nei nostri dati

Dalla grid search sul test (`results/.../grid_551bb722.md`). Sono **esplorativi** (scelte fatte sul test): servono a capire dove guardare, non vanno nel paper come risultati.

| Osservazione | Numeri | Perché conta |
|---|---|---|
| Il protocollo sposta più della distanza fra modelli | migliore combinazione contro default: d1-3B **+3,6** (71,84 → 75,47, una domanda sola), Rune +1,7, JPT +1,2, Winnow +0,8; fra i primi cinque della leaderboard ci sono 1,0 punti | RQ3: la leaderboard può essere più "protocollo" che "modello" |
| BAcc e AUC ordinano i modelli in modo diverso | elephant-1 ha la Macro AUC più alta (87,15) ma è secondo per BAcc (76,70 contro 76,95 di Winnow, Macro AUC 84,99) | RQ3/RQ5: la soglia fissa penalizza i modelli meno calibrati sul compito |
| La terza domanda (contraddizione) porta poco segnale | AUC invertita di `contradicted_detail` fra 0,62 e 0,83, contro 0,89–0,91 per le altre due | RQ4: i TDM riconoscono il "non supportato" ma non distinguono bene il "contraddetto" (cfr. il "terzo valore soppresso" di [Sys1Cal](https://arxiv.org/abs/2609.35342)) |
| Tre domande aiutano poco o niente | per Rune e d1-3B la migliore regola è una domanda sola; Jev con una domanda a 0,5 fa 78,6 ([2609.37647](https://arxiv.org/abs/2609.37647)), il nostro run a tre domande dev-tuned 79,09 | RQ4: la scomposizione "in domande" non è la scomposizione "in premesse" di PremiseCheck |
| Profili diversi a parità di media | Winnow forte su CNN (72,2), KnowLine debole (56,1) ma migliore su LFQA (85,2); elephant-1 primo su XSum e FactCheck | RQ4/RQ7: errori non sovrapposti, spazio per ensemble |
| Documenti lunghi | BAcc fra 67 e 71 oltre 16.000 caratteri per tutti i TDM, contro 82–87 fra 1.000 e 4.000 | RQ4: il limite è comune, non di un modello |

## 3. Modelli

### 3.1 I gruppi per base (il disegno di RQ2)

| Base | TDM (stato nella repo) | Base generica da leggere |
|---|---|---|
| **Qwen3.5-9B** | JPT-9B (fatto), elephant-1 (fatto), Nimble-9B-v3 (adapter pronto), d3-flash (adapter pronto), StartLux-9B-Instant (adapter pronto; base da verificare), Kev-9B (da aggiungere) | Qwen3.5-9B (nella cache HF) |
| **Qwen3.8-27B** | Clef (nella cache, adapter da scrivere), Nimble-v4-max (da aggiungere) | Qwen3.8-27B (da scaricare; ~55 GB in BF16: serve una GPU da 80 GB o FP8) |
| Gemma-4-12B | Winnow-12B (fatto: BF16 e NVFP4) | Gemma-4-12B |
| Gemma-4-26B-A4B | Rune-26B (fatto) | Gemma-4-26B-A4B |
| Qwen3.5-4B | KnowLine-4B (fatto) | Qwen3.5-4B |
| LFM2.5-VL-3B | d1-3B (fatto) | LFM2.5-VL-3B |
| Qwen3-8B | CLM-8B (fatto, encoder congelato + teste) | Qwen3-8B (nella cache) |

Il gruppo Qwen3.5-9B è il più prezioso: **sei ricette diverse** (LoRA semplice, attenzione bidirezionale a blocchi, testa con 255 codici, attenzione bidirezionale sul prompt, testa a puntatore) **sulla stessa base**. Misura direttamente quanto conta la ricetta.

Per ogni base, tre letture:
1. **ingenua**: il prompt del TDM corrispondente (o un prompt neutro comune), softmax sulle lettere delle opzioni, attenzione causale;
2. **calibrata senza training**: rotazione delle opzioni e divisione per il prior delle etichette (AnyJev L0, [2610.00831](https://arxiv.org/abs/2610.00831)), più temperatura fittata sul nostro dev;
3. il **TDM** con il suo runtime nativo (già così nella repo).

La differenza 3 − 2 è "cosa aggiunge l'addestramento da decision model oltre a ciò che si ottiene gratis"; la differenza 2 − 1 è "quanto vale la sola calibrazione della lettura". È esattamente la domanda dell'audit ([2609.32160](https://arxiv.org/abs/2609.32160)).

### 3.2 Gli altri gruppi

| Gruppo | Modelli | Stato |
|---|---|---|
| TDM chiusi | Jev 1.13 (fatto, da rifare con protocollo zero-shot), **Jev 1.14**; facoltativi: OpenAI Decisions (beta pubblica), d1-ultra, pplx-decider v1.2 via API | da fare (pochi dollari) |
| Altri TDM aperti | Rune-26B (fatto), d3-turbo (GPU 56 GiB+), StartLux-27B-AutoThink (pensa quando è incerto: interessante per RQ4), Drex v1.5, pplx-decider v1.2 (se i pesi sono scaricabili), Laya (encoder 0,4B) | in parte |
| Checker specializzati (B3) | FactCG-DeBERTa-L (fatto, 75,64), Bespoke-MiniCheck-7B (adapter fatto), MiniCheck-Flan-T5-L, AlignScore, Granite Guardian 3.3 | in parte |
| LLM generativi (B2) | un LLM piccolo che genera "Yes/No", con e senza ragionamento breve; un LLM chiuso economico su un sottoinsieme | da fare |

Per [08](08_benchmark_e_modelli.md) la lista completa e la selezione minima.

## 4. Dati

| Blocco | Dataset | Uso |
|---|---|---|
| Sviluppo | `dev_select` / `dev_confirm` dalle nostre fonti ([../fine_tuning_factcheck/00b_split_e_protocollo.md](../fine_tuning_factcheck/00b_split_e_protocollo.md)) | soglia, temperatura, scelta delle domande |
| Test principale | **LLM-AggreFact test** (11 fonti, 29.320 righe), zero-shot; media riportata **anche senza RAGTruth** | confronto con la leaderboard e con [2609.37647](https://arxiv.org/abs/2609.37647) |
| Fuori dominio | HaluBench (senza la parte RAGTruth), HalluMix, SciFact, Climate-FEVER, HealthVer, COVID-Fact, LLM-Oasis; FaithBench se ottenibile | generalizzazione e trasferimento della soglia; quasi tutti già nella cache HF |
| Diagnostico | [05_set_diagnostico.md](05_set_diagnostico.md), versione ridotta | RQ6 |

**Contaminazione** (T4): la Decision Index usa RAGTruth e HoVer, e i produttori ottimizzano per la Decision Index. Quindi RAGTruth va trattato come "probabilmente visto" per tutti i TDM recenti; HoVer non va usato come test fuori dominio per i TDM. Controllare anche la colonna `contamination_identifier` di AggreFact.

## 5. Protocollo

1. **Prompt**: per ogni TDM il suo formato e runtime nativi (già così nella repo: llm2jev, qev, Nimble engine, startlux-decide, ...). Per le basi, il formato del TDM corrispondente e un prompt neutro comune.
2. **Domande**: il pacchetto `config/questions/claim_support.json` (tre domande, scelte in origine per Jev sul dev di AggreFact: dichiararlo). Più la domanda singola `support_simple` come configurazione "neutra", e le sue varianti di polarità e parafrasi per RQ3.
3. **Soglia e temperatura**: fittate su `dev_select`, confermate su `dev_confirm`, applicate a tutti i test. Riportare **anche** soglia 0,5 (come [2609.37647](https://arxiv.org/abs/2609.37647)) e soglia fittata sul dev di AggreFact (come probabilmente fanno le voci della leaderboard).
4. **Test**: una sola esecuzione per configurazione congelata. La grid search sul test resta esplorativa e va dichiarata come tale.
5. **Jev**: versione esatta (1.13.0 e 1.14.0) e date delle chiamate; il controllo intermedio del run originale ([../../legacy/results/INTERIM_AMENDMENT.md](../../legacy/results/INTERIM_AMENDMENT.md)) dichiarato.
6. **Documenti lunghi**: stessa regola per tutti (80.000 caratteri, 75% inizio e 25% fine); i runtime che rifiutano invece di troncare (elephant-1, Nimble, d3) registrano i fallimenti: riportarli per modello e per fonte.
7. **Data di congelamento** della lista dei modelli (proposta: 30 novembre 2026).

## 6. Metriche

- BAcc per fonte e media (formato della leaderboard); AUC per fonte e Macro AUC.
- **Tasso di ammissione dei non supportati** (1 − recall dei non supportati) e recall dei supportati, al punto operativo: la metrica di deployment che la BAcc nasconde ([2610.20634](https://arxiv.org/abs/2610.20634)). Già nel `report.json` (`unsupported_recall`).
- Brier, ECE (10 intervalli), diagrammi di affidabilità; ECE dopo temperature scaling.
- Copertura selettiva: accuratezza contro percentuale coperta.
- Sensibilità al protocollo: intervallo dei risultati sulle varianti di protocollo, per modello; correlazione di rango fra le classifiche ottenute con protocolli diversi (tau di Kendall).
- Costo: token, latenza p50/p95 a cache fredda, throughput, energia per 1.000 righe (già misurati dal benchmark).
- Intervalli al 95% con bootstrap per documento; test appaiati (bootstrap appaiato o McNemar).

## 7. Esperimenti e analisi

| ID | Esperimento | Cosa mostra |
|---|---|---|
| A1 | Tabella principale: tutti i modelli, AggreFact (11 fonti, con e senza RAGTruth) + media fuori dominio | RQ1 |
| A2 | TDM contro la loro base, lettura ingenua e calibrata, per gruppo di base; dispersione fra ricette sulla stessa base | **RQ2** |
| A3 | Sensibilità al protocollo: soglia (0,5 / nostro dev / dev di AggreFact), polarità e parafrasi della domanda, una contro tre domande, domanda da sola contro dentro una richiesta con tre domande, troncamento; classifica sotto ogni protocollo | **RQ3** |
| A4 | Errori asimmetrici; una domanda a tre opzioni (supportato / contraddetto / non stabilito) contro le domande binarie, sulle fonti dove le due classi negative sono distinguibili | RQ4 |
| A5 | ECE per dominio; soglia dal nostro dev contro soglia in dominio, per fonte | RQ5 |
| A6 | Set diagnostico ridotto | RQ6 |
| A7 | Qualità contro costo (BAcc contro costo e contro energia per 1.000 claim) | RQ7 |
| A8 | Analisi degli errori: 100 errori per modello rappresentativo, categorizzati (entità, numeri, attribuzione, multi-hop, documento lungo, etichetta dubbia) | RQ4 |
| A9 | Accordo fra modelli: sovrapposizione degli errori fra TDM sulla stessa base, fra TDM e checker; ensemble semplice; cascata TDM → LLM contro escalation casuale a pari budget | RQ7 |
| A10 | Quantizzazione: Winnow BF16 / NVFP4 (fatto: 76,95 contro 77,49) | robustezza del deployment, in appendice |

## 8. Tabelle e figure previste

| N. | Contenuto |
|---|---|
| Tab. 1 | Modelli: base, ricetta di lettura, parametri, licenza, runtime, costo |
| Tab. 2 | LLM-AggreFact per fonte, tutti i modelli, protocollo unico |
| Tab. 3 | Fuori dominio + medie |
| Tab. 4 | TDM contro base (ingenua, calibrata), per gruppo (RQ2) |
| Tab. 5 | Sensibilità al protocollo, per modello (RQ3) |
| Tab. 6 | Set diagnostico ridotto (RQ6) |
| Fig. 1 | RQ2: per ogni base, la base ingenua, la base calibrata e i TDM come punti sullo stesso asse |
| Fig. 2 | RQ3: classifica sotto protocolli diversi (grafico a "bump"), con la banda della leaderboard |
| Fig. 3 | Diagrammi di affidabilità in dominio e fuori dominio |
| Fig. 4 | Qualità contro costo |
| Fig. 5 | Ammissione dei non supportati contro BAcc |

## 9. Lavoro e costi stimati

| Voce | Stima |
|---|---|
| Split, dev e test fuori dominio | 1 settimana (codice) |
| Run completi dei TDM con adapter pronto (d3-flash, StartLux, Nimble-v3, MiniCheck-7B) | 2–3 giorni GPU |
| Adapter nuovi (Clef, Nimble-v4, Kev-9B, basi generiche con le due letture) | 1,5 settimane |
| Inferenza su dev, fuori dominio e diagnostico | 3–5 giorni GPU |
| Jev 1.13 e 1.14 con il protocollo zero-shot | pochi dollari |
| Varianti di protocollo (RQ3) | quasi tutte senza inferenza (grid search); polarità e composizione della richiesta richiedono un'inferenza su un sottoinsieme |
| Analisi degli errori | 3 giorni |
| Scrittura | 2 settimane |
| **Totale** | **~7 settimane**, in parallelo con B-ctrl |
