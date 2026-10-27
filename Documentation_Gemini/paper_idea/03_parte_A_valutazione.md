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
| **RQ8** | **La classifica dei decision model predice il grounding?** La Decision Index (generale e il suo sottoinsieme di grounding) ordina i modelli come AggreFact e i test fuori dominio? | E9 di [11](11_design_esperimenti.md), senza inferenza in più |

RQ2 e RQ3 sono il cuore del paper: sono le domande nuove, e qualunque risultato esca è informativo. RQ8 è quasi gratis e utile a chiunque scelga un modello dalla classifica. Il design esperimento per esperimento (domande, dati, regole, celle, statistica) è in [11_design_esperimenti.md](11_design_esperimenti.md).

## 2. Indizi già visibili nei nostri dati

Dalla grid search sul test (`results/.../grid_551bb722.md`). Sono **esplorativi** (scelte fatte sul test): servono a capire dove guardare, non vanno nel paper come risultati.

| Osservazione | Numeri | Perché conta |
|---|---|---|
| Il protocollo sposta più della distanza fra modelli | migliore combinazione contro default: d1-3B **+3,6** (71,84 → 75,47, una domanda sola), JPT +1,2 (fuori vincolo anche Rune +1,7, Winnow +0,8); fra i primi cinque modelli fino a 9B della leaderboard ci sono 1,3 punti | RQ3: la leaderboard può essere più "protocollo" che "modello" |
| BAcc e AUC ordinano i modelli in modo diverso | KnowLine-4B ha Macro AUC più alta di JPT-9B (86,55 contro 84,48) ma BAcc più bassa (75,65 contro 76,04); elephant-1 ha la Macro AUC più alta di tutti (87,15) | RQ3/RQ5: la soglia fissa penalizza i modelli meno calibrati sul compito |
| La terza domanda (contraddizione) porta poco segnale | AUC invertita di `contradicted_detail` fra 0,72 e 0,83 per elephant-1 e KnowLine, contro 0,89–0,91 per le altre due | RQ4: i TDM riconoscono il "non supportato" ma non distinguono bene il "contraddetto" (cfr. il "terzo valore soppresso" di [Sys1Cal](https://arxiv.org/abs/2609.35342)) |
| Tre domande aiutano poco o niente | per d1-3B la migliore regola è una domanda sola; Jev (riferimento) con una domanda a 0,5 fa 78,6 ([2609.37647](https://arxiv.org/abs/2609.37647)), il nostro run a tre domande dev-tuned 79,09 | RQ4: la scomposizione "in domande" non è la scomposizione "in premesse" di PremiseCheck |
| Profili diversi a parità di media | KnowLine-4B debole su CNN (56,1) ma il migliore su LFQA (85,2); elephant-1 primo su XSum (77,6) e FactCheck (81,7) | RQ4/RQ7: errori non sovrapposti, spazio per ensemble |
| Documenti lunghi | BAcc fra 67 e 70 oltre 16.000 caratteri per elephant-1 e KnowLine, contro 82–85 fra 1.000 e 4.000 | RQ4: il limite è comune, e pesa di più sui modelli piccoli con contesto lungo |

## 3. Modelli

**Vincolo**: solo modelli fino a 9B parametri (taglia nominale: la classe "9B" di Qwen3.5 ha ~9,4B parametri con la torre visiva). Tutto gira su una GPU da 24–48 GB. I modelli più grandi compaiono solo come riferimento (§3.3).

### 3.1 I gruppi per base (il disegno di RQ2)

La lista definitiva, con le motivazioni e i numeri della Decision Index, è in [10_selezione_modelli.md](10_selezione_modelli.md). In sintesi:

| Base | TDM (livello A) | Base generica da leggere |
|---|---|---|
| **Qwen3.5-9B** | elephant-1 (fatto), StartLux-9B, d3-flash, Nimble-9B-v3 (pronti), OneJev 1.6-9B, Clef-flash, Tacit-9B (nuovi) | Qwen3.5-9B (nella cache) |
| Qwen3.5-9B-Base | JPT-9B (fatto), Kev-9B (nuovo) | Qwen3.5-9B-Base |
| **Qwen3.5-4B** | KnowLine-4B (fatto), StartLux-4B, pplx-decider-flash, d3-mini, OneJev 1.6-4B (nuovi) | Qwen3.5-4B |
| LFM2.5-VL-3B / LFM2.5-2.6B-Base | d1-3B (fatto), LFM2.5-2.6B-RLCD | LFM2.5-VL-3B / LFM2.5-2.6B-Base |
| **Gemma-4-E4B-it** (non Qwen) | Winnow-E4B, Strands Decider E4B, Gevva E4B (encoder) | Gemma-4-E4B-it |
| **Ministral-3-8B** (non Qwen) | Standard One 8B (Instruct), Jevstral 8B (Base) | Ministral-3-8B-Instruct e -Base |
| MiMo-V2.6-Distill-Qwen-9B | Drex v1.5 (nuovo) | MiMo-V2.6-Distill-Qwen-9B |
| DeBERTa-v3-large | GLiClass v4 edge (nuovo) | — (controparte: FactCG, checker sullo stesso backbone) |

Più le curve di taglia (livello B: StartLux, d3, OneJev, JPT, Kev fino a 0,8B), i controlli (CLM-8B, Laya) e due altri encoder (Amber su ModernBERT, Crivo su Ettin). Il blocco non Qwen serve a rispondere all'obiezione "è un confronto su Qwen": il 68% dei TDM fino a 9B della Decision Index, e tutti i primi 25, sono su Qwen; per questo il disegno principale è su Qwen e la generalità si verifica su Gemma-4-E4B, Ministral-3-8B e LFM2.5 ([10](10_selezione_modelli.md) §1.5; E2b di [11](11_design_esperimenti.md)).

Il gruppo Qwen3.5-9B è il più prezioso: **sette ricette diverse sulla stessa base** (fine-tuning completo con attenzione bidirezionale a blocchi, prompt bidirezionale, attenzione non causale con 255 codici, runtime OneJev, LoRA con 255 codici, testa congiunta, autodistillazione senza etichette). Misura direttamente quanto conta la ricetta. Il gruppo Qwen3.5-4B lo replica a metà taglia, e insieme al resto dà una **curva di taglia** da 0,4B a 9B.

Per ogni base, tre letture:
1. **ingenua**: il prompt del TDM corrispondente (o un prompt neutro comune), softmax sulle lettere delle opzioni, attenzione causale;
2. **calibrata senza training**: rotazione delle opzioni e divisione per il prior delle etichette (AnyJev L0, [2610.00831](https://arxiv.org/abs/2610.00831)), più temperatura fittata sul nostro dev;
3. il **TDM** con il suo runtime nativo (già così nella repo).

La differenza 3 − 2 è "cosa aggiunge l'addestramento da decision model oltre a ciò che si ottiene gratis"; la differenza 2 − 1 è "quanto vale la sola calibrazione della lettura". È esattamente la domanda dell'audit ([2609.32160](https://arxiv.org/abs/2609.32160)).

### 3.2 Gli altri gruppi (tutti fino a 9B)

| Gruppo | Modelli | Stato |
|---|---|---|
| Altri TDM | Drex v1.5 (9B, base MiMo distillata), Laya (encoder ~0,4B), Intern-Decision 0,8B / 2B | da aggiungere (curva di taglia) |
| Checker specializzati (B3) | FactCG-DeBERTa-L (fatto, 75,64), Bespoke-MiniCheck-7B (adapter fatto), MiniCheck-Flan-T5-L, AlignScore, Granite Guardian 3.3 (8B), HHEM-2.1 | in parte |
| LLM generativi (B2) | Qwen3.5-9B e Qwen3.5-4B in modalità generativa, con e senza ragionamento breve | da fare |

### 3.3 Riferimenti fuori vincolo o non eseguibili in locale (nessuna analisi su di loro)

| Riferimento | Perché | Costo |
|---|---|---|
| Jev 1.13 e 1.14 (chiuso) | il TDM di riferimento del filone; continuità con [2609.37647](https://arxiv.org/abs/2609.37647) | pochi dollari |
| Winnow-12B, Rune-26B-A4B | già misurati nella repo | zero (cache esistente) |
| Clef 27B, Nimble-v4-max, pplx-decider 27B, Nimble-v4 9B (solo API) | numeri della leaderboard, citati | zero |
| un LLM chiuso economico, su un sottoinsieme | riferimento generativo e cascata | decine di dollari |

Servono a rispondere a una domanda sola: **quanto distano i migliori modelli fino a 9B dai TDM grandi e chiusi** (Fig. 4).

Per [08](08_benchmark_e_modelli.md) la lista completa e la selezione minima.

## 4. Dati

| Blocco | Dataset | Uso |
|---|---|---|
| Sviluppo | `dev_select` / `dev_confirm` dalle nostre fonti ([../fine_tuning_factcheck/00b_split_e_protocollo.md](../fine_tuning_factcheck/00b_split_e_protocollo.md)) | soglia, temperatura, scelta delle domande |
| Test principale | **LLM-AggreFact test** (11 fonti, 29.320 righe), zero-shot; media riportata **anche senza RAGTruth** | confronto con la leaderboard e con [2609.37647](https://arxiv.org/abs/2609.37647) |
| Fuori dominio | HaluBench (senza la parte RAGTruth), HalluMix, SciFact, Climate-FEVER, HealthVer, COVID-Fact; FaithBench se ottenibile | generalizzazione e trasferimento della soglia; nella cache solo SciFact e HalluMix, gli altri da scaricare ([11](11_design_esperimenti.md) §3.3) |
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
| A10 | Quantizzazione di un 9B (BF16 contro FP8 o 4 bit), per il deployment su GPU piccole; il caso di Winnow (BF16 76,95, NVFP4 77,49) come riferimento | robustezza del deployment, in appendice |
| A11 | Decision Index contro le nostre misure (correlazione di rango, casi fuori diagonale) | RQ8 |

## 8. Tabelle e figure previste

| N. | Contenuto |
|---|---|
| Tab. 1 | Modelli: base, ricetta di lettura, parametri, licenza, runtime, memoria GPU, costo |
| Tab. 2 | LLM-AggreFact per fonte, tutti i modelli, protocollo unico |
| Tab. 3 | Fuori dominio + medie |
| Tab. 4 | TDM contro base (ingenua, calibrata), per gruppo (RQ2) |
| Tab. 5 | Sensibilità al protocollo, per modello (RQ3) |
| Tab. 6 | Set diagnostico ridotto (RQ6) |
| Fig. 1 | RQ2: per ogni base, la base ingenua, la base calibrata e i TDM come punti sullo stesso asse |
| Fig. 2 | RQ3: classifica sotto protocolli diversi (grafico a "bump"), con la banda della leaderboard |
| Fig. 3 | Diagrammi di affidabilità in dominio e fuori dominio |
| Fig. 4 | Qualità contro costo, con i riferimenti fuori vincolo (Jev, 27B della leaderboard) come linee orizzontali |
| Fig. 5 | Ammissione dei non supportati contro BAcc |

## 9. Lavoro e costi stimati

| Voce | Stima |
|---|---|
| Split, dev e test fuori dominio | 1 settimana (codice) |
| Run completi dei modelli con adapter pronto (d3-flash, StartLux-9B, Nimble-v3, MiniCheck-7B) | 2–3 giorni GPU |
| Adapter nuovi (Clef-flash, Tacit, pplx-decider-flash, Kev-9B/4B, Strands, basi generiche con le due letture) | 1,5–2 settimane (Tacit e le basi condividono la lettura di AnyJev) |
| Inferenza su dev, fuori dominio e diagnostico | 3–5 giorni GPU |
| Jev 1.13 e 1.14 con il protocollo zero-shot | pochi dollari |
| Varianti di protocollo (RQ3) | quasi tutte senza inferenza (grid search); polarità e composizione della richiesta richiedono un'inferenza su un sottoinsieme |
| Analisi degli errori | 3 giorni |
| Scrittura | 2 settimane |
| **Totale** | **~7 settimane**, in parallelo con B-ctrl |
