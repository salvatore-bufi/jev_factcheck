# Le opzioni di paper, a confronto

La domanda è: *creare un dataset nuovo e testarci i modelli? proporre un dataset? oppure altro?* Le opzioni realistiche sono cinque.

## Opzione 1 — Solo valutazione ("benchmarking paper")

**Cosa**: valutare TDM aperti e chiusi sul fact checking documento–claim con benchmark esistenti (LLM-AggreFact + test fuori dominio), con calibrazione, robustezza e costi, contro checker specializzati e LLM generici.

| Pro | Contro |
|---|---|
| Gran parte dei risultati c'è già (5 modelli su tutto il test di AggreFact) | i revisori delle conferenze principali tendono a considerarla un contributo debole se manca un'idea o un'analisi forte |
| Tempi brevi (4–6 settimane) | facile da superare: chiunque abbia accesso a Jev può fare qualcosa di simile |
| Costo basso (inferenza; Jev è costato ~2 $ per tutto) | |

**Adatta a**: preprint rapido, workshop (FEVER, TrustNLP), Findings se l'analisi è profonda.

## Opzione 2 — Valutazione + metodo di specializzazione (consigliata)

**Cosa**: l'Opzione 1, più il fine-tuning di Winnow-12B e JPT-9B con loss sulle lettere e replay, con la misura di quanto si perde sulle decisioni generali (oblio).

| Pro | Contro |
|---|---|
| Contributo completo: "dove sono i TDM oggi" + "come migliorarli senza romperli" | più lavoro: codice di training, GPU (L40S), molte corse |
| Risponde a un buco esplicito dell'audit (fine-tuning dei modelli aperti, attribuzione dei guadagni) | rischio che il fine-tuning non batta i checker specializzati: il paper resta valido se l'analisi è onesta, ma è meno forte |
| Il controllo con i modelli base (stesso training su Gemma-4-12B e Qwen3.5-9B) dà un risultato scientifico netto, in un senso o nell'altro | |

**Adatta a**: ACL / EMNLP (main o Findings).

## Opzione 3 — Proporre un dataset nuovo

**Cosa**: costruire un benchmark nuovo di fact checking (documenti, claim, etichette umane) e valutarci i modelli.

| Pro | Contro |
|---|---|
| Un buon dataset può avere molto impatto e molte citazioni | **costoso**: per ~2.000–5.000 esempi con doppia annotazione servono settimane di annotatori (o un budget di crowdsourcing qualificato) e un protocollo di accordo |
| Contributo chiaro e durevole | **i revisori sono severi**: chiedono motivazione del buco, guida di annotazione, accordo fra annotatori, analisi degli errori, datasheet, licenze, baseline forti |
| | il buco va dimostrato: LLM-AggreFact, HaluBench, HalluMix, RAGTruth, FaithBench coprono già molto |
| | allontana dalla domanda principale (i TDM) |

**Quando avrebbe senso**: solo se c'è un buco che i dataset esistenti non coprono e che i TDM rendono importante. Due candidati:
- **etichette graduate o distribuzionali** (più annotatori per esempio, con probabilità di supporto): l'unico modo per valutare davvero la calibrazione, che è la promessa dei TDM;
- **domini o lingue non coperti** (es. documenti in italiano, documenti amministrativi o legali lunghi).

**Adatta a**: NeurIPS Datasets & Benchmarks, LREC-COLING, oppure come secondo paper dopo il primo.

## Opzione 4 — Set diagnostico ricavato da test esistenti (contributo secondario consigliato)

**Cosa**: un insieme di test **controllati**, costruito trasformando esempi di test esistenti: coppie minime (si cambia un numero o un'entità nel claim), rimozione della frase di evidenza, distrattori aggiunti, evidenza spostata in un documento lungo, domande parafrasate, campi rinominati. Le trasformazioni sono automatiche (anche con LLM); un **campione viene validato a mano**.

| Pro | Contro |
|---|---|
| Costa poco: niente annotazione da zero, solo validazione di un campione (es. 500 esempi, 2 annotatori, 2–3 giorni) | non è un "dataset nuovo" pieno: va presentato come risorsa diagnostica |
| Misura proprio le proprietà promesse dai TDM (sensibilità all'evidenza, invarianza, calibrazione) | va costruito solo da test (mai da dati di training) e con licenze compatibili |
| Riutilizzabile da tutta la comunità, anche per altri checker | |

Dettagli: [05_set_diagnostico.md](05_set_diagnostico.md).

## Opzione 5 — Due paper separati

1. **Subito**: Opzione 1 come preprint e poi workshop o Findings (fissa la priorità, dato che il campo corre).
2. **Dopo**: Opzione 2 (fine-tuning) + Opzione 4 (set diagnostico) per ACL o EMNLP 2027.

| Pro | Contro |
|---|---|
| protegge dallo scoop | i due paper si sovrappongono: il secondo deve aggiungere molto per essere accettato |
| il primo esce in poche settimane | un preprint arXiv **non** impedisce la sottomissione a ACL/EMNLP; un paper già *accettato* a un workshop con atti sì, se il secondo è troppo simile: verificare le regole sulla doppia sottomissione |

## Tabella riassuntiva

| Opzione | Novità | Costo | Tempo | Rischio di rifiuto (conferenza principale) | Venue tipica |
|---|---|---|---|---|---|
| 1. Solo valutazione | media | basso | 4–6 sett. | alto | workshop, Findings |
| **2. Valutazione + metodo** | alta | medio | 3–4 mesi | medio | ACL/EMNLP main o Findings |
| 3. Dataset nuovo | alta se il buco è reale | alto | 4–6 mesi | medio-alto | D&B track, LREC |
| 4. Set diagnostico | media (come aggiunta) | basso-medio | 2–3 sett. | — (secondario) | dentro l'Opzione 2 |
| 5. Due paper | — | medio | preprint subito + 3–4 mesi | — | workshop + main |

## Raccomandazione

**Opzione 2 con il set diagnostico dell'Opzione 4, più un preprint anticipato della sola parte di valutazione.**

Motivi:
- usa il lavoro già fatto (cinque modelli valutati, infrastruttura, protocollo, dati di training pronti);
- risponde a tre buchi indicati dall'audit: fine-tuning dei modelli aperti, attribuzione dei guadagni (architettura/inizializzazione contro dati), stabilità della calibrazione fra domini;
- evita il costo e il rischio di un dataset nuovo, ma lascia una risorsa riutilizzabile (il set diagnostico);
- il preprint anticipato protegge dallo scoop.

Il dataset nuovo (Opzione 3) lo terrei come **secondo paper**, nella forma più distintiva: **etichette distribuzionali per la calibrazione** dei fact checker, eventualmente multilingue (inglese + italiano).

## Una domanda da chiarire prima di partire

**Contributo principale: "valutazione" o "metodo"?** Il titolo, l'abstract e l'ordine delle sezioni cambiano molto:
- se il fine-tuning dà un guadagno netto (es. Winnow sopra 79 su AggreFact zero-shot, senza oblio), il paper è **di metodo**, con la valutazione come motivazione;
- se il guadagno è modesto, il paper è **di analisi** ("cosa sanno e non sanno fare i TDM come fact checker"), con il fine-tuning come uno degli esperimenti.

Conviene decidere dopo le prime corse del pilota (Fase 1 di [../fine_tuning/07_esperimenti.md](../fine_tuning/07_esperimenti.md)), scrivendo intanto le sezioni comuni.
