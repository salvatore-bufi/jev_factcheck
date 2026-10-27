# Step 1 — Baseline: verdetto con il nuovo ordine, evidenza senza training

## Obiettivo

Sapere, prima di addestrare la testa:
1. quanto costa al verdetto mettere il claim prima del documento (zero-shot);
2. quanto si ottiene sull'evidenza **senza** training, con metodi in un forward;
3. un riferimento di qualità "alta" per l'evidenza, calcolato offline con molti forward.

## 1. Verdetto

| ID | Cosa | Perché |
|---|---|---|
| B-1D | decision model, una domanda, ordine nativo (documento → claim) | baseline del piano binario ([../fine_tuning_factcheck/01_baseline.md](../fine_tuning_factcheck/01_baseline.md)) |
| **B-ORD** | stesso modello, **claim → documento** | costo del cambio di formato prima del training |
| B-FT | modello fine-tunato del piano binario, se già disponibile | riferimento: la testa di evidenza non deve peggiorarlo |

Lettura di B-ORD: un calo di qualche decimo è normale (formato mai visto) e viene recuperato dal fine-tuning, che in questo piano usa il nuovo ordine. Un calo di più punti va segnalato: vorrebbe dire che il modello di partenza dipende molto dall'ordine e che il fine-tuning dovrà spostarlo di più.

Per Winnow il cambio d'ordine è nel JSON dello stato (`{"claim":...,"document":...}`); per JPT nelle righe dello stato (`claim:` prima di `document:`). La domanda resta quella scelta nel piano binario.

## 2. Evidenza senza training (un forward)

Tutte valutate su `dev_evid_select` con le metriche di [06](06_valutazione.md).

| ID | Metodo | Costo in produzione | Cosa dice |
|---|---|---|---|
| E-RAND | k frasi a caso (k = numero medio di frasi gold) | 0 | pavimento |
| E-LEX | BM25 / sovrapposizione di parole fra claim e frase, top-k o sopra soglia | trascurabile | quanto si ottiene con la sola somiglianza superficiale |
| E-EMB | coseno fra embedding del claim e della frase (es. un modello di embedding piccolo) | un encoder in più | somiglianza semantica, senza ragionamento |
| E-ATT | attenzione dalla posizione `Answer:` verso i token di ogni frase, sommata per frase, sulle teste/layer migliori (scelti su metà di `dev_evid_select`, misurati sull'altra metà) | 1 forward (serve accesso all'attenzione dell'ultima posizione) | quanto il modello "guarda" già l'evidenza |

E-LEX è il riferimento da battere con margine: se la testa addestrata non fa molto meglio della somiglianza lessicale, l'utente potrebbe ottenere quasi lo stesso con un BM25.

E-LEX ed E-EMB falliscono per costruzione sulle frasi ponte (non condividono parole con il claim): su `test_evid_sparse` ci si aspetta un calo forte. È la misura di quanto il problema sia reale.

## 3. Evidenza di riferimento offline (molti forward)

| ID | Metodo | Costo |
|---|---|---|
| E-ABL | ablazione greedy con il modello del verdetto: si parte dall'insieme vuoto e si aggiunge ogni volta la frase che alza di più P(supportato), finché si supera la soglia; poi si prova a togliere ogni frase dall'insieme e la si scarta se P resta sopra soglia | O(n · k) forward per riga: solo offline |

Serve a due cose:
- **riferimento** di qualità per la testa (quanto si avvicina a un metodo fedele per costruzione);
- **etichettatore** di dati di training dove non c'è evidenza gold ([02b](02b_dataset_generazione_llm.md) §3).

Su documenti lunghi E-ABL è costoso (n = 100 frasi, k = 3 → ~300 forward): si calcola su un campione di 300 righe per fonte.

## 4. Tabella attesa

| Metodo | F1 frase (FEVER) | F1 frase (HoVer) | Insieme completo (HoVer) | F1 su sparse | Costo |
|---|---:|---:|---:|---:|---|
| E-RAND | | | | | 0 |
| E-LEX | | | | | ~0 |
| E-EMB | | | | | encoder |
| E-ATT | | | | | 1 forward |
| E-ABL (offline) | | | | | O(n·k) forward |

## Criteri per passare allo step successivo

- B-ORD misurato; deciso che il piano procede con il claim prima.
- E-LEX ed E-ATT misurati su `dev_evid_select`; E-ABL su un campione.
