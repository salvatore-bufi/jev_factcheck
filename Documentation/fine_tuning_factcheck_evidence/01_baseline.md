# Step 1 — Baseline: il formato, l'evidenza senza training, il riferimento offline

## Obiettivo

Sapere, prima di addestrare la testa, per ciascuno dei tre modelli:
1. quanto costa al verdetto aggiungere il claim prima del documento, nelle due forme (claim prima, sandwich);
2. quanto si ottiene sull'evidenza **senza** training, con metodi in un forward;
3. un riferimento di qualità "alta" per l'evidenza, calcolato offline con molti forward.

## 1. Verdetto e formato

Misurati su `dev_select` per d1-3B, JPT-9B e Winnow-12B, con la domanda del piano binario ([../fine_tuning_factcheck/02d_varianti_domanda.md](../fine_tuning_factcheck/02d_varianti_domanda.md)).

| ID | Cosa | Perché |
|---|---|---|
| B-1D | modello di partenza, formato nativo | baseline del piano binario |
| **B-ORD** | modello di partenza, claim → documento | costo dell'ordine invertito |
| **B-SAND** | modello di partenza, sandwich (preambolo + prompt nativo) | costo del preambolo |
| B-FT | modello fine-tunato del piano binario, formato nativo | riferimento: la testa non deve peggiorarlo |
| **B-FT-SAND** | modello fine-tunato del piano binario, sandwich, **senza** riaddestrarlo | si può riusare così com'è? |

### Regola di scelta del formato

1. Se B-SAND è entro 0,3 punti di BAcc e AUC da B-1D (e B-FT-SAND da B-FT, se disponibile): **sandwich**, e il verdetto del piano binario si riusa senza riaddestrarlo.
2. Se il sandwich costa di più ma meno di B-ORD: sandwich, e il verdetto del piano binario si **riaddestra con il preambolo** (stessa ricetta, stesso mix; il preambolo è un'opzione del renderer).
3. Se B-ORD costa meno del sandwich (inatteso): claim prima, con verdetto riaddestrato.

La regola si applica per modello; ci si aspetta lo stesso esito sui tre, ma non è garantito (d1 legge lo stato come JSON indentato, JPT come righe, Winnow come JSON compatto).

## 2. Evidenza senza training (un forward)

Tutte valutate su `dev_evid_select` con le metriche di [06](06_valutazione.md), nel formato scelto in §1.

| ID | Metodo | Costo in produzione | Cosa dice |
|---|---|---|---|
| E-RAND | k frasi a caso (k = numero medio di frasi gold) | 0 | pavimento |
| E-LEX | BM25 / sovrapposizione di parole fra claim e frase, top-k o sopra soglia | trascurabile | quanto si ottiene con la sola somiglianza superficiale |
| E-EMB | coseno fra embedding del claim e della frase (un modello di embedding piccolo) | un encoder in più | somiglianza semantica, senza ragionamento |
| E-ATT | attenzione da `Answer:` verso i token di ogni frase, sommata per frase, negli **8 layer di attenzione piena** ([00c](00c_modelli_e_architetture.md) §1); si sceglie la combinazione migliore di layer e teste su metà di `dev_evid_select` e si misura sull'altra metà | 1 forward (riga di attenzione ricalcolata da un hook) | quanto il modello "guarda" già l'evidenza; è anche la prova che le feature dell'idea 1 contengono segnale |

E-LEX è il riferimento da battere con margine: se la testa addestrata non fa molto meglio della somiglianza lessicale, l'utente potrebbe ottenere quasi lo stesso con un BM25.

E-LEX ed E-EMB falliscono per costruzione sulle frasi ponte (non condividono parole con il claim): su `test_evid_sparse` ci si aspetta un calo forte. È la misura di quanto il problema sia reale.

E-ATT si misura anche con il formato nativo: se nel sandwich il segnale è più alto, è un indizio che il preambolo aiuta anche la localizzazione, non solo gli `h_k`.

## 3. Evidenza di riferimento offline (molti forward)

| ID | Metodo | Costo |
|---|---|---|
| E-ABL | ablazione greedy con il modello del verdetto: si parte dall'insieme vuoto e si aggiunge ogni volta la frase che alza di più P(supportato), finché si supera la soglia; poi si prova a togliere ogni frase dall'insieme e la si scarta se P resta sopra soglia | O(n · k) forward per riga: solo offline |

Serve a due cose:
- **riferimento** di qualità per la testa (quanto si avvicina a un metodo fedele per costruzione);
- **etichettatore** di dati di training dove non c'è evidenza gold ([02b](02b_dataset_generazione_llm.md) §3).

Costo: con n = 100 frasi e k = 3, ~300 forward per riga. Si calcola su 300 righe per fonte per ciascun modello. Come etichettatore dei dati di training si usa **d1-3B** (4–5 volte più veloce), con un controllo di accordo con JPT-9B su 300 righe ([02b](02b_dataset_generazione_llm.md) §3).

## 4. Tabella attesa (una per modello)

| Metodo | F1 frase (FEVER) | F1 frase (HoVer) | Insieme completo (HoVer) | F1 su sparse | Costo |
|---|---:|---:|---:|---:|---|
| E-RAND | | | | | 0 |
| E-LEX | | | | | ~0 |
| E-EMB | | | | | encoder |
| E-ATT (nativo) | | | | | 1 forward |
| E-ATT (sandwich) | | | | | 1 forward |
| E-ABL (offline) | | | | | O(n·k) forward |

## Criteri per passare allo step successivo

- B-ORD e B-SAND misurati per i tre modelli; formato deciso con la regola di §1.
- E-LEX ed E-ATT misurati su `dev_evid_select`; E-ABL su un campione.
