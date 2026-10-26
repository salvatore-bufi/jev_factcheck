# Step 7 — Piano degli esperimenti

## Obiettivo

Arrivare alla testa di evidenza finale con il minimo di corse. Quasi tutti gli esperimenti si fanno in **fase 1** (backbone congelato, stati salvati): durano minuti, non ore. Solo la conferma finale e la fase 2 richiedono GPU per molte ore.

## Regole

1. Una variazione per esperimento, registrata.
2. Si decide su `dev_evid_select` (F1 per frase, recall dell'insieme completo, F1 oltre 10 frasi) e, per la fase 2, anche sul verdetto in `dev_select`.
3. Si adotta una variazione se migliora l'F1 di ≥ 1 punto o la recall dell'insieme completo di ≥ 2 punti, senza peggiorare il verdetto.
4. Prima su JPT-9B, poi su Winnow-12B.

## Fase 0 — Baseline

| ID | Cosa |
|---|---|
| B-ORD | verdetto con claim → documento, zero-shot |
| E-RAND, E-LEX, E-EMB | evidenza senza modello |
| E-ATT | attenzione da `Answer:` (teste scelte su metà del dev) |
| E-ABL | ablazione greedy offline, su campione |

## Fase H — Testa di evidenza (fase 1, backbone congelato)

Riferimento **H0**: ultimo layer, aggregatore 2 layer, BCE pesata + ranking (γ = 0,5), mix `ev_head`.

| ID | Variazione | Domanda |
|---|---|---|
| **H0** | — | la testa batte E-LEX ed E-ATT? Se no, controllare rendering e posizioni prima di tutto |
| H1 | fase 2 (congiunto) invece di backbone congelato | quanto si guadagna sbloccando la LoRA, e quanto costa al verdetto |
| H2 | layer: ultimo / ~2/3 / concatenazione | quale rappresentazione localizza meglio l'evidenza |
| H3a | testa lineare per frase (senza aggregatore) | quanto serve l'aggregatore (atteso: crollo sull'evidenza sparsa) |
| H3b | solo pointer `qᵀ W h_k` | alternativa più semplice |
| H3c | aggregatore 1 / 2 / 4 layer | capacità necessaria |
| H4 | ordine nativo documento → claim (con verdetto addestrato in quell'ordine) | quanto vale mettere il claim prima |
| H5 | senza righe sparse costruite | l'augmentation serve? (atteso: calo oltre 10 frasi) |
| H6 | λ nella fase 2: 0,1 / 0,3 / 1 | equilibrio fra evidenza e verdetto |
| H7 | γ = 0 / 0,5 / 1; ω FEVER 0,3 / 1; unione vs "almeno un insieme" | gestione delle annotazioni incomplete |
| H8 | solo gold vs gold + construct vs + ablation | valore delle etichette generate e ricavate |
| H9 | senza embedding della posizione della frase | la posizione aiuta o crea un bias verso inizio/fine? |

**Come leggere H3a contro H0.** Se la testa lineare è vicina all'aggregatore su FEVER (evidenza di una frase) ma molto sotto su HoVer e sulle righe sparse, l'aggregatore sta facendo il lavoro previsto. Se è vicina anche sulle sparse, il claim prima del documento basta e l'aggregatore si può semplificare.

## Fase W — Winnow e conferma

| ID | Cosa |
|---|---|
| W0 | configurazione migliore di JPT su Winnow (fase 1) |
| W1 | fase 2 su Winnow, solo se su JPT H1 ha dato un guadagno chiaro |
| L1 | misura di latenza con e senza testa sul server di produzione |

## Controlli finali

| ID | Cosa |
|---|---|
| C1 | 3 seed della testa finale (fase 1: costo minimo) |
| C2 | sufficienza e necessità su tutti i test di evidenza |
| C3 | confronto con E-ABL: quanto la testa in un forward si avvicina al metodo da n forward |

## Calendario indicativo

| Settimana | Lavoro |
|---|---|
| 1 | builder con evidenza ([02a](02a_dataset_fonti_esistenti.md)), segmentazione e rendering con posizioni ([03](03_rendering_formato.md)), Fase 0 |
| 2 | etichette construct/ablation e righe sparse ([02b](02b_dataset_generazione_llm.md)); estrazione degli stati; H0–H9 |
| 3 | H1 (fase 2) se serve; Winnow (W0–W1); latenza |
| 4 | controlli, congelamento, test, export |

Prerequisito: il modello del verdetto con claim → documento ([../fine_tuning_factcheck/](../fine_tuning_factcheck/00_overview.md)); può procedere in parallelo, la Fase H si rifà sul modello finale.

## Registro

`finetune/experiments_evidence.md`:

| ID | Modello | Fase | Variazione | F1 frase | Insieme completo | F1 ≥ 10 frasi | Sufficienza | Δ verdetto | Decisione |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| E-LEX | — | — | — | | | | | — | baseline |
| H0 | JPT | 1 | — | | | | | 0 | |
