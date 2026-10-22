# Step 7 — Piano degli esperimenti

## Obiettivo

Arrivare alla configurazione finale con poche corse, una variazione alla volta, con criteri fissati prima. Rispetto al piano generale ([../fine_tuning/07_esperimenti.md](../fine_tuning/07_esperimenti.md)) c'è una fase nuova, **A (architettura)**, che risponde con numeri propri alla domanda "decision model, LLM base o encoder stile FactCG?", e una fase **S (distillazione)** opzionale. Spariscono gli esperimenti su replay e oblio.

## Regole

1. Ogni esperimento: ID, configurazione di riferimento, una sola variazione.
2. Si giudica su `dev_select` (BAcc macro, AUC, ECE, soglia ottimale, BAcc a 0,5) e sui test comportamentali. I test solo alla fine.
3. Una variazione si adotta se migliora la BAcc macro su `dev_select` di ≥ 0,5 punti senza peggiorare l'ECE di più di 0,02.
4. Prima su JPT-9B; le scelte vincenti si ripetono su Winnow-12B.

## Fase 0 — Baseline (senza training)

| ID | Cosa | Costo |
|---|---|---|
| B1 | decision model, una domanda (`support_simple`) | dalle cache sul dev di AggreFact già fatto; sul nostro dev ~1 h |
| B2 | tre domande | stessa cache di B1 |
| B3 | formulazioni Q2/Q3 ([02d](02d_varianti_domanda.md)) | ~1 h ciascuna |
| B4 | Gemma-4-12B e Qwen3.5-9B base, domanda scelta | ~2 h |
| B5–B7 | FactCG, MiniCheck, Bespoke-MiniCheck-7B | ~2 h |

**Uscita**: domanda di produzione scelta (B1 o B3); baseline vera congelata.

## Fase 1 — Pilota (JPT, `fc_pilot`, uno stadio)

Riferimento **P0**: LoRA r16, lr 2e-5, 1 epoca, CE, 4.096 token, domanda scelta.

| ID | Variazione | Domanda | Decisione |
|---|---|---|---|
| **P0** | — | il training funziona? | se non batte B1 su `dev_select`, fermarsi e ricontrollare rendering e loss |
| P1 | lr 5e-5 / 1e-5 | sensibilità | tenere il migliore |
| P2 | Brier | calibrazione con dati rumorosi | se ECE migliora senza perdere BAcc |
| P-LS | CE con smoothing 0,05 su dati generati e ANLI difficile | rumore delle etichette | idem |
| P-KL | + β·KL verso il modello di partenza (β = 0,05) | serve un ancoraggio? | solo se P0 peggiora test_ood-like (righe di `dev_select` delle fonti piccole) o l'ECE |
| P3 | SFT classico sul token di risposta (vocabolario intero) | quanto vale la loss sulle lettere | conferma della scelta |
| P-VAR | 10% di varianti di domanda | regolarizzazione | tenere solo se migliora `dev_select` |

## Fase A — Architettura (stessi dati: `fc_pilot`, poi il mix finale)

È la fase che risponde alla domanda di partenza ([00a](00a_scelta_approccio.md)).

| ID | Backbone | Uscita | Domanda |
|---|---|---|---|
| A1 | JPT-9B (decision model) | lettere | riferimento (= P0 o migliore della Fase 1) |
| A2 | Qwen3.5-9B base | lettere, stesso prompt | quanto vale partire da un decision model (controllo C1 anticipato) |
| A3 | Qwen3.5-9B base | **testa nuova** `Linear(d→2)` sull'ultimo token | cambiare la testa aiuta o peggiora? (atteso: peggio o uguale ad A2, con più dati necessari) |
| A4 | JPT-9B | testa nuova inizializzata da `w_vero − w_falso` | verifica pratica dell'equivalenza: deve dare lo stesso risultato di A1 |
| A5 | DeBERTa-v3-large, stile FactCG | testa nuova, chunk 550 | quanto vale la taglia a parità di dati |

**Come leggere la fase A.**

| Esito | Conclusione |
|---|---|
| A1 > A2 di ≥ 0,5 | il decision model aiuta: si prosegue con D1 (raccomandazione di [00a](00a_scelta_approccio.md)) |
| A1 ≈ A2 | il vantaggio è nei dati; il backbone si sceglie per licenza e costo (un base Apache 2.0 può essere preferibile a JPT, CC BY-NC) |
| A3 < A2 | conferma: non cambiare la testa |
| A1 ≈ A4 | conferma dell'equivalenza: la testa estratta si può usare per servire il modello ([08](08_merge_export.md)) |
| A5 ≥ A1 − 1 | l'encoder è quasi equivalente a un costo di gran lunga minore: valutare seriamente la fase S o direttamente E1 |

Costo: 4 corse in più sul pilota (A2–A5), ~4 h ciascuna; A5 sul portatile.

## Fase 2 — Dati (JPT, configurazione migliore)

| ID | Variazione | Domanda |
|---|---|---|
| D-MC | solo MiniCheck + ANLI difficile (≈ ricetta Bespoke) | quanto si ottiene con la sola ricetta pubblica |
| D-HARD | ANLI difficile contro ANLI casuale (stessa quantità) | la selezione degli esempi difficili serve? |
| D-D2C | + D2C esteso | migliorano le fonti di riassunti? |
| D-C2D | + C2D esteso | migliora "claim parziale"? |
| D-CG | + CG2C-MHQA + CG2C-Doc | migliora il multi-hop (HoVer, "evidenza divisa")? |
| D-RE | + `remove_evidence` su documenti lunghi | si abbassa la soglia ottimale? migliora "rimozione dell'evidenza"? |
| D-CUR | due stadi contro uno stadio con gli stessi dati | il curriculum serve? |
| D-SCALE | 25k / 50k / 100k | curva di apprendimento |
| D-FKG | + FactKG in prosa | aiuta? |
| D-NEG | FEVER senza negazioni esplicite | scorciatoie |

## Fase 3 — Winnow e formato

| ID | Variazione | Domanda |
|---|---|---|
| W0 | configurazione migliore di JPT su Winnow | il risultato si trasferisce? |
| F2 | 8.192 token invece di 4.096 | documenti lunghi |
| F3 | rank 8 / 32 / 64 | capacità dell'adapter |
| K1 | dimensione dei chunk in inferenza: 2k / 4k / 8k / intero | scelta di C su `dev_select` |
| K2 | `min` sulle frasi del claim × `max` sui chunk (Bespoke) | risposte lunghe |

## Fase 4 — Contrastive (opzionale)

| ID | Variazione |
|---|---|
| X1 | + `L_pair` sulle coppie C2D/CG2C/VitaminC (μ = 0,3) |
| X2 | μ ∈ {0,1; 0,3} e coppie generate |

## Fase S — Distillazione in un encoder (opzionale)

Solo se il costo d'inferenza è un requisito.

| ID | Cosa | Domanda |
|---|---|---|
| S1 | DeBERTa-v3-large addestrato sulle probabilità del modello finale sul mix (stesse righe di A5) | quanto recupera lo studente rispetto ad A5 (etichette dure)? |
| S2 | S1 + 200–500k coppie in più etichettate solo dall'insegnante (VitaminC, FEVER, HoVer interi, D2C non verificato) | più dati morbidi aiutano? |
| S3 | ModernBERT-large invece di DeBERTa | contesto 8k: meno chunk |

## Controlli finali (obbligatori)

| ID | Cosa |
|---|---|
| C1 | stessa ricetta finale su Gemma-4-12B base (per Winnow) e Qwen3.5-9B base (per JPT) — se la fase A è stata fatta sul pilota, va ripetuta sul mix finale |
| C2 | 3 seed della configurazione finale |
| C3 | FactCG, MiniCheck, Bespoke-MiniCheck-7B sulla stessa suite |

## Calendario indicativo

| Settimana | Lavoro |
|---|---|
| 1 | Step 0, Fase 0, `render.py`/`train.py`, ANLI difficile, `fc_pilot` |
| 2 | Fase 1 e Fase A su JPT |
| 3 | generazione D2C/C2D/CG2C (in parallelo dalla settimana 1 su un'altra GPU); Fase 2 |
| 4 | Fase 3 su Winnow |
| 5 | C1–C3, congelamento, test, export; Fase S se richiesta |

## Registro degli esperimenti

`finetune/experiments_fc.md`, una riga per corsa:

| ID | Modello | Mix | Variazione | BAcc dev | AUC dev | BAcc@0,5 | Soglia ott. | ECE | Solo claim | Rimozione evid. | Durata | Decisione |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| B1 | JPT | — | 1 domanda | | | | | | | | 0 | baseline |
| P0 | JPT | fc_pilot | — | | | | | | | | | |
