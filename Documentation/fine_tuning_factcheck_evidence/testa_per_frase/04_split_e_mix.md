# Passo 4 — Split e mix di training

## In breve

1. **Split**: si decide una volta quali righe servono per addestrare (train), per scegliere (dev) e per la misura finale (test), e non si cambia più. Si divide per documento (`doc_id`): lo stesso documento non può stare in due split.
2. **Mix** `ev_head`: le righe di training con il loro tipo di loss e i loro pesi.

Script: `evidence/data/make_splits.py`, `evidence/data/make_mix.py`. Uscita in `dataset/evidence/splits/` e `dataset/evidence/mixes/`, con un `manifest.json` (hash dei file, conteggi, seme 13).

## 1. Regole degli split

| Fonte | Train | Dev | Test |
|---|---|---|---|
| SciFact | 90% del train d'origine (per `doc_id`) | 10% del train d'origine | validation d'origine |
| HoVer | 95% del train d'origine (per `doc_id`) | 5% del train d'origine, solo righe `exact` | dev d'origine, solo righe `exact` |
| WiCE | train d'origine **senza** le righe con `aggrefact_overlap` | — | dev + test d'origine (tutte con `aggrefact_overlap`): test fuori dominio |

Perché:
- **WiCE**: le pagine del suo dev e test sono le stesse del dev e test di LLM-AggreFact, il benchmark finale; addestrare su di esse falserebbe quella misura. Restano utili come test fuori dominio (pagine web, non Wikipedia).
- **HoVer `db_split`** (pagine prese dal db, frase sfasata nel 2% dei casi): si usano in training con peso minore, ma non per misurare.
- I `doc_id` di dev e test non compaiono mai nel train; i dati generati e spostati nascono solo dal train.

## 2. I set

| Set | Contenuto | Uso |
|---|---|---|
| `train` | il resto | training della testa |
| `dev_evid_select` | metà dei positivi di dev (SciFact, HoVer), divisa per `doc_id` | tutte le scelte: configurazione, soglia di selezione, esperimenti |
| `dev_evid_confirm` | l'altra metà | conferma delle scelte |
| `dev_evid_neg` | negativi del dev con documenti lunghi | controllare che la testa non dia punteggi sopra soglia su claim non supportati |
| `test_evid_id` | test di HoVer (`exact`) e SciFact | misura finale |
| `test_evid_ood` | WiCE dev + test | misura finale fuori dominio |
| `test_evid_sparse` | da `test_evid_id`, con l'evidenza spostata lontano (passo 3, solo con frasi di disturbo del test) | evidenza lontana |

Per ogni set si salvano: frasi gold per riga, distanza fra la prima e l'ultima frase gold (in frasi: vicine / 2–9 / 10–19 / ≥ 20; e in token dall'inizio del documento: < 1.024 / 1.024–4.096 / > 4.096), lunghezza in token.

## 3. Il mix `ev_head`

Ogni riga riceve il tipo di loss dal suo `evidence_term` ([08](08_testa_e_loss.md) §6) e due pesi:
- **peso della riga** (`row_weight`): quanto conta la riga nella media del batch;
- **ω** (`omega`): quanto pesano le frasi **non** gold. Basso dove le annotazioni sono incomplete (frasi valide non segnate dagli annotatori).

| Famiglia | Termine | Peso della riga | ω |
|---|---|---:|---:|
| HoVer supportati, `exact` | `supervised` | 1,0 | 0,5 |
| HoVer supportati, `db_split` | `supervised` | 0,5 | 0,5 |
| SciFact SUPPORT | `supervised` | 1,0 | 0,5 |
| WiCE supportati (train) | `supervised` | 1,0 | 0,3 |
| Generati con posizioni note (passo 3) | `supervised` | 1,0 | 1,0 |
| Etichettati col modello (passo 3) | `supervised` | 0,5 | 0,5 |
| Negativi: HoVer NOT_SUPPORTED, SciFact CONTRADICT/NEI, WiCE not_supported | `mil_neg` | 1,0 | — |
| WiCE parzialmente supportati | `none`: fuori dal mix della testa | — | — |

ω è più basso per WiCE perché i suoi gruppi di evidenza sono spesso incompleti (altre righe della pagina possono supportare il claim).

Con le sole tre fonti il mix ha circa 14.000 positivi e 10.000 negativi (HoVer ne è la gran parte).

Vincoli che lo script controlla e stampa:

| Vincolo | Valore |
|---|---|
| Negativi | 20–45% delle righe (con le tre fonti sono ~40%: si può sottocampionare a 30%) |
| Positivi con 2 o più frasi gold | ≥ 40% |
| Positivi con frasi gold a 10 o più frasi di distanza | ≥ 25% |
| Positivi con evidenza oltre 1.024 token dall'inizio del documento | ≥ 15% |
| Documenti in comune con dev o test | 0 |
| Righe WiCE con `aggrefact_overlap` | 0 |

## 4. Comandi

```sh
python -m evidence.data.make_splits --out dataset/evidence/splits
python -m evidence.data.make_mix --recipe recipes/ev_head.json --out dataset/evidence/mixes/ev_head.jsonl
```

La ricetta (`recipes/ev_head.json`) contiene famiglie, numeri, pesi e seme: lo stesso mix si ricostruisce identico.
