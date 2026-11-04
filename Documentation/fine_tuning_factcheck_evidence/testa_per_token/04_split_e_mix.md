# Passo 4 — Split e mix

## In breve

1. **Split**: train / dev / test congelati, per documento (`doc_id`), con le fonti già costruite.
2. **Mix** `ev_tok_head`: le righe di training con il loro tipo di loss e i loro pesi.

Script: `evidence/data/make_splits.py`, `evidence/data/make_mix.py`. Uscita in `dataset/evidence/splits/` e `dataset/evidence/mixes/`.

## 1. Regole degli split

- Per `doc_id`: lo stesso documento sta in un solo split.
- **WiCE**: le righe con `meta.aggrefact_overlap = "dev"` o `"test"` (tutto il dev e il test di WiCE, più 2 righe del train) **non vanno in training**. Vanno in test (`test_evid_ood`), così restano utili come misura fuori dominio.
- **SciFact**: train d'origine → training e dev (90/10 per `doc_id`); validation d'origine → test.
- **HoVer**: train d'origine → training e dev (95/5 per `doc_id`); dev d'origine → test. Nei set di dev e test di HoVer si tengono solo le righe con `gold_alignment = exact`.
- Seme fisso (13); `manifest.json` con gli hash dei file e i conteggi.

## 2. I set

| Set | Contenuto | Uso |
|---|---|---|
| `train` | il resto | training della testa |
| `dev_evid_select` | positivi del dev (SciFact, HoVer) | scelte: configurazione, soglia di decodifica, esperimenti |
| `dev_evid_confirm` | metà del dev, separata per `doc_id` | conferma delle scelte |
| `dev_evid_neg` | negativi del dev con documenti lunghi | controllare che la testa non accenda tratti su claim non supportati |
| `test_evid_id` | test di HoVer (`exact`) e SciFact | misura finale |
| `test_evid_ood` | WiCE dev + test | misura finale fuori dominio (pagine web) |
| `test_evid_sparse` | da `test_evid_id`, con evidenza spostata lontano (passo 3) | evidenza lontana |

Per ogni set si salvano: frasi gold per riga, distanza fra la prima e l'ultima (in frasi e in token), lunghezza in token.

## 3. Il mix `ev_tok_head`

Ogni riga riceve dal suo `evidence_term` il tipo di loss ([08](08_testa_e_loss.md) §3), più due pesi:

| Famiglia | Termine | Peso della riga | ω (peso dei token non gold) |
|---|---|---:|---:|
| HoVer supportati, `exact` | `supervised` | 1,0 | 0,5 |
| HoVer supportati, `db_split` | `supervised` | 0,5 | 0,5 |
| SciFact SUPPORT | `supervised` | 1,0 | 0,5 |
| WiCE train supportati | `supervised` | 1,0 | 0,3 |
| C2D con intervalli esatti (passo 3) | `supervised` | 1,0 | 1,0 |
| Etichettati col modello (passo 3) | `supervised` | 0,5 | 0,5 |
| Negativi (HoVer NOT_SUPPORTED, SciFact CONTRADICT/NEI, WiCE not_supported) | `mil_neg` | 1,0 | — |
| WiCE parzialmente supportati | `none` (escluse dal mix della testa) | — | — |

ω è più basso per WiCE perché i suoi gruppi di evidenza sono spesso incompleti (altre righe della pagina possono supportare il claim).

Vincoli che lo script controlla e stampa:

| Vincolo | Valore |
|---|---|
| Negativi | 20–30% delle righe |
| Positivi con 2 o più frasi gold | ≥ 40% |
| Positivi con evidenza oltre 1.024 token dall'inizio del documento | ≥ 15% |
| Documenti in comune con dev o test | 0 |
| Righe WiCE con `aggrefact_overlap` nel training | 0 |

## 4. Comandi

```sh
python -m evidence.data.make_splits --out dataset/evidence/splits
python -m evidence.data.make_mix --recipe recipes/ev_tok_head.json --out dataset/evidence/mixes/ev_tok_head.jsonl
```
