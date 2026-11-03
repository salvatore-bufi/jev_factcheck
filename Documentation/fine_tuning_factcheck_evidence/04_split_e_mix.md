# Passo 4 — Split e mix

## Obiettivo

1. Congelare i blocchi di valutazione dell'evidenza, compatibili con gli split del piano binario.
2. Costruire i due mix di training: `ev_head` (testa su backbone congelato) ed `ev_joint` (LoRA + testa).

Script: `evidence/make_splits.py`, `evidence/make_mix.py`. Gli split sono gli stessi per i tre modelli.

## 1. Regole di split

- Si parte dagli split congelati del piano binario ([../fine_tuning_factcheck/00b_split_e_protocollo.md](../fine_tuning_factcheck/00b_split_e_protocollo.md)): una riga che lì è in dev o test resta in dev o test.
- Unità di split = `doc_id` (per FEVER e HoVer, l'insieme delle pagine): nessun documento in due split.
- HoVer: il dev ufficiale si divide a metà per `doc_id` (metà `dev`, metà `test`).
- FEVER: `train` del parquet → train; `paper_dev` → dev; `paper_test` → test.
- Le righe generate (§3 del passo [03](03_dati_generati.md)) nascono solo da righe di train; i loro distrattori vengono solo da train.

## 2. I blocchi di valutazione

| Blocco | Contenuto | Righe (obiettivo) | Uso |
|---|---|---:|---|
| `dev_evid_select` | dev, positivi con evidenza gold: FEVER, HoVer, C2D con posizioni + righe sparse costruite **dal dev** | 3.000 | scelte su testa, loss, esperimenti |
| `dev_evid_confirm` | come sopra, righe diverse | 2.000 | conferma, una volta per fase |
| `dev_evid_neg` | dev, negativi lunghi (FEVER REFUTES con `contra`, HoVer NOT_SUPPORTED) | 1.500 | frasi "accese" sui negativi |
| `test_evid_id` | test FEVER e HoVer, positivi con evidenza gold; 10% di C2D con posizioni | 3.000 | evidenza in distribuzione, una volta |
| `test_evid_ood` | SciFact (SUPPORT), WiCE (supported) | tutte | evidenza fuori dominio, una volta |
| `test_evid_sparse` | da `test_evid_id` con ≥ 2 frasi gold: estremi, distrattori (presi da documenti di test), lontano dal preambolo | 1.500 | evidenza lontana, una volta |

Le righe sparse di dev e test si costruiscono con `augment_sparse.py` sugli split rispettivi, con distrattori dello **stesso** split.

Per ogni blocco si salva la distribuzione di:
- numero di frasi gold per riga;
- distanza in frasi fra la prima e l'ultima frase gold (adiacenti / 2–9 / 10–19 / ≥ 20);
- distanza in token dell'ultima frase gold dall'inizio del documento, per ciascuno dei tre tokenizer (< 1.024 / 1.024–4.096 / > 4.096).

```sh
python -m evidence.make_splits --binary-splits <manifest del piano binario> --out data/evidence/splits/
```

Output: `data/evidence/splits/<blocco>/<fonte>.jsonl` e `manifest.json` (hash di ogni file, conteggi, distribuzioni).

## 3. Il mix `ev_head` (fasi 1a e 1b)

Ogni riga ha un `evidence_term` che dice quale termine di loss riceve ([08](08_testa_e_loss.md) §3):

| Famiglia | Righe (pilota) | `evidence_term` | `evidence_weight` | ω (peso delle frasi non gold) |
|---|---:|---|---:|---:|
| FEVER SUPPORTS | 6.000 | `supervised` | 1,0 | 0,3 |
| HoVer SUPPORTED | 5.000 | `supervised` | 1,0 | 0,5 |
| C2D / CG2C con posizioni | 2.000 | `supervised` | 1,0 | 1,0 |
| Ablazione | 3.000 | `supervised` | 0,5 (`ablation_partial`: 0,3) | 0,5 |
| Righe sparse positive | 4.000 | come l'origine | come l'origine | come l'origine |
| Negativi lunghi (FEVER REFUTES, HoVer NOT_SUPPORTED, C2D negativi, negativi sparsi) | 6.000 | `mil_neg` | 1,0 | — |
| Positivi senza evidenza (solo per l'esperimento H-MIL) | 3.000 | `mil_pos` | 1,0 | — |

Vincoli da verificare sul mix (lo script li stampa e si ferma se non sono rispettati):

| Vincolo | Valore |
|---|---|
| Righe positive con ≥ 2 frasi di evidenza | ≥ 40% |
| Righe positive con distanza ≥ 10 frasi fra le gold | ≥ 25% |
| Righe positive con evidenza oltre 1.024 token (tokenizer d1) | ≥ 15% |
| Righe positive non Wikipedia (C2D, CG2C, ablazione) | ≥ 30% |
| Righe con > 50 frasi | ≥ 15% |
| Negativi | 20–25% delle righe |
| Negativi con una frase lessicalmente vicina al claim (BM25 sopra la mediana del blocco) | ≥ 50% dei negativi |
| `doc_id` in comune con dev/test | 0 |

## 4. Il mix `ev_joint` (fase 2)

Il mix finale del piano binario ([../fine_tuning_factcheck/02e_mix_finale.md](../fine_tuning_factcheck/02e_mix_finale.md)), con:
- le righe che compaiono anche in `ev_head` che ne prendono `sentences`, `evidence`, `evidence_term`;
- le righe di `ev_head` che non ci sono (C2D/CG2C con posizioni, ablazione, sparse) aggiunte fino al 20–30% del totale;
- tutte le altre righe con `evidence_term = none` (contribuiscono solo al verdetto).

```sh
python -m evidence.make_mix --recipe recipes/ev_head.json  --out data/evidence/mixes/ev_head.jsonl
python -m evidence.make_mix --recipe recipes/ev_joint.json --out data/evidence/mixes/ev_joint.jsonl
```

La ricetta (`recipes/*.json`) elenca famiglie, numero di righe, seme e pesi della tabella di §3: il mix si ricostruisce identico dalla ricetta.

## Prodotto

- `data/evidence/splits/` con `manifest.json` congelato (hash).
- `data/evidence/mixes/ev_head.jsonl` (e `ev_joint.jsonl` quando serve), con le statistiche di §3.
