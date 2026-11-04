# Passo 2 — I dati dalle fonti e la divisione in frasi

## In breve

Si usano tre dataset pubblici con l'evidenza annotata: **SciFact**, **WiCE** e **HoVer** (il testo delle pagine di HoVer viene da HotpotQA distractor). Ogni fonte viene convertita in uno **stesso formato di riga**:

- il documento è diviso in frasi con il **nostro** divisore;
- le frasi di evidenza della fonte sono riportate sulle nostre frasi.

Codice (già presente prima dell'implementazione della testa e invariato): [`evidence/data/build.py`](../../../evidence/data/build.py), [`evidence/data/schema.py`](../../../evidence/data/schema.py), [`evidence/data/sources/`](../../../evidence/data/sources/) e il divisore [`evidence/sentences.py`](../../../evidence/sentences.py). Comando:

```sh
python -m evidence.data.build --source scifact wice hover --hover-db-fallback
```

## 1. Le fonti

| Fonte | Righe | Positivi / negativi | Documento | Frasi per documento (mediana) | Frasi gold per positivo |
|----------------|---------------:|--------------------------|-----------------------------------------|--------------------------------:|-----------------------------:|
| SciFact | 1.258 | 508 / 750 | titolo + abstract biomedico | 9 | 1,8 |
| WiCE | 5.294 | 3.100 / 2.194 | pagina web citata da Wikipedia | 68 | 2,6 |
| HoVer | 22.063 | 12.974 / 9.089 | circa 10 pagine Wikipedia (evidenza + disturbo) | 46 | 3,0 |

**Etichette per fonte:**

- **SciFact**: `SUPPORT` → 1; `CONTRADICT` e `NEI` → 0.
- **WiCE**: `supported` → 1; `not_supported` → 0; `partially_supported` → 0 senza loss di evidenza (`evidence_term = none`).
- **HoVer**: `SUPPORTED` → 1; `NOT_SUPPORTED` → 0.

**HoVer.** Le pagine mancanti in HotpotQA (claim a 3–4 salti) vengono prese dal database Wikipedia di HoVer, e la frase gold si ritrova con il nostro divisore. Queste righe hanno `meta.gold_alignment = "db_split"` (9.208); nel 2% dei casi la frase gold è sfasata di una. Le altre sono `"exact"` (12.855).

**WiCE e LLM-AggreFact.** WiCE è anche una fonte del test LLM-AggreFact. `meta.aggrefact_overlap` vale `"dev"` o `"test"` quando la pagina compare lì. Lo split ([04](04_split_e_mix.md)) impedisce che queste righe finiscano in training.

## 2. Il divisore in frasi (`evidence/sentences.py`)

`split(text)` restituisce gli intervalli $[s_k, e_k)$ di caratteri delle frasi. È l'**unica** divisione usata in tutta la pipeline: dati, rendering, training e server. Le regole:

1. **blocchi**: si va a capo dopo una riga vuota, prima di un elemento di elenco e dopo una riga senza punteggiatura finale (un titolo); una riga che finisce con virgola o trattino continua nella successiva;
2. **Punkt** (NLTK `punkt_tab`, inglese) dentro ogni blocco, con abbreviazioni in più (vol, no, fig, eq, approx, cf, pp, al, e.g, i.e, …);
3. frasi **oltre 60 parole** vengono spezzate su `; ` o `: ` vicino alla metà, poi su `, `, poi su uno spazio;
4. frasi **sotto 3 parole** vengono unite alla successiva (alla precedente se sono le ultime del paragrafo);
5. oltre **512 frasi** la riga si esclude (`MAX_SENTENCES`).

Il divisore **fa parte del modello**: la testa è addestrata su queste unità. Per questo il suo sha256 entra nel manifest dei dati, nel `meta.json` delle feature e nel `frozen.json` del rilascio ([12](12_export_e_servizio.md)).

### 2.1 Allineamento delle frasi gold

Le fonti indicano l'evidenza con le **loro** frasi. `sentences.align(spans, gold)` le riporta sulle nostre. La frase $k$ (intervallo $I_k$) corrisponde alla frase gold $g$ (intervallo $J_g$) se

$$
\lvert I_k \cap J_g \rvert \ \ge\ \tfrac{1}{2}\lvert I_k\rvert \quad\text{oppure}\quad \lvert I_k \cap J_g \rvert \ \ge\ \tfrac{1}{2}\lvert J_g\rvert .
$$

Se nessuna frase raggiunge la metà, si prende quella con la sovrapposizione maggiore. La frase della fonte corrisponde a **una sola** nostra frase nel 96,5% dei casi per SciFact, nel 97,3% per WiCE e nel 98,3% per HoVer. Il build si ferma se una fonte scende sotto il 90%.

## 3. Il formato di una riga

```json
{"id": "hover:train:…", "source": "hover", "orig_split": "train",
 "claim": "…", "document": "Kristian Zahrtmann\nKristian Zahrtmann was …",
 "label": 1, "sentences": [[0, 18], [19, 140], …],
 "evidence": [1, 9, 31], "evidence_sets": [[1, 9, 31]], "contra": [],
 "evidence_quality": "gold", "evidence_term": "supervised",
 "doc_id": "hover:5ab7a86d…", "meta": {"gold_alignment": "exact", "prompt_tokens": 1830, …}}
```

| Campo | Significato | Usato da |
|----------------------------|---------------------------------------|-------------------------------------|
| `sentences` | $[s_k, e_k)$ di ogni frase | rendering, metriche, server |
| `evidence` | $G$ = unione degli insiemi alternativi | loss, metriche |
| `evidence_sets` | insiemi alternativi (WiCE ne ha più di uno) | metriche "insieme completo" ed "esatto" |
| `evidence_term` | tipo di loss: `supervised`, `mil_neg`, `none` | mix, loss |
| `doc_id` | righe con lo stesso documento | split |
| `meta.gold_alignment` | `exact` / `db_split` (HoVer) | split, mix |
| `meta.aggrefact_overlap` | sovrapposizione con LLM-AggreFact (WiCE) | split, mix |
| `meta.prompt_tokens` | lunghezza stimata del prompt | statistiche |

Validazione: `schema.validate(row)`. Le frasi devono coprire tutto il documento senza sovrapporsi, e `evidence` deve essere l'unione di `evidence_sets`. Test: `tests/test_evidence_data.py`.
