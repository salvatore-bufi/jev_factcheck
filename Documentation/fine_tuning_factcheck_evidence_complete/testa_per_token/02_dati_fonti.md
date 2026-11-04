# Passo 2 — I dati dalle fonti

## In breve

Per addestrare e valutare la testa servono righe con documento, claim, etichetta e, per i positivi, **dove sono le frasi di evidenza** nel documento, in caratteri. Si usano tre dataset pubblici che hanno già l'evidenza annotata: **SciFact**, **WiCE**, **HoVer** (con HotpotQA distractor per il testo delle pagine).

La costruzione è già scritta, in `evidence/data/` (comando `python -m evidence.data.build`). Per la testa per token va aggiunto un campo: le frasi gold come intervalli di caratteri (§5).

## 1. Le fonti

| Fonte | Da dove | Righe | Positivi / negativi | Documento |
|---|---|---:|---|---|
| **SciFact** | `allenai/scifact_entailment` (cache HF), train + validation | 1.258 | 508 / 750 | titolo + abstract biomedico (mediana 9 frasi, ~590 token di prompt) |
| **WiCE** | `tasksource/wice` (cache HF), train + dev + test | 5.294 | 3.100 / 2.194 | la pagina web citata da Wikipedia, una riga per elemento (mediana 68 frasi, ~1.670 token, massimo 18.580) |
| **HoVer** | claim da `dataset/HoVer/data/hover_{train,dev}_release_v1.1.json`; testo da `hotpotqa/hotpot_qa` (distractor) | 22.063 | 12.974 / 9.089 | ~10 pagine Wikipedia: quelle dell'evidenza più pagine di disturbo (mediana 46 frasi, ~1.690 token) |

### SciFact
- `SUPPORT` → positivo; le frasi di *rationale* sono l'evidenza.
- `CONTRADICT` → negativo; le frasi che contraddicono si salvano a parte (`contra`), solo per analisi.
- `NEI` → negativo.
- Documento: titolo, a capo, frasi dell'abstract unite da spazio.

### WiCE
- `supported` → positivo; `supporting_sentences` dà **più insiemi alternativi** di frasi che dimostrano il claim.
- `not_supported` → negativo.
- `partially_supported` → etichetta 0 ma **nessuna loss di evidenza** (una parte del claim è supportata, quindi non si può chiedere che nessuna frase si accenda), oppure scartato con `--wice-partial drop`.
- Documento: gli elementi della pagina (titoli, righe, frasi) uniti da a capo.
- **WiCE è anche una fonte del test LLM-AggreFact.** Ogni riga ha `meta.aggrefact_overlap`: `"dev"` o `"test"` se la stessa pagina è in AggreFact dev o test, `false` altrimenti. Il confronto si fa sulla pagina, non sul claim, perché AggreFact usa i claim interi e WiCE li divide in pezzi. Misurato: tutto il dev di WiCE è in AggreFact dev, tutto il test di WiCE in AggreFact test, più 2 righe del train. Queste righe non vanno mai in training ([04](04_split_e_mix.md)).

### HoVer con HotpotQA
- HoVer dà i claim (2–4 "salti" fra pagine), le etichette `SUPPORTED` / `NOT_SUPPORTED` e le frasi di evidenza come (titolo della pagina, numero di frase), con la numerazione di HotpotQA.
- HotpotQA distractor, collegato con `hpqa_id`, dà per quella domanda le pagine già divise in frasi e ~8 pagine di disturbo. Il documento contiene le pagine dell'evidenza e quelle di disturbo, in ordine casuale fissato dall'id del claim: ogni pagina è titolo, a capo, testo; pagine separate da una riga vuota. L'evidenza è quindi spesso lontana: in 6.608 positivi su 12.974 le frasi gold distano 20 o più frasi.
- Per i claim a 3–4 salti, HotpotQA non ha tutte le pagine. Con `--hover-db-fallback` le pagine mancanti si prendono dal database Wikipedia di HoVer (`dataset/HoVer/data/wiki_wo_links.db`), e la frase n si trova dividendo il testo con il nostro divisore. Senza questa opzione si perde il 45% dei claim. Misurato su 5.488 frasi di pagine presenti in entrambe le fonti: la frase così trovata è quella giusta nel 98% dei casi; nel 2% è sfasata di una frase. Queste righe hanno `meta.gold_alignment = "db_split"`, le altre `"exact"`.

Il divisore in frasi entra **solo qui**, per creare le etichette delle pagine prese dal db: mai nel modello.

## 2. Preparazione

```sh
python -c "import nltk; nltk.download('punkt_tab')"        # divisore usato dal build (solo per i dati)
chmod -R go-w ~/nltk_data                                  # NLTK rifiuta una cartella dati scrivibile da altri utenti
hf download hotpotqa/hotpot_qa --repo-type dataset --include "distractor/*"
```

SciFact e WiCE devono essere nella cache HF; HoVer in `dataset/HoVer/`; LLM-AggreFact in `dataset/lytang--LLM-AggreFact/` (per il controllo su WiCE); il tokenizer di KnowLine-4B nella cache HF (per il filtro di lunghezza).

## 3. Il comando

```sh
python -m evidence.data.build --source scifact wice hover --hover-db-fallback
```

| Opzione | Default | Cosa fa |
|---|---|---|
| `--hover-db-fallback` | spento | pagine HoVer mancanti prese dal db (da tenere acceso) |
| `--max-tokens N` | 32.768 | scarta le righe il cui prompt supera N token del tokenizer di KnowLine-4B: documento + 2 × claim + 256 (il front end di KnowLine aggiunge 157 token, misurati) |
| `--wice-partial` | `negative` | WiCE parzialmente supportati: etichetta 0 senza loss di evidenza, o `drop` |
| `--limit N` | 0 | primi N record per fonte, per prove |

Uscita in `dataset/evidence/`: `<fonte>.jsonl`, `<fonte>.stats.json` (conteggi, scarti con motivo, lunghezze, distanze fra frasi gold), `manifest.json` (file letti, revisioni, sha256, opzioni, commit). Circa un minuto e mezzo.

Con il limite di default non si scarta nessuna riga per lunghezza. Con `--max-tokens 8192` si scartano 75 righe di WiCE; con 4.096, 503 di WiCE e 2 di HoVer.

## 4. Il formato di una riga oggi

```json
{"id": "hover:train:330ca632-…", "source": "hover", "orig_split": "train",
 "claim": "…", "document": "Kristian Zahrtmann\nKristian Zahrtmann was …\n\nOssian Elgström\n…",
 "label": 1,
 "sentences": [[0, 61], [62, 140], "…"],
 "evidence": [1, 9, 31], "evidence_sets": [[1, 9, 31]],
 "contra": [],
 "evidence_quality": "gold", "evidence_term": "supervised",
 "doc_id": "hover:5ab7a86d5542995dae37e986",
 "meta": {"orig_label": "SUPPORTED", "gold_alignment": "exact", "doc_tokens": 1534, "prompt_tokens": 1830, "…": "…"}}
```

| Campo | Significato |
|---|---|
| `label` | 1 = supportato, 0 = no |
| `sentences`, `evidence`, `evidence_sets` | frasi del nostro divisore e indici di quelle di evidenza: servono solo alla valutazione per frase ([10](10_valutazione.md) §3), la testa per token non li usa |
| `evidence_term` | che loss riceve la riga: `supervised` (positivo con evidenza), `mil_neg` (negativo: nessun token deve accendersi), `none` (nessuna loss di evidenza) |
| `doc_id` | righe con lo stesso documento (lo split si fa per `doc_id`) |
| `meta.prompt_tokens` | lunghezza stimata del prompt in token |

## 5. Il campo da aggiungere: `gold_char_sets`

La testa per token impara dalle frasi gold **come le dà la fonte**, in caratteri, senza passare dal nostro divisore. Il builder le calcola già (servono a ricavare `evidence`), ma non le salva. Si aggiungono due campi:

```json
"gold_char_sets": [[[412, 497], [3120, 3188]]],
"contra_char": []
```

- `gold_char_sets`: per ogni insieme alternativo di evidenza, gli intervalli `[inizio, fine)` di caratteri delle frasi gold (frasi di SciFact, elementi di WiCE, frasi HotpotQA di HoVer; per le pagine HoVer prese dal db, le frasi trovate dividendo il testo);
- `contra_char`: lo stesso per le frasi che contraddicono (SciFact).

Modifica in `evidence/data/schema.py`, funzione `assemble`:

```python
    row["gold_char_sets"] = [[list(span) for span in gold] for gold in gold_sets if gold]
    row["contra_char"] = [list(span) for span in contra]
```

e in `validate`:

```python
    for spans in row["gold_char_sets"] + [row["contra_char"]]:
        for s, e in spans:
            if not (0 <= s < e <= len(row["document"])) or not row["document"][s:e].strip():
                raise ValueError(f"{row['id']}: bad gold character span {(s, e)}")
    if row["label"] == 0 and row["gold_char_sets"]:
        raise ValueError(f"{row['id']}: label 0 with gold spans")
```

Poi un test in `tests/test_evidence_data.py` (il testo di ogni intervallo è la frase gold della fonte) e la ricostruzione dei dataset.

## 6. Controlli

| Controllo | Atteso |
|---|---|
| Ogni intervallo di `gold_char_sets` dentro il documento, non vuoto | tutte le righe |
| Testo degli intervalli = frase gold della fonte (spazi ai bordi a parte) | tutte le righe `exact` |
| 30 positivi per fonte letti a mano con gli intervalli evidenziati (`python -m evidence.data.show`) | le parti evidenziate dimostrano il claim in almeno 28 su 30 |
