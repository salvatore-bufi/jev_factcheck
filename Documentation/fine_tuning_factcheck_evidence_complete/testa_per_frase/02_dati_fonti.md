# Passo 2 — I dati dalle fonti

## In breve

Per addestrare e valutare la testa servono righe con documento, claim, etichetta, il documento **diviso in frasi** e, per i positivi, **quali frasi sono l'evidenza**. Si usano tre dataset pubblici che hanno già l'evidenza annotata: **SciFact**, **WiCE**, **HoVer** (con HotpotQA distractor per il testo delle pagine).

La costruzione è già scritta e funzionante, in `evidence/data/` (comando `python -m evidence.data.build`), e produce esattamente il formato che serve alla testa per frase: niente da aggiungere.

## 1. Le fonti

| Fonte | Da dove | Righe | Positivi / negativi | Documento |
|---|---|---:|---|---|
| **SciFact** | `allenai/scifact_entailment` (cache HF), train + validation | 1.258 | 508 / 750 | titolo + abstract biomedico (mediana 9 frasi) |
| **WiCE** | `tasksource/wice` (cache HF), train + dev + test | 5.294 | 3.100 / 2.194 | la pagina web citata da Wikipedia, una riga per elemento (mediana 68 frasi) |
| **HoVer** | claim da `dataset/HoVer/data/hover_{train,dev}_release_v1.1.json`; testo da `hotpotqa/hotpot_qa` (distractor) | 22.063 | 12.974 / 9.089 | ~10 pagine Wikipedia: quelle dell'evidenza più pagine di disturbo (mediana 46 frasi) |

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
- HotpotQA distractor, collegato con `hpqa_id`, dà per quella domanda le pagine già divise in frasi e ~8 pagine di disturbo. Il documento contiene le pagine dell'evidenza e quelle di disturbo, in ordine casuale fissato dall'id del claim: ogni pagina è titolo, a capo, testo; pagine separate da una riga vuota. L'evidenza è spesso lontana: in 6.608 positivi su 12.974 le frasi gold distano 20 o più frasi.
- Per i claim a 3–4 salti, HotpotQA non ha tutte le pagine. Con `--hover-db-fallback` le pagine mancanti si prendono dal database Wikipedia di HoVer (`dataset/HoVer/data/wiki_wo_links.db`), e la frase n si trova dividendo il testo con il nostro divisore. Senza questa opzione si perde il 45% dei claim. Misurato su 5.488 frasi di pagine presenti in entrambe le fonti: la frase così trovata è quella giusta nel 98% dei casi; nel 2% è sfasata di una frase. Queste righe hanno `meta.gold_alignment = "db_split"`, le altre `"exact"`.

## 2. Il divisore in frasi

`evidence/sentences.py`, funzione `split(testo)`, restituisce gli intervalli `[inizio, fine)` di caratteri delle frasi. È **l'unica** divisione usata in tutta la pipeline: dati, rendering, valutazione e server. Le regole:

1. si va sempre a capo dopo una riga vuota, prima di un elemento di elenco (`- `, `* `, `1. `) e dopo una riga che non finisce con punteggiatura (un titolo); una riga che finisce con virgola o trattino continua nella successiva;
2. dentro ogni blocco si usa NLTK Punkt (modello `punkt_tab`, inglese), con alcune abbreviazioni in più che il modello non conosce: vol, vols, no, fig, figs, eq, eqs, approx, ca, cf, mt, sgt, capt, gov, dept, pp, al, e.g, i.e (così "Vol. 2 No. 25" non viene spezzato);
3. frasi oltre 60 parole si spezzano su `; ` o `: ` vicino a metà, poi su `, `, poi su uno spazio;
4. frasi sotto 3 parole si uniscono alla successiva dello stesso paragrafo (alla precedente se sono le ultime);
5. oltre 512 frasi la riga si esclude.

**Le frasi di evidenza delle fonti si riportano sulla nostra divisione**: una nostra frase è evidenza se almeno metà dei suoi caratteri, o metà della frase della fonte, cade in una frase di evidenza della fonte. Così training e produzione vedono gli stessi confini. Misurato: la frase della fonte corrisponde a una sola nostra frase nel 96,5% dei casi (SciFact), 97,3% (WiCE), 98,3% (HoVer).

Se si cambia `sentences.py`, i dataset vanno ricostruiti e le teste riaddestrate: il suo sha256 sta nel manifest.

## 3. Preparazione e comando

```sh
python -c "import nltk; nltk.download('punkt_tab')"
chmod -R go-w ~/nltk_data
hf download hotpotqa/hotpot_qa --repo-type dataset --include "distractor/*"
python -m evidence.data.build --source scifact wice hover --hover-db-fallback
```

SciFact e WiCE devono essere nella cache HF; HoVer in `dataset/HoVer/`; LLM-AggreFact in `dataset/lytang--LLM-AggreFact/` (per il controllo su WiCE); il tokenizer di KnowLine-4B nella cache HF (per il filtro di lunghezza).

| Opzione | Default | Cosa fa |
|---|---|---|
| `--hover-db-fallback` | spento | pagine HoVer mancanti prese dal db (da tenere acceso) |
| `--max-tokens N` | 32.768 | scarta le righe il cui prompt supera N token del tokenizer di KnowLine-4B: documento + 2 × claim + 256 (il front end di KnowLine aggiunge 157 token, misurati) |
| `--wice-partial` | `negative` | WiCE parzialmente supportati: etichetta 0 senza loss di evidenza, o `drop` |
| `--limit N` | 0 | primi N record per fonte, per prove |

Uscita in `dataset/evidence/`: `<fonte>.jsonl`, `<fonte>.stats.json` (conteggi, scarti con motivo, lunghezze, distanze fra frasi gold, allineamento), `manifest.json` (file letti, revisioni, sha256, opzioni, commit, sha256 di `sentences.py`). Circa un minuto e mezzo.

Con il limite di default non si scarta nessuna riga per lunghezza (il prompt più lungo è di 18.580 token, in WiCE). Con `--max-tokens 8192` si scartano 75 righe di WiCE.

## 4. Il formato di una riga

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
| `sentences` | inizio e fine di ogni frase in caratteri (`document[s:e]`), dal nostro divisore |
| `evidence` | indici delle frasi di evidenza; vuoto se `label = 0` |
| `evidence_sets` | gruppi alternativi di frasi che dimostrano il claim (WiCE ne ha più di uno); `evidence` è la loro unione |
| `contra` | frasi che contraddicono il claim (SciFact), solo per analisi |
| `evidence_term` | che loss riceve la riga ([08](08_testa_e_loss.md) §6): `supervised` (positivo con evidenza), `mil_neg` (negativo: nessuna frase deve superare la soglia), `none` (nessuna loss di evidenza) |
| `doc_id` | righe con lo stesso documento (lo split si fa per `doc_id`) |
| `meta.gold_alignment` | `exact` o `db_split` (HoVer) |
| `meta.prompt_tokens` | lunghezza stimata del prompt in token |

I file si leggono anche con il benchmark della repo, perché hanno `document`, `claim`, `label`, `source`, `id`: `python benchmark.py --dataset evidence --split scifact --model knowline-4b --gpu 0`.

## 5. Statistiche dei dataset costruiti

| | SciFact | WiCE | HoVer |
|---|---:|---:|---:|
| Righe | 1.258 | 5.294 | 22.063 |
| Documenti diversi | 667 | 1.933 | 7.619 |
| Frasi per documento (mediana) | 9 | 68 | 46 |
| Frasi di evidenza per positivo (media) | 1,8 | 2,6 | 3,0 |
| Positivi con ≥ 2 frasi di evidenza | 51% | 70% | 100% |
| Positivi con frasi gold a ≥ 20 frasi di distanza | 3 | 583 | 6.608 |
| Token del prompt, mediana / massimo | 590 / 1.739 | 1.668 / 18.580 | 1.690 / 4.099 |

## 6. Controlli

| Controllo | Atteso |
|---|---|
| Le frasi coprono tutto il documento senza sovrapporsi | tutte le righe (lo verifica il build) |
| Frasi della fonte corrispondenti a una sola nostra frase | ≥ 90% per fonte (il build si ferma se sotto) |
| 30 positivi per fonte letti a mano (`python -m evidence.data.show --source hover --n 30`) | le frasi marcate dimostrano il claim in almeno 28 su 30 |
