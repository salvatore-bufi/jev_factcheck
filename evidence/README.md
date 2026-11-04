# evidence — dati per il fine-tuning con evidenza

> Tutta la pipeline della testa di evidenza (split, rendering, hook, estrazione, testa, training, valutazione, server) è spiegata in [code_report.md](code_report.md). Questo file descrive solo la costruzione dei dataset.

Codice per il piano [Documentation/fine_tuning_factcheck_evidence_complete/](../Documentation/fine_tuning_factcheck_evidence_complete/testa_per_frase/00_overview.md) (documentazione dell'implementazione: [Documentation/fine_tuning_factcheck_evidence/](../Documentation/fine_tuning_factcheck_evidence/README.md)): un decision model che, oltre al verdetto, indica le frasi del documento che dimostrano il claim.

Per ora c'è la prima parte: **portare i dataset in un formato unico**, pronto per la loss. Ogni riga dice dove inizia e finisce ogni frase del documento e quali frasi sono l'evidenza. Split, mix, rendering per modello e training vengono dopo (§7).

## 1. Cosa c'è

```text
evidence/
  sentences.py             divisione in frasi: l'unica usata in tutta la pipeline (dati, rendering, server)
  data/
    schema.py              il formato della riga, come si costruisce e come si valida
    sources/               una fonte per file: scifact.py, wice.py, hover.py
    length.py              lunghezza del prompt in token (filtro --max-tokens)
    stats.py, io.py        statistiche, scrittura dei file, manifest
    build.py               costruisce i dataset           python -m evidence.data.build
    show.py                stampa righe con l'evidenza    python -m evidence.data.show
tests/test_evidence_data.py   test offline (niente download, niente GPU)
dataset/evidence/             l'uscita (non va su git)
```

## 2. Le fonti

| Fonte | Da dove | Righe | Cosa porta |
|---|---|---:|---|
| **SciFact** | `allenai/scifact_entailment` (cache HF), train + validation | 1.258 | abstract biomedici corti; SUPPORT → positivo con le frasi di rationale; CONTRADICT e NEI → negativo |
| **WiCE** | `tasksource/wice` (cache HF), train + dev + test | 5.294 | pagine web citate da Wikipedia, documenti lunghi (mediana 68 frasi); più insiemi alternativi di evidenza |
| **HoVer** | claim da `dataset/HoVer/`, testo da `hotpotqa/hotpot_qa` (distractor) | 22.063 | claim a 2–4 hop, evidenza su più pagine con pagine di disturbo: il caso dell'evidenza lontana |

Per HoVer, HotpotQA dà le pagine già divise in frasi (con la stessa numerazione delle frasi di evidenza di HoVer) e ~8 pagine di disturbo per claim. Le pagine che HotpotQA non ha (claim a 3–4 hop) si prendono da `dataset/HoVer/data/wiki_wo_links.db` con `--hover-db-fallback`: senza, si perde il 45% dei claim.

**Attenzione a WiCE:** è anche una fonte del test LLM-AggreFact. Ogni riga WiCE ha `meta.aggrefact_overlap`:
- `"dev"` o `"test"` se la stessa pagina è in AggreFact dev o test;
- `false` se non c'è;
- `null` se AggreFact non è disponibile in locale.

Tutto il dev di WiCE è in AggreFact dev, tutto il test di WiCE in AggreFact test, più 2 righe del train. Quando si fa lo split, queste righe non devono finire in training.

## 3. Preparazione (una volta)

```sh
# dati della divisione in frasi (NLTK punkt_tab)
python -c "import nltk; nltk.download('punkt_tab')"
chmod -R go-w ~/nltk_data          # NLTK rifiuta una cartella dati scrivibile da altri utenti
                                   # (in alternativa: NLTK_DATA=<copia privata di nltk_data>)

# HotpotQA distractor (per HoVer)
hf download hotpotqa/hotpot_qa --repo-type dataset --include "distractor/*"
```

Gli altri dati devono già essere in locale:
- SciFact e WiCE nella cache HF;
- HoVer in `dataset/HoVer/`;
- AggreFact in `dataset/lytang--LLM-AggreFact/`, per il controllo su WiCE;
- il tokenizer di KnowLine-4B nella cache HF, per il filtro di lunghezza.

## 4. Costruire i dataset

```sh
python -m evidence.data.build --source scifact wice hover --hover-db-fallback
```

Ci vuole circa un minuto e mezzo. Opzioni:

| Opzione | Default | Cosa fa |
|---|---|---|
| `--source` | tutte | quali fonti costruire |
| `--hover-db-fallback` | spento | pagine HoVer mancanti in HotpotQA prese dal db di HoVer (consigliato: tenerlo acceso) |
| `--max-tokens N` | 32768 | scarta le righe il cui prompt supera N token (`0` = nessun filtro) |
| `--length-model` | `knowline-4b` | modello del cui tokenizer si contano i token |
| `--wice-partial` | `negative` | WiCE "parzialmente supportato": etichetta 0 senza loss di evidenza (`negative`) oppure scartato (`drop`) |
| `--limit N` | 0 | solo i primi N record per fonte, per prove veloci |
| `--out DIR` | `dataset/evidence` | dove scrivere |
| `--min-alignment` | 0.9 | il build fallisce se meno di questa quota di frasi di evidenza originali corrisponde a una sola nostra frase |

Il filtro di lunghezza stima il prompt come `token del documento + 2 × token del claim + 256`. Il claim compare due volte per via del preambolo. I 256 token coprono istruzioni, domanda e token di chat: il front end di KnowLine ne aggiunge 157, misurati. Il tokenizer è quello di Qwen3.5, comune ai 10 modelli dello studio. Con 32.768 oggi non si toglie nessuna riga: il prompt più lungo è di 18.580 token, in WiCE. Con `--max-tokens 8192` si tolgono 75 righe di WiCE, con 4096 se ne tolgono 503 di WiCE e 2 di HoVer.

Uscita in `dataset/evidence/`:

| File | Contenuto |
|---|---|
| `<fonte>.jsonl` | le righe |
| `<fonte>.stats.json` | righe per etichetta e split d'origine; righe scartate e perché; frasi e token per documento; frasi di evidenza per positivo; distanza fra prima e ultima frase di evidenza; quota di allineamento |
| `manifest.json` | file letti e revisioni, sha256 dell'uscita, opzioni, commit git, sha256 di `sentences.py` |

## 5. Il formato di una riga

```json
{"id": "hover:train:330ca632-…", "source": "hover", "orig_split": "train",
 "claim": "…", "document": "Kristian Zahrtmann\nKristian Zahrtmann was …\n\nOssian Elgström\n…",
 "label": 1,
 "sentences": [[0, 61], [62, 140], …],
 "evidence": [1, 9, 31], "evidence_sets": [[1, 9, 31]],
 "contra": [],
 "evidence_quality": "gold",
 "evidence_term": "supervised",
 "doc_id": "hover:5ab7a86d5542995dae37e986",
 "meta": {"orig_label": "SUPPORTED", "gold_alignment": "exact", "doc_tokens": 1534, "prompt_tokens": 1830, …}}
```

| Campo | Significato |
|---|---|
| `label` | 1 = il documento supporta il claim, 0 = no |
| `sentences` | inizio e fine di ogni frase, in caratteri (`document[s:e]`), sempre prodotte da `evidence/sentences.py` |
| `evidence` | indici delle frasi di evidenza; vuoto se `label = 0` |
| `evidence_sets` | gruppi alternativi di frasi che dimostrano il claim (WiCE ne ha più di uno); `evidence` è la loro unione |
| `contra` | frasi che contraddicono il claim (SciFact), solo per analisi |
| `evidence_term` | che termine di loss riceve la riga ([08_testa_e_loss.md](../Documentation/fine_tuning_factcheck_evidence_complete/testa_per_frase/08_testa_e_loss.md) §3): `supervised` (le frasi di evidenza sopra la soglia, le altre sotto), `mil_neg` (nessuna frase sopra la soglia), `none` (nessun termine di evidenza) |
| `doc_id` | righe con lo stesso documento: lo split si fa per `doc_id` |
| `meta.gold_alignment` | `exact` (frasi della fonte) o `db_split` (pagina HoVer presa dal db e divisa da noi) |
| `meta.doc_tokens`, `meta.prompt_tokens` | lunghezze in token (tokenizer di `--length-model`) |

Le frasi di evidenza delle fonti vengono riportate sulla **nostra** divisione in frasi: una nostra frase è evidenza se almeno metà dei suoi caratteri, o metà della frase della fonte, cade nella frase di evidenza. Così training e produzione vedono gli stessi confini.

Nella riga non ci sono pesi e posizioni in token: dipendono dal mix e dal modello e si aggiungono nei passi successivi.

**I file si leggono anche con il benchmark**, perché hanno `document`, `claim`, `label`, `source`, `id`:

```sh
python benchmark.py --dataset evidence --split scifact --model knowline-4b --gpu 0 --limit 32
```

## 6. Controllare i dati

```sh
python tests/test_evidence_data.py                              # test offline
python -m evidence.data.show --source hover --n 5               # 5 positivi a caso, evidenza marcata con >>
python -m evidence.data.show --source wice --label 0 --n 3      # negativi
python -m evidence.data.show --source scifact --id scifact:train:2:13734012 --full
```

Il piano chiede di leggere a mano 30 righe positive per fonte: le frasi marcate devono dimostrare il claim (almeno 28 su 30). Per HoVer conviene guardare anche le righe con `gold_alignment = db_split`, quelle che dipendono dalla nostra divisione delle pagine del db.

Divisione in frasi (`evidence/sentences.py`):
- si va a capo dopo una riga vuota, prima di un elemento di elenco e dopo una riga senza punteggiatura finale (un titolo);
- dentro ogni blocco si usa NLTK Punkt, con qualche abbreviazione in più (Vol., No., Fig., …);
- le frasi oltre 60 parole si spezzano;
- quelle sotto 3 parole si uniscono alla successiva.

**Se si cambia `sentences.py`, i dataset vanno ricostruiti** (e, più avanti, le teste riaddestrate): il suo sha256 sta nel manifest.

## 7. Cosa fare adesso

1. Rendere privata `~/nltk_data` (§3) e ricostruire i dataset con `--hover-db-fallback`, se non è già fatto.
2. Leggere a mano 30 positivi per fonte con `show` (§6) e annotare gli errori.
3. Scegliere il limite di token per il training (`--max-tokens`) in base alla memoria disponibile ([06](../Documentation/fine_tuning_factcheck_evidence_complete/testa_per_frase/06_rendering_ed_estrazione.md) §3 propone 16.384 per l'estrazione sui 4B).
4. **Prossimo codice da scrivere**, nell'ordine del piano:
   - **split** train/val/test per `doc_id`, tenendo fuori dal training le righe WiCE con `aggrefact_overlap` ([04](../Documentation/fine_tuning_factcheck_evidence_complete/testa_per_frase/04_split_e_mix.md));
   - **mix** con pesi e quote ([04](../Documentation/fine_tuning_factcheck_evidence_complete/testa_per_frase/04_split_e_mix.md) §3);
   - **rendering** con il preambolo e le posizioni in token di ogni frase, per KnowLine-4B ([05](../Documentation/fine_tuning_factcheck_evidence_complete/testa_per_frase/05_formato_prompt.md), [06](../Documentation/fine_tuning_factcheck_evidence_complete/testa_per_frase/06_rendering_ed_estrazione.md));
   - **estrazione dei vettori** e **training della testa** ([09](../Documentation/fine_tuning_factcheck_evidence_complete/testa_per_frase/09_training.md)).

### Aggiungere una fonte

1. Creare `evidence/data/sources/<nome>.py` con `NAME`, `load_raw(opts, prov)` (legge i file originali e li unisce) e `convert(raw, opts)`. `convert` restituisce `schema.assemble(...)` con le frasi di evidenza della fonte come intervalli di caratteri, oppure `schema.Skip("motivo")`.
2. Registrarla in `registry()` di `evidence/data/sources/__init__.py`.
3. Aggiungere a `tests/test_evidence_data.py` un test con 2–3 righe grezze scritte a mano.
