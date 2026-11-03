# Passo 2 — Documenti divisi in frasi, con le frasi di evidenza

## In breve

Per addestrare e valutare la testa servono righe fatte così: un documento, un claim, l'etichetta (supportato o no), il documento **diviso in frasi** e, se il claim è supportato, **quali frasi sono l'evidenza**. Queste righe si costruiscono da dataset pubblici già presenti sulla macchina.

Si scrivono due file:
- `evidence/sentences.py`: divide un testo in frasi. È **l'unica** funzione di divisione usata ovunque (dati, training, server): se in produzione le frasi fossero tagliate in modo diverso, la testa sbaglierebbe.
- `evidence/build_pairs.py`: costruisce le righe, fonte per fonte.

Uscita: `data/evidence/pairs/<fonte>_<split>.jsonl`.

## 1. Una riga

```json
{"id": "fever:train:75397", "source": "fever", "split": "train", "label": 1,
 "claim": "Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.",
 "document": "Nikolaj Coster-Waldau\nNikolaj Coster-Waldau (born 27 July 1970) is a Danish actor ...",
 "sentences": [[0, 21], [22, 134], [135, 290]],
 "evidence": [2], "evidence_sets": [[2]],
 "evidence_quality": "gold", "doc_id": "fever:Nikolaj_Coster-Waldau"}
```

| Campo | Contenuto |
|---|---|
| `label` | 1 = supportato, 0 = non supportato |
| `sentences` | dove inizia e finisce ogni frase, in caratteri |
| `evidence` | numeri delle frasi di evidenza (vuoto se `label = 0`) |
| `evidence_sets` | se ci sono più modi alternativi di dimostrare il claim (FEVER, WiCE), uno per insieme |
| `evidence_quality` | da dove viene l'evidenza: `gold` (annotata da persone), `construct` (nota perché i dati sono generati, [03](03_dati_generati.md)), `ablation` (ricavata col modello, [03](03_dati_generati.md)), `none` |
| `doc_id` | identifica il documento: lo stesso documento non può stare sia in training sia in test |

## 2. Come si divide in frasi

`sentences.split(testo)` restituisce gli intervalli delle frasi. Regole:
1. si va sempre a capo su una riga vuota, su un elemento di elenco e dopo un titolo (riga senza punto finale);
2. dentro ogni blocco si usa NLTK `sent_tokenize`;
3. frasi oltre 60 parole: si spezzano su `;` o `:` (altrimenti sulla virgola più vicina a metà);
4. frasi sotto 3 parole: si uniscono alla successiva;
5. oltre 512 frasi il documento si esclude dal training.

**Le frasi di evidenza si riportano sempre sulla nostra divisione.** Le fonti danno le loro frasi (FEVER, HotpotQA, SciFact, WiCE), ma servono solo a sapere *dove* nel testo sta l'evidenza. Una nostra frase è evidenza se almeno metà dei suoi caratteri, o metà della frase della fonte, cade dentro una frase di evidenza della fonte.

Controllo: per ogni fonte, almeno il 90% delle frasi di evidenza originali deve corrispondere a una sola nostra frase.

## 3. FEVER

Claim su Wikipedia, con le righe delle pagine che lo dimostrano.

| Cosa | Dove |
|---|---|
| Claim di training | `~/.cache/huggingface/hub/datasets--fever--fever/snapshots/5f57…/v1.0/train/0000.parquet` — **una riga per ogni frase di evidenza**, da raggruppare |
| Claim di dev e test | `dataset/Fever/data/paper_dev.jsonl`, `paper_test.jsonl` |
| Pagine | `dataset/Fever/data/wiki-pages/wiki-pages/wiki-*.jsonl`: il campo `lines` contiene le frasi numerate di ogni pagina |

Passi:
1. Creare una volta un indice sqlite `data/evidence/fever_pages.db` (id pagina → frasi), per trovare le pagine velocemente.
2. Raggruppare l'evidenza per claim: nel parquet, righe con lo stesso `id` e lo stesso `evidence_annotation_id` formano un insieme di evidenza.
3. Documento = le pagine citate dagli insiemi, in ordine casuale (seme fisso). Ogni pagina: titolo su una riga, poi le sue frasi.
4. Il testo di FEVER è "tokenizzato": va rimesso in forma normale. `-LRB-` → `(`, `-RRB-` → `)`, `-LSB-` → `[`, `-RSB-` → `]`, ` `` ` e `''` → `"`, `--` → `–`, niente spazio prima di `, . ; : ! ? ) 's n't`. Anche nel claim.
5. SUPPORTS → positivo con evidenza. REFUTES → negativo (si annota a parte quali frasi lo contraddicono, campo `contra`, utile in valutazione). NOT ENOUGH INFO → scartato (non ha pagine).

## 4. HoVer

Claim che richiedono 2–4 pagine Wikipedia: il caso tipico di evidenza sparsa.

| Cosa | Dove |
|---|---|
| Claim | `dataset/HoVer/data/hover_train_release_v1.1.json`, `hover_dev_release_v1.1.json`; `supporting_facts` = coppie (titolo pagina, numero di frase) |
| Pagine | `dataset/HoVer/data/wiki_wo_links.db`, tabella `documents(id, text)`: il testo **non** è diviso in frasi |
| Frasi originali e pagine di disturbo | dataset `hotpot_qa`, configurazione `distractor` (va scaricato: in cache c'è solo il README), cercato con il campo `hpqa_id` |

Il test di HoVer non ha etichette: si divide il dev in due metà (una per dev, una per test).

Passi:
1. Scaricare `hotpot_qa` (distractor) e indicizzarlo per `hpqa_id`.
2. Documento = le pagine di `supporting_facts` più le ~8 pagine di disturbo di HotpotQA per quella domanda, in ordine casuale. Così il documento è lungo e le frasi di evidenza sono distanti.
3. Dove sta l'evidenza: se la pagina è in HotpotQA, la frase numero i è la i-esima frase di HotpotQA, cercata nel testo. Se la pagina non c'è (capita con 3–4 pagine), si divide il testo con `sentences.py` e si prende la frase i, solo se il conto delle frasi è plausibile; altrimenti la riga va etichettata col modello ([03](03_dati_generati.md)) o scartata.
4. SUPPORTED → positivo; NOT_SUPPORTED → negativo.

Controllo: su 100 pagine presenti in entrambi, il testo del db deve coincidere con le frasi di HotpotQA unite (a meno degli spazi).

## 5. SciFact e WiCE (solo per il test, mai per il training)

| Fonte | File | Cosa si tiene |
|---|---|---|
| SciFact | `~/.cache/huggingface/hub/datasets--allenai--scifact_entailment/snapshots/bdc3…/default/validation/0000.parquet` (340 righe): `abstract` (lista di frasi), `evidence` (numeri), `verdict` | documento = titolo + abstract; `SUPPORT` → positivo, `CONTRADICT` → negativo |
| WiCE | `~/.cache/huggingface/hub/datasets--tasksource--wice/snapshots/4905…/test.jsonl`: `evidence` (frasi), `supporting_sentences` (insiemi), `label` | `supported` → positivo; `not_supported` → negativo; `partially_supported` scartato |

Servono a vedere se la testa funziona su testi diversi da quelli del training (articoli scientifici, pagine web). WiCE fa anche parte di AggreFact: per questo non entra mai nel training.

## 6. MiniCheck C2D e D2C

`~/.cache/huggingface/hub/datasets--lytang--C2D-and-D2C-MiniCheck/…/data/c2d-*.parquet` e `d2c-*.parquet`: documenti generati, con claim ed etichetta ma **senza** evidenza. Si dividono in frasi; ai positivi l'evidenza si aggiunge al passo [03](03_dati_generati.md).

## 7. Comandi

```sh
python -m evidence.build_pairs --source fever   --split train dev test
python -m evidence.build_pairs --source hover   --split train dev
python -m evidence.build_pairs --source scifact --split validation
python -m evidence.build_pairs --source wice    --split test
python -m evidence.build_pairs --source c2d d2c
```

Ogni comando scrive anche un `.stats.json`: righe per etichetta, righe scartate e perché, frasi per documento, frasi di evidenza per riga.

## 8. Controlli

| Controllo | Atteso |
|---|---|
| Le frasi coprono tutto il documento, senza sovrapporsi | tutte le righe |
| Frasi di evidenza originali corrispondenti a una sola nostra frase | ≥ 90% per fonte |
| 30 righe positive per fonte lette a mano: le frasi marcate dimostrano il claim | ≥ 28 su 30 |
| 10 documenti FEVER letti a mano | nessun `-LRB-`, spaziatura normale |
