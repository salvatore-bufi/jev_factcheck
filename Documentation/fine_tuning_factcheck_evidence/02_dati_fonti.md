# Passo 2 — Righe con frasi ed evidenza dalle fonti

## Obiettivo

Scrivere `evidence/sentences.py` ed `evidence/build_pairs.py`, e produrre `data/evidence/pairs/<fonte>_<split>.jsonl`: una riga per (documento, claim), con il documento diviso in frasi e, per i positivi, gli indici delle frasi di evidenza.

## 1. Il formato di una riga

```json
{"id": "fever:train:75397", "source": "fever", "split": "train", "label": 1,
 "claim": "Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.",
 "document": "Nikolaj Coster-Waldau\nNikolaj Coster-Waldau (born 27 July 1970) is a Danish actor ...",
 "sentences": [[0, 21], [22, 134], [135, 290]],
 "evidence": [2], "evidence_sets": [[2]],
 "evidence_quality": "gold", "doc_id": "fever:Nikolaj_Coster-Waldau", "pages": ["Nikolaj_Coster-Waldau"]}
```

| Campo | Contenuto |
|---|---|
| `sentences` | intervalli `[inizio, fine)` in caratteri sul `document`, prodotti **sempre** da `sentences.py` |
| `evidence` | unione degli insiemi gold, come indici in `sentences` |
| `evidence_sets` | gli insiemi alternativi (FEVER, WiCE); uno solo altrove |
| `evidence_quality` | `gold`, `construct`, `ablation`, `none` |
| `doc_id` | identifica il documento per gli split (stesso documento → stesso split) |
| `contra` (facoltativo) | per i negativi FEVER REFUTES: indici delle frasi che confutano; solo per analisi |

Negativi: `label = 0`, `evidence = []`, `evidence_quality = "none"`.

## 2. `evidence/sentences.py`

Una funzione, usata da builder, rendering, valutazione e server:

```python
def split(document: str) -> list[tuple[int, int]]:
    """Intervalli [inizio, fine) delle frasi; coprono il documento a meno degli spazi fra frasi."""
```

Regole, in ordine:
1. confini forzati: `\n\n`, inizio di riga di elenco (`- `, `* `, `1. `), ogni `\n` che segue una riga senza punteggiatura finale (titoli);
2. dentro ogni blocco: NLTK `sent_tokenize` (lingua inglese), riportato sugli offset con una ricerca in avanti nel testo;
3. frasi > 60 parole: spezzate su `; ` o `: `; se non basta, sulla virgola più vicina a metà;
4. frasi < 3 parole: unite alla successiva (all'ultima precedente se sono in fondo);
5. al massimo 512 frasi; oltre, la riga si segnala e si esclude dal training.

**Le etichette gold si riportano sulla segmentazione di `sentences.py`**, mai il contrario: le frasi date dalle fonti (righe FEVER, frasi HotpotQA, abstract SciFact, WiCE) servono solo a sapere dove sta l'evidenza. Mappatura:

```text
per ogni frase gold g (intervallo di caratteri nel documento ricostruito):
    evidenza ⊇ { k : |frase_k ∩ g| ≥ 0,5 · |frase_k|  oppure  |frase_k ∩ g| ≥ 0,5 · |g| }
```

Così training e produzione vedono gli stessi confini. Controllo: su ciascuna fonte, quota di frasi gold che finiscono in esattamente una frase nostra ≥ 90%; registrare il valore.

## 3. FEVER

| Cosa | Dove |
|---|---|
| Claim di train | `~/.cache/huggingface/hub/datasets--fever--fever/snapshots/5f57…/v1.0/train/0000.parquet` (311.431 righe: **una riga per elemento di evidenza**) |
| Claim di dev e test | `dataset/Fever/data/paper_dev.jsonl`, `paper_test.jsonl` (evidenza annidata: `[[[annot_id, ev_id, pagina, riga], …], …]`) |
| Pagine | `dataset/Fever/data/wiki-pages/wiki-pages/wiki-*.jsonl` (109 file); campo `lines` = righe `"<n>\t<frase>\t<link>…"` separate da `\n` |

Procedura:
1. **Indice delle pagine**: una volta, un sqlite `data/evidence/fever_pages.db` con (id pagina → `lines`), dalle 109 file.
2. **Raggruppare l'evidenza**: nel parquet di train, per `id` di claim raggruppare per `evidence_annotation_id`; ogni gruppo è un insieme di (`evidence_wiki_url`, `evidence_sentence_id`). In dev/test lo stesso dalla lista annidata.
3. **Documento**: le pagine citate da almeno un insieme, in ordine casuale (seme fisso dall'id). Ogni pagina = titolo (id con `_` → spazio) su una riga, poi le sue righe non vuote, una dopo l'altra separate da spazio; pagine separate da `\n\n`.
4. **Detokenizzazione** delle righe (il testo di FEVER è tokenizzato): `-LRB-` → `(`, `-RRB-` → `)`, `-LSB-` → `[`, `-RSB-` → `]`, `-LCB-` → `{`, `-RCB-` → `}`, `` `` `` e `''` → `"`, `--` → `–`, e spazi prima di `, . ; : ! ? ) ' 's n't` tolti. Anche il claim. Mentre si costruisce il documento, si conserva l'intervallo di caratteri di ogni riga (serve alla mappatura di §2).
5. **Etichette**: SUPPORTS → `label = 1`, `evidence_quality = gold`, insiemi riportati su `sentences` (§2). REFUTES → `label = 0`, `contra` = frasi delle righe citate. NOT ENOUGH INFO → scartato (non ha pagine).
6. Righe con un insieme che cita una pagina assente dall'indice: scartate e contate.

## 4. HoVer

| Cosa | Dove |
|---|---|
| Claim | `dataset/HoVer/data/hover_train_release_v1.1.json` (18.171), `hover_dev_release_v1.1.json`; campi `claim`, `supporting_facts` = [(titolo, indice di frase)], `label`, `num_hops`, `hpqa_id` |
| Pagine | `dataset/HoVer/data/wiki_wo_links.db`, tabella `documents(id, text)`: testo introduttivo **unito**, senza confini di frase |
| Segmentazione originale e distrattori | `hotpot_qa`, config `distractor` (da scaricare: in cache c'è solo il README): `context` = titoli + liste di frasi, per la domanda `hpqa_id` |

Il test di HoVer non ha etichette: il dev si divide a metà per `doc_id` (metà dev, metà test, [04](04_split_e_mix.md)).

Procedura:
1. `hf download hotpotqa/hotpot_qa --repo-type dataset` (config `distractor`, train e validation) e indice `hpqa_id` → `context`.
2. **Documento**: le pagine di `supporting_facts` più i paragrafi distrattori del `context` HotpotQA della domanda `hpqa_id` (8 circa: rendono il documento lungo e realistico), in ordine casuale con seme fisso. Ogni pagina = titolo su una riga, poi il testo; pagine separate da `\n\n`.
3. **Dove sta l'evidenza**: per una pagina presente nel `context` HotpotQA, la frase gold i è la i-esima frase di HotpotQA; se ne cerca l'intervallo nel testo della pagina (ricerca esatta, poi normalizzando gli spazi). Per una pagina assente dal `context` (accade nei 3–4 hop), si segmenta il testo del db con `sentences.py` e si prende la i-esima frase, solo se il numero di frasi coincide con quello atteso dal resto dei dati (indice gold < numero di frasi); altrimenti la riga va a `evidence_quality = ablation` (etichettata al [03](03_dati_generati.md)) o si scarta.
4. Le frasi gold si riportano su `sentences` con la mappatura di §2. Gli indici si calcolano **dopo** il mescolamento delle pagine.
5. SUPPORTED → `label = 1`, `gold`; NOT_SUPPORTED → `label = 0`.

Controllo specifico: su 100 pagine presenti in entrambi, il testo del db coincide con la concatenazione delle frasi HotpotQA (a meno degli spazi). Se no, si usa il testo HotpotQA per quelle pagine.

## 5. SciFact (solo test fuori dominio)

`~/.cache/huggingface/hub/datasets--allenai--scifact_entailment/snapshots/bdc3…/default/validation/0000.parquet` (340 righe): `abstract` (lista di frasi), `evidence` (indici), `verdict` (`SUPPORT`, `CONTRADICT`, `NEI`), `title`.

Documento = titolo + `\n` + frasi dell'abstract separate da spazio. Si tengono `SUPPORT` (positivi con evidenza) e `CONTRADICT` (negativi). Il train di SciFact **non** si usa.

## 6. WiCE (solo test fuori dominio)

`~/.cache/huggingface/hub/datasets--tasksource--wice/snapshots/4905…/test.jsonl`: `evidence` (lista di frasi della pagina citata), `supporting_sentences` (lista di insiemi di indici), `label` (`supported`, `partially_supported`, `not_supported`).

Documento = frasi di `evidence` unite da `\n` se sono righe di intestazione, da spazio altrimenti (si conservano gli intervalli). `supported` → positivo con `evidence_sets` = `supporting_sentences`; `not_supported` → negativo; `partially_supported` → scartato. WiCE è una fonte di AggreFact: mai in training.

## 7. MiniCheck C2D (senza evidenza)

`~/.cache/huggingface/hub/datasets--lytang--C2D-and-D2C-MiniCheck/snapshots/1f69…/data/c2d-*.parquet` (7.076 righe: `claim`, `doc`, `label`). Si segmenta il `doc`; positivi con `evidence_quality = none` (diventano `ablation` al [03](03_dati_generati.md)); negativi così come sono. Idem per D2C.

## 8. Comando

```sh
python -m evidence.build_pairs --source fever  --split train dev test
python -m evidence.build_pairs --source hover  --split train dev
python -m evidence.build_pairs --source scifact --split validation
python -m evidence.build_pairs --source wice   --split test
python -m evidence.build_pairs --source c2d d2c
```

Ogni esecuzione scrive anche `data/evidence/pairs/<fonte>_<split>.stats.json`: righe per etichetta, scartate (con motivo), frasi per documento (p50, p95), frasi gold per riga, quota di frasi gold mappate su una sola frase nostra.

## 9. Controlli prima di andare avanti

| Controllo | Atteso |
|---|---|
| `sentences` coprono `document` senza sovrapposizioni (spazi a parte) | tutte le righe |
| Gold mappate su una sola frase nostra | ≥ 90% per fonte |
| Indici di evidenza < numero di frasi | tutte le righe |
| Lettura a mano: 30 righe positive per fonte, le frasi marcate stabiliscono il claim | ≥ 28 su 30 |
| Lettura a mano: 10 documenti FEVER detokenizzati | nessun `-LRB-`, spaziatura normale |
