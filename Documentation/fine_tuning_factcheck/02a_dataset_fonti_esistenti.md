# Step 2a — Le fonti esistenti: cosa sono, come si trasformano, ANLI difficile

## Obiettivo

Avere, per ogni fonte pubblica utile, coppie (documento, claim, etichetta) nel formato comune; costruire il sottoinsieme **difficile** di ANLI; preparare HotpotQA e MuSiQue come materiale di partenza per la generazione CG2C ([02b](02b_dataset_generazione_llm.md)).

## 1. Il formato comune di una riga

Prodotto da [data_prep/build_pairs.py](../../data_prep/build_pairs.py):

```json
{"id": "anli:train:0fd0abfb-659e-4453-b196-c3a64d2d8267", "source": "anli", "split": "train",
 "document": "The Parma trolleybus system ... the system presently comprises four urban routes.",
 "claim": "The trolleybus system has over 2 urban routes",
 "label": 1, "orig_label": "entailment", "doc_id": "b03f7d2b59d9d58a", "meta": {"round": "r1"}}
```

| Campo | Regola |
|---|---|
| `id` | `<fonte>:<split>:<id originale>`; unico |
| `label` | 1 = supportato in ogni parte; 0 = contraddetto o non stabilito |
| `orig_label` | etichetta originale (es. SUPPORTS / REFUTES / NOT ENOUGH INFO) |
| `doc_id` | chiave del documento; se manca, i primi 16 caratteri dello SHA-1 del documento |
| `meta` | dettagli per analisi (round, dominio, troncamento, ...) |

Documenti oltre 24.000 caratteri vengono tagliati (si tiene l'inizio) e marcati `meta.truncated = true`.

Per questo piano si aggiunge, nelle righe dove si può ricavare, `meta.neg_type`:

- `contradicted` se `orig_label` è REFUTES / contradiction / Refuted / (sintetico) contradiction;
- `not_established` se è NOT ENOUGH INFO / neutral / Not Enough Evidence / (sintetico) NOT ENOUGH INFO;
- assente per positivi e negativi generici.

Serve a controllare la quota di negativi "non stabilito" (obiettivo ≥ 40% dei negativi, [02e](02e_mix_finale.md)).

## 2. Le fonti, una per una

### 2.1 MiniCheck C2D / D2C — `minicheck`

- **Origine**: Hugging Face `lytang/C2D-and-D2C-MiniCheck`, split `c2d` (7.076 righe) e `d2c` (7.319). Licenza MIT. Colonne: `claim`, `doc`, `label` (nessuna indicazione di dove sia l'evidenza).
- **Cos'è**: dati sintetici costruiti per questo compito (procedura in [02b](02b_dataset_generazione_llm.md)). C2D: documenti scritti da GPT-4 a partire da frasi che implicano il claim, con versioni "bucate". D2C: chunk di notizie reali con riassunti e versioni ablate.
- **Righe**: `meta.subset` = `c2d` o `d2c`; tutto emesso come `split = train` (lo Step 0 lo divide 80/10/10 per doc_id).
- **Ruolo**: C2D nello stadio 1, D2C nello stadio 2.

### 2.2 ANLI — `anli`

- **Origine**: `facebook/anli`, split `train_r1..r3`, `dev_r1..r3`, `test_r1..r3`. Licenza CC BY-NC 4.0.
- **Cos'è**: NLI avversariale: annotatori umani hanno scritto ipotesi che ingannavano modelli forti. Premessa = documento corto (Wikipedia, notizie), ipotesi = claim.
- **Mappatura**: entailment → 1; neutral → 0 (`not_established`); contradiction → 0 (`contradicted`).
- **Dimensioni**: 162.865 train, 3.200 dev, 3.200 test.
- **Ruolo**: **solo il sottoinsieme difficile** (§3), nello stadio 1.

### 2.3 SummEdits — `summedits`

- **Origine**: `Salesforce/summedits`, split unico `train`, 10 domini (notizie, podcast, dialoghi, documenti legali, scientifici, ...). Licenza CC BY 4.0.
- **Cos'è**: un documento e molti riassunti: uno corretto e versioni modificate da un LLM, etichettate da umani come coerenti o no. Errori sottili (un verbo, un numero, un'attribuzione).
- **Mappatura**: label 1 = coerente.
- **Esempio**: documento "Ann: Darling, buy some butter. Mike: Ok. ..." → "Ann **asks** Mike to buy butter" (1) contro "Ann **orders** Mike to buy butter" (0).
- **Ruolo**: stadio 2 (riassunti: il tipo di testo su cui i nostri modelli sbagliano di più).

### 2.4 HoVer — `hover`

- **Origine**: file locali `Datasets/HoVer/data/hover_{train,dev}_release_v1.1.json` + testi in `Datasets/HoVer/data/wiki_wo_links.db` (SQLite, tabella `documents(id, text)`).
- **Cos'è**: claim multi-hop (2–4 pagine Wikipedia) con `supporting_facts` (titolo, indice di frase).
- **Documento**: testo introduttivo di **tutte** le pagine citate, come `Title: <titolo>\n<testo>`, **in ordine casuale** (l'ordine originale rivelerebbe la catena).
- **Mappatura**: SUPPORTED → 1; NOT_SUPPORTED → 0 (generico: unisce contraddetti e non stabiliti).
- **Ruolo**: stadio 1 (multi-hop).

### 2.5 FEVER — `fever`

- **Origine**: parquet nella cache HF (`fever/fever`, revisione `refs/convert/parquet`: split `train`, `paper_dev`, `paper_test`) + pagine Wikipedia in `Datasets/Fever/data/wiki-pages/wiki-pages/*.jsonl` (campo `lines`, frasi numerate).
- **Documento**: le pagine intere (fino a 40 frasi, più quelle di evidenza) del primo insieme di evidenza gold, non la sola frase. Per NOT ENOUGH INFO, la pagina il cui titolo è il più lungo n-gramma contenuto nel claim. Token speciali convertiti (`-LRB-` → `(`).
- **Mappatura**: SUPPORTS → 1; REFUTES → 0 (`contradicted`); NOT ENOUGH INFO → 0 (`not_established`).
- **Limiti**: scorciatoie note (claim con "not", "never" tendono a REFUTES); i NEI sono negativi facili (pagina scelta per titolo). Quota bassa nel mix.

### 2.6 VitaminC — `vitaminc`

- **Origine**: `tals/vitaminc` (train 370.653, validation, test).
- **Cos'è**: coppie contrastive da revisioni di Wikipedia: stesso claim, due versioni dell'evidenza che differiscono per un numero, una data, un nome.
- **Documento**: `Title: <pagina>\n<evidenza>` (1–2 frasi). `doc_id = vitaminc:<case_id>`: le due versioni di un caso restano nello stesso split.
- **Ruolo**: stadio 1, quota ≤ 10% (documenti cortissimi).

### 2.7 AVeriTeC — `averitec`

- **Origine**: `Datasets/AVeriTeC/data/{train,dev}.json`.
- **Documento**: le coppie domanda–risposta di evidenza scritte dagli annotatori (con la fonte), **senza** la giustificazione (che rivelerebbe il verdetto), più intestazione con autore e data del claim.
- **Mappatura**: Supported → 1; Refuted → 0; Not Enough Evidence → 0; Conflicting escluso.
- **Ruolo**: stadio 2, piccola quota (claim reali).

### 2.8 FactKG — solo come esperimento

Oggi il "documento" è una lista di triple del grafo DBpedia: formato che il modello non vedrà mai in uso. Si usa solo dopo averlo trasformato in prosa (task `verbalize_kg`, esperimento D-FKG).

### 2.9 Escluse

| Fonte | Motivo |
|---|---|
| RAGTruth | 98% dei documenti in comune con LLM-AggreFact |
| WiCE | è una fonte di LLM-AggreFact |
| SciFact | tenuta come test_ood |
| LLM-AggreFact | la licenza vieta di addestrare su di esso |

## 3. ANLI difficile — `finetune/mine_hard.py`

### 3.1 Perché

MiniCheck non usa tutto ANLI: usa ~21.000 esempi **su cui modelli di NLI già addestrati sbagliavano**. Noi selezioniamo con il **nostro** modello di partenza. I motivi:

1. **Efficienza del segnale.** Con la cross-entropy il gradiente sul margine vale P(supportato) − y. Un esempio già risolto con P = 0,99 dà una spinta di 0,01, cento volte meno di un esempio sbagliato. La maggior parte di ANLI è già risolta dal modello: costa calcolo e insegna quasi nulla.
2. **Calibrazione fuori dal training.** La spinta degli esempi facili è piccola ma non si annulla mai (la loss di un esempio è minima solo per P → 1), e su migliaia di esempi si somma. Molti esempi di ANLI sono facili per indizi superficiali (premesse corte, parole in comune evidenti): il modello impara a rispondere 0,99 a quegli indizi. Sui documenti d'uso (riassunti, risposte RAG) gli stessi indizi compaiono in casi più difficili, e il modello risulta **sovra-confidente**: dice 0,99 dove ha ragione molto meno spesso. Un LLM da 3–12B può inoltre memorizzare gli esempi di training e farne crescere il margine oltre l'incertezza reale del compito.

Esempio: dopo il training il modello dà 0,98–0,99 a quasi tutti i claim che "sembrano" supportati, ma sul dev in quella fascia ha ragione l'88% delle volte; la soglia ottimale scivola verso un estremo e non si trasferisce da un dominio all'altro.

La selezione ha però l'effetto opposto da tenere sotto controllo: un insieme di soli esempi difficili è pieno di casi ambigui e di etichette sbagliate, e può rendere il modello sotto-confidente o insegnargli rumore. Per questo la procedura (§3.2) tiene anche una quota di esempi già risolti, bilancia le etichette e prevede il label smoothing (§3.3). L'effetto netto si misura: ECE e soglia ottimale su dev_select a ogni checkpoint, ed esperimento D-HARD ([07](07_esperimenti.md)), ANLI difficile contro ANLI casuale a parità di quantità.

### 3.2 Procedura

1. Prendere `data/finetune/splits/train/anli.jsonl` (solo train).
2. Calcolare p = P(supportato) con JPT-9B e la domanda di produzione (`finetune/score.py`, oppure lo stesso codice su un file arbitrario). Premesse corte (~130 token al 95° percentile): ~1 h con vLLM su una L40S per 160k righe.
3. Classificare ogni riga:

| Classe | Condizione |
|---|---|
| errore sicuro | (label 1 e p < 0,3) oppure (label 0 e p > 0,7) |
| incerto | 0,3 ≤ p ≤ 0,7 |
| giusto | il resto |

4. Tenere **tutti** gli errori sicuri e gli incerti, più un 20% (rispetto al totale tenuto) di righe giuste scelte a caso.
5. Bilanciare 50/50 sulle etichette e limitare a 20.000 righe.
6. Scrivere `data/finetune/splits/train/anli_hard.jsonl` con `meta.hard_class` e `meta.p0` (la probabilità del modello di partenza).

Codice di riferimento:

```python
import json, random
rng = random.Random(13)
recs = {json.loads(l)["id"]: json.loads(l) for l in open("results/fc/jpt/anli_train/answers.jsonl")}
rows = [json.loads(l) for l in open("data/finetune/splits/train/anli.jsonl")]
wrong, unsure, right = [], [], []
for r in rows:
    p = recs[r["id"]]["p"]; r["meta"]["p0"] = p
    if (r["label"] == 1 and p < 0.3) or (r["label"] == 0 and p > 0.7):
        r["meta"]["hard_class"] = "wrong"; wrong.append(r)
    elif 0.3 <= p <= 0.7:
        r["meta"]["hard_class"] = "unsure"; unsure.append(r)
    else:
        r["meta"]["hard_class"] = "right"; right.append(r)
keep = wrong + unsure
rng.shuffle(right)
keep += right[: len(keep) // 4]            # 20% del totale finale = 25% del resto
pos = [r for r in keep if r["label"] == 1]; neg = [r for r in keep if r["label"] == 0]
k = min(len(pos), len(neg), 10_000)
rng.shuffle(pos); rng.shuffle(neg)
out = pos[:k] + neg[:k]
with open("data/finetune/splits/train/anli_hard.jsonl", "w") as f:
    for r in out:
        r["source"] = "anli_hard"
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(len(wrong), len(unsure), len(right), "->", len(out))
```

### 3.3 Attenzioni

- **Etichette sbagliate**: parte degli "errori sicuri" sono errori di annotazione di ANLI. Leggere 50 righe di `wrong`: se più di ~10 hanno l'etichetta discutibile, nella loss usare label smoothing su questa fonte ([04](04_loss.md) §3).
- **Con il bias attuale** (il modello dice "sì" facilmente) gli errori sicuri saranno quasi tutti negativi con p > 0,7. Il bilanciamento 50/50 al punto 5 serve a non costruire una fonte di soli negativi.
- **Selezionare con JPT e addestrare Winnow** va bene (MiniCheck selezionava con modelli diversi da quelli addestrati). In alternativa, una selezione per modello.

### 3.4 Esempio illustrativo (costruito nello stile di ANLI, non una riga reale)

> Premessa: "... The company reported a net loss of $2 million in the third quarter, compared with a profit a year earlier."
> Claim: "The company was profitable in the third quarter of the previous year." → label 1

Un modello che guarda la somiglianza superficiale tende a dire "no" (la premessa parla di perdita); "compared with a profit a year earlier" invece stabilisce il claim.

## 4. HotpotQA e MuSiQue — materiale per CG2C-MHQA

Non diventano coppie direttamente: sono il punto di partenza della generazione ([02b](02b_dataset_generazione_llm.md) §4). Si usa **solo il train**.

### 4.1 HotpotQA

- **Caricamento**: `load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")` (~90.000 domande).
- **Campi**: `id`, `question`, `answer`, `type` (`bridge` o `comparison`), `level`, `supporting_facts` = `{"title": [...], "sent_id": [...]}`, `context` = `{"title": [...], "sentences": [[frasi], ...]}` (10 paragrafi: 2 gold + 8 distrattori).
- **Righe da costruire** (`data/finetune/raw/hotpotqa_seed.jsonl`):

```json
{"id": "hotpotqa:train:5a7a06935542990198eaf050", "question": "...", "answer": "Arthur's Magazine",
 "type": "comparison", "paragraphs": [{"title": "...", "sentences": ["...", "..."], "gold": true}, ...],
 "support": [["Arthur's Magazine", 0], ["First for Women", 0]],
 "doc_id": "hotpotqa:<titoli gold ordinati e uniti da |>"}
```

- Tenere preferibilmente `type = bridge` (catene vere); `comparison` va bene in quota minore.

### 4.2 MuSiQue

- **Caricamento**: file `musique_ans_v1.0_train.jsonl` del dataset `dgslibisey/MuSiQue` (`load_dataset("json", data_files="hf://datasets/dgslibisey/MuSiQue/musique_ans_v1.0_train.jsonl")`).
- **Campi**: `id` (es. `2hop__460946_294723`: il prefisso dice il numero di hop), `paragraphs` = lista di `{idx, title, paragraph_text, is_supporting}`, `question`, `question_decomposition` = lista di sotto-domande con `answer` e `paragraph_support_idx`, `answer`, `answerable`.
- Tenere solo `answerable = true`; 2–4 hop.
- Stessa riga di 4.1, con `paragraphs` dai paragrafi (frasi ottenute con lo splitter di [03](03_rendering_formato.md) §6) e `gold = is_supporting`.

### 4.3 Decontaminazione

Le pagine sono Wikipedia, come FEVER e HoVer: il rischio verso AggreFact è basso, ma il controllo a 13-grammi ([00b](00b_split_e_protocollo.md) §3) va fatto anche qui e contro test_ood.

## 5. Comandi

```sh
conda run -n factcheck python data_prep/build_pairs.py all --splits train dev test   # ri-esecuzione completa
conda run -n factcheck python finetune/score.py --model jpt --question finetune/question.json --split train_anli   # vedi nota
conda run -n factcheck python finetune/mine_hard.py
```

Nota: per far valutare a `score.py` il solo `train/anli.jsonl`, aggiungere un'opzione `--file <percorso>` che sostituisce la lettura della cartella dello split.

## 6. Controllo a mano

Su 20 righe per fonte (5 minuti a fonte):

- le righe con label 1 hanno davvero l'evidenza nel documento?
- per le righe con label 0, il motivo è chiaro (contraddizione o informazione mancante)?
- il documento è leggibile (niente markup, niente `-LRB-`)?

## Criteri per passare allo step successivo

- `anli_hard.jsonl` prodotto; distribuzione di p e quote di `wrong` / `unsure` / `right` registrate.
- `hotpotqa_seed.jsonl` e `musique_seed.jsonl` prodotti dal solo train e decontaminati (se si farà CG2C-MHQA).
- 20 righe per fonte lette.
