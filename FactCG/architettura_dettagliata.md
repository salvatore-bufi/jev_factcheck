# Architettura del modulo di Fact-Checking (FactCG, no retrieval)

> Documento di riferimento sull'architettura effettiva della pipeline, ricostruito
> leggendo il codice sorgente (`chunking.py`, `factcg_client.py`, `main.py`,
> `config.py`, `run_identity.py`). Dove il comportamento reale differisce da
> quanto suggerito dalla documentazione discorsiva (`README.md`,
> `overview.md`, `FactCG.md`) o da valori "attesi", la discrepanza è
> segnalata esplicitamente.

## 1. Obiettivo del sistema

Dato un insieme di **claim** e un corpus di **documenti** locali (`.txt`), il
sistema produce per ciascun claim un **verdetto binario** — `Supported` o
`Refuted` — confrontando il claim **direttamente** con ogni documento del
dataset, senza alcuna fase di retrieval a monte: stessa architettura
generale di `MiniCheck/` (vedi `MiniCheck/architettura_dettagliata.md`), ma
con `FactCG-DeBERTa-v3-Large` come classificatore al posto di
`bespoke-minicheck:7b` via Ollama.

È una reimplementazione del percorso di inferenza DeBERTa del progetto
**FactCG** (derenlei/FactCG — *"FactCG: Enhancing Fact Checkers with
Graph-Based Multi-Hop Data"*, NAACL 2025, vedi `FactCG.md`), che usa il
checkpoint pubblicato `yaxili96/FactCG-DeBERTa-v3-Large` caricato **in
locale in memoria** tramite `transformers`, non un LLM generativo servito
via HTTP. Le differenze rispetto all'originale sono riassunte nella
sezione 5.

## 2. Flusso end-to-end

```
input/<dataset>/documents/*.txt        input/<dataset>/claims/*.json
        |                                       |
        v                                       |
[1] Chunking sentence-aware                     |
    (chunking.py)                               |
        |                                       v
        v                              [2] Per ogni (documento, claim):
[3] Chunk raggruppati in batch                i chunk sono raggruppati in batch
    (batch_size=8), formattati con                e accoppiati al claim
    INSTRUCTION_TEMPLATE                         (main.score_document)
        |
        v
[4] Un solo forward pass per batch su
    FactCG-DeBERTa-v3-Large (transformers,
    locale, GPU/CPU) -> softmax(logits)
        |
        v
[5] Per documento: si tiene il chunk con support_prob massimo
    Tra tutti i documenti: si tiene quello con support_prob massimo
        |
        v
[6] verdetto = "Supported" se support_prob > soglia (0.5), altrimenti "Refuted"
        |
        v
[7] outputs/<dataset>/<run_id>/
    claim_<id>.json + summary.json + run_params.json
```

Il modello viene **caricato una sola volta per run** (`main.main()`,
`FactCGModel(...)` istanziato prima del ciclo sui claim) e riutilizzato per
ogni chiamata a `score_document`/`score_pairs` — a differenza di MiniCheck,
dove ogni chunk genera una chiamata HTTP indipendente a un processo Ollama
separato.

## 3. Architettura Dettagliata

### 3.1 Chunking (`chunking.py`)

- **Sentence-tokenization "piatta"**: `nltk.sent_tokenize(text)` sull'intero
  testo del documento in un colpo solo — a differenza di
  `MiniCheck/chunking.py` (`sent_tokenize_with_newlines`), qui **non** c'è
  uno split preliminare sui `\n` né una preservazione esplicita dei confini
  di paragrafo: `sent_tokenize` di NLTK tratta il testo come un unico
  blocco continuo.
- **Packing greedy**: identico nella struttura a MiniCheck — le frasi
  vengono accumulate finché la frase successiva non farebbe superare
  `chunk_size` **parole**; a differenza di MiniCheck però il conteggio usa
  `nltk.word_tokenize(sentence)` (tokenizzazione vera, non un semplice
  `.split()`), lo stesso proxy usato dall'`Inferencer.chunking_src`
  originale. Una singola frase più lunga del limite diventa comunque un
  chunk a sé (mai spezzata a metà).
- **Post-processing**: `[c.strip() for c in chunks if c.strip()]` — più
  semplice del corrispondente in MiniCheck (nessun marcatore `\n` da
  ripulire, dato che non ne vengono inseriti).
- **`chunk_size` di default = 550 parole** (`config.CHUNK_SIZE`), molto più
  piccolo delle ~18k parole di MiniCheck: qui il vincolo non è un context
  window LLM da 32k, ma `max_length = 2048` **token** del tokenizer DeBERTa
  (`config.MAX_LENGTH`, passato a `Inferencer` upstream) — 550 parole è lo
  stesso default conservativo della libreria originale (`chunking_src`).

### 3.2 Modulo di scoring per chunk (`factcg_client.py`)

- **`INSTRUCTION_TEMPLATE`**: stringa identica a `factcg/utils.py`
  dell'originale — un unico prompt testuale per coppia (documento, claim):
  `"{documento}\n\nChoose your answer: based on the paragraph above can we
  conclude that \"{claim}\"?\n\nOPTIONS:\n- Yes\n- No\nI think the answer is "`.
  ⚠️ Nonostante il formato "istruzione" (stile FLAN, condiviso con la
  variante FactCG-FT5 generativa), il modello qui è **DeBERTa**, che non
  genera testo libero: l'intero prompt formattato è semplicemente l'input
  di un classificatore a sequenza singola (`type_vocab_size: 0` nel
  `config.json` del modello — non un'encoding a coppia di frasi/segmenti).
- **Caricamento modello** (`FactCGModel.__init__`): `AutoConfig` con
  `num_labels=2`, `problem_type="single_label_classification"`, poi
  `AutoModelForSequenceClassification.from_pretrained(model_id, ...)` e
  `AutoTokenizer.from_pretrained(model_id, use_fast=True, ...)`, entrambi
  con `cache_dir=config.CACHE_DIR` (`FactCG/cache/`, **non condivisa** con
  `AICCTU/data/`). Device: `torch.device(device or ("cuda" if
  torch.cuda.is_available() else "cpu"))` — rilevamento automatico,
  override possibile via `--device`. Modello messo in `.eval()`.
- **`score_pairs`** (`@torch.no_grad()`): formatta l'intero batch con
  `INSTRUCTION_TEMPLATE`, tokenizza con `padding="longest"` (non
  `max_length` fisso: il padding è relativo al batch) e
  `truncation=True, max_length=self.max_length`, un solo forward pass per
  batch, poi `softmax(logits, dim=-1)`. Ritorna sia `support_probs`
  (`probs[:, 1]`, cioè **P(classe 1 = "Supported")**) sia `all_probs`
  (entrambe le probabilità softmax) per ogni coppia.
- **Convenzione dell'indice di classe**: l'indice 1 = "Supported" è
  verificato (nel docstring del modulo) contro l'esempio numerico ufficiale
  del progetto originale (claim inconsistente ≈0.065, consistente ≈0.784) —
  coerente direttamente con la convenzione `gold_label` di LLM-AggreFact
  usata da `scripts/prepare_aggregatefact.py` (1 = claim supportato).
- **Deviazione deliberata dal codice upstream**: `Inferencer` originale
  riassegna `tokenizer.pad_token = tokenizer.eos_token` — un workaround
  necessario per T5 (che non ha un pad token nativo) ereditato da un
  percorso di codice condiviso tra le tre varianti del modello. Questo
  client **non lo fa**: DeBERTa ha già un `pad_token` corretto (`'[PAD]'`,
  id 0), mentre `eos_token` qui alias erroneamente a `'[SEP]'` (id 2). Le
  posizioni di padding sono comunque mascherate dall'`attention_mask`
  indipendentemente dal token id letterale, quindi il comportamento
  numerico non cambia — ma il tokenizer resta coerente con se stesso invece
  di ereditare un workaround pensato per un'altra architettura.

### 3.3 Batching ed error handling (`main.py`)

- **`score_document`**: i chunk di un documento sono processati in batch di
  `args.batch_size` (default **8**, `config.BATCH_SIZE`) — scelto come
  "default conservativo per una GPU da 16GB a `max_length` token/esempio"
  (docstring `config.py`); ogni batch produce un forward pass unico via
  `model.score_pairs`.
- **Nessun retry, nessuna temperatura**: essendo un classificatore
  deterministico (softmax su logit calcolati da un forward pass), non
  esiste una nozione di "risposta non parsabile" — ogni chiamata produce
  sempre una probabilità valida. Coerentemente, `parse_args` non espone né
  `--temperature` né `--retries` (a differenza sia di AICCTU sia di
  MiniCheck).
- **A livello di claim**, un'eccezione imprevista viene comunque catturata
  in `main.main()`: il record scritto contiene solo `claim_id`, `claim` ed
  `error` (traceback troncato a 3 frame), e il run **prosegue** con il
  claim successivo — stesso pattern di AICCTU/MiniCheck.
- **`--resume`**: rilancia solo i claim il cui file di output non esiste
  ancora o contiene un campo `error`; i claim con un output valido sono
  considerati "fatti" e non rigenerati. Il modello viene comunque
  **ricaricato** all'avvio anche se tutti i claim risultano già completati
  (`FactCGModel(...)` è istanziato prima del ciclo, non lazy).

### 3.4 Selezione del documento migliore e verdetto (`main.py`)

Identico a MiniCheck (stesso `score_document`/`check_claim` a livello di
struttura, solo lo scoring per chunk cambia): per ogni documento si tiene il
chunk con `support_prob` massimo; tra tutti i documenti del dataset si
tiene quello con `support_prob` massimo in assoluto (`best_source`). Il
verdetto è `"Supported"` **se e solo se** `support_prob > args.threshold`
(default **0.5**, confronto stretto), altrimenti `"Refuted"` — nessuna
terza classe, coerente con `config.LABELS = ["Supported", "Refuted"]`.

### 3.5 Identità deterministica e namespacing (`run_identity.py`)

Come MiniCheck (e a differenza di AICCTU), **non esiste un
`index_identity`**: nessun embedding/indice da persistere prima del run.
Implementazione **identica** a quella di MiniCheck/AICCTU (stesso file,
copiato/adattato):

- **`run_identity(codebase, dataset, params)`** → hash SHA-256 (primi 8
  esadecimali) su *tutti* gli iperparametri che influenzano il verdetto:
  `llm_model`, `max_length`, `chunk_size`, `threshold`, oltre a `dataset` e
  `codebase`. ⚠️ Nota: `batch_size` **non** è incluso nei parametri
  hashati (`main.resolve_run` non lo passa a `run_identity`), pur essendo
  un flag CLI (`--batch-size`) — coerente col fatto che il batching è solo
  un dettaglio di esecuzione che non altera i punteggi prodotti (ogni
  esempio nel batch è comunque scored indipendentemente, salvo il
  `padding="longest"` relativo al batch, che in teoria potrebbe introdurre
  variazioni numeriche trascurabili tra batch di composizione diversa a
  parità di `max_length`/truncation).
- Cartella risultante: `outputs/<dataset>/<codebase>__<model-slug>--<hash8>/`,
  dove `codebase = config.MODEL_NAME = BASE_DIR.name` (cioè `"FactCG"`).
- **`slugify`**/**`short_hash`**/**`write_manifest`**: stessa
  implementazione di AICCTU/MiniCheck.

### 3.6 Schema di output

Identico nella forma a quello di MiniCheck (stesso schema, così
`evaluation/` legge entrambe le codebase senza modifiche), con l'aggiunta
di `probs` (entrambe le probabilità softmax) per ogni chunk:

| Campo | Contenuto |
|---|---|
| `claim_id`, `claim` | il claim in input |
| `claim_metadata` | tutti gli altri campi del claim in input (mai passati al modello) |
| `verdict` | `"Supported"` \| `"Refuted"` (soglia 0.5 su `support_prob`) |
| `support_prob` | probabilità di supporto massima tra tutti i documenti/chunk, in [0, 1] (softmax classe 1) |
| `best_source` | documento/chunk che ha prodotto `support_prob` (`file`, `chunk_index`, `text`) |
| `document_scores` | punteggio per **ogni** documento, con dettaglio chunk-per-chunk (`support_prob`, `probs`: entrambe le probabilità softmax) |
| `run_info` | modello, `max_length`, `chunk_size`, `threshold`, timestamp, durata |

Nessun `attempts`/`parse_ok` per chunk (a differenza di MiniCheck): un
forward pass deterministico non ha nulla da ritentare.

## 4. Namespacing dei dati a livello di repo

`input/` e `outputs/` vivono **un livello sopra** la cartella di questa
codebase (`config.REPO_ROOT = BASE_DIR.parent`), condivisi tra più
implementazioni e organizzati per **dataset** (`--dataset`, default
`"example"`):

- `input/<dataset>/claims/*.json` — claim da verificare;
- `input/<dataset>/documents/*.txt` — corpus su cui verificarli;
- `outputs/<dataset>/<run_id>/` — output per claim + `summary.json` +
  `run_params.json`.

Non esiste `data_store/` per questa codebase. `FactCG/cache/` (pesi HF
scaricati al primo avvio) è invece **locale e non namespaced**: vive dentro
`FactCG/`, non alla root del repo, e non è condivisa con le altre codebase.

## 5. Differenze rispetto al progetto FactCG originale

| Aspetto | Originale (derenlei/FactCG, `Inferencer(use_hf_ckpt=True)`) | Questa implementazione |
|---|---|---|
| Caricamento del checkpoint | Stesso repo HF (`yaxili96/FactCG-DeBERTa-v3-Large`) | Identico: `AutoModelForSequenceClassification.from_pretrained(...)` |
| `pad_token` | Riassegnato a `eos_token` (workaround per T5 in un percorso condiviso) | Lasciato quello nativo di DeBERTa (`[PAD]`); `attention_mask` rende il comportamento numericamente equivalente (§ 3.2) |
| Chunking | `Inferencer.chunking_src`, greedy sentence-packing a `chunk_size` parole (`nltk.word_tokenize`) | Stesso algoritmo (`chunking.py`), stesso default 550 parole |
| Ambito di applicazione | Libreria standalone, su coppie documento/claim fornite dal chiamante | Modulo del repo: scorre l'intero `documents/` per ogni claim di `input/<dataset>/claims/`, nessuna selezione a monte |
| Batching | Gestito dal chiamante | `--batch-size` (default 8) su tutti i chunk di un documento, non incluso nell'hash di `run_identity` (§ 3.5) |
| Output | `support_prob`/`pred_label` per la coppia valutata | Un JSON per claim con verdetto, punteggio per ogni documento/chunk (entrambe le probabilità softmax) e info di run |
| Identità/riproducibilità | Non prevista (chiamata di libreria) | `run_identity.py`: stessa configurazione → stessa cartella, manifest `run_params.json` |

Parametri invariati rispetto all'originale: `INSTRUCTION_TEMPLATE`,
`max_length = 2048`, soglia di verdetto 0.5, convenzione classe 1 =
"Supported".

## 6. Osservazioni / punti di attenzione emersi dal codice

- **`description_claude.md`, citato in quattro punti** (`README.md`,
  `overview.md`, il docstring di `config.py` e quello di
  `factcg_client.py`, tutti come riferimento a "note originali su dove
  trovare il checkpoint") **non esiste in questa working copy del repo** —
  verificato con una ricerca diretta nella cartella `FactCG/`. Le
  informazioni che quel file avrebbe dovuto documentare sono comunque
  presenti, duplicate, nel corpo di `README.md`/`overview.md` (sezione
  "Perché si scarica da Hugging Face invece di usare
  `ckpt/factcg_dbt.ckpt`"), quindi non è un blocco funzionale — ma un
  riferimento incrociato rotto da segnalare a chi mantiene la
  documentazione.
- `ckpt/factcg_dbt.ckpt` (il checkpoint PyTorch Lightning originale, con
  prefisso di stato `base_model.*` e quattro head: `bin_layer`,
  `bin_finetuned_layer`, `tri_layer`, `reg_layer`) è presente nella cartella
  `ckpt/` ma **non viene mai caricato** dal codice: `factcg_client.py`
  scarica sempre il checkpoint equivalente già convertito da Hugging Face
  Hub (`yaxili96/FactCG-DeBERTa-v3-Large`) — coerente con
  `Inferencer(use_hf_ckpt=True)`, il default upstream.
- Il `batch_size` altera solo l'ordine/raggruppamento delle chiamate al
  modello, non incluso nell'identità di run (§ 3.5): due run con
  `--batch-size` diverso ma stessi altri parametri **condividono lo stesso
  `run_id`** e quindi la stessa cartella di output — un run successivo con
  `--resume` e `batch_size` diverso salterebbe comunque i claim già
  completati, assumendo implicitamente che il batching non alteri i
  punteggi in modo rilevabile.
- Il confronto di soglia è **strettamente maggiore**
  (`support_prob > threshold`), identico a MiniCheck: un punteggio
  esattamente uguale a `threshold` produce `"Refuted"`.
- A differenza di `MiniCheck/chunking.py`, qui `sent_tokenize` non riceve
  alcun trattamento speciale per i confini di paragrafo (`\n\n`) del
  documento originale — un documento con più paragrafi viene trattato da
  NLTK come testo continuo ai fini della segmentazione in frasi.
