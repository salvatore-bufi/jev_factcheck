# FactCG — modulo di fact-checking diretto (no retrieval)

Modulo di fact-checking: verifica una lista di claim contro un corpus di
documenti locali confrontando **direttamente** ogni claim con ogni
documento, tramite `FactCG-DeBERTa-v3-Large`
(https://huggingface.co/yaxili96/FactCG-DeBERTa-v3-Large), un classificatore
binario di consistenza fattuale derivato dal progetto
[FactCG](https://github.com/derenlei/FactCG) (NAACL 2025).

A differenza di AICCTU, **non c'è retrieval**: nessun embedding, nessun
indice FAISS, nessun `data_store/`. Per ogni claim, ogni documento in
`documents/` viene spezzato in chunk (sentence-packing, ≤550 parole per
chunk) e confrontato per intero con la claim; il documento con la
probabilità di supporto più alta viene riportato come fonte migliore —
stessa architettura di `MiniCheck/`, con FactCG-DeBERTa-v3-Large al posto di
`bespoke-minicheck:7b`.

A differenza sia di AICCTU sia di MiniCheck (LLM serviti da Ollama), FactCG è
un modello di classificazione `transformers` caricato **in locale in
memoria** una sola volta per run (non un LLM generativo interrogato via
HTTP): è un classificatore deterministico, quindi non esistono `--temperature`
né `--retries` su risposte non parsabili — un forward pass produce sempre
una probabilità valida.

Questa cartella è **una codebase** tra potenzialmente più implementazioni
testate nello stesso repo (cartelle sorelle di `FactCG/`: `AICCTU/`,
`MiniCheck/`, ciascuna con lo stesso layout di codice). Claim e documenti
(`input/`) sono **condivisi** a livello di repo tra tutte le codebase,
organizzati per **dataset** (`input/<dataset>/claims`,
`input/<dataset>/documents`).

I risultati sono namespaced a livello di repo per dataset e per **identità
della configurazione** (codebase + modello + iperparametri, hashati
deterministicamente — vedi `run_identity.py`): stessa configurazione →
stessa cartella. Per questo la cartella **non** è autosufficiente/spostabile
a sé stante: dipende dalla root del repo (come AICCTU e MiniCheck).

## Struttura

```
FactChecker/                      # root del repo
├── input/
│   └── example/                          # DATASET_NAME (--dataset)
│       ├── claims/                       # uno o più .json con i claim da verificare
│       └── documents/                    # i .txt su cui verificarli
├── outputs/
│   └── example/
│       └── FactCG__FactCG-DeBERTa-v3-Large--a1b2c3d4/  # run_id = codebase__llm-slug--hash(TUTTI gli iperparametri)
│           ├── run_params.json               # config completa della run, in chiaro
│           ├── claim_<id>.json
│           └── summary.json
└── FactCG/                       # questa codebase
    ├── main.py                   # entry point (CLI)
    ├── config.py                 # tutti i default (modello, chunk size, dataset)
    ├── run_identity.py           # calcolo run_id, scrittura manifest
    ├── chunking.py               # sentence-tokenize + packing greedy in chunk
    ├── factcg_client.py          # prompt, caricamento modello transformers, softmax -> support_prob
    ├── requirements.txt
    ├── description_claude.md     # note originali su dove trovare il checkpoint
    └── ckpt/factcg_dbt.ckpt      # checkpoint originale (PyTorch Lightning, non usato direttamente: vedi sotto)
```

## Prerequisiti

1. **Python ≥ 3.10** con le dipendenze:
   ```bash
   pip install -r requirements.txt
   ```
2. Una **GPU** è fortemente consigliata (DeBERTa-v3-Large, 24 layer): il
   modello gira anche su CPU ma molto più lentamente. Rilevata
   automaticamente (`--device` per forzare `cpu`/`cuda`). Su un server
   multi-GPU condiviso, usare `--gpu <id>` per restringere il processo a una
   GPU fisica specifica via `CUDA_VISIBLE_DEVICES` (orthogonale a `--device`:
   `--gpu` decide quali GPU sono visibili, `--device` come/se usare CUDA tra
   quelle visibili) — evita che il modello finisca su una GPU usata da altri.
3. **Download del modello**: al primo avvio, i pesi (~1.7GB, formato
   `safetensors`) vengono scaricati da Hugging Face Hub
   (`yaxili96/FactCG-DeBERTa-v3-Large`) e messi in cache in `./cache/`
   (dentro questa cartella, non condivisa con `data/` di AICCTU). Le run
   successive riusano la cache, nessun nuovo download.

### Perché si scarica da Hugging Face invece di usare `ckpt/factcg_dbt.ckpt`

Il file `ckpt/factcg_dbt.ckpt` è il checkpoint originale PyTorch Lightning
(state dict con prefisso `base_model.*` + quattro head separate:
`bin_layer`, `bin_finetuned_layer`, `tri_layer`, `reg_layer` — solo
`bin_layer` è quella rilevante, `hyper_parameters.finetune_task == 'bin'`).
Caricarlo direttamente richiederebbe ricostruire a mano l'architettura
esatta (pooling, head, attivazioni) con il rischio concreto di un errore
silenzioso (numeri validi ma sbagliati). Il repository Hugging Face
`yaxili96/FactCG-DeBERTa-v3-Large` contiene **la stessa identica checkpoint**
(`pytorch_lightning_ckpt/factcg_dbt.ckpt`, verificato) già convertita nel
formato standard `transformers` (`model.safetensors` +
`DebertaV2ForSequenceClassification`) — ed è esattamente il percorso di
default che la libreria originale FactCG usa al suo interno
(`Inferencer(use_hf_ckpt=True)`, derenlei/FactCG). Questo modulo usa quindi
`AutoModelForSequenceClassification.from_pretrained("yaxili96/FactCG-DeBERTa-v3-Large", ...)`,
esattamente come da `description_claude.md`.

## Input

Percorsi relativi alla **root del repo** (condivisi tra tutte le codebase),
non a questa cartella, e organizzati per dataset tramite `--dataset`
(default `example`, vedi `config.DATASET_NAME`).

**Claim** — file `.json` in `input/<dataset>/claims/` (uno o più file,
ciascuno con un claim singolo o una lista). Campi obbligatori: `claim` e
`claim_id` (univoco). Qualsiasi altro campo è ammesso e viene ricopiato
nell'output come `claim_metadata` (non viene passato al modello: come
MiniCheck, FactCG confronta solo claim e testo del documento, non ha un
canale per metadati extra).

**Documenti** — file `.txt` in `input/<dataset>/documents/`. Ad ogni run,
**ogni** claim viene confrontato con **ogni** documento (nessuna
selezione/retrieval a monte); i documenti lunghi vengono spezzati in chunk
internamente (vedi `chunking.py`).

## Esecuzione

```bash
cd FactCG

# Run standard sul dataset di default ("example"): verifica tutti i claim contro tutti i documenti
python main.py

# Altro dataset
python main.py --dataset aggregatefact_test_n3

# Parametri diversi: finisce in una NUOVA cartella outputs/ (hash diverso)
python main.py --threshold 0.6 --chunk-size 400

# Riprende una run interrotta: salta i claim già completati con successo,
# ripete solo quelli mancanti o falliti (stessa identità di run richiesta)
python main.py --dataset aggregatefact_test_n3 --resume

# Directory di input/output alternative (bypassano la risoluzione automatica per dataset+hash)
python main.py --claims-dir /percorso/claims --documents-dir /percorso/docs --outputs-dir /percorso/out
```

Tutti i flag: `python main.py --help`.

## Output

In `../outputs/<dataset>/<run_id>/` viene scritto **un file per claim**
(`claim_<claim_id>.json`), un indice aggregato `summary.json` (claim_id →
verdetto, per valutazioni rapide) e un `run_params.json` con la
configurazione completa della run (dataset, codebase, modello, chunk size,
soglia, ecc. — lo stesso dizionario usato per calcolare `run_id`). `run_id`
è deterministico: `<codebase>__<model-slug>--<hash8>`, dove l'hash copre
*tutti* gli iperparametri della run.

Schema di `claim_<id>.json` (identico a MiniCheck, così `evaluation/` legge
entrambi senza modifiche):

| Campo | Contenuto |
|---|---|
| `claim_id`, `claim` | il claim in input |
| `claim_metadata` | tutti gli altri campi del claim in input |
| `verdict` | `Supported` \| `Refuted` (soglia 0.5 su `support_prob`) |
| `support_prob` | probabilità di supporto massima tra tutti i documenti/chunk, in [0, 1] (softmax classe 1 = "Supported") |
| `best_source` | documento/chunk che ha prodotto `support_prob` (`file`, `chunk_index`, `text`) |
| `document_scores` | punteggio per **ogni** documento del dataset (non solo il migliore), con dettaglio chunk-per-chunk (`probs`: entrambe le probabilità softmax) |
| `run_info` | modello, parametri, timestamp, durata |

Un errore imprevisto sul claim produce un file con campo `error`; il run
prosegue con i claim successivi (`--resume` in una run successiva ritenterà
solo questi).

## Esempio incluso

`../input/example/` (root del repo) contiene già claim e documenti di
esempio: un run "smoke test" completo si fa semplicemente con
`python main.py` appena installati i requisiti (usa il dataset `example` di
default; il primo avvio scarica il modello da Hugging Face Hub).
