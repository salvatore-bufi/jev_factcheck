# Overview dell'architettura

Questo modulo reimplementa l'algoritmo di inferenza del progetto
**FactCG** (derenlei/FactCG, NAACL 2025 —
https://github.com/derenlei/FactCG), usando il checkpoint pubblicato
`FactCG-DeBERTa-v3-Large` (https://huggingface.co/yaxili96/FactCG-DeBERTa-v3-Large)
come classificatore binario di consistenza fattuale. Come `MiniCheck/`, e a
differenza di `AICCTU/`, non c'è alcuna fase di retrieval: ogni claim viene
confrontato **direttamente** con **ogni** documento del dataset, chunk per
chunk, e il chunk con la probabilità di supporto più alta vince. La
differenza rispetto a `MiniCheck/` è il modello stesso: non un LLM
generativo servito via Ollama, ma un modello di **classificazione**
`transformers` caricato in locale in memoria, interrogato con un forward
pass deterministico (niente `--temperature`, niente `--retries`: ogni
inferenza produce sempre una probabilità valida).

La cartella `FactCG/` è una delle possibili codebase testate nel repo:
claim e documenti (`input/<dataset>/`) sono condivisi a livello di repo tra
tutte, organizzati per dataset. Non essendoci retrieval non c'è nulla da
persistere prima del run (niente embedding, niente indice FAISS, niente
`data_store/`): l'unico artefatto namespaced per dataset e per **hash
deterministico della configurazione** è la cartella di output
(`outputs/<dataset>/<run_id>/` — vedi `run_identity.py`), esattamente come
in MiniCheck.

## Flusso end-to-end

```
../input/<dataset>/documents/*.txt   ../input/<dataset>/claims/*.json
        │                                       │
        ▼                                       │
 [1] Chunking (sentence-aware,                   │
     packing greedy, ≤550 parole/chunk,          │
     nltk word_tokenize)                         │
        │                                       │
        ▼                                       ▼
 [2] Per ogni documento: i chunk sono raggruppati in batch
     (batch_size=8) e accoppiati al claim
        │
        ▼
 [3] Prompt-template FactCG (INSTRUCTION_TEMPLATE) tokenizzato e
     passato a FactCG-DeBERTa-v3-Large (transformers, locale, GPU/CPU)
        │
        ▼
 [4] softmax(logits) → support_prob = P(classe 1 = "Supported")
     (classificatore deterministico: nessun retry, nessuna temperatura)
        │
        ▼
 [5] Per ogni documento si tiene il chunk con support_prob massimo;
     tra tutti i documenti si tiene quello con support_prob massimo
        │
        ▼
 [6] verdetto = "Supported" se support_prob > soglia (0.5), altrimenti "Refuted"
        │
        ▼
 [7] ../outputs/<dataset>/<run_id>/claim_<id>.json (+ summary.json, run_params.json)
```

### [1] Chunking (`chunking.py`)

Ogni documento è spezzato in frasi (`nltk.sent_tokenize`) e le frasi vengono
impacchettate greedy in chunk finché la frase successiva non farebbe
sforare `chunk_size` **parole** (contate con `nltk.word_tokenize`, lo stesso
proxy usato dall'implementazione originale); una frase da sola più lunga del
limite diventa comunque un chunk a sé (mai spezzata a metà). Stesso
algoritmo di `MiniCheck/chunking.py`, con `chunk_size` di default più
piccolo (550 parole, contro le ~18k di MiniCheck) perché qui la lunghezza è
vincolata dal `max_length` del tokenizer DeBERTa (2048 token), non da un
context window da 32k come i modelli Ollama.

### [2]-[4] Scoring per chunk (`factcg_client.py`)

Non c'è retrieval: ogni claim viene valutato contro **tutti** i chunk di
**tutti** i documenti del dataset. A differenza di MiniCheck (una chiamata
HTTP a Ollama per chunk), qui il modello è caricato una sola volta per run
(`FactCGModel.__init__`, GPU se disponibile) e i chunk vengono scored **in
batch** (`--batch-size`, default 8): ogni coppia (chunk, claim) è formattata
con lo stesso `INSTRUCTION_TEMPLATE` della libreria originale
(`factcg/utils.py`) — non un'encoding a coppia di frasi, ma un'unica
stringa `"{documento}\n\nChoose your answer: ... {claim} ... OPTIONS: Yes/No ..."`
— tokenizzata con padding al più lungo del batch e troncata a `max_length`
(default 2048, come l'`Inferencer` upstream). Un singolo forward pass
produce i logits per l'intero batch; `support_prob = softmax(logits)[:, 1]`,
cioè la probabilità della classe 1 = "Supported" (verificato contro
l'esempio numerico ufficiale del progetto originale: claim inconsistente
≈0.065, consistente ≈0.784).

Una deviazione deliberata rispetto all'`Inferencer` originale: upstream
riassegna `tokenizer.pad_token = tokenizer.eos_token`, un workaround per T5
(che non ha un pad token nativo) ereditato da un percorso di codice
condiviso. DeBERTa ha già un `pad_token` corretto (`[PAD]`, id 0); qui viene
lasciato quello del tokenizer invece di sovrascriverlo con l'alias di
`eos_token` (`[SEP]`, id 2) — le posizioni di padding sono comunque
mascherate dall'`attention_mask`, quindi il comportamento non cambia, ma il
tokenizer resta coerente con se stesso.

### [5]-[6] Selezione del documento migliore e verdetto (`main.py`)

Identico a MiniCheck: per ogni documento si tiene il chunk con
`support_prob` massimo (`score_document`); tra tutti i documenti del
dataset si tiene quello con `support_prob` massimo in assoluto, riportato
come `best_source`. Il verdetto finale è binario — `Supported` se
`support_prob > threshold` (soglia di default 0.5), altrimenti `Refuted`.

## Perché si scarica da Hugging Face invece di usare `ckpt/factcg_dbt.ckpt`

Il file `ckpt/factcg_dbt.ckpt` è il checkpoint originale PyTorch Lightning
(state dict con prefisso `base_model.*` + quattro head separate: solo
`bin_layer` è quella rilevante). Caricarlo direttamente richiederebbe
ricostruire a mano l'architettura esatta (pooling, head, attivazioni), con
il rischio concreto di un errore silenzioso (numeri validi ma sbagliati). Il
repository Hugging Face `yaxili96/FactCG-DeBERTa-v3-Large` contiene la
stessa identica checkpoint già convertita nel formato standard
`transformers` (`model.safetensors` + `DebertaV2ForSequenceClassification`)
— lo stesso percorso di default usato internamente dalla libreria originale
(`Inferencer(use_hf_ckpt=True)`). Questo modulo usa quindi
`AutoModelForSequenceClassification.from_pretrained("yaxili96/FactCG-DeBERTa-v3-Large", ...)`,
scaricato al primo avvio e messo in cache in `./cache/` (self-contained
dentro `FactCG/`, non condivisa con `AICCTU/data/`). Vedi
`description_claude.md` per le note originali su dove trovare il
checkpoint.

## Differenze rispetto al progetto FactCG originale

| Aspetto | Originale (derenlei/FactCG) | Questo modulo |
|---|---|---|
| Caricamento del checkpoint | `Inferencer(use_hf_ckpt=True)`, stesso repo HF | Identico: `AutoModelForSequenceClassification.from_pretrained("yaxili96/FactCG-DeBERTa-v3-Large")` |
| `pad_token` | Riassegnato a `eos_token` (workaround per T5 in un percorso condiviso) | Lasciato quello nativo di DeBERTa (`[PAD]`); l'`attention_mask` rende il comportamento equivalente |
| Chunking | `Inferencer.chunking_src`, greedy sentence-packing a `chunk_size` parole | Stesso algoritmo (`chunking.py`), stesso default 550 parole |
| Ambito di applicazione | Libreria standalone, usata su coppie documento/claim fornite dal chiamante | **Modulo del repo**: scorre l'intero `documents/` per ogni claim di `input/<dataset>/claims/`, nessuna selezione a monte |
| Batching | Gestito dal chiamante | `--batch-size` (default 8) su tutti i chunk di un documento |
| Output | `support_prob`/`pred_label` per la coppia valutata | **Un JSON per claim** con verdetto, punteggio per ogni documento/chunk (`document_scores`, incluse entrambe le probabilità softmax) e info di run |
| Identità/riproducibilità | Non prevista (chiamata di libreria) | `run_identity.py`: stessa configurazione → stessa cartella `outputs/<dataset>/<run_id>/`, manifest `run_params.json` |

Parametri invariati rispetto all'originale: `INSTRUCTION_TEMPLATE`,
`max_length` 2048, soglia di verdetto 0.5, convenzione classe 1 =
"Supported" (coerente con il `gold_label` di LLM-AggreFact).

### [7] Identità di dataset/run (`run_identity.py`)

Come in MiniCheck (e a differenza di AICCTU) non esiste un
`index_identity`: non essendoci nulla da indicizzare/persistere prima del
run, c'è una sola identità, `run_identity(codebase, dataset, params)`,
calcolata su *tutti* gli iperparametri che influenzano il verdetto
(`llm_model`, `max_length`, `chunk_size`, `threshold`). Restituisce un nome
cartella breve e leggibile (slug del modello + hash a 8 cifre) più il
dizionario completo dei parametri, scritto come `run_params.json` accanto
agli output (`write_manifest`). Essendo l'hash deterministico, la stessa
configurazione rilanciata due volte produce lo stesso `run_id`.
`--resume` sfrutta la stessa proprietà: rilanciando la stessa
configurazione, i claim che hanno già un file di output senza `error`
vengono saltati, e solo quelli mancanti/falliti vengono ripetuti.

## Come si estende

- **Altro checkpoint/modello di classificazione**: cambiare `--model` con
  qualsiasi modello Hugging Face compatibile con
  `AutoModelForSequenceClassification` a 2 classi; il resto della pipeline
  non cambia.
- **Altri dataset**: aggiungere `input/<nuovo_dataset>/{claims,documents}` e
  lanciare con `--dataset <nuovo_dataset>`; nessuna modifica al codice.
- **Sweep di iperparametri**: ogni combinazione testata finisce
  automaticamente nel proprio `outputs/<dataset>/<run_id>/` grazie a
  `run_identity.py`, quindi si può ciclare su
  `--model`/`--chunk-size`/`--threshold`/`--batch-size`/... in uno script
  esterno senza rischio di sovrascrivere risultati precedenti.
- **Valutazione**: `../outputs/<dataset>/<run_id>/summary.json` dà subito
  verdetto/`support_prob` per claim se si dispone di etichette gold; i file
  per-claim contengono il dettaglio chunk-per-chunk (`document_scores`, con
  entrambe le probabilità softmax) per capire quale documento/porzione ha
  determinato il verdetto; `run_params.json` in ogni cartella permette di
  aggregare/confrontare più run per iperparametro senza doverli ricordare a
  memoria — stesso schema di MiniCheck, così `evaluation/` legge entrambe le
  codebase senza modifiche.
