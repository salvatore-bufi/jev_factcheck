# Struttura dei risultati

Questo file descrive tutto ciò che `benchmark.py` e `report.py` scrivono in `results/`: ogni file, ogni campo e da dove
viene. Gli esempi sono presi da run reali (JPT-9B, Winnow, d1-3B, FactCG su 32 righe di LLM-AggreFact test).

## Organizzazione delle cartelle

```
results/
└── <dataset>/                                  es. lytang--LLM-AggreFact
    └── <split>/                                es. test, dev, dev-jev4520
        ├── hash_questions.json                 hash delle domande  -> set di domande
        ├── hash_<modello>.json                 hash dei parametri  -> iperparametri del modello (uno per modello)
        ├── summary_<hash domande>.md           tabelle di confronto tra modelli (scritte da report.py)
        ├── summary_<hash domande>_custom.md    idem, con regola/soglia/temperatura cambiate (report.py con opzioni)
        └── <modello>_<hash parametri>_<hash domande>[-limitN][-tag]/
            ├── run.json            contesto del run: identità, opzioni, ambiente, tempi, riepilogo telemetria
            ├── answers.jsonl       una riga per esempio: le risposte a OGNI domanda
            ├── batches.jsonl       passaggio principale: una riga per batch
            ├── latency.jsonl       sonda di latenza: una riga per esempio, una riga alla volta
            ├── throughput.jsonl    righe/s a diverse dimensioni di batch
            ├── telemetry.jsonl     campioni GPU/memoria ogni 0,2 s
            ├── failures.jsonl      righe fallite e motivo
            └── report.json         metriche calcolate da tutti i file sopra (rigenerabile)
```

Esempio: `results/lytang--LLM-AggreFact/test/jpt-9b_5147a4ec_551bb722-limit32/`
- `jpt-9b`: cartella del modello in `models/`;
- `5147a4ec`: hash delle opzioni effettive del modello (vedi `hash_jpt-9b.json`);
- `551bb722`: hash delle domande (vedi `hash_questions.json`);
- `-limit32`: run su un sottoinsieme fisso di 32 righe (assente nei run completi); `-<tag>` se si usa `--tag`.

Stesso modello con altre opzioni (es. altra temperatura) → altro hash dei parametri → altra cartella, accanto alla prima.
Stesso modello e opzioni ma altre domande → altro hash delle domande → altra cartella.

**Cosa finisce in git:** solo i file piccoli (`run.json`, `report.json`, `hash_*.json`, `summary_*.md`). I file per riga
e la telemetria (`answers`, `batches`, `latency`, `throughput`, `telemetry`, `failures`) restano locali (`.gitignore`).

---

## Registri degli hash

### `hash_questions.json`

Un oggetto `hash → voce`, scritto all'avvio di ogni run. Se l'hash esiste già non viene toccato; se è nuovo si aggiunge.

| Campo | Significato |
|---|---|
| *(chiave)* | 8 caratteri esadecimali: sha256 del JSON canonico (chiavi ordinate, senza spazi) delle **sole domande** |
| `name` | nome del set (campo `name` del file in `config/questions/`) |
| `questions` | le domande esatte, come le ricevono i modelli |
| `first_used` | data e ora UTC del primo run con queste domande |

La regola di decisione (`decision`) **non** entra nell'hash: si applica al momento del report e si può cambiare senza
rifare l'inferenza.

### `hash_<modello>.json`

Stessa logica, per gli iperparametri. Esempio (`hash_jpt-9b.json`):

```json
{"5147a4ec": {"model": "jpt-9b", "model_name": "JPT-9B",
              "options": {"revision": "b447cc7...", "temperature": 1.087, "max_model_len": 32768, "gpu_util": 0.92,
                          "max_num_seqs": 32, "max_batched_tokens": 8192, "batch": 64},
              "first_used": "2026-10-13T12:59:12+00:00"}}
```

| Campo | Significato |
|---|---|
| *(chiave)* | 8 caratteri esadecimali: sha256 del JSON canonico delle **opzioni effettive** |
| `model` | cartella del modello in `models/` |
| `model_name` | nome mostrato nelle tabelle |
| `options` | opzioni effettive: default dell'adapter, poi `config/models/<modello>.json`, poi `--opt` / `--batch` |
| `first_used` | data e ora UTC del primo run con queste opzioni |

Le opzioni variano per modello (temperatura, contesto massimo, offload su CPU, revisioni di modello o encoder, batch,
porta del server, ...). Il loro significato è in `config/models/<modello>.json` (campo `_comment`) e nel README del
modello. Un valore `"auto"` (es. `cpu_offload_gb`) entra nell'hash come `"auto"`; il valore effettivamente usato su
quella GPU è in `run.json` → `model`.

---

## `answers.jsonl` — le risposte

**Riga 1 (intestazione)**: `{"identity": {...}}`, la stessa `identity` di `run.json` (vedi sotto). Serve alla ripresa:
se la cartella esiste ma l'identità è diversa (es. dati cambiati), il run viene rifiutato invece di mescolare risultati.

**Righe successive**: una per esempio valutato con successo (le righe fallite sono in `failures.jsonl`).

```json
{"id": "test:90", "source": "AggreFact-CNN", "label": 1,
 "claim_chars": 381, "claim_words": 65, "doc_chars": 2403, "doc_words": 410, "truncated": false,
 "prompt_tokens": {"support_simple": 699, "unsupported_detail": 749, "contradicted_detail": 742},
 "answers": {"support_simple": {"type": "noul", "keys": ["true", "false"],
                                "probs": {"true": 0.9725, "false": 0.0275},
                                "logits": [-0.0242, -3.8992], "temperature": 1.087}, "...": {}},
 "batch": 0}
```

| Campo | Significato |
|---|---|
| `id` | identificativo della riga nel dataset (`<split>:<indice>`, o la colonna `id` del dataset se esiste) |
| `source` | fonte/sotto-dataset della riga (colonna `dataset`; `all` se manca). È la base delle metriche per fonte |
| `label` | etichetta vera: 1 = il documento supporta il claim, 0 = no |
| `claim_chars`, `claim_words` | lunghezza del claim in caratteri e parole (spazi) |
| `doc_chars`, `doc_words` | lunghezza del documento **originale** in caratteri e parole |
| `truncated` | `true` se il documento superava `--max-doc-chars` (80.000) ed è stato accorciato (75% iniziale + 25% finale) |
| `prompt_tokens` | token del prompt, contati col tokenizer del modello. Vedi le differenze per modello sotto |
| `answers` | per **ogni** domanda del set, la risposta completa (tabella seguente) |
| `batch` | numero del batch del passaggio principale in cui la riga è stata valutata (vedi `batches.jsonl`) |

**Una risposta (`answers.<domanda>`)**

| Campo | Significato |
|---|---|
| `type` | `noul` (sì/no), `choice` (una opzione tra N) o `score` (livello su scala ordinata) |
| `keys` | le chiavi delle opzioni, **nell'ordine in cui il modello le legge**. `noul`: sempre `true`/`false`, ma l'ordine dipende dal modello (JPT-9B: `["true","false"]`, cioè A = Sì; Rune/Winnow/CLM/d1: `["false","true"]`). `choice`: i nomi delle opzioni. `score`: `"0"`, `"1"`, ... |
| `probs` | probabilità di ogni opzione (sommano a 1), come il modello le produce con la sua temperatura. Per una `noul` la probabilità del "sì" è `probs.true` |
| `logits` | i punteggi grezzi delle opzioni, **nello stesso ordine di `keys`**, prima della temperatura; `null` se il modello non li espone. Servono per ricalcolare le probabilità con un'altra temperatura (`report.py --temperature`) |
| `temperature` | la temperatura con cui `probs` è stato calcolato da `logits` (`null` se non applicabile) |

Differenze per modello:

| Modello | `logits` | `temperature` | `prompt_tokens` |
|---|---|---|---|
| JPT-9B, Rune, Winnow-BF16 | log-probabilità dei token delle lettere delle opzioni (differiscono dai logit per una costante, che il softmax elimina) | 1.087 / 2.0 / 1.0 | dizionario: token del prompt di ciascuna domanda |
| Winnow-12B NVFP4 (server) | logit delle opzioni restituiti dal server (`diagnostics`) | 1.0 | **intero**: token di input dell'intera richiesta (documento una volta + le tre domande) |
| CLM-8B | `scala × coseno` tra stato e testo dell'opzione | 1.0 | dizionario: token del testo dello stato per ciascuna domanda |
| d1-3B | `null` (la sua API restituisce solo probabilità) | `null` | dizionario: token del prompt di ciascuna domanda |
| FactCG (baseline) | `null` | `null` | `null` (ha `chunks` interni, non salvati) |

FactCG ignora le domande: la sua unica "risposta" è `factcg_support` (probabilità massima di supporto tra i pezzi del
documento), giudicata con la sua regola (`> 0.5`).

---

## `batches.jsonl` — il passaggio principale

Una riga per batch del passaggio principale (tutte le righe dello split, in batch di `batch` righe).

```json
{"batch": 0, "rows": 32, "ok": 32, "failed": 0, "prompt_tokens": 91791, "seconds": 10.5733,
 "t_start": 1791896387.869, "t_end": 1791896398.444}
```

| Campo | Significato |
|---|---|
| `batch` | numero progressivo del batch (prosegue dopo una ripresa) |
| `rows` | righe inviate al modello in questo batch |
| `ok`, `failed` | righe con risposta valida / fallite |
| `prompt_tokens` | somma dei token dei prompt delle righe riuscite (tutte le domande) |
| `seconds` | tempo reale della chiamata al modello per l'intero batch (`perf_counter`) |
| `t_start`, `t_end` | inizio e fine in tempo Unix (secondi), per allinearlo con `telemetry.jsonl` |

Il throughput principale di `report.json` è `somma(ok) / somma(seconds)`. Il tempo per singola riga dentro un batch
non esiste: le righe sono elaborate insieme.

---

## `latency.jsonl` — la latenza

Misura separata dal passaggio principale: un campione fisso (200 righe di default, `--latency-sample`) scelto con un
seed e distribuito su tutte le fonti, valutato **una riga alla volta** (tutte le sue domande), con la cache dei
prefissi svuotata prima di ogni riga (documento "a freddo"), dopo un riscaldamento. È la latenza che vedrebbe una
singola richiesta.

```json
{"id": "test:5573", "source": "ClaimVerify", "label": 0, "claim_chars": 165, "claim_words": 29,
 "doc_chars": 17432, "doc_words": 2695, "truncated": false, "seconds": 1.32853, "ok": true,
 "prompt_tokens": {"support_simple": 6093, "unsupported_detail": 6143, "contradicted_detail": 6136}}
```

| Campo | Significato |
|---|---|
| `id`, `source`, `label`, `claim_*`, `doc_*`, `truncated` | come in `answers.jsonl` |
| `seconds` | tempo reale dall'invio della riga al ritorno di tutte le risposte. Per Winnow-NVFP4 include la richiesta HTTP locale |
| `ok` | `true` se la riga ha avuto risposta |
| `error` | presente solo se `ok` è `false`: il motivo |
| `prompt_tokens` | come in `answers.jsonl` |

Le risposte della sonda non vengono salvate: quelle valide per l'accuratezza sono in `answers.jsonl`. La sonda si fa
una volta per cartella; se è già completa, una ripresa la salta.

---

## `throughput.jsonl` — throughput contro dimensione del batch

Un campione fisso (64 righe di default, `--throughput-sample`), valutato più volte con batch diversi
(`--throughput-sizes`, default `1,8,32`). Mostra quanto conviene raggruppare le richieste.

```json
{"batch_size": 4, "rows": 8, "failed": 0, "seconds": 2.1612, "rows_per_s": 3.702}
```

| Campo | Significato |
|---|---|
| `batch_size` | righe per chiamata al modello |
| `rows` | righe del campione |
| `failed` | righe fallite |
| `seconds` | tempo reale per elaborare tutto il campione con quella dimensione di batch |
| `rows_per_s` | `rows / seconds` |

---

## `telemetry.jsonl` — GPU e memoria nel tempo

Un campione ogni 0,2 secondi (`--telemetry-interval` nel runner), dalla libreria NVML per la GPU del benchmark e da
`/proc` per la memoria del processo e dei suoi figli (vLLM e il server di Winnow girano in processi figli).

```json
{"t": 1791896388.039, "phase": "main", "rss_mb": 4741.6, "power_w": 164.348, "gpu_util": 100, "mem_util": 23,
 "mem_used_mb": 22795, "temp_c": 69, "sm_clock_mhz": 2070, "mem_clock_mhz": 14001, "pcie_gen": 5, "throttle": 4}
```

| Campo | Significato |
|---|---|
| `t` | istante in tempo Unix (secondi) |
| `phase` | fase del run in quel momento: `start`, `load` (caricamento del modello), `warmup`, `latency`, `throughput_bs<N>`, `main` |
| `rss_mb` | memoria RAM residente del processo e dei suoi figli, in MB |
| `power_w` | potenza assorbita dalla GPU, in watt |
| `gpu_util` | utilizzo dei core della GPU nell'ultimo intervallo, in % |
| `mem_util` | utilizzo della banda di memoria della GPU, in % |
| `mem_used_mb` | memoria GPU occupata **sull'intera scheda**, in MB (vLLM riserva in anticipo la quota `gpu_util`, quindi è la memoria riservata, non solo quella usata) |
| `temp_c` | temperatura della GPU, in °C |
| `sm_clock_mhz`, `mem_clock_mhz` | frequenza dei core e della memoria, in MHz (cali indicano throttling) |
| `pcie_gen` | generazione PCIe corrente del collegamento (a riposo può scendere) |
| `throttle` | maschera di bit NVML dei motivi di rallentamento (valori principali, sommabili): 0 = nessuno; 1 = GPU inattiva, 4 = limite di potenza software, 8 = rallentamento hardware, 32 = termico software, 64 = termico hardware, 128 = freno di potenza |

---

## `failures.jsonl` — le righe fallite

Una riga per ogni fallimento (anche ripetuti tra una ripresa e l'altra). Una riga fallita **non riceve mai un punteggio
inventato**: non è in `answers.jsonl`, e una ripresa la riprova.

```json
{"id": "test:2766", "phase": "main", "error": "prompt of 12284 tokens exceeds max_model_len 8192", "at": "2026-10-13T..."}
```

| Campo | Significato |
|---|---|
| `id` | riga del dataset |
| `phase` | fase in cui è fallita (`main`) |
| `error` | motivo: prompt troppo lungo per il contesto del modello, out of memory, errore HTTP o timeout del server, ecc. |
| `at` | data e ora UTC |

---

## `run.json` — il contesto del run

Riscritto all'avvio e alla fine di ogni sessione (una ripresa aggiunge una sessione).

| Campo | Significato |
|---|---|
| `identity` | ciò che identifica i risultati: `model`, `options` (effettive), `params_hash`, `questions_hash`, `dataset`, `split`, `data_sha256` (impronta dei file di dati), `limit`, `max_doc_chars`, `runner_version` |
| `model_name`, `model_size`, `model_kind` | nome e dimensione per le tabelle; `decision` o `baseline` (FactCG) |
| `default_decision` | regola propria di una baseline (FactCG: `factcg_support > 0.5`); `null` per i decision model |
| `questions` | il set di domande completo, con la sua `decision` (regola di default per il report) |
| `data` | da dove vengono i dati: `files`, `revision`, `sha256`, `rows_total` (righe dello split), `limit` |
| `model_config` | `file` di config letto, i suoi `values` e le sostituzioni da riga di comando (`command_line`) |
| `environment` | `host`, `platform`, `python`, `cpu`, `cpu_count`, `ram_total_gb`; `git` (`commit`, `branch`, `dirty` = modifiche non committate); `cuda_visible_devices`; `packages` (versioni di torch, vllm, transformers, ...); `gpu` (`index`, `name`, `memory_total_mb`, `driver`, `cuda_driver`, `power_limit_w`, `pcie_gen_max`, `pcie_width_max`) |
| `gpu_busy_at_start` | altri processi sulla GPU all'avvio: `compute` (bloccano il run, salvo `--allow-busy-gpu`) e `graphics` (es. il compositor del desktop, solo annotati) |
| `status` | `running`, `complete` (tutte le righe valutate), `incomplete` (alcune fallite), `error`, `interrupted` (Ctrl-C) |
| `sessions` | una voce per ogni avvio (vedi sotto) |
| `model` | descrizione dal modello stesso: repository, revisione, protocollo del prompt, temperatura, offload effettivo, hash dei pesi, ecc. |

**Una sessione (`sessions[i]`)**

| Campo | Significato |
|---|---|
| `started_at`, `finished_at` | inizio e fine (UTC) |
| `argv`, `gpu` | comando del runner e GPU fisica usata |
| `load_seconds` | tempo di caricamento del modello (a freddo: avvio di vLLM o del server, lettura dei pesi, compilazione) |
| `first_call_seconds` | tempo della prima chiamata dopo il caricamento (selezione dei kernel, grafi CUDA) |
| `warmup_seconds` | tempi delle chiamate di riscaldamento, escluse dalle misure |
| `main_rows_scored`, `main_rows_failed`, `main_seconds` | righe valutate, fallite e tempo del passaggio principale in questa sessione |
| `error` | presente se la sessione si è chiusa con un errore |
| `telemetry` | riepilogo di `telemetry.jsonl` **per fase**: `samples`, `seconds`, `energy_j` (energia GPU, integrale della potenza), `power_mean_w`, `power_max_w`, `gpu_util_mean`, `gpu_mem_peak_mb`, `rss_peak_mb`, `temp_max_c`, `sm_clock_mean_mhz`, `sm_clock_min_mhz` |

---

## `report.json` — le metriche

Calcolato alla fine del run e ricalcolabile in ogni momento con `python report.py`, senza inferenza. Usa la regola di
decisione del set di domande (o quella propria della baseline). Le metriche sono frazioni (0–1), non percentuali.

| Sezione | Contenuto |
|---|---|
| `run`, `model`, `model_name`, `model_size`, `model_kind`, `dataset`, `split`, `questions_hash`, `params_hash`, `options`, `status` | identificazione |
| `decision` | la regola applicata: `combine` (`min`, `mean`, `product`, `single`), `inputs` (domanda e cosa prenderne: `yes`, `no`, `option:K`, `score`), `threshold` |
| `temperature_override` | `null`, la temperatura passata a `report.py`, oppure `"n/a"` se il modello non salva logit |
| `coverage` | `rows_expected` (righe dello split o del limite), `rows_scored`, `complete`, `failures_logged` |
| `accuracy` | sulla regola: `macro_balanced_accuracy` (**media non pesata della balanced accuracy per fonte**, la metrica della classifica LLM-AggreFact; `null` se una fonte ha una sola classe, normale con pochi esempi), `balanced_accuracy`, `accuracy`, `roc_auc` (sul punteggio continuo, aggregati su tutte le righe), `supported_recall` (supportati riconosciuti), `unsupported_recall` (non supportati riconosciuti), `n`, `macro_roc_auc` (media dell'AUC per fonte) |
| `by_source` | per ogni fonte: `n`, `balanced_accuracy`, `accuracy`, `roc_auc`, `supported_recall`, `unsupported_recall` |
| `by_length` | le stesse metriche (aggregate) per fasce di `claim_chars` (0-50, 50-100, 100-200, 200-400, >=400), `doc_chars` (0-1000, ..., 8000-16000, >=16000) e `prompt_tokens` (0-512, ..., 4096-8192, >=8192; per riga si usa il prompt più lungo tra le domande) |
| `per_question` | ogni domanda da sola: `type`; per una `noul` `mean_p_yes` e `auc_yes_vs_supported` (AUC di P(sì) contro l'etichetta: sotto 0,5 per le domande "invertite", come `unsupported_detail`); per `choice`/`score` `mean_probs` |
| `latency` | da `latency.jsonl`: `overall_seconds` (`n`, `mean`, `p50`, `p90`, `p95`, `p99`, `min`, `max`), `failed`, e le stesse statistiche `by_source`, `by_doc_chars`, `by_prompt_tokens` |
| `throughput` | `batch_size_sweep` (le righe di `throughput.jsonl`); `main`: `rows`, `seconds`, `rows_per_s`, `questions_per_s`, `prompt_tokens_per_s`, `batch_seconds` (statistiche dei tempi per batch), `batch_size` |
| `resources` | `load_seconds` e `first_call_seconds` (uno per sessione), `gpu_mem_peak_mb`, `rss_peak_mb`, `main_energy_j`, `energy_per_1k_rows_j`, `main_power_mean_w`, `main_power_max_w`, `main_gpu_util_mean`, `main_temp_max_c`, `main_sm_clock_min_mhz`, `gpu` |

Per cambiare regola, soglia o temperatura senza rifare l'inferenza:

```sh
python report.py --threshold 0.5                 # altra soglia
python report.py --rule support_simple           # una sola domanda
python report.py --decision regola.json          # regola qualsiasi
python report.py --temperature 2                 # ricalcola dalle probabilità dai logit (non per d1-3B e FactCG)
python report.py --calibrate-on dev-jev4520      # soglia di ogni modello adattata sul run di quello split
```

Con queste opzioni `report.json` non viene toccato: il risultato va in `summary_<hash>_custom.md`.

---

## `summary_<hash domande>.md` — il confronto

Scritto da `report.py` (e alla fine di ogni `benchmark.py`) per ogni set di domande dello split, con una riga per run:

1. **Balanced accuracy (%) per fonte**, nel formato di `table.md` (migliore in grassetto, secondo sottolineato), con
   soglia usata e AUC media per fonte. Un run incompleto o su un sottoinsieme è segnalato nel nome; più varianti dello
   stesso modello sono distinte dall'hash dei parametri.
2. **Ogni domanda da sola**: AUC di P(sì) contro l'etichetta.
3. **Velocità e risorse**: latenza p50/p95/p99, righe/s, domande/s, token/s, caricamento, prima chiamata, picco di memoria
   GPU e RAM, energia per 1.000 righe, potenza media, fallimenti.
4. **Throughput contro dimensione del batch**.
5. **Balanced accuracy e latenza p50 per fascia di lunghezza del documento**.

---

## `grid_<hash domande>.md` e `grid_<hash domande>.csv` — la grid search

Scritti da `grid_search.py` nella cartella dello split (`results/<dataset>/<split>/`), uno per set di domande. Ricalcolano i
risultati di ogni modello per **tutte le combinazioni** di regola di decisione, temperatura e soglia, partendo solo dai file
`answers.jsonl` e `run.json` dei run: nessuna inferenza.

```sh
python grid_search.py                                                     # griglia di default
python grid_search.py --thresholds 0.05:0.95:0.05 --temperatures default,0.5,1,1.5,2
python grid_search.py --models jpt-9b rune-26b --rules rule,support_simple
```

Di default entrano solo i run `complete` e non `-limitN` (`--include-incomplete`, `--include-limited` per includerli).

### I tre parametri

| Parametro | Valori | Significato |
|---|---|---|
| `rule` | `rule` | la regola del set di domande (per `claim_support`: min di P(sì) di `support_simple`, 1 − P(sì) di `unsupported_detail`, 1 − P(sì) di `contradicted_detail`). Per la baseline FactCG è la sua regola propria (`factcg_support`) |
| | `support_simple`, `unsupported_detail`, `contradicted_detail` | una sola domanda, presa nel verso in cui la usa la regola: `support_simple` come P(sì); le altre due come 1 − P(sì) (cioè "probabilità che **non** ci sia il problema") |
| `temperature` | `default` | le probabilità salvate in `answers.jsonl`, cioè con la temperatura propria del modello (JPT 1.087, Rune 2, Winnow 1, CLM 1, d1-3B e FactCG nessuna) |
| | un numero | probabilità ricalcolate dai logit salvati: `softmax(logits / T)`. Non esiste per i modelli senza logit (d1-3B, FactCG): quelle combinazioni mancano |
| `threshold` | 0–1 | il claim è "supportato" se il punteggio della regola è **strettamente maggiore** della soglia |

Una combinazione è quindi (modello, regola, temperatura, soglia). La ROC AUC non dipende dalla soglia: è la stessa su tutte
le righe con uguali modello, regola e temperatura.

### `grid_<hash>.md`

1. **Intestazione**: dataset, split, set di domande, griglia usata, numero di combinazioni e tempo di calcolo.
2. **Summary**: due righe per modello.
   - `default`: regola del set, temperatura `default`, soglia del set (0,30). Coincide con `report.json` e con la tabella
     di `summary_<hash>.md`.
   - `best Average`: la combinazione con la media per fonte più alta. In grassetto il valore migliore di ogni colonna.
3. **Una sezione per modello**: tutte le sue combinazioni, ordinate dalla migliore (o le prime N con `--top N`).

Colonne (valori in **percentuale**):

| Colonna | Significato |
|---|---|
| `Model`, `Setting`, `Rule`, `T`, `Threshold` | il modello e la combinazione. `Rule` scrive per esteso la regola (es. `min of 3`, `support_simple (P(yes))`, `unsupported_detail (1 - P(yes))`) |
| `Average` | **macro balanced accuracy**: media non pesata della balanced accuracy delle 11 fonti (la metrica della classifica LLM-AggreFact, la colonna "Average" di `table.md`) |
| `Bal. acc.` | balanced accuracy su tutte le righe insieme: (recall dei supportati + recall dei non supportati) / 2 |
| `Accuracy` | frazione di predizioni corrette su tutte le righe (dominata dalle fonti grandi e dalla classe più frequente) |
| `ROC AUC` | AUC del punteggio continuo su tutte le righe (indipendente dalla soglia) |
| `Macro ROC AUC` | media delle AUC delle 11 fonti |
| `CNN` … `RAGTruth` | balanced accuracy di ciascuna fonte, nell'ordine di `table.md` |

> **Attenzione**: la riga `best` è scelta sullo stesso split su cui è misurata, quindi è una stima **ottimistica**. Serve
> per esplorare (quale regola e quale soglia funzionano meglio per un modello), non come risultato finale: una scelta va
> fatta su uno split di sviluppo e poi confermata sul test.

### `grid_<hash>.csv`

Una riga per combinazione, per tutti i modelli insieme. È il file da usare per analisi automatiche: contiene tutto ciò che
c'è nel `.md` e di più. I valori sono **frazioni 0–1** (non percentuali); una cella vuota significa "non definito" (es.
una fonte con una sola classe).

| Colonna | Significato |
|---|---|
| `model` | cartella del modello in `models/` (es. `jpt-9b`) |
| `model_name` | nome per le tabelle (es. `JPT-9B`; con `[hash parametri]` se ci sono più varianti dello stesso modello) |
| `run` | cartella del run da cui vengono i dati (es. `jpt-9b_5147a4ec_551bb722`) |
| `rule` | `rule` oppure l'id di una domanda (vedi sopra) |
| `rule_definition` | la regola esatta in JSON: `combine` (`min`, `single`, ...), `inputs` (domanda e `take`: `yes` = P(sì), `no` = 1 − P(sì)) |
| `temperature` | `default` oppure il valore numerico |
| `threshold` | la soglia |
| `macro_balanced_accuracy` | media non pesata della balanced accuracy per fonte (= `Average`) |
| `balanced_accuracy`, `accuracy` | su tutte le righe insieme |
| `roc_auc`, `macro_roc_auc` | AUC su tutte le righe e media delle AUC per fonte |
| `supported_recall` | frazione dei claim supportati (label 1) predetti come supportati |
| `unsupported_recall` | frazione dei claim non supportati (label 0) predetti come non supportati |
| `bacc_<fonte>` | balanced accuracy della fonte, col nome originale del dataset (`AggreFact-CNN`, `AggreFact-XSum`, `TofuEval-MediaS`, `TofuEval-MeetB`, `Wice`, `Reveal`, `ClaimVerify`, `FactCheck-GPT`, `ExpertQA`, `Lfqa`, `RAGTruth`) |
| `auc_<fonte>` | ROC AUC della fonte |

Corrispondenza tra nomi nel CSV e colonne di `table.md`: AggreFact-CNN = CNN, AggreFact-XSum = XSum, TofuEval-MediaS =
MediaS, TofuEval-MeetB = MeetB, Wice = WiCE, Reveal = REVEAL, FactCheck-GPT = FactCheck, Lfqa = LFQA; le altre uguali.

### Leggerli in un altro repository

Il CSV è autosufficiente: per analizzare la grid search altrove bastano `grid_<hash>.csv` (e, per leggibilità,
`grid_<hash>.md`). Per risalire al contesto di un run (opzioni, GPU, versioni) servono anche `run.json` e `report.json`
della cartella indicata nella colonna `run`, e `hash_questions.json` per il testo delle domande.

```python
import pandas as pd

df = pd.read_csv("grid_551bb722.csv")

# la combinazione migliore per modello, secondo la media per fonte
best = df.loc[df.groupby("model")["macro_balanced_accuracy"].idxmax()]

# la configurazione del benchmark (regola del set, temperatura propria, soglia 0.30)
default = df[(df.rule == "rule") & (df.temperature == "default") & (df.threshold.round(2) == 0.30)]

# l'effetto della soglia per un modello, con regola e temperatura fissate
curve = df[(df.model == "jpt-9b") & (df.rule == "rule") & (df.temperature == "default")][["threshold", "macro_balanced_accuracy"]]
```

Nota su `temperature`: la colonna mescola il testo `default` e numeri, quindi pandas la legge come testo; per filtrare un
valore numerico usare ad esempio `df.temperature == "1.5"` (i valori interi sono scritti `1.0`, `2.0`), oppure convertire
con `pd.to_numeric(df.temperature, errors="coerce")`, che trasforma `default` in `NaN`.
