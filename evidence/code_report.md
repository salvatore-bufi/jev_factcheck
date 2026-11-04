# Code report — testa di evidenza per frase (KnowLine-4B)

Questo documento spiega il codice in `evidence/`: cosa fa ogni file, come si collegano, in che ordine si lanciano, e cosa ci si deve aspettare. Il piano da cui nasce è in [Documentation/fine_tuning_factcheck_evidence/testa_per_frase/](../Documentation/fine_tuning_factcheck_evidence/testa_per_frase/00_overview.md). Le decisioni prese lungo la strada sono in [CLAUDE_README.md](../CLAUDE_README.md).

## 1. L'idea in tre righe

KnowLine-4B legge `documento + claim + domanda` e dà P(supportato) all'ultimo token, con **un solo forward**. Durante quello stesso forward catturiamo, per ogni frase del documento, alcuni vettori che il modello calcola comunque. Una piccola rete (la **testa**, circa 3,7 M parametri) li trasforma in un punteggio per frase e in una soglia per documento: le frasi sopra soglia sono l'evidenza.

Il modello è congelato: il verdetto non cambia. Si addestra solo la testa.

## 2. Il workflow

```text
 dataset/evidence/{scifact,wice,hover}.jsonl            (già costruiti: evidence/data/build.py)
        │
        ▼  STEP 1  evidence/data/make_splits.py   → dataset/evidence/splits/*.jsonl
        ▼  STEP 1  evidence/data/make_mix.py      → dataset/evidence/mixes/ev_head.jsonl   (con evidence/recipes/ev_head.json)
        │
        ▼  STEP 4  evidence/check_preamble.py     → dataset/evidence/preamble_choice.json  (che preambolo usare)
        │
        ▼  STEP 5  evidence/extract.py            → data/evidence/features/knowline-4b/<set>/   (GPU, ore)
        │            usa: render.py (prompt + token delle frasi), model.py (forward con hook)
        │
        ├─▶ STEP 6  evidence/baselines.py         → data/evidence/runs/baselines.json
        │
        ▼  STEP 8  evidence/train_head.py         → data/evidence/runs/knowline-4b/<id>/       (minuti)
        │            usa: features.py (lettura), head.py (rete), losses.py, metrics.py, configs/<id>.json
        │
        ▼  STEP 9  evidence/eval_evidence.py      → <run>/eval/<set>/metrics.json
        ▼  STEP 9  evidence/latency.py            → <run>/latency.json
        │
        ▼  STEP 11 evidence/release.py            → releases/<nome>/
        ▼  STEP 11 evidence/serve.py              → server HTTP  POST /v1/evidence
```

## 3. I file, uno per uno

### Configurazione

| File | Cosa fa |
|---|---|
| `config.py` | Tutte le costanti in un posto solo: percorsi, revisione di KnowLine, la **domanda** del verdetto (`support_simple`, identica a `config/questions/q1_only.json`), la soglia 0,5, i preamboli candidati (`P-A`, `P-B`), i layer letti (19, 23, 27, 31), i layer ad attenzione piena, i limiti di token. `preamble_template()` restituisce il preambolo scelto da `check_preamble.py`, o `P-A` se la scelta non è ancora stata fatta. |
| `recipes/ev_head.json` | La ricetta del mix: per ogni famiglia di righe (es. HoVer positivi `exact`) il termine di loss, il peso della riga e ω. Vince la **prima** famiglia che corrisponde. Un esperimento sui dati (H7: ω, H12: peso di `db_split`) è una ricetta nuova, non codice nuovo. |
| `configs/*.json` | Un file per esperimento della testa (doc 11): layer letti, parti (`end`/`mean`), opzioni della testa, loss, training. Ce ne sono 12: `h0`, `h_loss`, `h_mil`, `h_mil1`, `h2a`, `h2b`, `h2c`, `h3a`, `h3c1`, `h3c4`, `h9`, `h_pair`. |

### Dati (STEP 1)

| File | Input → Output | Cosa fa |
|---|---|---|
| `data/make_splits.py` | `<fonte>.jsonl` → `splits/<set>.jsonl` + `.stats.json` + `manifest.json` | Divide **per documento** (`doc_id`), con un hash deterministico (`unit()`): lo stesso documento cade sempre nello stesso split. Regole: SciFact unito e 80/10/10 (strada B); HoVer train d'origine 90/10 train/dev, dev d'origine come test (solo righe `exact`); WiCE dev e test d'origine come test fuori dominio, e mai in train le righe che si sovrappongono a LLM-AggreFact. Un documento in due stadi resta solo nel più "alto" (test > dev > train). Il `check()` finale si ferma se trova una sovrapposizione. |
| `data/make_mix.py` | `splits/train.jsonl` + ricetta → `mixes/<nome>.jsonl` + manifest | Applica la ricetta. Ogni riga riceve `evidence_term`, `row_weight` e `omega`; le WiCE parziali escono; i negativi si possono sottocampionare (`neg_fraction`). Stampa i vincoli della doc 04 §3 e si ferma se c'è una fuga di documenti verso dev o test. |

### Dal testo al modello (STEP 2–3)

| File | Cosa fa |
|---|---|
| `render.py` | **Il file più delicato.** Costruisce il prompt di KnowLine con il codice degli autori (`knowline_layout.render_request`, importato dallo snapshot HF) e trova, per ogni frase, il primo e l'ultimo token nel prompt. Passi: (1) il preambolo va in testa al turno utente dello stato; senza preambolo il prompt è identico byte per byte a quello del benchmark; (2) `document_in_prompt` ricostruisce il documento come lo scrive KnowLine, con le righe indentate di 2 spazi, e la mappa `origin` dal carattere del prompt al carattere originale; (3) si trova l'**ultima** copia del documento, perché KnowLine scrive due volte le richieste fino a 3.000 token; (4) gli offset del tokenizer danno i caratteri originali di ogni token (`token_char_spans`); (5) ogni token va alla frase che contiene il suo ultimo carattere non spazio (`sentence_token_spans`). `check_row` verifica che i token decodificati di ogni frase ridiano la frase. `python -m evidence.render 200` lo controlla su 200 righe per set. |
| `model.py` | Carica KnowLine (bf16) e fa **un forward** che restituisce il verdetto e le feature. Due meccanismi: (1) **hook** sui layer 19, 23, 27 e 31 (`SentenceTap`): per ogni frase calcolano subito sulla GPU il vettore dell'ultimo token (`h_end`) e la media dei token (`h_mean`, con somme cumulative), più il vettore della posizione di risposta (`q`); (2) una **funzione di attenzione** registrata in transformers (`sdpa_capture`): fa l'attenzione normale e, negli 8 layer pieni, ricalcola una riga di pesi (risposta → tutti i token). `attention_per_sentence` somma quei pesi per frase: 8 × 16 = 128 numeri. `python -m evidence.model --selftest` controlla verdetto invariato, riassunti uguali a `hidden_states` e attenzione uguale alla modalità "eager". |

### Estrazione e lettura delle feature (STEP 5)

| File | Cosa fa |
|---|---|
| `extract.py` | Per ogni riga: render, poi forward, poi salvataggio. Scrive shard `safetensors` da 250 righe, con **un tensore per layer** (`h_end.19`, `h_end.23`, …): così chi legge solo il layer 31 legge un ottavo dei dati. In più `index.jsonl` (una riga per riga: dove sta, frasi gold, pesi, posizioni in token, P(supportato)), `meta.json` (impostazioni e hash del codice: il training rifiuta di mescolare feature diverse) e `skipped.json`. Riprende dagli shard già completi. Le righe di training troppo lunghe (> 16.384 token) vengono **accorciate** (`shorten`/`rebuild`): i positivi perdono frasi non gold, a partire dalle più lontane dall'evidenza, e i negativi perdono la fine. Dev e test troppo lunghi vengono saltati. |
| `features.py` | `FeatureStore`: legge le feature dal disco **solo quando servono** (safetensors in mmap), e solo i layer e le parti richiesti. Le feature complete (circa 100 GB) non devono stare in RAM. |

### Testa, loss, metriche (STEP 7)

| File | Cosa fa |
|---|---|
| `head.py` | `EvidenceHead`, stadio per stadio: (A) miscela dei layer con pesi appresi più LayerNorm; (B) vettore di ogni frase = proiezione di `[h_end; h_mean]` + proiezione di `[att; log n_tok]` + posizione; vettore di risposta `x0` da `q`; (C) transformer **non causale** a 2 layer su `[x0, frasi]`, in cui ogni frase vede tutte le altre (è questo che risolve l'evidenza sparsa); (D) un'unica uscita lineare dà `s[k]` per le frasi e `s0` (la soglia del documento) per `x0`. Ogni pezzo ha un interruttore (`use_mix`, `use_mean`, `use_att`, `use_pos`, `layers=0`, `use_pair`). `load_run` ricarica una testa addestrata. |
| `losses.py` | `ref_loss` (H0): BCE pesata per frase (le gold pesano α = 10) + termine di ordinamento, con soglia fissa. `zlpr_loss` (principale): una frase è evidenza se e solo se `s[k] > s0`. `supervised` spinge le gold sopra `s0` e le altre sotto (pesate ω); `mil_neg` spinge tutte sotto `s0` sui claim non supportati. `pair_loss` serve a H-PAIR. |
| `metrics.py` | `select` (frasi con punteggio > soglia, al massimo 5, almeno 1); `sentence_metrics` (precisione, recall, F1, insieme completo, esatto); `summarize`, che calcola tutte le metriche della doc 10 su una lista di **predizioni** (formato comune a baseline, training e valutazione): F1 per distanza in frasi e in token, negativi accesi, recall sui positivi incerti, `score` = media di F1 e insieme completo, usato come criterio del checkpoint. L'evidenza si misura solo sui positivi che il verdetto giudica supportati. |

### Baseline, training, valutazione (STEP 6, 8, 9)

| File | Cosa fa |
|---|---|
| `baselines.py` | E-RAND (casuale), E-LEX (BM25 claim–frase, scritto a mano perché `rank_bm25` non è installato), E-ATT (le 8 migliori combinazioni layer/testa/normalizzazione dell'attenzione, scelte sulla metà A del set), E-ABL (ablazione greedy con il modello: il gruppo più piccolo di frasi che da solo fa dire "supportato"). Tutto si sceglie sulla metà A e si misura sulla metà B. E-EMB è escluso (manca `sentence-transformers`). |
| `train_head.py` | Fase 1a: righe raggruppate per lunghezza, AdamW, riscaldamento e coseno, clipping del gradiente, arresto anticipato con il checkpoint migliore su `dev_evid_select`. Dopo il training sceglie la **regola di selezione** sul dev: τ_e per la loss di riferimento, δ per ZLPR (δ ≠ 0 solo se guadagna almeno 1 punto). Misura poi confirm e neg, salva la run e aggiunge una riga a `data/evidence/runs/registry.md`. `--overfit 64` è il controllo da fare prima di tutto: la testa deve imparare 64 righe a memoria. |
| `eval_evidence.py` | Metriche su un set con la regola salvata, metriche del verdetto (BAcc, AUC, ECE) e, con `--sufficiency`, sufficienza e necessità (2 forward per riga: documento ridotto alle sole frasi mostrate, e documento senza di esse). |
| `check_preamble.py` | STEP 4: verdetto con nessun preambolo, `P-A` e `P-B` sugli stessi dev. Sceglie il preambolo più vicino a "nessuno" in AUC (a parità, il più corto) e dice se resta nelle tolleranze della doc 05. |
| `latency.py` | Latenza A (senza preambolo) / B (con preambolo) / C (con preambolo + evidenza), p50 e p95. Obiettivo: C ≤ B + 3% (o + 2 ms). |

### Rilascio e server (STEP 11)

| File | Cosa fa |
|---|---|
| `release.py` | Crea `releases/<nome>/`: testa, config, regola, `decision.json` (tutte le impostazioni), una copia del divisore in frasi **con i parametri `punkt_tab`**, `frozen.json` (hash), `report.md` e `MODEL_CARD.md`. `--verify` confronta gli hash con la macchina attuale. |
| `serve.py` | `Service.answer(document, claim)`: split, poi render, poi un forward, poi P(supportato); se è supportato, testa, selezione e posizioni nel documento originale. Server HTTP minimo (`POST /v1/evidence`, `GET /health`) e `--demo`. All'avvio da una release verifica gli hash. È lo stesso codice della valutazione. |

### Test

| File | Cosa controlla |
|---|---|
| `tests/test_evidence_splits.py` | Regole di split e mix su righe scritte a mano. |
| `tests/test_evidence_render.py` | Mappa frasi → token su documenti difficili (virgolette, backslash, tab, `<`, emoji, cirillico, greco, giapponese, `\r\n`); prompt identico al benchmark senza preambolo; frasi lette dalla seconda copia. Serve il tokenizer nella cache HF, non la GPU. |
| `tests/test_evidence_head.py` | Forme, invarianza al riempimento, esempi numerici della doc (1,975; 2,526 → 1,473), gradiente zero per le righe `none`, esempi delle metriche. |

Si lanciano con `python tests/<file>.py`; con pytest installato, anche `python -m pytest tests`.

## 4. I comandi, in ordine

```sh
# 0. test (CPU, secondi)
for t in tests/test_evidence_*.py; do python $t; done

# 1. split e mix (CPU, ~10 s)
python -m evidence.data.make_splits
python -m evidence.data.make_mix --recipe evidence/recipes/ev_head.json

# 2-3. controlli su rendering e modello (CPU / GPU, ~1 min)
python -m evidence.render 400
python -m evidence.model --selftest

# 4. scelta del preambolo (GPU, ~1.200 righe x 3 forward ≈ 25 min)
python -m evidence.check_preamble

# 5. estrazione (GPU; ~2,5 righe/s su RTX 5090 Laptop: mix ≈ 2,3 h, dev ≈ 8 min, test ≈ 25 min; ~120 GB)
F=data/evidence/features/knowline-4b; S=dataset/evidence/splits
python -m evidence.extract --rows dataset/evidence/mixes/ev_head.jsonl $S/dev_evid_select.jsonl \
    $S/dev_evid_confirm.jsonl $S/dev_evid_neg.jsonl
python -m evidence.extract --rows $S/test_evid_id.jsonl $S/test_evid_ood.jsonl      # solo alla fine (test)

# 6. baseline (CPU per rand/lex/att; GPU per abl)
python -m evidence.baselines --features $F/dev_evid_select --rows $S/dev_evid_select.jsonl --method rand lex att
python -m evidence.baselines --features $F/dev_evid_select --rows $S/dev_evid_select.jsonl --method abl --per-source 100

# 8. training: prima il controllo, poi H0 e gli esperimenti uno alla volta (doc 11)
python -m evidence.train_head --config evidence/configs/h0.json --train $F/ev_head --overfit 64
for c in h0 h_loss h_mil h2a h2b h2c; do
  python -m evidence.train_head --config evidence/configs/$c.json --train $F/ev_head \
      --dev $F/dev_evid_select --confirm $F/dev_evid_confirm --dev-neg $F/dev_evid_neg
done
cat data/evidence/runs/registry.md        # i numeri per decidere; la colonna "decision" la compili tu

# 9. valutazione della configurazione scelta (sui dev; i test una volta sola, a configurazione congelata)
python -m evidence.eval_evidence --run data/evidence/runs/knowline-4b/<id> --features $F/dev_evid_select \
    --rows $S/dev_evid_select.jsonl --sufficiency 300
python -m evidence.latency --run data/evidence/runs/knowline-4b/<id> --n 1000

# 11. rilascio e server
python -m evidence.release --run data/evidence/runs/knowline-4b/<id> --name knowline-4b-fc-ev-v1
python -m evidence.eval_evidence --run releases/knowline-4b-fc-ev-v1 --features $F/test_evid_id \
    --rows $S/test_evid_id.jsonl --sufficiency 0
python -m evidence.serve --run releases/knowline-4b-fc-ev-v1 --port 8090
```

Per gli esperimenti con più semi: `--seed 14 --tag seed14`. Per H4 (senza preambolo): estrarre con `--preamble none --suffix _nopre` e addestrare su quelle feature (anche il dev va estratto allo stesso modo).

## 5. Cosa è stato verificato (2026-11-04, RTX 5090 Laptop)

| Controllo | Risultato |
|---|---|
| Split | nessun documento condiviso fra train, dev e test; dev_evid_select 391 positivi, dev_evid_confirm 391, dev_evid_neg 394, test_evid_id 1.902, test_evid_ood 1.884, train 20.717 |
| Mix `ev_head` | 20.249 righe; negativi 38,9%; positivi con ≥ 2 gold 93%; con gold a ≥ 10 frasi 66%; nessuna fuga |
| Rendering | token di ogni frase = frase su 1.185 righe di dev e WiCE e su 1.500 righe del mix; prompt identico al benchmark senza preambolo |
| Self-test del modello | verdetto con hook = senza hook (ΔP = 0); `h_end`/`h_mean` uguali a `hidden_states` (diff. relativa 0,003, bf16); attenzione per frase contro "eager": correlazione 0,9994 |
| Accorciamento | righe da 18.700 a 16.100 token, frasi gold conservate |
| Training di prova (300 righe di train, 300 di dev) | 64 righe imparate a memoria (F1 = 1); H0 F1 0,66, H-MIL 0,68, H2c 0,78; negativi accesi da 0,95 (H0) a 0,35 (H2c) |
| Baseline (stesso campione) | E-RAND 0,07, E-LEX 0,60, E-ATT 0,75 |
| Server | stesse frasi della valutazione su 60/60 righe; l'esempio di Nancy Howell trova le frasi 0 e 4 |
| Latenza | C − B = +1,8 ms (+0,5%), testa 1,1 ms, divisione in frasi 1,3 ms |

Questi numeri vengono da campioni minuscoli: servono solo a dire che la pipeline funziona. I numeri veri vengono dall'estrazione completa. Gli artefatti di prova sono stati cancellati.

## 6. Scelte e limiti da conoscere

- **Il verdetto è KnowLine-4B così com'è**: non c'è una LoRA del fine-tuning binario. La domanda è `support_simple`, con soglia 0,5. Per usare un modello fine-tunato in futuro, cambiano `load_model` in `model.py` e `config.QUESTION`.
- **Solo KnowLine-4B.** Per gli altri modelli servirebbero un renderer per ciascuno (stato in JSON: la mappa `escape_with_map` della doc 06) e la loro maschera di attenzione dentro `sdpa_capture`.
- **Fuori perimetro**: dati generati del passo 03 (quindi anche `test_evid_sparse` e le prove di comportamento della doc 10 §4), fase 1b, fase 2, E-EMB, la sezione 14.
- **Il preambolo cambia le feature.** Prima dell'estrazione completa va lanciato `check_preamble.py`: se sceglie `P-B`, tutte le feature vanno estratte con `P-B`.
- **Batch 1 nel modello**: semplice, e senza riempimento da gestire nelle maschere. Basta per l'estrazione (circa 2,5 righe/s) e per il server.
- **Modificare `sentences.py`, `render.py` o `model.py` invalida le feature** (hash in `meta.json`) e le release (hash in `frozen.json`). È voluto.
- `data/` e `releases/` sono nel `.gitignore`.
