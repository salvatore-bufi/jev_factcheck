# CLAUDE_README — stato del lavoro "testa di evidenza per frase"

> File per riprendere la sessione. Chi apre una nuova conversazione (Claude o l'utente) legge questo file per primo,
> poi `evidence/code_report.md` (spiegazione del codice, comandi in ordine). Va aggiornato alla fine di ogni blocco.

## 1. Che cosa stiamo facendo

Implementiamo il piano in `Documentation/fine_tuning_factcheck_evidence/testa_per_frase/` (file 00–15), **solo per KnowLine-4B**.

Al decision model si aggiunge una piccola testa (circa 3,7 M parametri) che, dallo **stesso forward** del verdetto, dà un punteggio a ogni frase del documento e indica le frasi di evidenza quando il claim è supportato. Il modello è congelato, quindi il verdetto non cambia.

## 2. Metodo di lavoro (deciso dall'utente)

- **2026-11-04, prima regola** (poi abbandonata): costruire tutto pezzo per pezzo, con l'utente che crea i file.
- **2026-11-04, regola attuale**: **Claude scrive tutto il codice**, l'utente lo studia dopo. Il codice deve essere SEMPLICE e ben commentato:
  - un'intestazione per file (cosa fa, dove sta nel workflow, come si lancia, cosa legge e scrive);
  - una docstring per funzione con cosa fa, input e output.
- La spiegazione di tutto è in `evidence/code_report.md`, da tenere aggiornato.
- Lingua con l'utente: italiano. Commenti nel codice: inglese, come il resto della repo.

## 3. Decisioni prese (con l'utente)

- **Perimetro**: tutta la pipeline obbligatoria (split → server), solo KnowLine-4B.
  - Fuori: passo 03 (dati generati, quindi niente `test_evid_sparse` né prove di comportamento), altri modelli, fase 1b, fase 2, sezione 14.
  - E-EMB escluso (manca `sentence-transformers`).
- **Verdetto**: KnowLine-4B così com'è, perché nella repo non esiste una LoRA del fine-tuning binario.
  - Una domanda, `support_simple` (identica a `config/questions/q1_only.json`), soglia 0,5.
  - Un forward per riga.
- **SciFact, strada B**: train e validation uniti, poi 80/10/10 per `doc_id`. Lo split d'origine è per claim: 295 righe su 919 del train condividono l'abstract con la validation. Il test SciFact quindi **non** coincide con la validation ufficiale.
- **HoVer**:
  - test = dev d'origine, solo righe `exact`;
  - dev = **10%** dei documenti del train d'origine, solo `exact` (con il 5% `dev_evid_select` aveva solo 207 positivi);
  - le righe `db_split` fuori dal train si scartano;
  - le 6 righe del train con documento nel test si tolgono.
- **WiCE**: dev e test d'origine vanno in `test_evid_ood`. Nel train solo `aggrefact_overlap is False`; le righe `none` (parzialmente supportate) restano negli split e le toglie il mix.
- **Altre regole degli split**:
  - `dev_evid_neg` contiene tutti i negativi del dev;
  - split per hash `sha256(seed:salt:doc_id)`;
  - la distanza in token si calcola dopo il rendering.
- **File diversi dalla doc**: `models.py` diventa `model.py`, `feature_dataset.py` diventa `features.py`. Ricette e config stanno in `evidence/recipes/` ed `evidence/configs/`.
- **Feature**: un tensore per layer negli shard, per leggere solo i layer necessari. Shard da 250 righe; lettura in mmap, perché con 62 GB di RAM le feature complete non entrano.
- **`.gitignore`**: aggiunti `/data/` e `/releases/`.

## 4. Stato degli step

| # | Step | File | Stato |
|---|---|---|---|
| 1 | Split e mix | `data/make_splits.py`, `data/make_mix.py`, `recipes/ev_head.json` | ✅ scritti e **lanciati** (output in `dataset/evidence/splits/`, `mixes/`) |
| 2 | Rendering | `render.py` | ✅ verificato (frasi = token su circa 2.700 righe) |
| 3 | Hook e forward | `model.py` | ✅ self-test superato |
| 4 | Scelta del preambolo | `check_preamble.py` | ✅ scritto, ⏳ **da lanciare** (circa 25 min GPU) |
| 5 | Estrazione | `extract.py`, `features.py` | ✅ provata su 300 righe, ⏳ **estrazione completa da lanciare** (circa 2,5 h, 120 GB) |
| 6 | Baseline | `baselines.py` | ✅ provata su un campione |
| 7 | Testa e loss | `head.py`, `losses.py`, `metrics.py` | ✅ test superati |
| 8 | Training 1a | `train_head.py`, `configs/*.json` (12 esperimenti) | ✅ provato (overfit 64 = F1 1; training di prova) |
| 9 | Valutazione e latenza | `eval_evidence.py`, `latency.py` | ✅ provati (sufficienza; latenza +0,5%) |
| 10 | Esperimenti | registro `data/evidence/runs/registry.md` | ⏳ dopo l'estrazione |
| 11 | Rilascio e server | `release.py`, `serve.py` | ✅ provati (server = valutazione su 60/60 righe) |
| — | Test | `tests/test_evidence_{splits,render,head}.py` | ✅ tutti superati |
| — | Report | `evidence/code_report.md` | ✅ |

Gli artefatti di prova (feature di 300 righe, run `*_smoke`, `releases/smoke-test`) sono stati **cancellati**: `data/` è vuota.

## 5. Dove siamo arrivati

- **Ultimo aggiornamento:** 2026-11-04
- **Fatto:** tutto il codice è scritto e provato su piccoli campioni.
- **Prossima azione (dell'utente, o di Claude se richiesto):**
  1. studiare il codice partendo da `evidence/code_report.md`;
  2. lanciare `python -m evidence.check_preamble`;
  3. lanciare l'estrazione completa del mix e dei dev (comandi in code_report §4);
  4. lanciare baseline e training H0 → esperimenti, compilando la colonna "decision" del registro.
- Niente è stato committato: i file sono non tracciati, sul branch `finetune_evidence`.

## 6. Questioni aperte

- La sonda per layer (doc 15 §3) non è implementata a parte: i pesi della miscela (H2a) e gli esperimenti fanno lo stesso lavoro. Da aggiungere se si vuole fissare i layer prima.
- E-ABL su 5 righe ha dato F1 0,45 con insieme completo 0: KnowLine sembra dire "supportato" anche con una sola frase. Da verificare sul set completo.
- Se `check_preamble` dice `acceptable: false`, il preambolo non resta nelle tolleranze. Non c'è una LoRA da riaddestrare: si decide se tenerlo comunque (H4 misura quanto aiuta l'evidenza).

## 7. Diario

- 2026-11-04: letta tutta la doc; deciso il metodo di lavoro; decise le regole degli split (SciFact B, HoVer al 10%).
- 2026-11-04: l'utente passa al metodo "Claude scrive tutto". Scritta e provata l'intera pipeline (sezione 4); scritto `evidence/code_report.md`.
