# Fine-tuning di un fact checker binario: guida operativa completa

## 0. Come usare questa guida

Questa guida spiega **come fare**, passo per passo, il fine-tuning di un decision model che, dato un documento e un'affermazione (claim), dice se il documento la supporta. È scritta per essere **autosufficiente**: non serve leggere i paper di MiniCheck o FactCG. Ogni procedura riporta dati di partenza, comandi, prompt, codice di riferimento, controlli e criteri per passare al passo successivo.

I modelli sono **i 10 decision model e i 2 backbone dello studio focalizzato** ([../paper_idea/15_focused_10_tdm_experiment_guide.md](../paper_idea/15_focused_10_tdm_experiment_guide.md) §5.2). Il fine-tuning serve anche all'esperimento B-ctrl del paper ([../paper_idea/04_parte_B_finetuning.md](../paper_idea/04_parte_B_finetuning.md)): stessa ricetta su un decision model e sul suo backbone.

La cartella [summary/](summary/00_overview.md) contiene la versione riassuntiva dello stesso piano; i numeri dei passi coincidono.

| File | Contenuto |
|---|---|
| `00_overview.md` | questo file: il compito, i modelli, il glossario, l'ambiente, la mappa del lavoro |
| `00a_scelta_approccio.md` | perché si parte da un decision model con una sola domanda; cosa sono MiniCheck e FactCG |
| `00b_split_e_protocollo.md` | come costruire e congelare train / dev / test, con il codice |
| `01_baseline.md` | come misurare i modelli prima del training |
| `02a_dataset_fonti_esistenti.md` | le fonti pubbliche, come si trasformano in coppie, ANLI difficile |
| `02b_dataset_generazione_llm.md` | come generare i dati alla MiniCheck (C2D, D2C) e alla FactCG (CG2C), con i prompt |
| `02c_replay_e_forgetting.md` | perché il replay non serve e come ancorare la calibrazione |
| `02d_varianti_domanda.md` | come scegliere e congelare la domanda |
| `02e_mix_finale.md` | come costruire i due training set (stadio 1 e stadio 2) |
| `03_rendering_formato.md` | il prompt, la maschera di attenzione e la lettura di ciascun modello, e il chunking |
| `04_loss.md` | la funzione obiettivo, con derivazioni ed esempi numerici |
| `05_training.md` | lo script di training, la configurazione LoRA, l'hardware |
| `06_valutazione.md` | metriche, test comportamentali, report |
| `07_esperimenti.md` | il piano delle prove e come decidere |
| `08_merge_export.md` | fusione dei pesi, soglia finale, servizio con il runtime degli autori |
| `09_rischi_checklist.md` | cosa può andare storto e checklist completa |

Il PDF unico con tutte le sezioni è in `pdf/fine_tuning_factcheck_guida_completa.pdf`.

## 1. Il compito

**Input**: un documento (testo in prosa, anche lungo) e un claim (una o più frasi).
**Output**: P(supportato) ∈ [0, 1], e la decisione `P > soglia`.

**Etichetta** (convenzione di LLM-AggreFact, usata ovunque in questa guida):

- **1 = supportato**: ogni parte del claim è stabilita dal documento (parafrasi fedeli e implicazioni dirette contano);
- **0 = non supportato**: almeno una parte è **contraddetta** dal documento **oppure non stabilita** (il documento non dice nulla in proposito).

Tre esempi con lo stesso documento:

> Documento: "The meeting was moved to Tuesday at 10:00. It will be held in Rome."

| Claim | Etichetta | Perché |
|---|:-:|---|
| "The meeting is on Tuesday in Rome." | 1 | tutte le parti sono nel documento |
| "The meeting is on Wednesday." | 0 | contraddetto (il documento dice martedì) |
| "The meeting lasts two hours." | 0 | non stabilito (il documento non parla di durata) |

Il documento è **sempre fornito**: non c'è retrieval. Il modello non deve usare conoscenze proprie: un claim vero nel mondo ma assente dal documento è 0.

## 2. I modelli

Tutti e 12 sono **Qwen3.5**: 32 layer, di cui 24 Gated DeltaNet (attenzione lineare, ricorrente) e 8 ad attenzione piena (layer 3, 7, 11, 15, 19, 23, 27, 31). Lo stato nascosto è lungo 4.096 numeri nel gruppo 9B e 2.560 nel gruppo 4B. Ciò che cambia da un modello all'altro è la **ricetta**: il prompt, la maschera di attenzione (causale o bidirezionale sul prompt) e il modo in cui si legge la risposta.

| ID | Modello | Base | Attenzione sul prompt | Lettura della risposta | Ruolo in questo piano | Licenza |
|---|---|---|---|---|---|---|
| A11 | **KnowLine-4B** (`PelaAI/KnowLine-4B-Gen5.5`) | Qwen3.5-4B | causale; richieste brevi scritte due volte | token delle etichette (A = Yes, B = No), temperatura già nei pesi | **si itera qui** | Apache-2.0 |
| A1 | **elephant-1** (`baselabs/elephant-1`) | Qwen3.5-9B | bidirezionale a blocchi nei layer pieni | lettere A/B dalla testa del vocabolario | **modello principale a 9B** | MIT |
| A2 | StartLux-9B-Instant | Qwen3.5-9B | bidirezionale sul prompt | lettere A/B, temperatura per tipo di domanda | estensione | CC BY-NC |
| A3 | d3-flash | Qwen3.5-9B | non causale nei layer pieni | 255 codici di risposta | estensione | Apache-2.0 |
| A4 | OneJev1.6-9B | Qwen3.5-9B | causale | lettere A/B (`slot_lm`) | estensione | Apache-2.0 |
| A5 | Nimble-9B-v3 | Qwen3.5-9B + LoRA r64 | causale, prefisso condiviso | 255 codici di risposta | estensione | CC BY-NC |
| A7 | Clef-Flash | Qwen3.5-9B | come caricato | testa congiunta propria (4 layer) | estensione | Apache-2.0 |
| A12 | StartLux-4B-Instant | Qwen3.5-4B | bidirezionale sul prompt | lettere A/B | estensione | CC BY-NC |
| A13 | pplx-decider-flash | Qwen3.5-4B | non causale nei layer pieni | testa di decisione propria a 255 vie | estensione | Apache-2.0 |
| A14 | Jeffan-4B | Qwen3.5-4B | bidirezionale nei layer pieni | codici A/B dalla testa del vocabolario | estensione | Apache-2.0 |
| BB9 | Qwen3.5-9B | — | causale | lettere A/B con prompt neutro | controllo di A1–A7 (B-ctrl) | Apache-2.0 |
| BB4 | Qwen3.5-4B | — | causale | lettere A/B con prompt neutro | controllo di A11–A14 (B-ctrl) | Apache-2.0 |

Perché questi ruoli:
- **KnowLine-4B** è il primo del gruppo 4B per il Decision Index, è causale e legge la risposta da due token della testa del vocabolario: è il caso più semplice da addestrare, entra in 24 GB anche per il training, e le prove costano poco;
- **elephant-1** è il primo del gruppo 9B e la coppia principale di B-ctrl (elephant-1 contro Qwen3.5-9B), con licenza MIT;
- gli altri 8 si addestrano con la ricetta finale per completare B-ctrl su più ricette: si raggruppano per tipo di lettura ([03](03_rendering_formato.md) §2).

Ogni modello si addestra e si serve **con il codice dei suoi autori** (prompt, maschera, lettura), come nei benchmark della repo: altrimenti si addestrerebbe un modello diverso.

## 3. La soluzione in una pagina

1. **Modello di partenza**: un decision model, interrogato con **una sola domanda sì/no**. La probabilità si ricava, in un solo forward, dai punteggi delle **due opzioni** nella posizione della risposta (due righe della testa del vocabolario, o la testa propria del modello). La domanda si sceglie sul nostro dev fra quelle dello studio ([02d](02d_varianti_domanda.md)).
2. **Dati**: fonti pubbliche più dati generati con un LLM secondo le ricette di MiniCheck (C2D, D2C) e FactCG (CG2C), e un sottoinsieme "difficile" di ANLI.
3. **Training**: adapter LoRA, loss di cross-entropy **solo sui punteggi delle due opzioni**, curriculum in due stadi (prima dati "logici", poi dati realistici).
4. **Valutazione**: split nostri congelati; tutte le scelte sul nostro dev; LLM-AggreFact solo come test finale, zero-shot.
5. **Export**: pesi fusi caricabili dal runtime degli autori, soglia e temperatura fissate; verifica con l'adapter del benchmark.

## 4. Glossario

| Termine | Significato in questa guida |
|---|---|
| **Claim** | l'affermazione da verificare |
| **Documento** | il testo contro cui si verifica il claim |
| **Coppia** | (documento, claim, etichetta) |
| **Decision model (TDM)** | LLM addestrato a rispondere a domande tipizzate con una probabilità per opzione, in un forward, senza generare testo |
| **Backbone** | il LLM di partenza di un decision model, prima del training da decision model (qui Qwen3.5-9B e Qwen3.5-4B) |
| **Ricetta** | come un gruppo ha trasformato il backbone in decision model: dati, training, attenzione, lettura |
| **Lettura** | come si ottiene la probabilità: dalle righe della testa del vocabolario per le lettere/codici delle opzioni, o da una testa propria |
| **Attenzione bidirezionale sul prompt** | ogni token del prompt vede anche quelli dopo (nei layer ad attenzione piena); la maschera va riprodotta anche in training |
| **Domanda `noul`** | domanda con due opzioni, vero/falso (sì/no); la risposta è una probabilità |
| **Logit** | punteggio grezzo che il modello assegna a un'opzione (o a un token) |
| **Temperatura T** | si divide il logit per T prima della softmax: T > 1 rende le probabilità meno estreme |
| **Margine** s | differenza fra il logit di "supportato" e quello di "non supportato", divisa per T; P(supportato) = σ(s) |
| **Soglia** | valore sopra il quale P(supportato) diventa "supportato" |
| **Forward / prefill** | un passaggio del modello sul prompt; il costo d'inferenza di un decision model è un solo prefill |
| **LoRA** | metodo di fine-tuning che congela i pesi e aggiunge, per alcune matrici W, due matrici piccole B·A addestrabili: W' = W + (α/r)·B·A |
| **QLoRA** | LoRA con i pesi congelati caricati a 4 bit per risparmiare memoria |
| **Rank r, alpha α** | dimensione interna di LoRA e fattore di scala |
| **Merge (fusione)** | calcolare W' = W + (α/r)·B·A e salvare pesi normali, senza adapter |
| **Chunk** | pezzo di documento; per documenti lunghi si valuta ogni chunk e si prende il massimo |
| **doc_id** | chiave che identifica il documento; righe con lo stesso doc_id stanno sempre nello stesso split |
| **Split** | partizione dei dati: train (si addestra), dev (si decide), test (si guarda una volta) |
| **dev_select / dev_confirm** | due metà del dev: sulla prima si sceglie, sulla seconda si conferma |
| **test_id / test_ood** | test delle fonti usate in training / di fonti mai viste |
| **Decontaminazione** | rimozione dal training di righe che si sovrappongono ai test |
| **BAcc** (balanced accuracy) | ½ (recall dei positivi + recall dei negativi) |
| **BAcc macro** | media non pesata della BAcc delle fonti |
| **AUC** | probabilità che un positivo a caso abbia punteggio più alto di un negativo a caso |
| **ECE** | errore di calibrazione: scarto medio fra confidenza e accuratezza, per fasce di confidenza |
| **Brier** | media di (p − y)² |
| **Zero-shot** (per AggreFact) | il modello non ha visto nulla di AggreFact, nemmeno per scegliere la soglia |
| **C2D, D2C** | ricette di dati sintetici di MiniCheck, spiegate in [02b](02b_dataset_generazione_llm.md) |
| **CG2C** | ricetta di FactCG: claim multi-hop ricavati da un grafo di entità, spiegata in [02b](02b_dataset_generazione_llm.md) |
| **Multi-hop** | claim che si verifica solo combinando più fatti del documento |
| **Stadio 1 / stadio 2** | le due fasi del curriculum di training ([02e](02e_mix_finale.md)) |
| **B-ctrl** | esperimento del paper: stessa ricetta di fine-tuning su un decision model e sul suo backbone |

## 5. Ambiente

### 5.1 Macchine

| Macchina | GPU | Uso |
|---|---|---|
| Portatile | RTX 5090 Laptop, 24 GB | sviluppo, test, training dei modelli 4B (LoRA in bf16) |
| Server (`ailab-server2`, Slurm) | L40S da 45 GB | training dei modelli 9B, generazione dati, valutazione |

Sul server i job si lanciano con `gpu-run --batch`, che li mette in coda come job `sbatch` (non dipendono dalla sessione). Il log va rediretto dentro il comando:

```sh
gpu-run --batch --name ft-knowline-p0 -- bash -c 'accelerate launch finetune/train.py ... > logs/ft-knowline-p0.out 2>&1'
```

Il limite di tempo di `gpu-run` è 24 ore (`--time` per cambiarlo).

### 5.2 Software

Ambiente conda `factcheck` (transformers 5.14, vLLM, torch, datasets, accelerate, nltk, scikit-learn). Da installare per il training:

```sh
pip install peft flash-linear-attention causal-conv1d
```

`flash-linear-attention` e `causal-conv1d` servono ai layer Gated DeltaNet di Qwen3.5: senza, transformers usa un'implementazione lenta in torch. Alcuni modelli hanno un ambiente proprio per l'inferenza (StartLux: `.venv` con vLLM 0.31 e il plugin bidirezionale; OneJev: `.venv` con Flide): per il training si usa l'ambiente principale con transformers ([05](05_training.md) §2).

### 5.3 Pesi dei modelli

Tutti dalla cache di Hugging Face, alle revisioni fissate nei `config/models/<modello>.json` e negli adapter della repo (`models/<modello>/`). I due backbone: `Qwen/Qwen3.5-9B` @ `c2022362…` e `Qwen/Qwen3.5-4B` @ `851bf6e8…`. Riferimenti esterni: FactCG-DeBERTa-v3-Large e Bespoke-MiniCheck-7B (adapter già nella repo).

### 5.4 Dati e risultati

| Percorso | Contenuto |
|---|---|
| `data/finetune/raw/<fonte>.jsonl` | coppie per fonte, prodotte da `data_prep/build_pairs.py` |
| `dataset/lytang--LLM-AggreFact/` | LLM-AggreFact (dev, test, AF-4k): **solo per valutare e decontaminare** |
| `results/lytang--LLM-AggreFact/<split>/<modello>_<hash>/` | run dello studio focalizzato (RUN-01…RUN-15): `answers.jsonl` riga per riga, formato in [README_RESULTS.md](../../README_RESULTS.md) |
| `dataset/` | FEVER, HoVer, AVeriTeC, … in locale |

## 6. Struttura delle cartelle che verrà creata

```text
data/finetune/
  raw/                       # coppie per fonte + fonti nuove e generate
  splits/                    # Step 0: train / dev_select / dev_confirm / test_id / test_ood / test_long + manifest.json
  gen/                       # Step 2b: output grezzi della generazione, per famiglia
  mix_fc_pilot/, mix_fc_stage1/, mix_fc_stage2/     # Step 2e
  rendered/<modello>/<mix>/  # Step 3
finetune/
  question.json              # Step 2d: la domanda congelata
  sentences.py, render.py, losses.py, train.py, quick_eval.py, calibrate.py, behavioral.py,
  mine_hard.py, make_splits.py, merge.py, report.py
  outputs/<esperimento>/     # Step 5: adapter, log, config
  release/<modello>-fc-<versione>/   # Step 8
  experiments_fc.md          # registro degli esperimenti
```

## 7. Il flusso di lavoro

```text
Step 0  split congelati ──────────────┐ (prima di tutto: nessuna generazione né baseline prima)
Step 1  baseline sul nostro dev       │
Step 2a fonti esistenti + ANLI difficile
Step 2b generazione C2D / D2C / CG2C  │ (solo dal train)
Step 2c (niente replay)               │
Step 2d domanda congelata             │
Step 2e mix stadio 1 e stadio 2       │
Step 3  rendering                     │
Step 4  loss                          │
Step 5  training LoRA    ◄─────────┐  │
Step 6  valutazione sul dev        │  │
Step 7  esperimenti ───────────────┘  │
Step 8  merge, soglia finale, test una volta, export
```

| Step | Input | Output | Codice |
|---|---|---|---|
| 0 | `data/finetune/raw/`, AggreFact, fonti test_ood | `data/finetune/splits/` + manifest | `make_splits.py` |
| 1 | split, modelli | tabella di baseline | `benchmark.py` sul nostro dev, `calibrate.py` |
| 2a | split di train | `anli_hard`, builder HotpotQA/MuSiQue | `mine_hard.py`, `build_pairs.py` |
| 2b | split di train, LLM locale | famiglie generate e verificate | `llm_gen/` esteso |
| 2d | dev | `finetune/question.json` | — |
| 2e | split e famiglie | `mix_fc_*` | `make_mix.py` esteso |
| 3 | mix | esempi tokenizzati, per modello | `render.py` |
| 4–5 | dati renderizzati | adapter | `losses.py`, `train.py` |
| 6 | adapter | metriche | `benchmark.py` sui pesi fusi, `behavioral.py`, `report.py` |
| 7 | — | configurazione finale | — |
| 8 | adapter scelto | modello rilasciato, report | `merge.py` |

## 8. Convenzioni

- **Etichette**: sempre 1 = supportato, 0 = non supportato. Ogni punteggio è sempre P(supportato), qualunque sia l'ordine delle opzioni del modello (es. KnowLine e elephant-1: A = sì; d3-flash, Jeffan, pplx: A = no).
- **Percentuali**: due decimali (77,42).
- **Seed**: 13 per la preparazione dei dati; per il training 3 seed nella configurazione finale (13, 14, 15).
- **Mai addestrare** su LLM-AggreFact (lo vieta la sua licenza) né sulle fonti di test_ood. Attenzione: alcuni decision model potrebbero aver visto RAGTruth (è nel Decision Index pubblico): AggreFact si riporta anche senza RAGTruth, come nello studio.
- **Mai scegliere** nulla guardando un test.
- Ogni file prodotto ha accanto un `.stats.json` o un `manifest.json` con numero di righe e hash SHA-256.

## 9. Obiettivi misurabili

| Obiettivo | Soglia |
|---|---|
| LLM-AggreFact zero-shot, BAcc macro, una sola domanda | > 77,4 (Bespoke-MiniCheck-7B); puntare a ≥ 78,5 |
| AUC macro su AggreFact | > 85,0 |
| Soglia ottimale sul nostro dev | fra 0,3 e 0,7 |
| ECE sul nostro dev | non peggiore della baseline |
| test_ood | nessun peggioramento rispetto alla baseline |
| Test "solo claim" | P(supportato) media < 0,15 |
| Test "rimozione dell'evidenza" | la probabilità scende sotto soglia in > 80% dei casi |
| B-ctrl | decision model fine-tunato e backbone fine-tunato confrontati sugli stessi dati, con la stessa ricetta |
| Costo | un forward per (documento o chunk, claim) |
