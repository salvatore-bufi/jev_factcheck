# Fine-tuning di un fact checker binario: guida operativa completa

## 0. Come usare questa guida

Questa guida spiega **come fare**, passo per passo, il fine-tuning di un modello che, dato un documento e un'affermazione (claim), dice se il documento la supporta. È scritta per essere **autosufficiente**: non serve leggere i paper di MiniCheck o FactCG né fare ricerche. Ogni procedura riporta dati di partenza, comandi, prompt, codice di riferimento, controlli e criteri per passare al passo successivo.

La cartella [summary/](summary/00_overview.md) contiene la versione riassuntiva dello stesso piano (decisioni e numeri). Questa guida la estende con i dettagli operativi; i numeri dei passi coincidono.

| File | Contenuto |
|---|---|
| `00_overview.md` | questo file: il compito, il glossario, l'ambiente, la mappa del lavoro |
| `00a_scelta_approccio.md` | perché si parte da un decision model con una sola domanda; cosa sono MiniCheck e FactCG |
| `00b_split_e_protocollo.md` | come costruire e congelare train / dev / test, con il codice |
| `01_baseline.md` | come misurare i modelli prima del training |
| `02a_dataset_fonti_esistenti.md` | le fonti pubbliche, come si trasformano in coppie, ANLI difficile |
| `02b_dataset_generazione_llm.md` | come generare i dati alla MiniCheck (C2D, D2C) e alla FactCG (CG2C), con i prompt |
| `02c_replay_e_forgetting.md` | perché il replay non serve e come ancorare la calibrazione |
| `02d_varianti_domanda.md` | come scegliere e congelare la domanda |
| `02e_mix_finale.md` | come costruire i due training set (stadio 1 e stadio 2) |
| `03_rendering_formato.md` | il prompt esatto di ciascun modello, token per token, e il chunking |
| `04_loss.md` | la funzione obiettivo, con derivazioni ed esempi numerici |
| `05_training.md` | lo script di training, la configurazione LoRA, l'hardware |
| `06_valutazione.md` | metriche, test comportamentali, report |
| `07_esperimenti.md` | il piano delle prove e come decidere |
| `08_merge_export.md` | fusione dei pesi, export, soglia finale, distillazione |
| `09_rischi_checklist.md` | cosa può andare storto e checklist completa |

Il PDF unico con tutte le sezioni è in `pdf/fine_tuning_factcheck_guida_completa.pdf`.

I risultati misurati dei modelli candidati su LLM-AggreFact (run di ottobre 2026, con d1-3B e CLM-8B) sono analizzati in [../risultati_aggrefact_2026-10.md](../risultati_aggrefact_2026-10.md); questa guida ne usa i numeri.

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

## 2. La soluzione in una pagina

1. **Modello di partenza**: un *decision model*, interrogato con **una sola domanda sì/no**. Il modello legge il prompt e la probabilità si ricava dai logit di **due risposte** (le lettere A/B per JPT e Winnow, le parole yes/no per d1-3B) nella posizione della risposta: un solo forward, nessun testo generato. Tre candidati:
   - **d1-3B** (LiquidAI, 3B): quasi la stessa AUC dei modelli grandi, 4–5 volte più veloce, entra in una GPU da 24 GB anche per il training: **il modello per iterare** e il candidato per la produzione se conta il costo;
   - **JPT-9B**: il riferimento a 9B (licenza non commerciale);
   - **Winnow-12B**: il candidato per la massima accuratezza (Apache 2.0).
   La domanda singola da preferire, dai risultati sul test, è `unsupported_detail` letta come 1 − P(sì) ([02d](02d_varianti_domanda.md)); la scelta definitiva si fa sul nostro dev.
2. **Dati**: le fonti pubbliche già preparate nella repo (`data_prep/`) più dati generati con un LLM secondo le ricette di MiniCheck (C2D, D2C) e FactCG (CG2C), e un sottoinsieme "difficile" di ANLI.
3. **Training**: adapter LoRA, loss di cross-entropy **solo sui logit delle due lettere**, curriculum in due stadi (prima dati "logici", poi dati realistici).
4. **Valutazione**: split nostri congelati; tutte le scelte sul nostro dev; LLM-AggreFact solo come test finale, zero-shot.
5. **Export**: pesi fusi, soglia e temperatura fissate, servizio con vLLM come oggi nei benchmark.

## 3. Glossario

| Termine | Significato in questa guida |
|---|---|
| **Claim** | l'affermazione da verificare |
| **Documento** | il testo contro cui si verifica il claim |
| **Coppia** | (documento, claim, etichetta) |
| **Decision model** | LLM addestrato a rispondere a domande tipizzate scegliendo una lettera fra opzioni; la probabilità si legge dai logit delle lettere. JPT-9B e Winnow-12B lo sono |
| **Domanda `noul`** | tipo di domanda dei decision model con due opzioni, vero/falso (o sì/no); la risposta è una probabilità |
| **Logit** | punteggio grezzo che il modello assegna a ogni token del vocabolario nella posizione successiva |
| **Softmax** | trasforma logit in probabilità: p_k = e^{z_k} / Σ_j e^{z_j} |
| **Temperatura T** | si divide il logit per T prima della softmax: T > 1 rende le probabilità meno estreme |
| **Margine** s | differenza fra il logit di "vero" e quello di "falso", divisa per T; P(supportato) = σ(s) = 1/(1+e^{−s}) |
| **Soglia** | valore sopra il quale P(supportato) diventa "supportato" |
| **Forward / prefill** | un passaggio del modello sul prompt; il costo d'inferenza di un decision model è un solo prefill |
| **LoRA** | metodo di fine-tuning che congela i pesi e aggiunge, per alcune matrici W, due matrici piccole B·A addestrabili: W' = W + (α/r)·B·A |
| **QLoRA** | LoRA con i pesi congelati caricati a 4 bit per risparmiare memoria |
| **Rank r, alpha α** | dimensione interna di LoRA e fattore di scala (α/r moltiplica l'aggiornamento) |
| **Merge (fusione)** | calcolare W' = W + (α/r)·B·A e salvare pesi normali, senza adapter |
| **Chunk** | pezzo di documento; per documenti lunghi si valuta ogni chunk e si prende il massimo |
| **doc_id** | chiave che identifica il documento; righe con lo stesso doc_id stanno sempre nello stesso split |
| **Split** | partizione dei dati: train (si addestra), dev (si decide), test (si guarda una volta) |
| **dev_select / dev_confirm** | due metà del dev: sulla prima si sceglie, sulla seconda si conferma |
| **test_id / test_ood** | test delle fonti usate in training / di fonti mai viste |
| **Decontaminazione** | rimozione dal training di righe che si sovrappongono ai test |
| **13-gramma** | sequenza di 13 parole consecutive; si usano per misurare la sovrapposizione fra documenti |
| **BAcc** (balanced accuracy) | ½ (recall dei positivi + recall dei negativi); non premia chi dice sempre la classe più frequente |
| **BAcc macro** | media non pesata della BAcc delle fonti |
| **AUC** | probabilità che un positivo a caso abbia punteggio più alto di un negativo a caso; indipendente dalla soglia |
| **ECE** | errore di calibrazione: scarto medio fra confidenza e accuratezza, per fasce di confidenza |
| **Brier** | media di (p − y)²: errore quadratico delle probabilità |
| **Zero-shot** (per AggreFact) | il modello non ha visto nulla di AggreFact, nemmeno per scegliere la soglia |
| **C2D, D2C** | ricette di dati sintetici di MiniCheck ("claim → documento", "documento → claim"), spiegate in [02b](02b_dataset_generazione_llm.md) |
| **CG2C** | ricetta di FactCG ("context graph → claim"): claim multi-hop ricavati da un grafo di entità, spiegata in [02b](02b_dataset_generazione_llm.md) |
| **Multi-hop** | claim che si verifica solo combinando più fatti del documento (es. A lavora in B, B è in C ⇒ A lavora in C) |
| **Stadio 1 / stadio 2** | le due fasi del curriculum di training ([02e](02e_mix_finale.md)) |

## 4. Ambiente

### 4.1 Macchine

| Macchina | GPU | Uso |
|---|---|---|
| Portatile | RTX 5090 Laptop, 24 GB; 62 GB di RAM | sviluppo, test, QLoRA di JPT, encoder, generazione piccola |
| Server | 2× L40S, 45 GB ciascuna | training di Winnow e JPT in bf16, generazione dati in volume, valutazione |

### 4.2 Software

Tutto si esegue nell'ambiente conda `factcheck` (`conda run -n factcheck python ...`). Versioni presenti (verificate): transformers 5.14.1, vLLM 0.28.0, torch 2.13.0+cu130, datasets 5.0.1, accelerate 1.15.0, bitsandbytes 0.50.2, networkx 3.6.1, nltk 3.10.1, scikit-learn 1.9.0, openai 3.10.0.

Da installare prima del training:

```sh
conda run -n factcheck pip install peft          # LoRA; verificare che la versione installata supporti transformers 5.14
conda run -n factcheck pip install rank_bm25     # solo per la baseline lessicale facoltativa
```

Nota: `conda run` non passa lo standard input, quindi gli script vanno scritti in file e lanciati, non passati con heredoc.

### 4.3 Pesi dei modelli

| Modello | Dove | Note |
|---|---|---|
| d1-3B | cache HF, `LiquidAI/d1-3B`, revisione **`051bcc464b01b9f92942b364d9586b0ef5912432`** | LFM2.5-VL-3B (Liquid AI) post-addestrato come decision model; codice proprio (`trust_remote_code`); licenza LFM Open License v1.0 (commerciale solo sotto 10 M$ di fatturato annuo) |
| JPT-9B | cache HF, `kirp/jpt-9b`, revisione **`b447cc7ee105c0a76a22f8fde8ecf05074dc8be0`** | Qwen3.5-9B con una LoRA già fusa; licenza CC BY-NC 4.0 |
| Winnow-12B bf16 | `winnow-12b-bf16/model/` (creato da `python winnow-12b-bf16/convert.py` dal GGUF BF16) | Gemma-4-12B fine-tunato; Apache 2.0 |
| Qwen3.5-9B base (controllo) | `Qwen/Qwen3.5-9B` (da scaricare) | Apache 2.0 |
| Gemma-4-12B-it (controllo) | `google/gemma-4-12B-it` (da scaricare) | Apache 2.0 |
| FactCG-DeBERTa-v3-Large | `yaxili96/FactCG-DeBERTa-v3-Large` | MIT; benchmark in `FactCG/` |
| MiniCheck-Flan-T5-Large | `lytang/MiniCheck-Flan-T5-Large` | MIT |
| Bespoke-MiniCheck-7B | `bespokelabs/Bespoke-MiniCheck-7B` (già nella cache HF) | |

### 4.4 Dati già presenti nella repo

| Percorso | Contenuto |
|---|---|
| `data/finetune/raw/<fonte>.jsonl` | coppie di 11 fonti, prodotte da `data_prep/build_pairs.py` |
| `data/finetune/mix_pilot`, `mix_core`, `mix_extended` | mix del piano precedente (non usati qui: questo piano ricostruisce gli split) |
| `lytang___llm-aggre_fact/` | cache di LLM-AggreFact (dev e test): **solo per valutare e decontaminare** |
| `results/lytang--LLM-AggreFact/test/<modello>_<hash parametri>_<hash domande>/` | run completi su AggreFact test di JPT-9B, Winnow-12B-BF16, Rune-26B, d1-3B, CLM-8B: `answers.jsonl` (risposte alle tre domande, riga per riga), `report.json`, `run.json`, latenza e telemetria; formato in [README_RESULTS.md](../../README_RESULTS.md). Non ci sono run sul dev: la vecchia cartella `results/runs/` (con le cache del dev di Jev) non esiste più |
| `Datasets/` | FEVER, HoVer, FactKG, AVeriTeC, ... in locale |

## 5. Struttura delle cartelle che verrà creata

```text
data/finetune/
  raw/                       # coppie per fonte (esiste) + fonti nuove e generate
  splits/                    # Step 0: train / dev_select / dev_confirm / test_id / test_ood / test_long + manifest.json
  gen/                       # Step 2b: output grezzi della generazione, per famiglia
  mix_fc_pilot/, mix_fc_stage1/, mix_fc_stage2/     # Step 2e
  rendered/<modello>/<mix>/  # Step 3
finetune/
  question.json              # Step 2d: la domanda congelata
  sentences.py, render.py, losses.py, train.py, score.py, calibrate.py, behavioral.py,
  mine_hard.py, make_splits.py (o in data_prep/), merge.py, report.py
  outputs/<esperimento>/     # Step 5: adapter, log, config
  release/<modello>-fc-<versione>/   # Step 8
  experiments_fc.md          # registro degli esperimenti
results/fc/<modello>/<split>/answers.jsonl     # cache delle valutazioni nuove
```

## 6. Il flusso di lavoro

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

| Step | Input | Output | Codice | Tempo indicativo |
|---|---|---|---|---|
| 0 | `data/finetune/raw/`, AggreFact, fonti test_ood | `data/finetune/splits/` + manifest | `make_splits.py` | 1–2 giorni di codice |
| 1 | split, modelli | tabella di baseline, domanda scelta | `score.py`, `calibrate.py` | ~1 giorno di GPU |
| 2a | split di train | `anli_hard`, builder HotpotQA/MuSiQue | `mine_hard.py`, `build_pairs.py` | ~1 giorno |
| 2b | split di train, LLM locale | famiglie generate e verificate | `llm_gen/` esteso | 2–4 giorni (in parallelo agli altri step) |
| 2d | dev | `finetune/question.json` | — | ore |
| 2e | split e famiglie | `mix_fc_*` | `make_mix.py` esteso | ore |
| 3 | mix | esempi tokenizzati | `render.py` | ore |
| 4–5 | dati renderizzati | adapter | `losses.py`, `train.py` | 4–30 h per corsa |
| 6 | adapter | metriche | `score.py`, `behavioral.py`, `report.py` | 1–3 h per valutazione sul dev |
| 7 | — | configurazione finale | — | 2–3 settimane |
| 8 | adapter scelto | modello rilasciato, report | `merge.py` | 1–2 giorni |

## 7. Convenzioni

- **Etichette**: sempre 1 = supportato, 0 = non supportato. Ogni punteggio è sempre P(supportato), anche quando il modello ha le opzioni in ordine diverso (Winnow: A = falso; JPT: A = sì).
- **Percentuali**: le metriche si riportano in percentuale con due decimali (77,42).
- **Seed**: 13 per la preparazione dei dati (come `build_pairs.py` e `make_mix.py`); per il training 3 seed nella configurazione finale (13, 14, 15).
- **Mai addestrare** su LLM-AggreFact (lo vieta la sua licenza) né sulle fonti di test_ood.
- **Mai scegliere** nulla guardando un test.
- Ogni file prodotto ha accanto un `.stats.json` o un `manifest.json` con numero di righe e hash SHA-256.

## 8. Obiettivi misurabili

| Obiettivo | Soglia |
|---|---|
| LLM-AggreFact zero-shot, BAcc macro, una sola domanda | > 77,4 (Bespoke-MiniCheck-7B); puntare a ≥ 78,5 |
| AUC macro su AggreFact | > 85,0 |
| Soglia ottimale sul nostro dev | fra 0,3 e 0,7 (oggi 0,90–0,95 con la domanda singola) |
| ECE sul nostro dev | non peggiore della baseline |
| test_ood | nessun peggioramento rispetto alla baseline |
| Test "solo claim" | P(supportato) media < 0,15 |
| Test "rimozione dell'evidenza" | la probabilità scende sotto soglia in > 80% dei casi |
| Costo | un forward per (documento o chunk, claim) |
