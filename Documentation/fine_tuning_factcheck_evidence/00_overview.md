# Fact checking con evidenza evidenziata — piano operativo

## Cosa si costruisce

Un decision model che, dato un documento e un claim, restituisce in **un solo forward** (nessun token generato):
- P(supportato) e il verdetto, come nel piano binario ([../fine_tuning_factcheck/](../fine_tuning_factcheck/00_overview.md));
- **se il claim è supportato**, le frasi del documento che lo stabiliscono.

Se il verdetto è "non supportato", non si evidenzia nulla.

```json
{"p_supported": 0.93, "supported": true,
 "evidence": [{"sentence": 0,  "start": 0,    "end": 61,   "score": 0.91},
              {"sentence": 39, "start": 4210, "end": 4262, "score": 0.84}]}
```

`start` / `end` sono offset di carattere sul documento originale.

Il caso da gestire bene è l'**evidenza sparsa**: due frasi lontane che solo insieme stabiliscono il claim.

> Claim: "Nancy R. Howell teaches at a seminary in Kansas City."
> S1: "Saint Paul School of Theology is a seminary located in Kansas City, Missouri."
> … 38 frasi irrilevanti …
> S40: "Nancy R. Howell is Professor of Theology at Saint Paul School of Theology."
> Evidenza: S1 e S40.

## Modelli

| Modello | Ruolo | d | Layer di attenzione piena (0-indicizzati) |
|---|---|---:|---|
| **d1-3B** | tutti gli esperimenti si fanno prima qui | 2.048 | 2, 5, 9, 13, 17, 21, 24, 27 (gli altri 22 sono convoluzioni corte) |
| **JPT-9B** | conferma a 9B | 4.096 | 3, 7, 11, 15, 19, 23, 27, 31 (gli altri 24 sono Gated DeltaNet) |
| **Winnow-12B** (BF16) | candidato di massima qualità | 3.840 | 5, 11, 17, 23, 29, 35, 41, 47 (gli altri 40 a finestra di 1.024 token) |

Il verdetto di ciascun modello è quello già addestrato nel piano binario (domanda, risposte, soglia, temperatura). Caricamento e hook in [01](01_ambiente_e_modelli.md).

## L'architettura

```text
prompt: [preambolo: "Claim under review: <claim>"] [prompt nativo del piano binario: documento S1…Sn, claim, domanda] Answer:
                                     │
                    forward unico del decision model (pesi del piano binario)
                                     │
   per ogni frase k (dagli offset del tokenizer, senza marcatori nel testo):
     h_end[k]  = stati nell'ultimo token della frase, 4 layer
     h_mean[k] = media degli stati dei token della frase, 4 layer
     att[k]    = massa di attenzione di "Answer:" sui token della frase, 8 layer pieni × teste
   q = stati in "Answer:", 4 layer            ──► logit delle risposte ──► P(supportato)
                                     │
   testa: mix dei layer → proiezione a 256 → transformer NON causale a 2 layer su [x_0 = q, x_1 … x_n]
                                     │
   s_k = logit della frase k;  s_0 = logit di x_0 = soglia di questa riga
                                     │
   mostrate le frasi con s_k > s_0 (min 1, max 5), solo se P(supportato) > soglia del verdetto
```

- Il **preambolo** fa sì che ogni frase sia letta sapendo quale claim si verifica. Il resto del prompt è identico a quello del piano binario, quindi il verdetto resta utilizzabile.
- L'**aggregatore non causale** fa comunicare le frasi fra loro e con `q`: la prima frase "sa" dell'ultima.
- La **soglia della riga** `s_0` adatta il numero di frasi mostrate al caso; non c'è una soglia fissa da tarare.

## Sequenza di lavoro

| # | Passo | File | Prodotto |
|---|---|---|---|
| 1 | Ambiente, modelli, hook | [01](01_ambiente_e_modelli.md) | modelli caricabili con hook verificati; test di identità del verdetto |
| 2 | Righe con frasi ed evidenza dalle fonti | [02](02_dati_fonti.md) | `data/evidence/pairs/*.jsonl` |
| 3 | Etichette generate, ablazione, righe sparse | [03](03_dati_generati.md) | righe `construct`, `ablation`, `sparse` |
| 4 | Split e mix | [04](04_split_e_mix.md) | blocchi dev/test congelati; mix `ev_head` ed `ev_joint` |
| 5 | Formato del prompt (preambolo) | [05](05_formato_prompt.md) | preambolo scelto; `question.json` esteso |
| 6 | Rendering ed estrazione delle feature | [06](06_rendering_ed_estrazione.md) | feature salvate su disco |
| 7 | Baseline | [07](07_baseline.md) | numeri di riferimento (E-LEX, E-ATT, E-ABL) |
| 8 | Testa e loss | [08](08_testa_e_loss.md) | `evidence/head.py`, `evidence/losses.py` testati |
| 9 | Training | [09](09_training.md) | teste addestrate (fasi 1a, 1b, 2) |
| 10 | Valutazione | [10](10_valutazione.md) | `evidence/eval_evidence.py`, report |
| 11 | Esperimenti | [11](11_esperimenti.md) | configurazione finale per i tre modelli |
| 12 | Export e servizio | [12](12_export_e_servizio.md) | server, `decision.json`, rilascio |
| — | Checklist | [13](13_checklist.md) | |
| — | Una strada da esplorare | [14](14_strada_da_esplorare.md) | |

## Codice e dati

Tutto il codice nuovo va in `evidence/` (pacchetto Python nella radice della repo), i dati in `data/evidence/`:

```text
evidence/
  sentences.py          # segmentazione unica, usata ovunque
  build_pairs.py        # FEVER, HoVer, SciFact, WiCE → righe con frasi ed evidenza    (02)
  gen_marked.py         # C2D e CG2C con posizioni note                                 (03)
  label_ablation.py     # etichette per ablazione con d1-3B                             (03)
  augment_sparse.py     # righe sparse                                                  (03)
  make_splits.py, make_mix.py                                                           (04)
  render.py             # prompt con preambolo + intervalli di token per frase          (05, 06)
  models.py             # caricamento dei tre modelli, hook di stati e attenzione       (01, 06)
  extract.py            # feature su disco                                              (06)
  baselines.py          # E-RAND, E-LEX, E-EMB, E-ATT, E-ABL                            (07)
  head.py, losses.py                                                                    (08)
  train_head.py         # fasi 1a e 1b                                                  (09)
  train_joint.py        # fase 2                                                        (09)
  eval_evidence.py, latency.py                                                          (10)
  serve.py                                                                              (12)
data/evidence/
  pairs/  splits/  mixes/  features/<modello>/  runs/<modello>/<id>/
```

Sul server i job GPU si lanciano con `gpu-run --batch` (job `sbatch`, non dipende dalla sessione). La redirezione del log va dentro il comando, altrimenti l'output finisce in `slurm-<id>.out`:

```sh
gpu-run --batch --name ev-extract-d1 -- bash -c \
  'python -m evidence.extract --model d1-3b --mix ev_head > logs/ev-extract-d1.out 2>&1'
```

## Obiettivi

| Obiettivo | Soglia |
|---|---|
| Verdetto | invariato rispetto al piano binario (BAcc e AUC su `dev_select` entro 0,3 punti) |
| F1 per frase | ≥ E-LEX + 15 punti; indicativo ≥ 0,75 su FEVER e HoVer |
| Recall dell'insieme completo (tutte le frasi gold mostrate) | ≥ 0,65 su HoVer |
| Calo con la distanza fra frasi gold | < 5 punti di F1 fra "adiacenti" e "≥ 20 frasi" |
| Sufficienza (le frasi mostrate, da sole, danno "supportato") | ≥ 85% |
| Latenza | ≤ +3% del solo verdetto (d1-3B: ≤ +3% o ≤ 2 ms, il maggiore dei due) |
