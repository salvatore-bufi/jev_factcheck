# Passo 9 — Training

## In breve

Si addestra la testa in tre modi, dal più semplice al più costoso. Si passa al successivo solo se il precedente non basta.

| Fase | Il modello | Su cosa si addestra la testa | Quando |
|---|---|---|---|
| **1a** | congelato, non in memoria | sui vettori salvati al passo 6 | quasi sempre: ogni prova dura minuti |
| **1b** | congelato, in memoria | sui vettori ricalcolati a ogni passo | solo per l'esperimento che ha bisogno dei vettori di tutti i token (H2d) |
| **2** | addestrato anche lui, con una LoRA | come 1a | solo se, alla fine degli esperimenti, gli obiettivi non sono raggiunti |

Nelle fasi 1a e 1b il verdetto **non può cambiare**: il modello non viene toccato.

Ogni prova scrive in `data/evidence/runs/<modello>/<id>/`: `config.json` (tutte le opzioni), `head.safetensors` (pesi della testa), `log.jsonl` (andamento), `dev_metrics.json`.

## 1. Fase 1a

| Impostazione | Valore |
|---|---|
| Ottimizzatore | AdamW, learning rate 1e-3 con riscaldamento iniziale (5%) e discesa a coseno, weight decay 0,01 |
| Epoche | fino a 10; ci si ferma se per 2 epoche `dev_evid_select` non migliora |
| Batch | 64 righe |
| Dropout | 0,1; in più, per ogni frase, la feature di attenzione si azzera con probabilità 0,1 (per non dipendere da una sola testa) |
| Checkpoint tenuto | quello con la media più alta di F1 e insieme completo su `dev_evid_select` |

```sh
python -m evidence.train_head --phase 1a --model knowline-4b --features data/evidence/features/knowline-4b \
    --mix ev_head --config configs/h0.json --run h0
```

Ogni esperimento è un file in `configs/` con le opzioni della testa e della loss.

Prima della prima prova: la testa deve riuscire a imparare a memoria 64 righe (F1 ≈ 1 su quelle righe). Se non ci riesce, c'è un errore nei dati o nel codice.

## 2. Quali frasi si mostrano

Si decide dopo il training, su `dev_evid_select`, guardando solo le righe che il verdetto giudica supportate:

| Loss usata | Regola | Cosa si sceglie |
|---|---|---|
| loss di partenza | probabilità della frase > `τ_e` | il valore di `τ_e` che dà l'F1 migliore |
| loss con soglia per riga | `s_k > s_0 + δ` | δ fra −1 e 1; si tiene δ diverso da 0 solo se guadagna almeno 1 punto di F1 |

Sempre almeno 1 frase (la migliore) e al massimo 5. Si conferma su `dev_evid_confirm`: l'F1 non deve scendere di più di 1 punto.

## 3. Che cosa guardare durante il training

| Cosa | Su | Segnale di problema | Cosa fare |
|---|---|---|---|
| F1 e insieme completo | `dev_evid_select` | non supera E-LEX dopo 2 epoche | controllare le posizioni dei token ([06](06_rendering_ed_estrazione.md) §6) prima di cambiare la testa |
| F1 per distanza fra frasi di evidenza | righe spostate | crolla oltre 10 frasi | più righe spostate; esperimenti H3c, H-PAIR |
| F1 per distanza dal preambolo | `dev_evid_select` | crolla oltre 1.024 o oltre 4.096 token | più righe lontane; attenzione come feature (H2b) |
| Frasi mostrate per riga | `dev_evid_select` | molto più o molto meno delle frasi di evidenza | rivedere δ o il peso dei negativi |
| Negativi con frasi sopra soglia | `dev_evid_neg` | oltre il 30% | peso dei negativi più alto |
| Positivi incerti (P appena sopra la soglia del verdetto) | `dev_evid_select` | la testa non mostra quasi nulla | peso dei negativi più basso |
| Pesi dei 4 layer | testa | tutto su un layer | basta quel layer (meno spazio, meno latenza) |

## 4. Fase 1b

Il modello è caricato in memoria e congelato; a ogni passo si rifà il forward per avere i vettori. Serve per un solo esperimento:
- **H2d**: invece della media fissa dei token di una frase, una media pesata in cui la testa decide quali token contano (per esempio il nome proprio dentro una frase lunga). Richiede i vettori di tutti i token, che non si salvano su disco.

| Impostazione | Gruppo 4B | Gruppo 9B |
|---|---|---|
| GPU | 24 GB | 45 GB |
| Righe per passo | 4–8 (accumulando fino a 32) | 1–2 (accumulando fino a 32) |
| Lunghezza massima | 8.192 token | 8.192 token |
| Learning rate | 5e-4 | 5e-4 |
| Partenza | la testa migliore della fase 1a | idem |

## 5. Fase 2: training anche del modello

Solo se gli obiettivi non sono raggiunti dopo gli esperimenti della fase 1.

| Impostazione | Valore |
|---|---|
| Partenza | LoRA del verdetto del piano binario + testa migliore della fase 1 |
| Learning rate | LoRA 5e-6 – 1e-5 (basso: il verdetto è già addestrato); testa 2e-4 |
| Loss | loss del verdetto del piano binario + 0,3 × loss di evidenza |
| Dati | `ev_joint` |
| Epoche | 1 |

Il verdetto ora può cambiare, quindi:
- prima del primo passo si controlla che il verdetto sia identico a quello di partenza;
- a ogni checkpoint si misura il verdetto su `dev_select`; un checkpoint si accetta solo se BAcc resta entro 0,3 punti ed ECE entro 0,01 da quello del piano binario;
- se la calibrazione si sposta, si aggiunge un termine che tiene le probabilità vicine a quelle del verdetto di partenza.

```sh
gpu-run --batch --name ev-joint-knowline -- bash -c \
  'python -m evidence.train_joint --model knowline-4b --mix ev_joint --config configs/<migliore>.json \
     --lambda 0.3 --run joint > logs/ev-joint-knowline.out 2>&1'
```
