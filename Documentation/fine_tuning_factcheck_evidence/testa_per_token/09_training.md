# Passo 9 — Training

## In breve

Come nel piano per frase, tre modi dal più semplice al più costoso; si passa al successivo solo se il precedente non basta.

| Fase | Il modello | Ingresso della testa | Quando |
|---|---|---|---|
| **1a** | congelato, non in memoria | vettori ridotti con la PCA, salvati al passo 6 | quasi tutti gli esperimenti |
| **1b** | congelato, in memoria | vettori completi ricalcolati a ogni passo; la testa impara la sua proiezione al posto della PCA | conferma (H9): la PCA perde qualcosa? |
| **2** | LoRA addestrata con la testa | come 1b, con gradiente | solo se gli obiettivi non sono raggiunti |

Nelle fasi 1a e 1b il verdetto non cambia. Ogni prova scrive in `data/evidence_tok/runs/<modello>/<id>/` configurazione, pesi della testa, log, metriche dev.

## 1. Fase 1a

| Impostazione | Valore |
|---|---|
| Ottimizzatore | AdamW, lr 5e-4, warmup 5%, coseno, weight decay 0,01 |
| Epoche | fino a 10; ci si ferma se per 2 epoche `dev_evid_select` non migliora |
| Batch | 16 righe, raggruppate per lunghezza (si riempiono fino alla più lunga del batch) |
| Dropout | 0,1 nel transformer; 0,1 sulle feature di attenzione (intere, per token) |
| Checkpoint tenuto | media più alta di F1 sui caratteri e frasi gold colpite, su `dev_evid_select` |
| Loss | [08](08_testa_e_loss.md) §3, media sulle righe pesata con `row_weight` |

```sh
python -m evidence.token_train --phase 1a --model knowline-4b --features data/evidence_tok/features/knowline-4b \
    --mix ev_tok_head --config configs/tok_h0.json --run h0
```

Prima della prima prova: la testa deve imparare a memoria 64 righe (frasi gold colpite ≈ 1, niente acceso fuori). Se non ci riesce, c'è un errore in `tok_char`, nelle etichette o nel codice.

Memoria: per una riga di 6.000 token la testa usa poche decine di MB (il transformer lavora su ~375 blocchi). La fase 1a gira anche su CPU, più comodamente su una GPU qualsiasi.

## 2. Scegliere i parametri della decodifica

Dopo il training, su `dev_evid_select`, fra le righe supportate che il verdetto giudica supportate, griglia su:

| Parametro | Valori |
|---|---|
| δ (scarto dalla soglia) | da −1 a 1, passo 0,1 |
| buco massimo | 0, 2, 3, 5 token |
| lunghezza minima | 2, 4, 6 token |
| tratti massimi | 3, 5, 8 |

Criterio: media di F1 sui caratteri e frasi gold colpite. Si conferma su `dev_evid_confirm` (non più di 1 punto sotto). I valori vanno in `decision.json`.

## 3. Che cosa guardare durante il training

| Cosa | Su | Segnale di problema | Cosa fare |
|---|---|---|---|
| F1 sui caratteri, frasi gold colpite | `dev_evid_select` | non supera E-LEX dopo 2 epoche | controllare `tok_char` ed etichette prima di toccare la testa |
| Tratti per riga, lunghezza media dei tratti | `dev_evid_select` | tratti troppo lunghi (paragrafi interi) o troppi frammenti | peso di `L_out`, `L_tv`; parametri della decodifica |
| Frasi gold colpite per distanza fra le frasi gold | righe sparse | crollo oltre 10 frasi | contesto globale (H3), più righe sparse |
| Righe con tratti sopra soglia | `dev_evid_neg` | oltre il 30% | peso di `mil_neg` più alto |
| Pesi dei layer | testa | tutto su un layer | basta quel layer (meno spazio su disco) |

## 4. Fase 1b: vettori completi, proiezione appresa

Il modello è in memoria, congelato. A ogni passo si rifà il forward, si prendono i vettori **completi** dei token del documento (2.560 o 4.096 numeri) e la testa li proietta con una matrice appresa (`nn.Linear(D, 256)`, inizializzata con la PCA).

| Impostazione | Gruppo 4B | Gruppo 9B |
|---|---|---|
| GPU | 24 GB | 45 GB |
| Righe per passo | 2–4 (accumulando fino a 16) | 1 (accumulando fino a 16) |
| Lunghezza massima | 8.192 token | 8.192 token |
| Partenza | la testa migliore della fase 1a | idem |
| Epoche | 2 | 1 |

## 5. Fase 2: training anche del modello

Come nel piano per frase ([../testa_per_frase/09_training.md](../testa_per_frase/09_training.md) §5): partenza dalla LoRA del verdetto del piano binario e dalla testa della fase 1; lr della LoRA 5e-6 – 1e-5; loss = loss del verdetto + 0,3 × loss di evidenza; dati del piano binario più le righe con evidenza. Il verdetto si controlla a ogni checkpoint: si accetta solo entro 0,3 punti di BAcc e 0,01 di ECE dal piano binario.

## 6. Test prima di ogni nuova configurazione

| Test | Atteso |
|---|---|
| Imparare a memoria 64 righe | frasi gold colpite ≈ 1, F1 sui caratteri > 0,9 |
| H0 riprodotto con le opzioni spente | stessi numeri (stesso seme) |
| Righe `none` | loss e gradiente zero |
| Batch con righe di lunghezze molto diverse | stessi punteggi di una riga alla volta (a meno di 1e-4) |
