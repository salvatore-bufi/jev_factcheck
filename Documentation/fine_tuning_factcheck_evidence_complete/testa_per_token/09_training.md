# Passo 9 — Training

## In breve

Si addestra la testa in tre modi, dal più semplice al più costoso; si passa al successivo solo se il precedente non basta.

| Fase | Il decision model | Ingresso della testa | Quando |
|---|---|---|---|
| **1a** | congelato, non in memoria | vettori ridotti con la PCA, salvati al passo 6 | quasi tutti gli esperimenti: ogni prova dura minuti |
| **1b** | congelato, in memoria | vettori completi ricalcolati a ogni passo; la testa impara la sua riduzione al posto della PCA | conferma (H9): la PCA perde informazione? |
| **2** | addestrato anche lui, con una LoRA | come 1b, con gradiente nel modello | solo se dopo gli esperimenti gli obiettivi non sono raggiunti |

Nelle fasi 1a e 1b il verdetto **non può cambiare**: il modello non viene toccato.

Ogni prova scrive in `data/evidence_tok/runs/<modello>/<id>/`: `config.json` (opzioni della testa, della loss, iperparametri, hash delle feature e del mix), `head.safetensors`, `log.jsonl`, `dev_metrics.json`.

## 1. Fase 1a: il ciclo di training

Ogni riga del mix porta: le feature salvate (`h`, `att`, `q`), le etichette dei token (`tok_label`), gli intervalli di token delle frasi gold (`gold_tok_spans`), il tipo di loss (`evidence_term`), ω e il peso della riga.

1. **Batch per lunghezza**: le righe si ordinano per numero di token e si raggruppano a 16, così il riempimento è minimo; l'ordine dei batch è casuale e cambia a ogni epoca.
2. **Riempimento**: ogni batch si porta alla riga più lunga; `mask` segna i token veri. La testa è invariante al riempimento (verificato: differenza ~1e-7).
3. **Loss per riga** ([08](08_testa_e_loss.md) §6), moltiplicata per il peso della riga; media sul batch divisa per la somma dei pesi delle righe con loss.
4. **Ottimizzazione**: AdamW, clipping del gradiente a 1, learning rate con riscaldamento lineare (primo 5% dei passi) e discesa a coseno.
5. **A fine epoca**: valutazione su `dev_evid_select` con la decodifica e le metriche del passo 10; si tiene il checkpoint migliore; ci si ferma dopo 2 epoche senza miglioramento.

```python
import math
import random

import torch
import torch.nn.functional as F

from head import TokenEvidenceHead
from losses import token_evidence_loss


def collate(rows, device):
    """rows: dict con h [n, L, 256], att [n, 128], q [L, D], tok_label [n], gold_tok_spans, evidence_term, omega, row_weight.
    -> tensori riempiti fino al documento più lungo del batch."""
    n = max(r["h"].shape[0] for r in rows)
    B, L = len(rows), rows[0]["h"].shape[1]
    h = torch.zeros(B, n, L, rows[0]["h"].shape[2])
    att = torch.zeros(B, n, rows[0]["att"].shape[1])
    y = torch.zeros(B, n)
    mask = torch.zeros(B, n)
    for i, r in enumerate(rows):
        k = r["h"].shape[0]
        h[i, :k], att[i, :k], y[i, :k], mask[i, :k] = r["h"].float(), r["att"].float(), r["tok_label"].float(), 1
    q = torch.stack([r["q"].float() for r in rows])
    return h.to(device), att.to(device), q.to(device), y.to(device), mask.to(device)


def batches(rows, size, seed):
    """Righe di lunghezza simile insieme (meno riempimento), ordine dei batch casuale."""
    order = sorted(range(len(rows)), key=lambda i: rows[i]["h"].shape[0])
    groups = [order[i:i + size] for i in range(0, len(order), size)]
    random.Random(seed).shuffle(groups)
    return groups


def batch_loss(model, rows, device):
    h, att, q, y, mask = collate(rows, device)
    s, s0 = model(h, att, q, mask)
    losses = [token_evidence_loss(s[i], s0[i], y[i], mask[i], r["gold_tok_spans"], r["evidence_term"], r["omega"])
              * r["row_weight"] for i, r in enumerate(rows)]
    weight = sum(r["row_weight"] for r in rows if r["evidence_term"] != "none")
    return torch.stack(losses).sum() / max(weight, 1e-6)


def train(rows, dev_rows, d_q, epochs=10, batch=16, lr=5e-4, seed=13, device="cuda", patience=2, evaluate=None, **head_kw):
    torch.manual_seed(seed)
    model = TokenEvidenceHead(d_q=d_q, n_layers=rows[0]["h"].shape[1], **head_kw).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * math.ceil(len(rows) / batch)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda k: min(1.0, k / max(1, 0.05 * steps)) * 0.5 * (1 + math.cos(math.pi * min(k, steps) / steps)))
    best, best_state, bad = -1.0, None, 0
    for epoch in range(epochs):
        model.train()
        for idx in batches(rows, batch, seed + epoch):
            loss = batch_loss(model, [rows[i] for i in idx], device)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        score = evaluate(model, dev_rows) if evaluate else -loss.item()
        print(f"epoch {epoch}  loss {loss.item():.4f}  dev {score:.4f}")
        if score > best:
            best, best_state, bad = score, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, best
```

Questo ciclo è stato eseguito su feature finte (120 righe di 200–900 token, segnale nascosto in alcune componenti dei token gold): converge in poche epoche e trova le frasi gold.

Nel codice vero le righe si leggono dagli shard del passo 6 (`evidence/token_dataset.py`, che restituisce gli stessi dizionari) e la funzione `evaluate` usa `decode` e le metriche del passo 10.

| Iperparametro | Valore |
|---|---|
| Learning rate | 5e-4, riscaldamento 5%, coseno |
| Weight decay | 0,01 |
| Batch | 16 righe raggruppate per lunghezza |
| Epoche | fino a 10, arresto dopo 2 senza miglioramento |
| Dropout | 0,1 nel transformer; 0,1 sulle feature di attenzione (intere, per token) |
| Criterio del checkpoint | media di frasi gold colpite e F1 sui caratteri su `dev_evid_select` |
| Seme | 13 (3 semi per la configurazione finale: 13, 14, 15) |

```sh
python -m evidence.token_train --phase 1a --model knowline-4b --features data/evidence_tok/features/knowline-4b \
    --mix ev_tok_head --config configs/tok_h0.json --run h0
```

Memoria: per una riga di 6.000 token la testa usa poche decine di MB. La fase 1a gira su qualunque GPU, anche su CPU per righe brevi.

**Prima della prima prova**: la testa deve imparare a memoria 64 righe (frasi gold colpite ≈ 1, nulla acceso fuori). Se non ci riesce, c'è un errore in `tok_char`, nelle etichette o nel codice.

## 2. Scegliere i parametri della decodifica

Dopo il training, su `dev_evid_select`, solo sulle righe supportate che il verdetto giudica supportate, griglia su:

| Parametro | Valori |
|---|---|
| δ (scarto dalla soglia) | da −1 a 1, passo 0,1 |
| buco massimo | 0, 2, 3, 5 token |
| lunghezza minima | 2, 4, 6 token |
| tratti massimi | 3, 5, 8 |

Criterio: media di frasi gold colpite e F1 sui caratteri. Conferma su `dev_evid_confirm`: non più di 1 punto sotto. I valori vanno in `decision.json`.

## 3. Che cosa guardare durante il training

| Cosa | Su | Segnale di problema | Cosa fare |
|---|---|---|---|
| Frasi gold colpite, F1 sui caratteri | `dev_evid_select` | non superano E-LEX dopo 2 epoche | controllare `tok_char` ed etichette prima di toccare la testa |
| Tratti per riga, token per tratto | `dev_evid_select` | paragrafi interi, o tanti frammenti | pesi di `L_out` e `L_tv`; decodifica |
| Frasi colpite per distanza fra le frasi gold | righe con evidenza lontana | crollo oltre 10 frasi | contesto globale (H3); più righe spostate |
| Righe con tratti sopra soglia | `dev_evid_neg` | oltre il 30% | peso di `mil_neg` più alto |
| Pesi α dei layer | testa | uno vicino a 1 | basta quel layer (metà spazio su disco) |

## 4. Fase 1b: vettori completi, riduzione appresa

Il modello è in memoria e congelato. A ogni passo si rifà il forward (senza gradiente nel modello) e si prendono i vettori **completi** dei token del documento (D numeri). La testa li riduce con una matrice appresa, `nn.Linear(D, 256)` per layer, **inizializzata con la PCA** (media e proiezione), al posto della proiezione fissa. Tutto il resto è uguale.

| Impostazione | Gruppo 4B | Gruppo 9B |
|---|---|---|
| GPU | 24 GB | 45 GB |
| Righe per passo | 2–4 (accumulando fino a 16) | 1 (accumulando fino a 16) |
| Lunghezza massima | 8.192 token (si escludono le righe oltre: oggi 75 di WiCE) | 8.192 token |
| Partenza | la testa migliore della fase 1a | idem |
| Learning rate | 2e-4 (proiezione 5e-5) | idem |
| Epoche | 2 | 1 |

## 5. Fase 2: training anche del modello

Solo se gli obiettivi non sono raggiunti dopo gli esperimenti della fase 1. Il modello del verdetto è il decision model con la LoRA del fine-tuning binario (rango 16 sui moduli di attenzione, DeltaNet e MLP di Qwen3.5). Si continua ad addestrare quella LoRA insieme alla testa.

| Impostazione | Valore |
|---|---|
| Partenza | LoRA del verdetto + testa migliore della fase 1 |
| Learning rate | LoRA 5e-6 – 1e-5 (basso: il verdetto è già addestrato); testa 2e-4 |
| Loss | loss del verdetto (cross-entropy sui due punteggi sì/no, su tutte le righe) + 0,3 × loss di evidenza |
| Dati | i dati del fine-tuning binario più le righe con evidenza |
| Epoche | 1 |
| GPU | 4B: 24 GB; 9B: 45 GB |

Il verdetto ora può cambiare, quindi:
- prima del primo passo si controlla che sia identico a quello di partenza;
- a ogni checkpoint si misurano BAcc, AUC ed ECE sul dev del fine-tuning binario; un checkpoint si accetta solo se BAcc resta entro 0,3 punti ed ECE entro 0,01;
- se la calibrazione si sposta, si aggiunge un termine che tiene le probabilità vicine a quelle del verdetto di partenza: β · KL(p_partenza ‖ p), β = 0,1.

## 6. Test prima di ogni nuova configurazione

| Test | Atteso |
|---|---|
| Imparare a memoria 64 righe | frasi gold colpite ≈ 1, F1 sui caratteri > 0,9 |
| H0 riprodotto con le opzioni spente | stessi numeri (stesso seme) |
| Righe `none` | loss e gradiente zero |
| Batch con righe di lunghezze molto diverse | stessi punteggi di una riga alla volta (differenza < 1e-4) |
