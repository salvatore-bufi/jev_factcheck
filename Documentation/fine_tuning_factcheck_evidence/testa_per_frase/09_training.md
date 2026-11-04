# Passo 9 — Training

## In breve

Si addestra la testa in tre modi, dal più semplice al più costoso; si passa al successivo solo se il precedente non basta.

| Fase | Il decision model | Ingresso della testa | Quando |
|---|---|---|---|
| **1a** | congelato, non in memoria | vettori delle frasi salvati al passo 6 | quasi tutti gli esperimenti: ogni prova dura minuti |
| **1b** | congelato, in memoria | vettori ricalcolati a ogni passo | solo per l'esperimento che ha bisogno dei vettori di tutti i token (H2d, media pesata dei token) |
| **2** | addestrato anche lui, con una LoRA | come 1a, con gradiente nel modello | solo se dopo gli esperimenti gli obiettivi non sono raggiunti |

Nelle fasi 1a e 1b il verdetto **non può cambiare**: il modello non viene toccato.

Ogni prova scrive in `data/evidence/runs/<modello>/<id>/`: `config.json` (opzioni della testa, della loss, iperparametri, hash delle feature e del mix), `head.safetensors`, `log.jsonl`, `dev_metrics.json`.

## 1. Fase 1a: il ciclo di training

Ogni riga del mix porta: le feature salvate (`q`, `h_end`, `h_mean`, `att`, `n_tok`), le etichette delle frasi (`gold`), il tipo di loss (`evidence_term`), ω e il peso della riga.

1. **Batch per lunghezza**: le righe si ordinano per numero di frasi e si raggruppano a 64, così il riempimento è minimo; l'ordine dei batch è casuale e cambia a ogni epoca.
2. **Riempimento**: ogni batch si porta alla riga con più frasi; `mask` segna le frasi vere. La testa è invariante al riempimento (verificato: differenza ~1e-7).
3. **Loss** ([08](08_testa_e_loss.md) §6.3) per riga, moltiplicata per il peso della riga; somma divisa per la somma dei pesi delle righe con loss.
4. **Ottimizzazione**: AdamW, clipping del gradiente a 1, learning rate con riscaldamento lineare (primo 5% dei passi) e discesa a coseno.
5. **A fine epoca**: valutazione su `dev_evid_select` con la selezione e le metriche del passo 10; si tiene il checkpoint migliore; ci si ferma dopo 2 epoche senza miglioramento.

```python
import math
import random

import torch

from head import EvidenceHead
from losses import zlpr_loss

TERMS = {"supervised": 0, "mil_neg": 1, "mil_pos": 2, "none": 3}


def collate(rows, device):
    """rows: dict con q [L, D], h_end [n, L, D], h_mean [n, L, D], att [n, 128], n_tok [n], gold [n] (0/1),
    omega, evidence_term, row_weight. -> tensori riempiti fino alla riga con più frasi del batch."""
    n = max(r["h_end"].shape[0] for r in rows)
    B, (L, D) = len(rows), rows[0]["q"].shape
    h_end, h_mean = torch.zeros(B, n, L, D), torch.zeros(B, n, L, D)
    att, n_tok = torch.zeros(B, n, rows[0]["att"].shape[1]), torch.ones(B, n)
    gold, mask, omega = torch.zeros(B, n), torch.zeros(B, n), torch.ones(B, n)
    for i, r in enumerate(rows):
        k = r["h_end"].shape[0]
        h_end[i, :k], h_mean[i, :k], att[i, :k] = r["h_end"].float(), r["h_mean"].float(), r["att"].float()
        n_tok[i, :k], gold[i, :k], mask[i, :k], omega[i, :k] = r["n_tok"].float(), r["gold"].float(), 1, r["omega"]
    q = torch.stack([r["q"].float() for r in rows])
    term = torch.tensor([TERMS[r["evidence_term"]] for r in rows])
    weight = torch.tensor([r["row_weight"] for r in rows], dtype=torch.float)
    return [t.to(device) for t in (q, h_end, h_mean, att, n_tok, mask, gold, omega, term, weight)]


def batches(rows, size, seed):
    order = sorted(range(len(rows)), key=lambda i: rows[i]["h_end"].shape[0])
    groups = [order[i:i + size] for i in range(0, len(order), size)]
    random.Random(seed).shuffle(groups)
    return groups


def batch_loss(model, rows, device):
    q, h_end, h_mean, att, n_tok, mask, gold, omega, term, weight = collate(rows, device)
    s, s0, _ = model(q, h_end, h_mean, att, n_tok, mask)
    loss = zlpr_loss(s, s0, gold, mask, omega, term) * weight
    active = (term != TERMS["none"]).float() * weight
    return loss.sum() / active.sum().clamp_min(1e-6)


def train(rows, dev_rows, d_model, epochs=10, batch=64, lr=1e-3, seed=13, device="cuda", patience=2, evaluate=None,
          **head_kw):
    torch.manual_seed(seed)
    model = EvidenceHead(d_model=d_model, n_mix=rows[0]["q"].shape[0], n_att=rows[0]["att"].shape[1], **head_kw).to(device)
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

Questo ciclo è stato eseguito su feature finte (300 righe di 10–80 frasi, segnale nascosto nelle frasi gold, un quarto di negativi): in 5 epoche arriva a F1 = 1 sul dev finto, con arresto anticipato e ripristino del checkpoint migliore.

Nel codice vero le righe si leggono dagli shard del passo 6 (`evidence/feature_dataset.py`, che restituisce gli stessi dizionari) e la funzione `evaluate` usa `select` e le metriche del passo 10.

| Iperparametro | Valore |
|---|---|
| Learning rate | 1e-3, riscaldamento 5%, coseno |
| Weight decay | 0,01 |
| Batch | 64 righe raggruppate per lunghezza |
| Epoche | fino a 10, arresto dopo 2 senza miglioramento |
| Dropout | 0,1 nel transformer; 0,1 sulle feature di attenzione (intere, per frase) |
| Criterio del checkpoint | media di F1 per frase e insieme completo su `dev_evid_select` |
| Seme | 13 (3 semi per la configurazione finale: 13, 14, 15) |

```sh
python -m evidence.train_head --phase 1a --model knowline-4b --features data/evidence/features/knowline-4b \
    --mix ev_head --config configs/h0.json --run h0
```

La fase 1a non ha bisogno del modello in memoria: gira su qualunque GPU, anche su CPU.

**Prima della prima prova**: la testa deve imparare a memoria 64 righe (F1 ≈ 1 su quelle righe). Se non ci riesce, c'è un errore nelle posizioni delle frasi, nelle etichette o nel codice.

## 2. La regola di selezione delle frasi

Si fissa dopo il training, su `dev_evid_select`, solo sulle righe supportate che il verdetto giudica supportate:

| Loss usata | Regola | Cosa si sceglie |
|---|---|---|
| loss di riferimento (H0) | σ(s[k]) > τ_e | τ_e che dà l'F1 per frase migliore |
| loss con soglia del documento | s[k] > s0 + δ | δ fra −1 e 1, passo 0,1; si tiene δ ≠ 0 solo se guadagna almeno 1 punto di F1 |

Sempre almeno 1 frase (la migliore) se il verdetto è "supportato" e al massimo 5. Conferma su `dev_evid_confirm`: F1 non più di 1 punto sotto. I valori vanno in `decision.json`.

## 3. Che cosa guardare durante il training

| Cosa | Su | Segnale di problema | Cosa fare |
|---|---|---|---|
| F1 e insieme completo | `dev_evid_select` | non superano E-LEX dopo 2 epoche | controllare le posizioni delle frasi ([06](06_rendering_ed_estrazione.md) §7) prima di cambiare la testa |
| F1 per distanza fra le frasi gold | righe con evidenza lontana | crollo oltre 10 frasi | più righe spostate; transformer a 4 layer; H-PAIR |
| F1 per distanza dall'inizio del documento | `dev_evid_select` | crollo oltre 1.024 o 4.096 token | più righe lontane; attenzione come feature |
| Frasi mostrate per riga | `dev_evid_select` | molte più o molte meno delle frasi gold | δ, ω |
| Negativi con frasi sopra soglia | `dev_evid_neg` | oltre il 30% | peso di `mil_neg` più alto |
| Recall sui positivi incerti (P appena sopra la soglia del verdetto, fino a +0,15) | `dev_evid_select` | molto sotto la media | peso di `mil_neg` più basso |
| Pesi α dei layer | testa | uno sotto il 10% | togliere quel layer (meno spazio, meno costo) |

## 4. Fase 1b: con il modello in memoria

Il modello è caricato e congelato; a ogni passo si rifà il forward (senza gradiente) per avere i vettori. Serve per un solo esperimento:
- **H2d**: invece della media fissa dei token di una frase, una media pesata in cui la testa decide quali token contano (per esempio il nome proprio dentro una frase lunga). Per ogni frase: proiezione `d → 256` dei vettori dei suoi token, punteggio di ogni token contro una query `u = W_u [c ; q]` (`c` = media dei token del preambolo), softmax dentro la frase, somma pesata. Richiede i vettori di tutti i token, che non si salvano su disco.

| Impostazione | Gruppo 4B | Gruppo 9B |
|---|---|---|
| GPU | 24 GB | 45 GB |
| Righe per passo | 4–8 (accumulando fino a 32) | 1–2 (accumulando fino a 32) |
| Lunghezza massima | 8.192 token | 8.192 token |
| Learning rate | 5e-4 | 5e-4 |
| Partenza | la testa migliore della fase 1a | idem |

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
- se la calibrazione si sposta, si aggiunge un termine che tiene le probabilità vicine a quelle del verdetto di partenza: β · KL(p_partenza ‖ p), β = 0,1;
- se il termine `mil_neg` sposta la calibrazione, lo si applica senza gradiente verso il modello (agisce solo sulla testa).

## 6. Test prima di ogni nuova configurazione

| Test | Atteso |
|---|---|
| Imparare a memoria 64 righe | F1 ≈ 1 sul training |
| H0 riprodotto con le opzioni spente | stessi numeri (stesso seme) |
| Righe `none` | loss e gradiente zero |
| Batch con righe di lunghezze molto diverse | stessi punteggi di una riga alla volta (differenza < 1e-4) |
