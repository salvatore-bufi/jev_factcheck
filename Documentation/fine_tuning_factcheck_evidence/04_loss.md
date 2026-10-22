# Step 4 — La loss: verdetto + evidenza per frase

## Obiettivo

Definire la loss che addestra la testa di evidenza (e, nel training congiunto, il verdetto) tenendo conto di tre fatti: l'evidenza esiste solo per i positivi; le frasi di evidenza sono poche (classe molto sbilanciata); le annotazioni gold sono incomplete.

## 1. La struttura

```text
L = L_verdetto + λ · L_evidenza
L_verdetto = CE binaria sul margine delle lettere        (piano binario, 04_loss.md)
L_evidenza = media sulle righe con evidence_mask = 1 di  w_riga · [ L_BCE + γ · L_rank ]
```

- Con backbone **congelato** (prima fase, [05](05_training.md)) `L_verdetto` non c'è: il verdetto non cambia.
- Nel training **congiunto** λ ∈ [0,2; 0,5], scelto su `dev_select` + `dev_evid_select`.
- `w_riga` = `evidence_weight` (1,0 gold e construct, 0,5 ablation).

## 2. BCE per frase con classi sbilanciate

Per una riga con n frasi e punteggi `e_k = σ(o_k)`:

```text
L_BCE = (1/n) Σ_k  [ α · y_k · (−ln e_k)  +  (1 − y_k) · ω · (−ln(1 − e_k)) ]
```

- Tipicamente 2 frasi di evidenza su 30–100: senza correzione la testa impara a dire "no" a tutte.
- `α` = peso dei positivi, intorno a (frasi non evidenza)/(frasi evidenza) sul mix, limitato a 20.
- `ω` = peso dei negativi, < 1 per le fonti con annotazioni incomplete (§3).

Esempio: riga con 40 frasi, evidenza {1, 40}, α = 10, ω = 1. Se la testa dà 0,02 a tutte:
- contributo positivi: 10 · 2 · (−ln 0,02) = 10 · 2 · 3,91 = 78,2
- contributo negativi: 38 · (−ln 0,98) = 38 · 0,020 = 0,77
- L_BCE = (78,2 + 0,77)/40 = **1,97**: la loss spinge su S1 e S40.

## 3. Annotazioni incomplete: ranking invece di "tutto il resto è falso"

Una frase non annotata in FEVER può essere evidenza valida. Con la BCE piena verrebbe penalizzata. Due correttivi:

1. **ω basso sulle fonti incomplete** (FEVER: 0,3; HoVer: 0,5; construct: 1,0).
2. **Loss di ranking**: ogni frase gold deve stare sopra le frasi non gold più alte, non necessariamente sopra tutte.

```text
L_rank = media sulle frasi gold g di  softplus( m − o_g + max_{j ∉ gold} o_j )
```

Con margine m = 1: se la frase gold ha logit 0,5 e la migliore non gold 1,2, L = softplus(1 − 0,5 + 1,2) = softplus(1,7) = 1,87; quando la gold sale a 2,5, L = softplus(−0,3) = 0,55.

Il `max` sulle non gold concentra il segnale sulla frase "concorrente" più pericolosa, e una sola frase non gold valida non viene schiacciata per tutte. γ ∈ {0; 0,5; 1}: esperimento H7.

## 4. Insiemi alternativi (FEVER)

Quando ci sono più insiemi gold alternativi (`evidence_sets`), il target è l'**unione**: si vogliono evidenziare tutte le frasi che supportano. Variante (H7): loss "almeno un insieme", cioè il minimo della loss fra gli insiemi; più fedele alle annotazioni, più complessa.

## 5. Nessuna evidenza sui negativi

Le righe con label 0 hanno `evidence_mask = 0`: nessun termine di evidenza. In produzione la testa produce comunque dei punteggi, ma non vengono mostrati quando il verdetto è "non supportato" ([08](08_merge_export.md)).

## 6. Codice di riferimento

```python
import torch
import torch.nn.functional as F

def evidence_loss(logits, target, mask, pos_w, neg_w, gamma=0.5, margin=1.0):
    """logits, target, mask: [B, N] (mask = frase reale); pos_w: scalare; neg_w: [B] per fonte."""
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    w = torch.where(target > 0, torch.as_tensor(pos_w, device=logits.device), neg_w[:, None])
    l_bce = (bce * w * mask).sum(1) / mask.sum(1)
    neg = logits.masked_fill((target > 0) | (mask == 0), float("-inf")).amax(1, keepdim=True)
    rank = F.softplus(margin - logits + neg) * (target > 0)
    l_rank = rank.sum(1) / (target > 0).sum(1).clamp(min=1)
    return l_bce + gamma * l_rank            # [B]; poi media pesata con w_riga sulle righe con evidenza
```

## Criteri per passare allo step successivo

- Funzioni testate sugli esempi numerici di questo file.
- α calcolato sul mix; ω per fonte fissato.
