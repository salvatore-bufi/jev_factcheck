# Step 4 — La funzione obiettivo, con derivazioni ed esempi numerici

## Obiettivo

Definire esattamente che cosa si ottimizza. Regola guida: **si addestra esattamente ciò che si legge in inferenza**, cioè la softmax sui logit delle due risposte (lettere o parole sì/no) nella posizione `answer_pos`. Il resto del vocabolario non entra nella loss.

## 1. Dal modello alla probabilità

All'ultima posizione del prompt il modello produce lo stato nascosto finale h (dopo la normalizzazione finale; dimensione 2.048 per d1-3B, 4.096 per JPT, 3.840 per Winnow). La testa del vocabolario dà un logit per token:

```text
d1-3B:   z_k = E[k] · h                                     (embedding legate)
JPT:     z_k = W_lm[k] · h                                  (lm_head separata)
Winnow:  z_k = 30 · tanh( (E[k] · h) / 30 )                 (embedding legate + softcapping)
```

Dei ~128.000–260.000 logit si prendono solo quelli delle risposte sì e no. Una risposta può avere più forme (d1-3B: `yes`/`Yes` e `no`/`No`/`NO`): il suo logit è il **massimo** del gruppo, come fa il modello in inferenza. Poi:

```text
z_sì = max(z_k, k ∈ gruppo sì)        z_no = max(z_k, k ∈ gruppo no)
margine "sì" m = (z_sì − z_no) / T
take = yes (Q1–Q3):   s = m           take = no (Q4):   s = −m
P(supportato) = σ(s) = 1 / (1 + e^(−s))
```

| Modello | gruppo sì | gruppo no | T |
|---|---|---|---|
| d1-3B | `yes` 11683, `Yes` 12447 | `no` 2243, `No` 4547, `NO` 19598 | 1 |
| JPT-9B | `" A"` 357 | `" B"` 417 | 1,087 |
| Winnow-12B | `B` 236799 (true) | `A` 236776 (false) | 1 |

Esempio con Q4 (d1-3B): z_yes = 0,4, z_Yes = 1,1, z_no = 2,0, z_No = 1,5, z_NO = −3 → z_sì = 1,1, z_no = 2,0 → m = −0,9 → s = +0,9 → P(supportato) = 0,71 (il modello dice "no, nessun dettaglio non supportato").

Esempio (Winnow): z_A = 2,1, z_B = 0,3 → s = −1,8 → P = 1/(1 + e^1,8) = 0,142.

Per JPT e senza softcapping, s = (W[357] − W[417]) · h / T: una regressione logistica su h con pesi già inizializzati ([00a](00a_scelta_approccio.md) §5).

## 2. La loss di default: cross-entropy binaria

Con y ∈ {0, 1} l'etichetta:

```text
L_CE = −[ y · ln σ(s) + (1 − y) · ln(1 − σ(s)) ]
dL_CE / ds = σ(s) − y
```

È identica alla cross-entropy della softmax a due classi (`-log_softmax(z/T)[giusta]`): le due scritture sono la stessa funzione.

**Esempio 1 — il difetto attuale.** Coppia **non** supportata (y = 0); il modello di partenza dà s = 2,2:

- P = σ(2,2) = 0,90
- L = −ln(1 − 0,90) = −ln 0,10 = 2,30
- gradiente su s = 0,90 − 0 = +0,90 (spinge s verso il basso con quasi la massima forza)

Dopo alcuni passi s = −0,5 → P = 0,38, L = −ln 0,62 = 0,48, gradiente +0,38. È il meccanismo che riporta la soglia ottimale da 0,90–0,95 verso 0,5.

**Esempio 2 — un esempio già giusto.** y = 1, s = 3 → P = 0,953, L = 0,049, gradiente −0,047: il modello quasi non si sposta.

## 3. Perché non l'SFT classico

L'SFT classico calcola la cross-entropy del token giusto **su tutto il vocabolario**. Situazione: P(A) = 0,05, P(B) = 0,40 e 0,55 sparso su altri token (a capo, "B.", "true", ...). Target B.

| Loss | Calcolo | Valore |
|---|---|---:|
| SFT classico | −ln 0,40 | 0,92 |
| CE sulle due lettere | P(B fra A e B) = 0,40/0,45 = 0,889 → −ln 0,889 | 0,12 |

La decisione che si legge in inferenza è già giusta e sicura (0,889), ma l'SFT continua a spingere per spostare su B la massa degli altri token: modifica parti del modello che non si leggono e allarga la distanza fra z_B e tutti gli altri logit, compreso z_A → probabilità sovra-confidenti. L'esperimento P3 ([07](07_esperimenti.md)) misura quanto si perde.

## 4. Varianti contro il rumore delle etichette

Dati generati e ANLI difficile contengono etichette sbagliate. Con la CE un esempio sbagliato ma "sicuro" spinge al massimo.

| Loss | Formula | Gradiente su s, per y = 1 e σ(s) = 0,01 |
|---|---|---:|
| CE | −ln σ(s) | −0,99 |
| CE con label smoothing ε = 0,05 | target y' = y(1 − ε) + ε/2 = 0,975 | σ(s) − y' = −0,965 |
| Brier | 2 (σ(s) − y)² | 4 (σ(s) − y) σ(s) (1 − σ(s)) = 4 · (−0,99) · 0,01 · 0,99 ≈ −0,04 |

- **Label smoothing**: tiene le probabilità lontane da 0 e 1 (P massima raggiungibile senza penalità: 0,975); utile per la calibrazione.
- **Brier**: quasi nessuna spinta sugli errori sicuri: robusto al rumore, ma lento a correggere errori veri. Coerente con il training originale di JPT.

Scelta iniziale: **CE**. Esperimenti P2 (Brier) e P-LS (smoothing ε = 0,05 solo su fonti generate e ANLI difficile).

## 5. Termini opzionali

### 5.1 Ancoraggio al modello di partenza (P-KL)

```text
L += β · [ p0 · ln(p0 / p) + (1 − p0) · ln((1 − p0) / (1 − p)) ]
```

p0 = P(supportato) del modello di partenza (precalcolata, campo `teacher_p`), p = σ(s). Esempio: p0 = 0,80, p = 0,60 → 0,092. β ∈ {0,05; 0,1}. Dettagli e motivazione in [02c](02c_replay_e_forgetting.md).

### 5.2 Loss contrastiva su coppie minime (X1–X2)

Molte righe generate hanno una "gemella" quasi identica con etichetta opposta (stesso `pair_id`): documento intero contro documento senza una frase (C2D, CG2C, `remove_evidence`), claim originale contro claim modificato (`contradict_claim`), le due revisioni di VitaminC.

```text
L_pair = −ln σ(s_pos − s_neg) = softplus(−(s_pos − s_neg))
```

Esempio: s_pos = 1,2 e s_neg = 0,9 → differenza 0,3 → L = −ln σ(0,3) = 0,55. Con differenza 3 → 0,05. Spinge il modello a guardare proprio la frase o la parola che cambia.

Peso μ ∈ {0,1; 0,3}. I due membri devono stare nello stesso micro-batch (il campionatore li accoppia, [05](05_training.md) §5).

## 6. La loss totale

```text
L = media_batch[ CE (o Brier, o CE con smoothing) ]
  + β · media_batch[ KL(p0 ‖ p) ]          β = 0 di default
  + μ · media_coppie[ L_pair ]             μ = 0 di default
```

## 7. Codice di riferimento: `finetune/losses.py`

```python
import torch
import torch.nn.functional as F

def margin(h_answer, w_yes, w_no, temperature, softcap=None, take="yes"):
    """h_answer [B, d] (stato finale in answer_pos); w_yes [n_sì, d], w_no [n_no, d]: righe della testa
    del vocabolario per i token di ciascuna risposta -> margine s [B] a favore di "supportato"."""
    z_yes, z_no = h_answer @ w_yes.T, h_answer @ w_no.T          # [B, n_sì], [B, n_no]
    if softcap:                                                  # Winnow: 30 * tanh(z / 30)
        z_yes, z_no = softcap * torch.tanh(z_yes / softcap), softcap * torch.tanh(z_no / softcap)
    m = (z_yes.max(-1).values - z_no.max(-1).values).float() / temperature
    return m if take == "yes" else -m

def binary_loss(s, y, kind="ce", smoothing=0.0):
    if kind == "ce":
        y = y * (1 - smoothing) + smoothing / 2
        return F.binary_cross_entropy_with_logits(s, y)
    if kind == "brier":
        return (2 * (torch.sigmoid(s) - y) ** 2).mean()
    raise ValueError(kind)

def anchor_kl(s, p0, eps=1e-6):
    p = torch.sigmoid(s).clamp(eps, 1 - eps); p0 = p0.clamp(eps, 1 - eps)
    return (p0 * (p0 / p).log() + (1 - p0) * ((1 - p0) / (1 - p)).log()).mean()

def pair_loss(s_pos, s_neg):
    return F.softplus(-(s_pos - s_neg)).mean()
```

## 8. Test numerici: `finetune/test_losses.py`

```python
import math, torch
from losses import binary_loss, anchor_kl, pair_loss
t = lambda x: torch.tensor([float(x)])
assert abs(binary_loss(t(2.2), t(0)).item() - 2.3051) < 1e-3          # esempio 1 (σ(2,2) = 0,9002)
assert abs(binary_loss(t(-0.5), t(0)).item() - 0.4741) < 1e-3
assert abs(binary_loss(t(3.0), t(1)).item() - 0.0486) < 1e-3          # esempio 2
p0, p = 0.8, 0.6
assert abs(anchor_kl(t(math.log(p / (1 - p))), t(p0)).item() - 0.0915) < 1e-3
assert abs(pair_loss(t(1.2), t(0.9)).item() - 0.5544) < 1e-3
s = t(math.log(0.01 / 0.99)); s.requires_grad_(True)
binary_loss(s, t(1), "brier").backward()
assert abs(s.grad.item() - (-0.0392)) < 1e-3                          # Brier quasi non spinge
```

(Valori attesi verificati eseguendo le funzioni di §7 con torch 2.13.)

## 9. Verifica che il margine ridotto coincida con il forward completo

Prima del training ([05](05_training.md) §8): su 50 prompt, `margin(h, w_yes, w_no, T, softcap)` deve coincidere con `(max logits[:, gruppo sì] − max logits[:, gruppo no]) / T` del forward completo del modello (differenza < 1e-3 in fp32). Per d1-3B si confronta anche con la P(sì) restituita dall'API `system_one` (entro 0,01). Per Winnow verifica anche che il softcapping sia applicato come fa il modello (se nella classe di Gemma-4 c'è anche uno scaling dei logit, va aggiunto).

## Criteri per passare allo step successivo

- `test_losses.py` passa.
- Verifica §9 superata per entrambi i modelli.
