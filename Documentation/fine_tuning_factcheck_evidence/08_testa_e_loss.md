# Passo 8 — La testa e la loss

## In breve

- La **testa** (`evidence/head.py`) riceve i vettori salvati al passo 6 e restituisce un punteggio per ogni frase (`s_k`) e un punteggio-soglia per la riga (`s_0`).
- La **loss** (`evidence/losses.py`) dice alla testa quanto sta sbagliando, così il training la corregge.

Ogni parte della testa si può accendere o spegnere con un'opzione: tutti gli esperimenti ([11](11_esperimenti.md)) usano lo stesso codice con opzioni diverse.

## 1. La testa, passo per passo

Per una riga con n frasi:

1. **Combina i layer.** Per ogni frase ci sono vettori da 4 layer. La testa li somma con 4 pesi che impara (normalizzati in modo che sommino a 1).
2. **Costruisce un vettore per frase** da 256 numeri:
   - vettore dell'ultimo token e media dei token della frase, affiancati e moltiplicati per una matrice;
   - più l'attenzione ricevuta dalla frase (8 layer × teste, più la lunghezza della frase), moltiplicata per un'altra matrice;
   - più un vettore che indica la posizione della frase nel documento (frase 0, 1, 2, …).
3. **Costruisce il vettore `x_0`** da `q` (la posizione di risposta), allo stesso modo.
4. **Fa comunicare tutto**: `[x_0, frase 0, …, frase n−1]` passano in 2 layer di transformer in cui ognuno vede tutti gli altri. Dopo, il vettore della frase 1 "sa" anche della frase 40.
5. **Dà i punteggi**: uno stesso strato finale trasforma ogni vettore in un numero. Il numero di `x_0` è la soglia `s_0`; quelli delle frasi sono `s_k`.

Facoltativo (esperimento H-PAIR): un punteggio in più per ogni **coppia** di frasi, che premia una frase se c'è un'altra frase con cui "fa coppia". Pensato per l'evidenza su più frasi.

```python
import torch
import torch.nn as nn

class EvidenceHead(nn.Module):
    def __init__(self, d_model, n_mix, n_att, d=256, layers=2, heads=4, max_sent=512,
                 use_mix=True, use_mean=True, use_att=True, use_pos=True, use_pair=False, pair_rank=64):
        super().__init__()
        self.use_mix, self.use_mean, self.use_att, self.use_pos, self.use_pair = use_mix, use_mean, use_att, use_pos, use_pair
        self.mix = nn.Parameter(torch.zeros(n_mix))                     # pesi dei layer (punto 1)
        self.ln = nn.LayerNorm(d_model)                                  # i layer hanno scale diverse
        self.proj_h = nn.Linear(d_model * (2 if use_mean else 1), d)     # [ultimo token ; media] → 256
        self.proj_q = nn.Linear(d_model, d)                              # q → 256
        self.proj_a = nn.Linear(n_att + 1, d)                            # attenzione + log(numero di token) → 256
        self.pos = nn.Embedding(max_sent, d)                             # posizione della frase
        self.kind = nn.Embedding(2, d)                                   # 0 = x_0, 1 = frase
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)                # tutti vedono tutti (punto 4)
        self.out = nn.Linear(d, 1)                                       # punteggio (punto 5)
        self.pair = nn.ModuleList([nn.Linear(d, pair_rank, bias=False), nn.Linear(d, pair_rank, bias=False)])
        self.pair_tau = nn.Parameter(torch.tensor(0.1))

    def mixed(self, x):                                                  # x: [..., n_mix, D]
        if not self.use_mix:
            return self.ln(x[..., -1, :])                                # solo l'ultimo layer
        return self.ln((self.mix.softmax(0)[:, None] * x).sum(-2))

    def forward(self, q, h_end, h_mean, att, n_tok, sent_mask):
        """q: [B, L, D]; h_end, h_mean: [B, N, L, D]; att: [B, N, n_att]; n_tok, sent_mask: [B, N] (1 = frase vera).
        Restituisce s: [B, N], s0: [B], S: [B, N, N] o None."""
        h = self.mixed(h_end)
        if self.use_mean:
            h = torch.cat([h, self.mixed(h_mean)], -1)
        x_h = self.proj_h(h) + self.kind.weight[1]
        if self.use_pos:
            x_h = x_h + self.pos(torch.arange(h.size(1), device=h.device))
        if self.use_att:
            x_h = x_h + self.proj_a(torch.cat([att, n_tok.clamp_min(1).float().log()[..., None]], -1))
        x_0 = (self.proj_q(self.mixed(q)) + self.kind.weight[0])[:, None]
        pad = torch.cat([torch.zeros_like(sent_mask[:, :1]), 1 - sent_mask], 1).bool()
        X = self.encoder(torch.cat([x_0, x_h], 1), src_key_padding_mask=pad)
        o = self.out(X)[..., 0]
        s, s0, S = o[:, 1:], o[:, 0], None
        if self.use_pair:
            Xs = X[:, 1:]
            S = self.pair[0](Xs) @ self.pair[1](Xs).transpose(1, 2)    # punteggio di ogni coppia
            invalid = ~sent_mask.bool()[:, None, :] | torch.eye(S.size(1), dtype=torch.bool, device=S.device)
            support = (S + s.detach()[:, None, :]).masked_fill(invalid, float("-inf")).amax(-1)
            s = s + self.pair_tau * support.nan_to_num(neginf=0.0)      # nessun partner: nessun contributo
        return s, s0, S
```

Circa 2–4 milioni di parametri. Righe con meno frasi si riempiono fino alla più lunga del batch (`sent_mask = 0` sulle frasi di riempimento).

**Configurazione di partenza (H0)**: solo l'ultimo layer, solo l'ultimo token, niente attenzione, niente coppie; loss del §2; soglia fissa. Da qui gli esperimenti aggiungono un pezzo alla volta.

## 2. Loss di partenza (H0): per frase, con soglia fissa

Ogni frase è trattata come un sì/no indipendente ("è evidenza?"), con due correzioni:
- le frasi di evidenza sono poche (2 su 50, per esempio): senza correzione la testa imparerebbe a dire sempre "no". Si pesano di più (`alpha`, ~10);
- un termine di **ordinamento**: ogni frase di evidenza deve avere un punteggio più alto, di un margine, della migliore frase non di evidenza.

In uso si mostrano le frasi con probabilità sopra una soglia `τ_e`, scelta sul dev.

```python
import torch.nn.functional as F

def ref_loss(s, gold, sent_mask, omega, alpha=10.0, gamma=0.5, margin=1.0):
    """s, gold, sent_mask, omega: [B, N] (omega = peso delle frasi non di evidenza). Restituisce [B]."""
    bce = F.binary_cross_entropy_with_logits(s, gold.float(), reduction="none")
    w = torch.where(gold > 0, torch.full_like(s, alpha), omega) * sent_mask
    l_bce = (bce * w).sum(1) / sent_mask.sum(1)
    best_neg = s.masked_fill((gold > 0) | (sent_mask == 0), float("-inf")).amax(1, keepdim=True)
    rank = F.softplus(margin - s + best_neg).nan_to_num(posinf=0.0) * (gold > 0)
    l_rank = rank.sum(1) / (gold > 0).sum(1).clamp(min=1)
    return l_bce + gamma * l_rank
```

## 3. Loss con soglia per riga (quella da provare in H-LOSS)

Qui la regola è: **una frase è evidenza se il suo punteggio supera quello della soglia `s_0` della stessa riga.** La loss spinge:
- le frasi di evidenza **sopra** `s_0`, le altre **sotto** (righe `supervised`);
- **tutte** le frasi sotto `s_0` nei claim non supportati (righe `mil_neg`): così la testa impara che una frase che somiglia al claim ma non lo dimostra non va mostrata;
- almeno una frase sopra `s_0`, ma poche, nei positivi senza evidenza nota (righe `mil_pos`, solo nell'esperimento H-MIL).

Vantaggi: non c'è una soglia fissa da tarare, e il numero di frasi mostrate si adatta al caso (con una catena di 3 frasi la soglia può scendere per farle passare tutte).

In formula, con G = frasi di evidenza e ω = peso delle frasi non di evidenza:

```text
supervised:  log(1 + Σ_{i ∉ G} ω_i · e^{s_i − s_0})  +  log(1 + Σ_{j ∈ G} e^{s_0 − s_j})
mil_neg:     μ_neg · log(1 + Σ_i e^{s_i − s_0})                                  μ_neg = 0,3
mil_pos:     μ_pos · [ log(1 + e^{s_0 − LSE_j s_j}) + ρ · media_i σ(s_i − s_0) ]   μ_pos = 0,3, ρ = 0,1
```

Il primo termine cresce se qualche frase non di evidenza sta sopra la soglia, il secondo se qualche frase di evidenza sta sotto.

```python
def zlpr_loss(s, s0, gold, sent_mask, omega, term, mu_neg=0.3, mu_pos=0.3, rho=0.1):
    """term: [B] con 0 = supervised, 1 = mil_neg, 2 = mil_pos, 3 = nessun termine. Restituisce [B]."""
    ninf = torch.finfo(s.dtype).min
    real = sent_mask.bool()
    d = s - s0[:, None]
    is_neg = (gold == 0) & real
    l_neg = F.softplus(torch.logsumexp((d + omega.clamp_min(1e-6).log()).masked_fill(~is_neg, ninf), 1))
    l_pos = F.softplus(torch.logsumexp((-d).masked_fill(~(gold > 0), ninf), 1)) * (gold.sum(1) > 0)
    supervised = l_neg + l_pos
    lse = torch.logsumexp(d.masked_fill(~real, ninf), 1)
    mil_neg = mu_neg * F.softplus(lse)
    spars = (torch.sigmoid(d) * sent_mask).sum(1) / sent_mask.sum(1)
    mil_pos = mu_pos * (F.softplus(-lse) + rho * spars)
    out = torch.stack([supervised, mil_neg, mil_pos, torch.zeros_like(lse)], 1)
    return out.gather(1, term[:, None]).squeeze(1)
```

La loss di ogni riga si moltiplica per il suo peso (`evidence_weight`) e si fa la media sul batch.

### Esempio numerico (anche come test automatico)

40 frasi, evidenza = frasi 1 e 40, soglia `s_0 = 0`, ω = 1:

| | frase 1 | frase 40 | una frase che somiglia al claim | altre 37 | Loss |
|---|---:|---:|---:|---:|---:|
| prima | 2,0 | −0,5 (sotto soglia: verrebbe persa) | 0,5 (sopra soglia: verrebbe mostrata) | −3 | **2,526** |
| dopo | 2,0 | 1,5 | −1,0 | −3 | **1,473** |

La loss scende quando la frase 40 sale sopra la soglia e la frase che somiglia scende sotto.

Test per la loss di partenza: 40 frasi, evidenza {1, 40}, α = 10, tutte con probabilità 0,02 → parte per frase = **1,975**.

Il codice di questo file è stato eseguito su dati finti: forme giuste, gradienti, caso con una sola frase, e i valori sopra tornano.

## 4. Termine aggiuntivo per le coppie (solo nell'esperimento H-PAIR)

**Coppie (H-PAIR).** Nelle righe con 2 o più frasi di evidenza, il punteggio di coppia deve essere alto per le coppie di frasi di evidenza e basso fra una frase di evidenza e le 5 frasi non di evidenza più "simili". Peso 0,2.

```python
def pair_loss(S, gold, sent_mask, n_hard=5):
    g = gold.bool() & sent_mask.bool()
    pos_pairs = g[:, :, None] & g[:, None, :] & ~torch.eye(S.size(1), dtype=torch.bool, device=S.device)
    l_pos = F.binary_cross_entropy_with_logits(S[pos_pairs], torch.ones_like(S[pos_pairs]))
    neg_cand = S.masked_fill(~(g[:, :, None] & (~g & sent_mask.bool())[:, None, :]), float("-inf"))
    hard = neg_cand.topk(min(n_hard, S.size(1)), -1).values
    hard = hard[torch.isfinite(hard)]
    l_neg = F.binary_cross_entropy_with_logits(hard, torch.zeros_like(hard))
    return l_pos + l_neg
```

## 5. Test

`tests/test_evidence_losses.py`: i valori del §3 (tolleranza 0,001); con tutte le frasi molto sotto soglia `mil_neg` ≈ 0; una riga senza termine dà loss e gradiente zero.
`tests/test_evidence_head.py`: forme, frasi di riempimento ignorate, riga con una sola frase, con e senza coppie.
