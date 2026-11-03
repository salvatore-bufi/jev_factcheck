# Passo 8 — La testa e la loss

## Obiettivo

Scrivere e testare `evidence/head.py` (la testa) ed `evidence/losses.py` (le loss). La testa ha un flag per ogni componente, così tutti gli esperimenti di [11](11_esperimenti.md) usano lo stesso codice.

## 1. La testa

```python
import torch
import torch.nn as nn

class EvidenceHead(nn.Module):
    def __init__(self, d_model, n_mix, n_att, d=256, layers=2, heads=4, max_sent=512,
                 use_mix=True, use_mean=True, use_att=True, use_pos=True, use_pair=False, pair_rank=64):
        super().__init__()
        self.use_mix, self.use_mean, self.use_att, self.use_pos, self.use_pair = use_mix, use_mean, use_att, use_pos, use_pair
        self.mix = nn.Parameter(torch.zeros(n_mix))                     # pesi dei layer (softmax); ignorati se use_mix=False
        self.ln = nn.LayerNorm(d_model)                                  # i layer hanno scale diverse
        self.proj_h = nn.Linear(d_model * (2 if use_mean else 1), d)     # [fine frase ; media frase]
        self.proj_q = nn.Linear(d_model, d)
        self.proj_a = nn.Linear(n_att + 1, d)                            # attenzione 8×H + log(numero di token)
        self.pos = nn.Embedding(max_sent, d)
        self.kind = nn.Embedding(2, d)                                   # 0 = x_0, 1 = frase
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)                # attenzione piena, non causale
        self.out = nn.Linear(d, 1)                                       # stessa uscita per le frasi e per x_0
        self.pair = nn.ModuleList([nn.Linear(d, pair_rank, bias=False), nn.Linear(d, pair_rank, bias=False)])
        self.pair_tau = nn.Parameter(torch.tensor(0.1))

    def mixed(self, x):                                                  # x: [..., n_mix, D]
        if not self.use_mix:
            return self.ln(x[..., -1, :])                                # solo l'ultimo layer del mix
        return self.ln((self.mix.softmax(0)[:, None] * x).sum(-2))

    def forward(self, q, h_end, h_mean, att, n_tok, sent_mask):
        """q: [B, L, D]; h_end, h_mean: [B, N, L, D]; att: [B, N, n_att]; n_tok, sent_mask: [B, N].
        Restituisce s: [B, N] (logit delle frasi), s0: [B] (soglia della riga), S: [B, N, N] o None."""
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
            S = self.pair[0](Xs) @ self.pair[1](Xs).transpose(1, 2)    # [B, N, N]
            invalid = ~sent_mask.bool()[:, None, :] | torch.eye(S.size(1), dtype=torch.bool, device=S.device)
            support = (S + s.detach()[:, None, :]).masked_fill(invalid, float("-inf")).amax(-1)
            s = s + self.pair_tau * support.nan_to_num(neginf=0.0)      # nessun partner: nessun contributo
        return s, s0, S
```

| Parametro | Default | Note |
|---|---|---|
| `n_mix` | 4 | layer del mix ([01](01_ambiente_e_modelli.md) §2) |
| `n_att` | 8 × H (d1: 256; JPT e Winnow: 128) | |
| `d`, `layers`, `heads` | 256, 2, 4 | |
| Parametri totali | ~2–4M | dominati da `proj_h` (2 · D · 256) |

Configurazione H0 (il riferimento di partenza): `use_mix=False, use_mean=False, use_att=False, use_pair=False`, loss di riferimento (§2), soglia fissa.

Ogni frase reale ha `sent_mask = 1`; le righe del batch si riempiono fino al massimo `n` del batch.

## 2. Loss di riferimento (H0)

Solo righe `supervised`; regola di produzione: `σ(s_k) > τ_e`, con `τ_e` scelta sul dev.

```python
import torch.nn.functional as F

def ref_loss(s, gold, sent_mask, omega, alpha=10.0, gamma=0.5, margin=1.0):
    """BCE pesata + ranking. s, gold, sent_mask, omega: [B, N]. Restituisce [B]."""
    bce = F.binary_cross_entropy_with_logits(s, gold.float(), reduction="none")
    w = torch.where(gold > 0, torch.full_like(s, alpha), omega) * sent_mask
    l_bce = (bce * w).sum(1) / sent_mask.sum(1)
    best_neg = s.masked_fill((gold > 0) | (sent_mask == 0), float("-inf")).amax(1, keepdim=True)
    rank = F.softplus(margin - s + best_neg).nan_to_num(posinf=0.0) * (gold > 0)
    l_rank = rank.sum(1) / (gold > 0).sum(1).clamp(min=1)
    return l_bce + gamma * l_rank
```

`alpha` = rapporto fra frasi non evidenza ed evidenza sul mix, al massimo 20.

## 3. Loss con soglia della riga (ZLPR + MIL)

Una frase è evidenza se e solo se `s_k > s_0`. Un termine per tipo di riga (`evidence_term`, [04](04_split_e_mix.md) §3):

```text
supervised:  log(1 + Σ_{i ∉ G} ω_i · e^{s_i − s_0})  +  log(1 + Σ_{j ∈ G} e^{s_0 − s_j})
mil_neg:     μ_neg · log(1 + Σ_i e^{s_i − s_0})
mil_pos:     μ_pos · [ log(1 + e^{s_0 − LSE_j s_j})  +  ρ · (1/n) Σ_i σ(s_i − s_0) ]
none:        0
```

```python
def zlpr_loss(s, s0, gold, sent_mask, omega, term, mu_neg=0.3, mu_pos=0.3, rho=0.1):
    """term: [B] con 0 = supervised, 1 = mil_neg, 2 = mil_pos, 3 = none. Restituisce [B]."""
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

`softplus(logsumexp(x)) = log(1 + Σ e^x)`. La loss della riga si moltiplica per `evidence_weight` e si fa la media sul batch.

### Test numerico (da mettere in `tests/`)

Riga `supervised` con 40 frasi, G = {S1, S40}, ω = 1, `s_0` = 0:

| | s_1 | s_40 | una non gold | altre 37 | Termine negativi | Termine positivi | Totale |
|---|---:|---:|---:|---:|---:|---:|---:|
| caso 1 | 2,0 | −0,5 | 0,5 | −3 | ln(1 + 1,649 + 1,842) = 1,502 | ln(1 + 0,135 + 1,649) = 1,024 | **2,526** |
| caso 2 | 2,0 | 1,5 | −1,0 | −3 | ln(1 + 0,368 + 1,842) = 1,166 | ln(1 + 0,135 + 0,223) = 0,306 | **1,473** |

Test per la loss di riferimento: 40 frasi, evidenza {1, 40}, α = 10, ω = 1, tutte le frasi a σ(s) = 0,02 → `l_bce` = (10 · 2 · 3,912 + 38 · 0,0202)/40 = **1,975**.

Altri test:
- `mil_neg` con tutte le frasi a −10 rispetto a `s_0`: ≈ 0;
- riga `none`: esattamente 0, e nessun gradiente verso la testa;
- `supervised` con ω = 0 per una frase non gold molto alta: quella frase non cambia la loss.

## 4. Termini ausiliari (solo negli esperimenti che li usano)

### Coppie (H-PAIR)

Per le righe `supervised` con |G| ≥ 2, sulle `S` restituite dalla testa:

```python
def pair_loss(S, gold, sent_mask, n_hard=5):
    """Le coppie gold × gold devono avere S alto; per ogni gold, le n_hard non gold con S più alto devono averlo basso."""
    g = gold.bool() & sent_mask.bool()
    pos_pairs = g[:, :, None] & g[:, None, :] & ~torch.eye(S.size(1), dtype=torch.bool, device=S.device)
    l_pos = F.binary_cross_entropy_with_logits(S[pos_pairs], torch.ones_like(S[pos_pairs]))
    neg_cand = S.masked_fill(~(g[:, :, None] & (~g & sent_mask.bool())[:, None, :]), float("-inf"))
    hard = neg_cand.topk(min(n_hard, S.size(1)), -1).values
    hard = hard[torch.isfinite(hard)]
    l_neg = F.binary_cross_entropy_with_logits(hard, torch.zeros_like(hard))
    return l_pos + l_neg
```

Peso 0,2 rispetto alla loss principale.

### Fedeltà (H-FAITH)

Definita sui margini del verdetto calcolati con la maschera per frase ([09](09_training.md) §5), con `p_k = σ(s_k − s_0)`:

```text
L_faith = KL( σ(z_full) ‖ σ(z_suff) )  +  softplus( z_comp − z_full + δ )        δ = 1
```

- `z_full`: margine del verdetto sul documento intero;
- `z_suff`: margine con bias `log(p_k + 1e-4)` sulle chiavi della frase k (restano "visibili" solo le frasi scelte);
- `z_comp`: margine con bias `log(1 − p_k + 1e-4)` (vengono "tolte" le frasi scelte).

Si usa solo sulle righe positive giudicate supportate, sempre **insieme** a `zlpr_loss` (o con `mil_pos` e la sparsità): da sola ha una soluzione banale (`p_k = 1` per tutte). Peso 0,1–0,3.

## Prodotto

- `evidence/head.py`, `evidence/losses.py`.
- `tests/test_evidence_losses.py` con i valori di §3 (tolleranza 1e-3) e i test di §3; `tests/test_evidence_head.py`: forme, maschere, nessun gradiente da righe `none`, con `use_pair` e senza.
