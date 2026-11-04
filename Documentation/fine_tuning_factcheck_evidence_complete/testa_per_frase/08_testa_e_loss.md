# Passo 8 — L'architettura della testa e la loss

## In breve

Questo passo descrive per intero la rete da implementare e addestrare, la *testa di evidenza per frase*, e la loss con cui si addestra. Due file:
- `evidence/head.py`: la testa. Riceve i vettori delle frasi e della posizione di risposta e restituisce un punteggio per frase (`s`) e la soglia del documento (`s0`);
- `evidence/losses.py`: la loss.

Ogni parte della testa si accende o spegne con un'opzione: tutti gli esperimenti ([11](11_esperimenti.md)) usano lo stesso codice. Tutto il codice di questo file è stato eseguito su dati finti: forme, gradienti, una riga sola o con righe più lunghe nello stesso batch, valori numerici delle loss, un ciclo di training completo. Parametri e tempi sono misurati.

## 1. La rete in una figura

Notazione: B = righe nel batch, n = frasi del documento (riempite fino alla riga con più frasi del batch), L = layer letti (4: 19, 23, 27, 31), D = dimensione del modello (2.560 per KnowLine-4B, 4.096 per elephant-1), d = 256.

```text
 INGRESSI (dal forward del decision model, passo 1)
   q      [B, L, D]     vettore della posizione di risposta, 4 layer
   h_end  [B, n, L, D]  vettore dell'ultimo token di ogni frase, 4 layer
   h_mean [B, n, L, D]  media dei token di ogni frase, 4 layer
   att    [B, n, 128]   attenzione della posizione di risposta sulla frase (8 layer pieni × 16 teste)
   n_tok  [B, n]        numero di token della frase
   mask   [B, n]        1 = frase vera, 0 = riempimento
        │
 A. MISCELA DEI LAYER    m(v) = LN( Σ_l softmax(α)_l · v[l] )        stessi pesi α e stessa LN per h_end, h_mean, q
        │
 B. VETTORE DELLA FRASE  x[k] = W_h [ m(h_end[k]) ; m(h_mean[k]) ]            [B, n, 256]
                              + W_a [ att[k] ; log n_tok[k] ]
                              + P[k]  (posizione della frase, appresa)  + T_frase
        │
    VETTORE DI RISPOSTA  x0   = W_q m(q) + T_risposta                         [B, 256]
        │
 C. COMUNICAZIONE        transformer encoder NON causale, 2 layer, 4 teste, FFN 1.024, pre-norm, dropout 0,1
                         sulla sequenza [x0, x[0], …, x[n−1]]: ogni frase vede tutte le altre e x0
                         → X0 [B, 256],  X [B, n, 256]
        │
 D. PUNTEGGI             s[k] = w_o · X[k] + b_o                      [B, n]
                         s0   = w_o · X0  + b_o                       [B]      soglia del documento
        │
 (facoltativo, H-PAIR)   S[k, j] = (U_1 X[k]) · (U_2 X[j])   rango 64     [B, n, n]
                         s[k] ← s[k] + τ · max_{j ≠ k} ( S[k, j] + s[j] )   (s[j] senza gradiente)
        │
 SELEZIONE               frasi con s[k] > s0 + δ, al massimo 5, almeno 1 (la migliore) se il verdetto è "supportato"
```

## 2. Gli ingressi

| Tensore | KnowLine-4B | elephant-1 | Da dove |
|---|---|---|---|
| `q` | [B, 4, 2.560] | [B, 4, 4.096] | hook sui layer 19, 23, 27, 31, posizione di risposta |
| `h_end` | [B, n, 4, 2.560] | [B, n, 4, 4.096] | hook, ultimo token di ogni frase |
| `h_mean` | [B, n, 4, 2.560] | [B, n, 4, 4.096] | hook, media dei token di ogni frase |
| `att` | [B, n, 128] | [B, n, 128] | righe di attenzione degli 8 layer pieni, sommate per frase |
| `n_tok` | [B, n] | [B, n] | rendering |

n tipico: 9 (SciFact), 46 (HoVer), 68 (WiCE); massimo 512.

## 3. Strato per strato

### A. Miscela dei layer
- Quattro pesi appresi α, che partono da 0 (quindi 25% ciascuno); `softmax(α)` li rende positivi a somma 1. La stessa miscela vale per `h_end`, `h_mean` e `q`, così frasi e posizione di risposta sono rappresentate allo stesso modo.
- Una LayerNorm dopo la miscela: i layer hanno scale molto diverse (l'ultimo ha valori più grandi).
- Perché 4 layer e quali: [15](15_recap_architettura_layer_loss.md) §3. Dopo il training, i pesi α dicono quali layer servono (se uno ha meno del 10%, si può togliere).

### B. Vettore della frase
- `W_h` (2D → 256): ultimo token e media affiancati. L'ultimo token riassume la frase letta fino a lì (in un modello causale vede tutta la frase e ciò che la precede); la media pesa tutti i token allo stesso modo e non dipende dalla punteggiatura finale.
- `W_a` (129 → 256): le 128 feature di attenzione più il logaritmo del numero di token. Una frase lunga raccoglie più attenzione solo perché ha più token: con `log n_tok` la testa può correggere. Durante il training le feature di attenzione si possono azzerare per frase con probabilità 0,1 (dropout), così la testa non dipende da una sola testa di attenzione.
- `P[k]`: vettore appreso della posizione della frase nel documento (0 … 511). Aiuta a trattare titoli e frasi iniziali; l'esperimento H9 controlla che non crei un bias verso inizio o fine.
- `T_frase`, `T_risposta`: vettori di "tipo", per distinguere le frasi da `x0` nella sequenza.

### C. Comunicazione fra le frasi
- Transformer encoder **non causale**: ogni elemento della sequenza vede tutti gli altri, in avanti e all'indietro. 2 layer, d = 256, 4 teste, feed-forward 1.024, pre-norm, dropout 0,1. Le frasi di riempimento sono mascherate.
- È il pezzo che risolve l'**evidenza sparsa**: nel modello causale la frase 1 non vede la frase 40; qui sì. Nell'esempio di Nancy Howell, la frase sul seminario (che da sola non nomina Nancy Howell) può "sapere" che più avanti c'è la frase su Nancy Howell e il seminario.
- `x0` porta il punto di vista del verdetto: la posizione di risposta ha visto tutto il prompt e sa su cosa si è basata la decisione.

### D. Punteggi e soglia
- Un'unica uscita lineare `w_o` (256 → 1) dà `s[k]` per ogni frase e, applicata a `X0`, la soglia `s0`. Stessa uscita per entrambi: punteggi e soglia sono sulla stessa scala, e `s[k] − s0` ha senso.
- Una frase è evidenza se `s[k] > s0`. La soglia è diversa per ogni documento: con un claim che richiede 3 frasi, `X0` (che sa quante parti servono) può abbassarla per farle passare tutte.

### Facoltativo: punteggio per coppie (H-PAIR)
- Per ogni coppia di frasi un punteggio bilineare a rango 64; ogni frase riceve, moltiplicato per τ (appreso, parte da 0,1), il massimo fra "punteggio della coppia + punteggio dell'altra frase" sulle altre frasi. Premia una frase quando c'è un'altra frase con cui "fa coppia".
- Si usa solo se, dopo gli esperimenti principali, l'insieme completo su HoVer resta sotto 0,65.

## 4. Parametri e costo (misurati)

| Componente | KnowLine-4B | elephant-1 |
|---|---:|---:|
| A. miscela e normalizzazione | 5.124 | 8.196 |
| B. `W_h` (2D → 256) | 1.310.976 | 2.097.408 |
| B. `W_a` (129 → 256) | 33.280 | 33.280 |
| B. posizione (512 frasi) e tipi | 131.584 | 131.584 |
| `W_q` (D → 256) | 655.616 | 1.048.832 |
| C. transformer | 1.579.520 | 1.579.520 |
| D. uscita | 257 | 257 |
| **Totale (senza coppie)** | **3.716.357** (15 MB in fp32) | **4.899.077** (19,6 MB) |
| Coppie (solo H-PAIR) | + 32.769 | + 32.769 |

Tempo misurato della testa (GPU locale RTX 5090 Laptop, fp32, una riga): **0,67 ms** con 46 frasi, 0,83 ms con 200, 1,55 ms con 500. Il forward del modello su un documento di ~2.000 token costa decine o centinaia di millisecondi: la testa aggiunge meno dell'1–2%.

## 5. Il codice della testa

```python
import torch
import torch.nn as nn

class EvidenceHead(nn.Module):
    def __init__(self, d_model, n_mix, n_att, d=256, layers=2, heads=4, max_sent=512,
                 use_mix=True, use_mean=True, use_att=True, use_pos=True, use_pair=False, pair_rank=64, att_dropout=0.1):
        super().__init__()
        self.use_mix, self.use_mean, self.use_att, self.use_pos, self.use_pair = use_mix, use_mean, use_att, use_pos, use_pair
        self.att_dropout = att_dropout
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
            att = att.float()
            if self.training and self.att_dropout:                         # intere, per frase: non dipendere da una sola testa
                att = att * (torch.rand_like(att[..., :1]) >= self.att_dropout)
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

Configurazione di partenza dell'esperimento **H0**: `use_mix=False, use_mean=False, use_att=False, use_pair=False` (solo l'ultimo layer, solo l'ultimo token, niente attenzione), loss di riferimento (§6.2), soglia fissa. Gli esperimenti ([11](11_esperimenti.md)) accendono un pezzo alla volta.

## 6. La loss

### 6.1 I tipi di riga

Ogni riga del mix ha un `evidence_term` ([04](04_split_e_mix.md) §3):

| Tipo | Righe | Che cosa chiede la loss |
|---|---|---|
| `supervised` | positivi con evidenza nota | le frasi gold sopra la soglia, le altre sotto |
| `mil_neg` | negativi | **nessuna** frase sopra la soglia: una frase che somiglia al claim ma non lo dimostra non va mostrata |
| `mil_pos` | positivi senza evidenza nota (solo esperimento H-MIL) | almeno una frase sopra la soglia, ma poche |
| `none` | righe senza evidenza utilizzabile (WiCE parzialmente supportati) | niente |

Due pesi per riga: `row_weight` (quanto conta la riga) e ω (quanto pesano le frasi non gold: basso dove le annotazioni sono incomplete).

### 6.2 Loss di riferimento (H0): per frase, con soglia fissa

Ogni frase è un sì/no indipendente ("è evidenza?"), con due correzioni:
- le frasi gold sono poche (2 su 50): senza correzione la testa imparerebbe a dire sempre "no", quindi pesano di più (α, ~10);
- un termine di **ordinamento**: ogni frase gold deve superare di un margine la migliore frase non gold.

In uso si mostrano le frasi con σ(s) > τ_e, una soglia fissa scelta sul dev. Solo righe `supervised`.

```python
import torch
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

Esempio (test): 40 frasi, evidenza {1, 40}, α = 10, tutte con probabilità 0,02: parte per frase = **1,975**, quasi tutta dalle due frasi gold (10 · 2 · 3,91 / 40): la loss spinge su quelle due.

### 6.3 Loss con soglia del documento (ZLPR + MIL)

La regola è: **una frase è evidenza se e solo se `s[k] > s0`**. La loss (ZLPR di Su et al., con la soglia appresa come "classe soglia") spinge:
- in `supervised`: le frasi gold sopra `s0`, tutte le altre sotto;
- in `mil_neg`: tutte le frasi sotto `s0`;
- in `mil_pos`: almeno una frase sopra `s0` (massimo "morbido", LSE), più un termine di sparsità.

```text
supervised:  log(1 + Σ_{i ∉ G} ω_i · e^(s_i − s0))  +  log(1 + Σ_{j ∈ G} e^(s0 − s_j))
mil_neg:     0,3 · log(1 + Σ_i e^(s_i − s0))
mil_pos:     0,3 · [ log(1 + e^(s0 − LSE_j s_j))  +  0,1 · media_i σ(s_i − s0) ]
none:        0
```

Il primo termine di `supervised` cresce se qualche frase non gold sta sopra la soglia, il secondo se qualche frase gold sta sotto. Rispetto alla loss di riferimento: nessuna soglia fissa da tarare, numero di frasi mostrate adattato al documento, e i negativi entrano nel training.

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

Esempio numerico (test): 40 frasi, evidenza = frasi 1 e 40, `s0 = 0`, ω = 1:

| | frase 1 | frase 40 | una frase simile al claim | altre 37 | Loss |
|---|---:|---:|---:|---:|---:|
| prima | 2,0 | −0,5 (sotto soglia: verrebbe persa) | 0,5 (sopra soglia: verrebbe mostrata) | −3 | **2,526** |
| dopo | 2,0 | 1,5 | −1,0 | −3 | **1,473** |

La loss scende quando la frase 40 sale sopra la soglia e la frase simile scende sotto.

Rischio del termine `mil_neg`: `x0` conosce il verdetto, quindi la testa può imparare "se il verdetto è no, alza la soglia". È coerente con l'obiettivo (l'evidenza si mostra solo sui supportati), ma sui positivi incerti (P appena sopra la soglia del verdetto) la testa può spegnersi: si controlla la recall in quella fascia ([09](09_training.md) §3) e si abbassa il peso 0,3 se serve.

### 6.4 Termine per le coppie (solo H-PAIR)

Nelle righe con 2 o più frasi gold, il punteggio di coppia deve essere alto per le coppie di frasi gold e basso fra una frase gold e le 5 frasi non gold più "vicine". Peso 0,2.

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

## 7. La selezione delle frasi

```python
def select(s, s0, delta=0.0, max_shown=5, min_shown=1):
    """Frasi mostrate: s[k] > s0 + delta, al massimo max_shown (le più alte), almeno min_shown (la migliore)."""
    order = sorted(range(len(s)), key=lambda k: s[k], reverse=True)
    chosen = [k for k in order if s[k] > s0 + delta][:max_shown]
    return sorted(chosen or order[:min_shown])
```

δ si sceglie sul dev ([09](09_training.md) §2); con la loss di riferimento la regola è σ(s) > τ_e.

## 8. Test (`tests/test_evidence_head.py`)

- forme dei tensori, frasi di riempimento ignorate, riga con una sola frase, con e senza coppie;
- i punteggi di una riga non cambiano se nel batch c'è una riga con più frasi;
- gli esempi numerici delle loss (tolleranza 1e-3): 1,975; 2,526; 1,473; `mil_neg` con tutte le frasi molto sotto soglia ≈ 0; una riga `none` dà loss e gradiente zero;
- con tutte le opzioni spente la testa è la configurazione H0.
