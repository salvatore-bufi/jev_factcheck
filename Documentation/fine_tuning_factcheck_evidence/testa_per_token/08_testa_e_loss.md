# Passo 8 — L'architettura della testa, la loss, la decodifica

## In breve

Questo passo descrive per intero la rete da implementare e addestrare, la *testa di evidenza per token*, e come si trasformano i suoi punteggi in tratti di testo. Tre file:
- `evidence/token_head.py`: la testa;
- `evidence/token_losses.py`: la loss;
- `evidence/decode.py`: dai punteggi ai tratti (la stessa funzione in valutazione e in produzione).

Tutto il codice di questo file è stato eseguito su dati finti: forme dei tensori, gradienti, righe di lunghezza diversa nello stesso batch, valori numerici, decodifica e un ciclo di training completo tornano. I numeri di parametri e tempi sono misurati, non stimati.

## 1. La rete in una figura

Notazione: B = righe nel batch, N = token del documento (riempiti fino al più lungo del batch), L = layer letti (2), D = dimensione del modello (2.560 per KnowLine-4B, 4.096 per elephant-1), d = 256, K = blocchi = ⌈N / 16⌉.

```text
 INGRESSI (dal forward del decision model)
   h    [B, N, L, 256]   vettori dei token del documento, layer 23 e 31, già ridotti con la PCA
   att  [B, N, 128]      attenzione della posizione di risposta sul token (8 layer pieni × 16 teste)
   q    [B, L, D]        vettore della posizione di risposta, layer 23 e 31
   mask [B, N]           1 = token vero, 0 = riempimento
        │
 A. MISCELA DEI LAYER    h̄[t] = LN( Σ_l softmax(α)_l · h[t, l] )          [B, N, 256]
                         q̄    = LN( Σ_l softmax(α)_l · q[l] )             [B, D]
        │
 B. VETTORE DEL TOKEN    x[t] = W_h h̄[t] + W_a att[t] + W_q q̄            [B, N, 256]
        │
 C. CONTESTO LOCALE      3 blocchi residui:  x ← x + GELU( Conv1d_k5,dil(LN(x)) ),  dil = 1, 2, 4
                         ogni token vede 29 token intorno                    [B, N, 256]
        │
 D. BLOCCHI              b[k] = media dei token k·16 … k·16+15 (solo quelli veri)   [B, K, 256]
                         + posizione del blocco P[k] + tipo "blocco"
        │
 E. CONTESTO GLOBALE     sequenza [x0, b[1], …, b[K]],  x0 = W_q q̄ + tipo "risposta"
                         transformer NON causale, 2 layer, 4 teste, FFN 1.024, pre-norm
                         → X0 [B, 256],  X_b [B, K, 256]: ogni blocco ha visto tutto il documento e q
        │
 F. RITORNO AI TOKEN     z[t] = GELU( W_r [ x[t] ; X_b[blocco di t] ] )      [B, N, 256]
        │
 G. PUNTEGGI             s[t] = w_o · z[t] + b_o        (−10.000 sui token di riempimento)    [B, N]
                         s0   = w_o · X0  + b_o         soglia del documento                   [B]
```

## 2. Gli ingressi

| Tensore | KnowLine-4B | elephant-1 | Da dove |
|---|---|---|---|
| `h` | [B, N, 2, 256] | [B, N, 2, 256] | hook sui layer 23 e 31, proiezione PCA fissa ([01](01_ambiente_e_modelli.md) §5–6) |
| `att` | [B, N, 128] | [B, N, 128] | riga di attenzione della posizione di risposta, 8 layer × 16 teste ([01](01_ambiente_e_modelli.md) §7) |
| `q` | [B, 2, 2.560] | [B, 2, 4.096] | stato della posizione di risposta, layer 23 e 31 |
| `mask` | [B, N] | [B, N] | 1 sui token del documento |

N tipico: ~1.500 (HoVer), ~600 (SciFact), da poche centinaia a ~18.000 (WiCE).

## 3. Strato per strato

### A. Miscela dei layer
- Due pesi appresi α (partono da 0, quindi 50/50); `softmax(α)` li rende positivi a somma 1. La stessa miscela vale per `h` e per `q`.
- Una LayerNorm per ciascuno: i layer hanno scale molto diverse (l'ultimo ha valori più grandi).
- Perché: l'ultimo layer è specializzato a predire la risposta; il layer 23 rappresenta meglio il contenuto. Dopo il training, i pesi α dicono quale serve di più (se uno va a ~1, si può tenere solo quello).

### B. Vettore del token
- `W_h` (256 → 256): il contenuto del token.
- `W_a` (128 → 256): dove il modello ha guardato per decidere. Durante il training le 128 feature di un token si azzerano con probabilità 0,1 (dropout per token), così la testa non dipende da una sola testa di attenzione.
- `W_q` (D → 256): il claim e il verdetto, uguali per tutti i token della riga. Sommandolo, ogni token è "letto alla luce" del claim anche nei layer della testa.
- I token di riempimento si azzerano.

### C. Contesto locale
- 3 convoluzioni 1D lungo i token, finestra 5, dilatazione 1, 2, 4: campo visivo di 1 + 4·(1 + 2 + 4) = 29 token.
- Ognuna dentro un blocco residuo pre-norm: `x ← x + GELU(Conv(LN(x)))`; i riempimenti si riazzerano dopo ogni blocco.
- Perché: la stessa parola è evidenza in una frase e non in un'altra; serve vedere le parole vicine. Le convoluzioni sono parallele (veloci su GPU) e non dipendono da confini di frase.

### D. Blocchi
- I token si raggruppano in blocchi **fissi** di 16, per posizione (token 0–15, 16–31, …). Il vettore del blocco è la media dei suoi token veri. Un documento di 1.500 token dà 94 blocchi; uno di 18.000, 1.125.
- Ogni blocco riceve un vettore di posizione appreso `P[k]` (fino a 4.096 blocchi, cioè 65.536 token) e un vettore di "tipo".
- Perché: per far comunicare parti lontane serve un'attenzione su tutto il documento. Su migliaia di token costa troppo; su ~100–1.000 blocchi è trascurabile. I blocchi non tagliano il risultato: l'uscita resta per token.

### E. Contesto globale
- Sequenza `[x0, b[1], …, b[K]]`: `x0` rappresenta la posizione di risposta (W_q q̄ + tipo "risposta").
- Transformer encoder **non causale**: ogni elemento vede tutti gli altri, in avanti e all'indietro. 2 layer, d = 256, 4 teste, feed-forward 1.024, pre-norm, dropout 0,1. Maschera di riempimento sui blocchi vuoti.
- Uscita: `X0` (la posizione di risposta, dopo aver visto tutti i blocchi) e `X_b` (ogni blocco, dopo aver visto tutti gli altri e `q`).
- Perché: è qui che "frase 1" e "frase 40" si vedono. `x0` porta il punto di vista del verdetto: sa quali parti hanno pesato.

### F–G. Ritorno ai token e punteggi
- Ogni token concatena il suo vettore locale `x[t]` con il vettore globale del suo blocco `X_b[⌊t/16⌋]` → `W_r` (512 → 256) → GELU.
- Un'unica uscita lineare `w_o` (256 → 1) dà `s[t]` per ogni token e, applicata a `X0`, la soglia `s0`. Stessa uscita per entrambi: punteggi e soglia sono sulla stessa scala, e `s[t] − s0` ha senso.
- Un token è evidenza se `s[t] > s0`.

## 4. Parametri e costo (misurati)

| Componente | Parametri KnowLine-4B | Parametri elephant-1 |
|---|---:|---:|
| A. miscela e normalizzazioni | 5.634 | 8.706 |
| B. `W_h` | 65.792 | 65.792 |
| B. `W_a` | 33.024 | 33.024 |
| B. `W_q` | 655.616 | 1.048.832 |
| C. convoluzioni (+ norme) | 985.344 | 985.344 |
| D. posizione dei blocchi e tipi | 1.049.088 | 1.049.088 |
| E. transformer | 1.579.520 | 1.579.520 |
| F–G. ritorno e uscita | 131.585 | 131.585 |
| **Totale** | **4.505.603** (18 MB in fp32) | **4.901.891** (19,6 MB) |

Costo per una riga di 2.000 token: convoluzioni ~4 GFLOP, transformer sui 125 blocchi ~0,2 GFLOP; la proiezione PCA negli hook ~5 GFLOP. Il forward del modello su quella riga (4B, letta due volte) costa ~30 TFLOP: la testa aggiunge meno dello 0,05% dei calcoli.

Tempo misurato della testa (GPU locale RTX 5090 Laptop, fp32, una riga da 6.000 token): **3,2 ms**.

## 5. Il codice della testa

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenEvidenceHead(nn.Module):
    def __init__(self, d_q, n_layers=2, d_in=256, n_att=128, d=256, block=16, kernel=5, conv_layers=3,
                 enc_layers=2, heads=4, max_blocks=4096, use_att=True, use_local=True, use_global=True, att_dropout=0.1):
        super().__init__()
        self.block, self.use_att, self.use_local, self.use_global = block, use_att, use_local, use_global
        self.att_dropout = att_dropout                                   # in training: feature di attenzione di un token azzerate
        self.mix = nn.Parameter(torch.zeros(n_layers))                 # pesi dei layer (softmax)
        self.ln_h, self.ln_q = nn.LayerNorm(d_in), nn.LayerNorm(d_q)
        self.proj_h, self.proj_a, self.proj_q = nn.Linear(d_in, d), nn.Linear(n_att, d), nn.Linear(d_q, d)
        self.convs = nn.ModuleList([nn.Conv1d(d, d, kernel, padding=2 ** i * (kernel // 2), dilation=2 ** i)
                                    for i in range(conv_layers)])     # contesto locale: ~30 token
        self.conv_norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(conv_layers)])
        self.pos, self.kind = nn.Embedding(max_blocks, d), nn.Embedding(2, d)
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, enc_layers)          # contesto globale, sui blocchi
        self.back = nn.Linear(2 * d, d)
        self.out = nn.Linear(d, 1)

    def mixed(self, x, ln):                                            # x: [..., L, D]
        return ln((self.mix.softmax(0)[:, None] * x).sum(-2))

    def forward(self, h, att, q, mask):
        """h: [B, N, L, 256]; att: [B, N, 128]; q: [B, L, Dq]; mask: [B, N] (1 = token del documento)
        -> s: [B, N] punteggi dei token, s0: [B] soglia del documento."""
        m = mask.unsqueeze(-1).float()
        qv = self.proj_q(self.mixed(q, self.ln_q))                    # [B, d]
        x = self.proj_h(self.mixed(h, self.ln_h)) + qv[:, None]
        if self.use_att:
            att = att.float()
            if self.training and self.att_dropout:                         # intere, per token: non dipendere da una sola testa
                att = att * (torch.rand_like(att[..., :1]) >= self.att_dropout)
            x = x + self.proj_a(att)
        x = x * m
        if self.use_local:
            for conv, norm in zip(self.convs, self.conv_norms):       # convoluzioni residue, padding a zero
                x = x + F.gelu(conv(norm(x).transpose(1, 2)).transpose(1, 2)) * m
        B, N, d = x.shape
        nb = -(-N // self.block)
        xb = F.pad(x, (0, 0, 0, nb * self.block - N)).view(B, nb, self.block, d)
        mb = F.pad(m, (0, 0, 0, nb * self.block - N)).view(B, nb, self.block, 1)
        blocks = (xb * mb).sum(2) / mb.sum(2).clamp_min(1)              # media dei token di ogni blocco
        block_mask = mb.sum(2).squeeze(-1) > 0                           # [B, nb]
        x0 = qv + self.kind.weight[0]
        if self.use_global:
            seq = torch.cat([x0[:, None], blocks + self.pos(torch.arange(nb, device=x.device)) + self.kind.weight[1]], 1)
            pad = torch.cat([torch.zeros_like(block_mask[:, :1]), ~block_mask], 1)
            X = self.encoder(seq, src_key_padding_mask=pad)
            X0, Xb = X[:, 0], X[:, 1:]
        else:
            X0, Xb = x0, blocks
        ctx = Xb.repeat_interleave(self.block, dim=1)[:, :N]            # ogni token riceve il contesto del suo blocco
        z = F.gelu(self.back(torch.cat([x, ctx], -1)))
        s = self.out(z).squeeze(-1).masked_fill(mask == 0, -1e4)
        s0 = self.out(X0).squeeze(-1)
        return s, s0
```

Configurazione di partenza dell'esperimento **H0**: un solo layer (31), `use_att=False, use_local=False, use_global=False`, cioè un punteggio per token dal suo vettore e da `q`. Gli esperimenti ([11](11_esperimenti.md)) accendono un pezzo alla volta.

## 6. La loss

### 6.1 Il problema delle etichette

Le fonti annotano **frasi**, non parole: in una frase gold tutti i token hanno etichetta 1, anche "is", "the" o una data accessoria. Se si chiedesse alla testa di accendere ogni token gold, imparerebbe a evidenziare sempre frasi intere, e una testa che evidenzia solo la parte utile verrebbe penalizzata.

Per questo la loss ha quattro termini con ruoli diversi:
- **`L_tok`**, per token, con peso moderato: dà la direzione generale;
- **`L_hit`**: ogni frase gold dev'essere **colpita**, cioè almeno un suo token chiaramente sopra soglia. È ciò che conta per l'utente (e per l'insieme completo nei casi multi-hop);
- **`L_out`**: **fuori** dalle frasi gold nessun token deve superare la soglia. Pesato con ω, perché le annotazioni possono essere incomplete (una frase valida non segnata);
- **`L_tv`**: token vicini con punteggi simili, per tratti contigui invece che a macchia di leopardo.

### 6.2 Le formule

Con `d[t] = s[t] − s0` (positivo = sopra soglia), per una riga:

```text
supervised:  L = L_tok + 1,0 · L_hit + 0,5 · ω · L_out + 0,1 · L_tv
  L_tok = Σ_t w_t · BCE(σ(d[t]), y[t]) / Σ_t w_t          w_t = α = 3 sui token gold, ω sugli altri
  L_hit = media sulle frasi gold g di softplus(1 − max_{t ∈ g} d[t])
  L_out = softplus(1 + max_{t ∉ gold} d[t])
  L_tv  = media_t |σ(d[t+1]) − σ(d[t])|
mil_neg:     L = 0,3 · softplus(1 + max_t d[t])               claim non supportato: nessun token sopra soglia
none:        L = 0                                            nessuna loss di evidenza
```

Il margine 1 vuol dire "chiaramente sopra / sotto", non appena sopra. Le righe hanno lunghezze diverse: la loss si calcola **riga per riga** e si fa la media pesata con il peso della riga (`row_weight`, [04](04_split_e_mix.md) §3).

### 6.3 Il codice

```python
import torch
import torch.nn.functional as F


def token_evidence_loss(s, s0, y, mask, gold_spans, term, omega, alpha=3.0, margin=1.0,
                        b_hit=1.0, b_out=0.5, b_tv=0.1, mu_neg=0.3):
    """Una riga alla volta (i documenti hanno lunghezze diverse).
    s: [N] punteggi dei token; s0: [] soglia; y: [N] etichette 0/1; mask: [N]; gold_spans: [(a, b)] token, estremi inclusi;
    term: "supervised" | "mil_neg" | "none"; omega: peso dei token fuori dalle frasi gold."""
    d = (s - s0)[mask.bool()]
    y = y[mask.bool()].float()
    if term == "none":
        return s.sum() * 0.0
    if term == "mil_neg":                                              # nessun token sopra la soglia
        return mu_neg * F.softplus(d.max() + margin)
    w = torch.where(y > 0, torch.full_like(d, alpha), torch.full_like(d, omega))
    l_tok = (F.binary_cross_entropy_with_logits(d, y, reduction="none") * w).sum() / w.sum()
    l_hit = torch.stack([F.softplus(margin - d[a:b + 1].max()) for a, b in gold_spans]).mean()  # ogni frase gold colpita
    outside = d[y == 0]
    l_out = F.softplus(outside.max() + margin) if outside.numel() else d.sum() * 0.0      # niente acceso fuori
    p = torch.sigmoid(d)
    l_tv = (p[1:] - p[:-1]).abs().mean() if p.numel() > 1 else d.sum() * 0.0            # tratti contigui
    return l_tok + b_hit * l_hit + b_out * omega * l_out + b_tv * l_tv
```

### 6.4 Esempio numerico (anche test automatico)

10 token, frase gold sui token 2–5, `s0 = 0`, ω = 0,5, punteggi `−3; −3; 2; 0,5; −0,5; 2; −3; −3; −3; 1,5`. La frase gold è colpita (il token 2 vale 2), ma il token 9, fuori, è acceso (1,5) e il token 4, dentro, è sotto soglia. Loss: **1,4015**. La stessa riga come negativo (`mil_neg`): **0,9146**, perché il token più alto (2) sta sopra soglia.

## 7. La decodifica: dai punteggi ai tratti

1. token con `s[t] − s0 > δ` (δ = 0 di default, scelto sul dev, [09](09_training.md) §2);
2. tratti di token consecutivi; si uniscono tratti separati da al massimo 3 token;
3. si tolgono i tratti sotto 4 token (parole isolate);
4. si tengono i 5 tratti con punteggio medio più alto, in ordine di posizione;
5. se il verdetto è "supportato" e non resta nulla, si mostra il tratto intorno al token migliore;
6. ogni tratto diventa `[inizio, fine)` di caratteri con `tok_char` ([06](06_rendering_ed_estrazione.md) §1), allargato alla parola intera; il punteggio del tratto è σ(media di `s − s0`).

```python
import math


def decode(s, s0, tok_char, text, delta=0.0, max_gap=3, min_len=4, max_spans=5, force_one=True):
    """s: punteggi dei token del documento (lista); s0: soglia; tok_char: [(inizio, fine)] in caratteri del documento.
    -> lista di {"start", "end", "score"} sul testo originale, in ordine di posizione."""
    on = [x - s0 > delta for x in s]
    runs, i, n = [], 0, len(s)
    while i < n:                                            # tratti consecutivi sopra soglia
        if on[i]:
            j = i
            while j + 1 < n and on[j + 1]:
                j += 1
            runs.append([i, j]); i = j + 1
        else:
            i += 1
    merged = []
    for a, b in runs:                                       # si chiudono i buchi piccoli
        if merged and a - merged[-1][1] - 1 <= max_gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    merged = [r for r in merged if r[1] - r[0] + 1 >= min_len]
    if not merged and force_one and n:                      # supportato: almeno un tratto, intorno al token migliore
        k = max(range(n), key=lambda t: s[t])
        merged = [[max(0, k - min_len), min(n - 1, k + min_len)]]
    score = lambda r: sum(s[r[0]:r[1] + 1]) / (r[1] - r[0] + 1) - s0
    merged = sorted(sorted(merged, key=score, reverse=True)[:max_spans])
    out = []
    for a, b in merged:
        start, end = tok_char[a][0], tok_char[b][1]
        while start > 0 and not text[start - 1].isspace():  # si allarga alla parola intera
            start -= 1
        while end < len(text) and text[end].isalnum():
            end += 1
        while start < end and text[start].isspace():
            start += 1
        out.append({"start": start, "end": end, "score": 1 / (1 + math.exp(-score([a, b])))})
    return out
```

Esempio (anche test): testo "Rose Byrne is an Australian actress. She starred in Get Him to the Greek in 2010.", token 0–6 sopra soglia tranne il 3 (un buco di un token), token 12 sopra soglia da solo → un solo tratto: "Rose Byrne is an Australian actress." Il buco è chiuso, il token isolato è tolto.

I parametri (δ, buco massimo, lunghezza minima, numero massimo di tratti) si scelgono sul dev e si scrivono in `decision.json` ([12](12_export_e_servizio.md)).

## 8. Test (`tests/test_token_head.py`)

- forme dei tensori con righe di lunghezza diversa nello stesso batch; i token di riempimento hanno punteggio −10.000; i punteggi di una riga non cambiano se nel batch c'è una riga più lunga (a meno di 1e-4);
- l'esempio numerico della loss (tolleranza 1e-3); `none` dà zero e nessun gradiente;
- l'esempio della decodifica; un documento senza token sopra soglia, con `force_one`, dà un tratto; i tratti non escono dal testo;
- con tutte le opzioni spente la testa è la configurazione H0.
