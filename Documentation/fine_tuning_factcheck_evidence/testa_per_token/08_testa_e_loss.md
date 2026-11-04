# Passo 8 — La testa, la loss, la decodifica

## In breve

Tre file:
- `evidence/token_head.py`: la testa. Riceve i vettori dei token del documento e restituisce un punteggio per token (`s`) e la soglia del documento (`s0`);
- `evidence/token_losses.py`: la loss, che dice alla testa quanto sbaglia;
- `evidence/decode.py`: trasforma i punteggi in tratti di testo (inizio e fine in caratteri). È la stessa funzione in valutazione e in produzione.

Ogni parte della testa si accende o spegne con un'opzione: gli esperimenti ([11](11_esperimenti.md)) usano lo stesso codice.

Il codice di questo file è stato eseguito su dati finti: forme, gradienti, righe di lunghezza diversa nello stesso batch, valori numerici e decodifica tornano.

## 1. La testa, passo per passo

1. **Vettore del token**: i vettori dei 2–4 layer (già ridotti a 256 dalla PCA, [01](01_ambiente_e_modelli.md) §3) si sommano con pesi appresi; si aggiungono l'attenzione che la posizione di risposta dà al token (128 numeri) e il vettore `q` (uguale per tutti i token: dice alla testa qual è il claim e il verdetto). Tutto proiettato a 256.
2. **Contesto locale**: 3 convoluzioni 1D residue con dilatazione 1, 2, 4 e finestra 5: ogni token vede ~30 token intorno. Serve a capire che una parola è evidenza perché sta in una certa frase.
3. **Contesto globale**: i token si raggruppano in **blocchi fissi di 16** (per posizione, non per significato); ogni blocco è la media dei suoi token. Un transformer non causale a 2 layer lavora su `[q, blocco 1, …, blocco B]`: ogni blocco vede tutti gli altri e `q`. È ciò che permette di trovare evidenza sparsa (frase 1 e frase 40). Con 2.000 token sono 125 blocchi: costo trascurabile.
4. **Ritorno ai token**: ogni token riceve il contesto del suo blocco, lo combina con il proprio vettore e produce il suo punteggio `s[t]`. Dallo stato di `q` dopo il transformer esce la soglia `s0`.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenEvidenceHead(nn.Module):
    def __init__(self, d_q, n_layers=2, d_in=256, n_att=128, d=256, block=16, kernel=5, conv_layers=3,
                 enc_layers=2, heads=4, max_blocks=4096, use_att=True, use_local=True, use_global=True):
        super().__init__()
        self.block, self.use_att, self.use_local, self.use_global = block, use_att, use_local, use_global
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
            x = x + self.proj_a(att.float())
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

| Parametro | Default | Note |
|---|---|---|
| `n_layers` | 2 | layer letti (23 e 31); H6 prova 4 |
| `d`, `block` | 256, 16 | H7 prova blocchi da 8 e 32 |
| `conv_layers`, `kernel` | 3, 5 | finestra locale ~30 token |
| `enc_layers`, `heads` | 2, 4 | |
| Parametri | ~4,5 M (con `d_q` = 2.560) | |

**Configurazione di partenza H0**: un solo layer, niente attenzione, niente contesto locale né globale (`use_att=False, use_local=False, use_global=False`): un punteggio per token dal suo vettore e da `q`. Gli esperimenti aggiungono un pezzo alla volta.

## 2. Il problema delle etichette

Le fonti annotano **frasi**, non parole: in una frase gold tutti i token hanno etichetta 1, anche quelli irrilevanti ("is", "the", la data accessoria). Chiedere che ogni token gold superi la soglia insegnerebbe alla testa a evidenziare frasi intere sempre, e penalizzerebbe una testa che evidenzia solo la parte utile.

Per questo la loss ha più termini: uno per token, con peso moderato, e due che parlano di **frasi gold** e di **ciò che sta fuori**, che sono le cose che contano per l'utente:
- ogni frase gold deve essere **colpita**: almeno un suo token chiaramente sopra soglia;
- **fuori** dalle frasi gold nessun token deve superare la soglia (con peso ω, perché le annotazioni possono essere incomplete);
- i punteggi di token vicini devono essere simili (tratti contigui, non a macchia di leopardo).

## 3. La loss

Con `d[t] = s[t] − s0` (positivo = sopra soglia), per una riga:

```text
supervised:  L = L_tok + 1,0 · L_hit + 0,5 · ω · L_out + 0,1 · L_tv
  L_tok = BCE pesata per token fra σ(d[t]) e l'etichetta (peso α = 3 sui token gold, ω sugli altri)
  L_hit = media sulle frasi gold g di softplus(1 − max_{t ∈ g} d[t])         ogni frase gold colpita
  L_out = softplus(max_{t ∉ gold} d[t] + 1)                                 niente acceso fuori
  L_tv  = media |σ(d[t+1]) − σ(d[t])|                                        tratti contigui
mil_neg:     L = 0,3 · softplus(max_t d[t] + 1)                              claim non supportato: nessun token sopra soglia
none:        L = 0
```

I documenti hanno lunghezze molto diverse: la loss si calcola **riga per riga** e poi si fa la media pesata con il peso della riga (`row_weight`, [04](04_split_e_mix.md) §3).

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

Esempio numerico (anche come test automatico): 10 token, frase gold sui token 2–5, `s0 = 0`, ω = 0,5, punteggi `−3; −3; 2; 0,5; −0,5; 2; −3; −3; −3; 1,5` (il token 9, fuori dalla frase gold, è acceso): loss **1,4015**. La stessa riga come negativo (`mil_neg`): **0,9146** (il token più alto, 2, sta sopra soglia).

## 4. La decodifica: dai punteggi ai tratti

1. token con `s[t] − s0 > δ` (δ = 0 di default, scelto sul dev, [09](09_training.md) §2);
2. tratti di token consecutivi; si uniscono tratti separati da al massimo 3 token;
3. si tolgono i tratti sotto 4 token (parole isolate);
4. si tengono i 5 tratti con punteggio medio più alto, in ordine di posizione;
5. se il verdetto è "supportato" e non resta nulla, si mostra il tratto intorno al token migliore;
6. ogni tratto diventa `[inizio, fine)` di caratteri con `tok_char` ([06](06_rendering_ed_estrazione.md) §1), allargato alla parola intera.

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

Esempio: testo "Rose Byrne is an Australian actress. She starred in Get Him to the Greek in 2010.", token 0–6 sopra soglia tranne il 3 (un buco di un token), token 12 sopra soglia da solo → un tratto: "Rose Byrne is an Australian actress." (il buco è chiuso, il token isolato è tolto).

Parametri della decodifica (δ, buco massimo, lunghezza minima, numero massimo di tratti) si scelgono sul dev e si scrivono in `decision.json` ([12](12_export_e_servizio.md)).

## 5. Test

`tests/test_token_head.py`:
- forme e maschere con righe di lunghezza diversa nello stesso batch; i token di riempimento hanno punteggio −10.000 e non influenzano gli altri;
- la loss dell'esempio numerico (tolleranza 1e-3); `none` dà zero e nessun gradiente;
- la decodifica dell'esempio; un documento senza token sopra soglia e `force_one` produce un tratto; i tratti non escono dal testo.
