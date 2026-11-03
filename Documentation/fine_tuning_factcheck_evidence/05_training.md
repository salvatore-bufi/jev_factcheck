# Step 5 — Training: la testa di evidenza, in tre fasi

## Obiettivo

Addestrare la testa di evidenza sopra il modello del verdetto, senza peggiorarlo e senza aggiungere costo in inferenza. Tre fasi:

| Fase | Backbone | Ingresso della testa | Per cosa |
|---|---|---|---|
| **1a** | congelato, stati **salvati** su disco | feature di [03](03_rendering_formato.md) §4 | quasi tutti gli esperimenti: minuti per prova |
| **1b** | congelato, forward **in linea** a ogni passo | stati di tutti i token | ciò che non si può salvare: pooling attentivo (H2d), fedeltà (H-FAITH) |
| **2** | LoRA aggiornata insieme alla testa | come 1a, dal forward con gradiente | solo se 1a/1b non bastano |

Ordine dei modelli: **d1-3B** per tutti gli esperimenti; JPT-9B e Winnow-12B ripetono la configurazione migliore ([07](07_esperimenti.md)).

## Prerequisito

Il modello del verdetto del piano binario, nel formato scelto in [02d](02d_varianti_domanda.md): riusato così com'è o riaddestrato con il preambolo ([01](01_baseline.md) §1). Se non è ancora pronto, la fase 1a si può provare sul modello di partenza (stesso formato), sapendo che andrà rifatta sul modello fine-tunato (l'estrazione costa 1–3 ore).

## 1. L'architettura della testa

```python
class EvidenceHead(nn.Module):
    def __init__(self, d_model, n_mix, n_att, d=256, layers=2, heads=4, max_sent=512,
                 use_mean=True, use_att=True, use_pair=False, pair_rank=64):
        super().__init__()
        self.mix = nn.Parameter(torch.zeros(n_mix))                 # pesi scalari dei layer (softmax)
        self.ln = nn.LayerNorm(d_model)                              # i layer hanno scale diverse
        n_in = d_model * (2 if use_mean else 1)
        self.proj_h = nn.Linear(n_in, d)                             # [fine frase ; media frase]
        self.proj_q = nn.Linear(d_model, d)
        self.proj_a = nn.Linear(n_att + 1, d) if use_att else None   # attenzione 8×H + log(numero di token)
        self.pos = nn.Embedding(max_sent, d)
        self.type = nn.Embedding(2, d)                               # 0 = x_0, 1 = frase
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)            # attenzione PIENA (non causale)
        self.out = nn.Linear(d, 1)                                   # stessa uscita per frasi e per x_0
        self.pair = (nn.Linear(d, pair_rank, bias=False), nn.Linear(d, pair_rank, bias=False)) if use_pair else None
        self.use_mean = use_mean

    def mixed(self, x):                                              # x: [..., n_mix, D]
        return self.ln((self.mix.softmax(0)[:, None] * x).sum(-2))

    def forward(self, q, h_end, h_mean, att, att_len, sent_mask):
        """q: [B, n_mix, D]; h_end, h_mean: [B, N, n_mix, D]; att: [B, N, n_att]; -> s: [B, N], s0: [B]."""
        h = self.mixed(h_end)
        if self.use_mean:
            h = torch.cat([h, self.mixed(h_mean)], -1)
        x_h = self.proj_h(h) + self.pos(torch.arange(h.size(1), device=h.device)) + self.type.weight[1]
        if self.proj_a is not None:
            x_h = x_h + self.proj_a(torch.cat([att, att_len.clamp_min(1).log()[..., None]], -1))
        x_0 = (self.proj_q(self.mixed(q)) + self.type.weight[0])[:, None]
        pad = torch.cat([torch.zeros_like(sent_mask[:, :1]), 1 - sent_mask], 1).bool()
        X = self.encoder(torch.cat([x_0, x_h], 1), src_key_padding_mask=pad)
        o = self.out(X)[..., 0]
        s, s0 = o[:, 1:], o[:, 0]
        if self.pair is not None:                                    # H-PAIR (idea 3)
            S = self.pair[0](X[:, 1:]) @ self.pair[1](X[:, 1:]).transpose(1, 2)
            S = S.masked_fill(~sent_mask.bool()[:, None, :] | torch.eye(S.size(1), device=S.device, dtype=torch.bool), float("-inf"))
            s = s + self.tau * (S + s.detach()[:, None, :]).amax(-1).nan_to_num(neginf=0.0)  # n = 1: nessun partner
        return s, s0
```

(`self.tau` del termine a coppie è un parametro scalare inizializzato a 0,1; i moduli di `self.pair` vanno registrati in un `nn.ModuleList`. Lo schema è il riferimento, non il codice finale.)

| Componente | Valore | Perché |
|---|---|---|
| Mix di layer | 4 layer per modello ([00c](00c_modelli_e_architetture.md) §2), pesi appresi | idea 1: l'ultimo layer è specializzato sulla lettera; gli altri localizzano meglio |
| Fine frase + media frase | concatenati | idea 2, gradino fisso: la media non dipende dalla punteggiatura finale |
| Feature di attenzione | 8 layer pieni × teste (+ log della lunghezza) | idea 1: dove il modello guarda per decidere |
| `x_0` come soglia | `s_0` dalla stessa uscita lineare | idea 6, [04](04_loss.md) §3 |
| 2 layer, 4 teste, attenzione piena | | le frasi comunicano fra loro e con `x_0`: evidenza sparsa |
| Termine a coppie (facoltativo) | rango 64, `max` sui partner, `stop_grad` sul punteggio del partner | idea 3, solo in H-PAIR |
| Parametri | ~2–4M (d1: d = 2.048; JPT 4.096; Winnow 3.840) | trascurabili |

Ogni componente si può spegnere con un flag: è così che H0 (solo ultimo layer, solo fine frase, niente attenzione, soglia fissa) e i gradini H2 usano lo stesso codice.

### Variante: pooling attentivo dentro la frase (H2d, fase 1b)

Al posto della media fissa, il pooling guidato da `[claim; q]` di [idea.md](idea.md) idea 2: proiezione d → 256 di tutti i token del documento (una moltiplicazione per riga), punteggi `uᵀ v_{k,i}` con `u = W [c ; q]` (c = media degli stati del preambolo), softmax dentro ogni frase, somma pesata concatenata allo stato di fine frase. Serve la fase 1b, perché richiede gli stati di tutti i token. Si prova solo se in H2c la media fissa guadagna sulle frasi lunghe (> 25 parole).

## 2. Fase 1a — stati salvati (il cavallo da lavoro)

1. **Estrazione** ([03](03_rendering_formato.md) §4): un forward del modello del verdetto su `ev_head` e sui dev; si salvano `q`, `h_end`, `h_mean`, `att` per i layer del mix.
   - Spazio: ~40 GB per d1, ~80 GB per JPT e Winnow con 4 layer ([00c](00c_modelli_e_architetture.md) §2).
   - Tempo: un'inferenza sul mix; per d1 meno di un'ora, per JPT e Winnow ~1–3 h su L40S.
2. **Training della testa sugli stati salvati**: niente LLM in memoria.

| Iperparametro | Valore |
|---|---|
| Learning rate | 1e-3, warmup 5%, cosine |
| Epoche | 5–10 con early stopping su `dev_evid_select` |
| Batch | 64 righe |
| Loss | H0: BCE + ranking; poi quella scelta in H-LOSS ([04](04_loss.md)) |
| Regolarizzazione | dropout 0,1; weight decay 0,01; dropout delle feature di attenzione 0,1 (per non dipendere da una sola testa) |

## 3. Fase 1b — backbone congelato in linea

Quando: H2d (pooling attentivo) o H-FAITH. Il modello del verdetto è in memoria, in `torch.no_grad()` per H2d; con gradiente rispetto alla sola maschera per H-FAITH.

| Parametro | d1-3B | JPT-9B / Winnow-12B |
|---|---|---|
| GPU | 24 GB | 48 GB |
| Batch | 4–8 righe, accumulo fino a 32 | 1–2 righe, accumulo fino a 32 |
| Learning rate della testa | 5e-4 | 5e-4 |
| Epoche | 1–2 | 1 |

## 4. H-FAITH — fedeltà con maschera per frase (fase 1b, solo d1-3B e Winnow)

Implementazione della versione corretta di [00a](00a_scelta_approccio.md) §8:

1. Forward 1 (no grad): stati, `s`, `s_0`, margine `z_full`.
2. `p_k = σ(s_k − s_0)` (con gradiente verso la testa).
3. Forward 2 e 3, con bias additivo sulle chiavi della frase k per tutte le query fuori dalla frase k:
   - sufficienza: `log(p_k + ε)`; completezza: `log(1 − p_k + ε)`;
   - in tutti i layer di attenzione: gli 8 pieni per d1 (le convoluzioni restano), tutti i 48 per Winnow;
   - si mascherano solo le frasi del documento: preambolo, claim dello stato e domanda restano visibili.
4. `L_faith` di [04](04_loss.md) §6, sommata al termine supervised; il gradiente passa per il backbone congelato fino a `p_k`.

Costo: due forward + backward attraverso il backbone per passo, ~3–4× un passo di H2d. Su d1-3B con documenti fino a 4k token è fattibile su 24 GB con gradient checkpointing; su Winnow serve una GPU da 48 GB e documenti più corti.

Il bias si applica con una maschera additiva in SDPA (FlashAttention non accetta bias arbitrari): più lento, accettabile in training. In produzione questo ramo non esiste.

## 5. Fase 2 — training congiunto (solo se le fasi 1 non bastano)

Quando: F1 o recall dell'insieme completo sotto gli obiettivi ([00_overview.md](00_overview.md) §6), soprattutto sulle righe sparse di `dev_evid_select`.

| Parametro | Valore |
|---|---|
| Inizializzazione | adapter LoRA del verdetto + testa della fase 1 |
| Moduli LoRA | gli stessi del piano binario |
| Learning rate | LoRA 5e-6 – 1e-5 (il verdetto è già addestrato); testa 2e-4 |
| Loss | `L_verdetto + λ · L_evidenza`, λ = 0,3 |
| Dati | `ev_joint` |
| Epoche | 1 |
| Durata | come una corsa del piano binario + ~10–20% |
| GPU | d1: 24 GB; JPT e Winnow: 48 GB (o QLoRA su 24 GB per JPT) |

Implementazione: nel forward del modello con LoRA si leggono gli stati dei layer del mix con hook, nelle posizioni di `sent_tok_spans` e `answer_pos`, e la riga di attenzione degli 8 layer pieni; le lettere si calcolano solo in `answer_pos` con le righe della testa del vocabolario (come nel piano binario).

## 6. Cosa controllare

| Metrica | Su cosa | Allarme |
|---|---|---|
| F1 per frase, recall dell'insieme completo | `dev_evid_select` | non sale sopra E-LEX ([01](01_baseline.md)) |
| F1 per fascia di distanza fra frasi gold | righe sparse di `dev_evid_select` | crollo oltre 10 frasi → aggregatore o dati sparsi insufficienti |
| F1 per distanza in token dal preambolo | `dev_evid_select` | crollo oltre 1.024 token (Winnow) o oltre i 4k → claim che non arriva alle frasi lontane |
| Numero medio di frasi mostrate | `dev_evid_select` | molto sopra o sotto il numero gold → `s_0` (o `τ_e`) da rivedere |
| Quota di negativi con almeno una frase sopra `s_0` | `dev_evid_neg` | alta: la testa segue la somiglianza lessicale |
| Pesi del mix di layer | testa addestrata | tutto su un layer: il mix si può ridurre a quel layer (meno spazio, meno latenza) |
| **Verdetto** (solo fase 2) | `dev_select` | BAcc −0,3 o ECE +0,01 rispetto al modello del piano binario |

## 7. La soglia di evidenza

- Con la loss ZLPR: regola `s_k > s_0`, nessuna soglia da scegliere. Si verifica su `dev_evid_select` che uno spostamento globale `s_k > s_0 + δ` con δ ∈ [−1, 1] non guadagni più di 1 punto di F1; se guadagna, si fissa δ sul dev.
- Con la loss di riferimento: `τ_e` che massimizza l'F1 per frase su `dev_evid_select`.

In entrambi i casi: al massimo 5 frasi mostrate; almeno 1 se il verdetto è "supportato" (la migliore). Conferma su `dev_evid_confirm`.

## 8. Test prima del training

| Test | Cosa verifica |
|---|---|
| Stati estratti = stati del forward in produzione | stessa funzione di rendering, stesso modello, differenza < 1e-3 |
| Overfit di 64 righe | la testa arriva a F1 ≈ 1 sul training (con e senza `s_0`) |
| Gradino zero | con mix, media e attenzione spenti, la testa coincide con H0 |
| Fase 2: passo zero | prima di aggiornare, verdetto identico al modello del piano binario |
| H-FAITH: maschera tutta a 1 | con `p_k = 1` per tutte, `z_suff = z_full` entro 1e-3 |

## Criteri per passare allo step successivo

- Testa che supera E-LEX ed E-ATT su `dev_evid_select` con margine, confermata su `dev_evid_confirm`, per i tre modelli.
- Verdetto invariato (fasi 1) o entro i limiti (fase 2).
