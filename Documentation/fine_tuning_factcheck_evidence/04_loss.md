# Step 4 — La loss: verdetto + evidenza con soglia dell'istanza

## Obiettivo

Definire la loss che addestra la testa di evidenza (e, nel training congiunto, il verdetto) tenendo conto di quattro fatti:
1. l'evidenza esiste solo per i positivi, ma i negativi dicono qualcosa di certo: **nessuna** frase stabilisce il claim;
2. le frasi di evidenza sono poche (classe molto sbilanciata) e il loro numero varia da riga a riga;
3. le annotazioni gold sono incomplete;
4. con una soglia fissa, la frase ponte (punteggio più basso) si perde.

Ci sono due loss: quella di **riferimento** (BCE pesata + ranking, il piano precedente, usata in H0) e quella **candidata** (ZLPR con soglia dell'istanza e MIL, da [idea.md](idea.md) idea 6). H-LOSS ([07](07_esperimenti.md)) decide.

## 1. La struttura

```text
L = L_verdetto + λ · L_evidenza
L_verdetto = CE binaria sul margine delle lettere                     (piano binario, 04_loss.md)
L_evidenza = media sulle righe di  w_riga · termine(riga)
```

- Fasi 1a e 1b (backbone congelato): `L_verdetto` non c'è.
- Fase 2: λ ∈ [0,2; 0,5], scelto su `dev_select` + `dev_evid_select`.
- `w_riga` = `evidence_weight` (1,0 gold e construct, 0,5 ablation; [02e](02e_mix_finale.md)).
- `termine(riga)` dipende da `evidence_term` ([03](03_rendering_formato.md) §1).

Notazione: `s_k` = logit della frase k; `s_0` = logit di `x_0` (lo stato di `Answer:` dopo l'aggregatore, passato per la stessa uscita lineare); `G` = frasi gold; `ω_i` = peso della frase non gold i.

## 2. Loss di riferimento (H0): BCE pesata + ranking

Solo sulle righe `supervised`; negativi mascherati; soglia fissa `τ_e` scelta sul dev.

```text
L_BCE  = (1/n) Σ_k [ α · y_k · (−ln σ(s_k)) + (1 − y_k) · ω_k · (−ln(1 − σ(s_k))) ]
L_rank = media su g ∈ G di softplus( m − s_g + max_{j ∉ G} s_j )
termine = L_BCE + γ · L_rank          (α ≈ non evidenza / evidenza, max 20; γ = 0,5; m = 1)
```

Esempio: 40 frasi, evidenza {1, 40}, α = 10, ω = 1, la testa dà 0,02 a tutte → L_BCE = (10 · 2 · 3,91 + 38 · 0,020)/40 = **1,97**: la loss spinge su S1 e S40. Ranking con m = 1: gold a 0,5 e migliore non gold a 1,2 → softplus(1,7) = 1,87; gold a 2,5 → softplus(−0,3) = 0,55.

## 3. Loss candidata: ZLPR con soglia dell'istanza

Ogni riga ha la sua soglia `s_0`. Una frase è evidenza se e solo se `s_k > s_0`. Tre termini, uno per tipo di riga:

```text
supervised:  log(1 + Σ_{i ∉ G} ω_i · e^{s_i − s_0})  +  log(1 + Σ_{j ∈ G} e^{s_0 − s_j})
mil_neg:     μ_neg · log(1 + Σ_{i} e^{s_i − s_0})
mil_pos:     μ_pos · [ log(1 + e^{s_0 − LSE_j s_j})  +  ρ · (1/n) Σ_i σ(s_i − s_0) ]
```

| Termine | Dice | Note |
|---|---|---|
| supervised | tutte le gold sopra `s_0`, tutte le altre sotto | ZLPR di Su et al., con `ω_i` dentro la somma: `log(1 + Σ e^{s_i − s_0 + ln ω_i})` |
| mil_neg | nessuna frase sopra `s_0` | è il termine supervised con G vuoto: non serve una loss a parte. μ_neg = 0,3 di default |
| mil_pos | almeno una frase sopra `s_0` (con LSE, il massimo "morbido"), ma poche | solo in H-MIL; ρ = 0,1 regola la sparsità |

Perché la soglia dell'istanza:
- **niente `τ_e`** da tarare: la regola di produzione è `s_k > s_0`;
- **numero di frasi adattivo**: in un documento multi-hop `x_0` (che ha visto tutta la catena) può abbassare `s_0` e far passare la frase ponte con punteggio più basso; in un caso semplice la alza;
- i negativi, con il loro termine, insegnano a tenere `s_0` sopra le frasi che **somigliano** al claim ma non lo stabiliscono.

Esempio numerico, riga supervised con 40 frasi, G = {1, 40}, ω = 1, `s_0` = 0:

| Situazione | s_1 | s_40 | distrattore | altre 37 | Termine negativi | Termine positivi | Totale |
|---|---:|---:|---:|---:|---:|---:|---:|
| prima | 2,0 | −0,5 (ponte) | 0,5 | −3 | ln(1 + 1,65 + 1,84) = 1,50 | ln(1 + 0,14 + 1,65) = 1,02 | **2,53** |
| dopo | 2,0 | 1,5 | −1,0 | −3 | ln(1 + 0,37 + 1,84) = 1,17 | ln(1 + 0,14 + 0,22) = 0,31 | **1,47** |

Nella prima riga la frase ponte (−0,5) sta sotto la soglia: con una soglia fissa a 0 si perderebbe. Il termine positivo la spinge sopra, e quello negativo spinge giù il distrattore.

Le 37 frasi facili a −3 pesano ancora 1,84 nella somma: con n = 150 peserebbero 4 volte tanto. Il termine non è normalizzato per n di proposito (ogni frase sopra soglia è un errore che l'utente vede), ma i documenti lunghi hanno una loss più alta: se H-LOSS mostra che i documenti lunghi dominano il gradiente, si divide il termine negativo per log(n).

### Vincoli di produzione

La regola `s_k > s_0` si combina con i vincoli del prodotto ([08](08_merge_export.md)): se il verdetto è "supportato" si mostra almeno una frase (la migliore, anche se sotto `s_0`), al massimo 5.

### Rischio del termine sui negativi

`x_0` conosce il verdetto: la testa può imparare "se il verdetto è no, alza `s_0`". È coerente con la definizione, ma vicino alla soglia del verdetto (positivi incerti) la testa può spegnersi del tutto. Controllo: recall dell'insieme completo sulle righe con P(supportato) fra la soglia e soglia + 0,15. Se cala con il termine MIL, μ_neg si abbassa.

## 4. Annotazioni incomplete

Una frase non annotata in FEVER può essere evidenza valida. Correttivi, per entrambe le loss:

1. **ω basso sulle fonti incomplete**: FEVER 0,3; HoVer 0,5; construct 1,0; ablation 0,5.
2. Nella loss di riferimento, il **ranking** con il `max` sulle non gold: una sola frase non gold valida non viene schiacciata per tutte.
3. Nella ZLPR il termine negativo è una somma, quindi una frase valida non annotata viene spinta giù; ω basso la protegge in parte. Variante (H7): escludere dal termine la frase non gold con punteggio più alto quando E-ABL la indica come sufficiente.

## 5. Insiemi alternativi (FEVER)

Con più insiemi gold (`evidence_sets`) il target di default è l'**unione**. Variante (H7): "almeno un insieme", cioè il minimo del termine supervised fra gli insiemi; più fedele alle annotazioni, più complessa.

## 6. Termini ausiliari (solo negli esperimenti condizionati)

### Coppie (H-PAIR)

Con il termine bilineare a coppie ([05](05_training.md) §1), per le righe con |G| ≥ 2:

```text
L_pair = media su k ∈ G di  BCE(S_kj, 1) per j ∈ G \ {k}  +  BCE(S_kj, 0) per i 5 j ∉ G con S_kj più alto
```

Peso 0,2 rispetto al termine principale. Su FEVER (quasi sempre una frase) è vuoto: il segnale viene da HoVer, CG2C, C2D.

### Fedeltà (H-FAITH)

Con il backbone congelato e la maschera per frase ([00a](00a_scelta_approccio.md) §8, [05](05_training.md) §4), sulle righe positive giudicate supportate, con `p_k = σ(s_k − s_0)`:

```text
z_full  = margine del verdetto sul documento intero
z_suff  = margine con bias log(p_k + ε) sulle chiavi della frase k
z_comp  = margine con bias log(1 − p_k + ε)
L_faith = KL(σ(z_full) ‖ σ(z_suff))  +  softplus(z_comp − z_full + δ)        (δ = 1)
```

Attenzione alla soluzione banale: `p_k = 1` per tutte soddisfa la sufficienza, e la completezza è soddisfatta "togliendo tutto". Per questo L_faith non si usa mai da sola: si aggiunge al termine supervised (o mil_pos con la sparsità ρ), con peso 0,1–0,3.

### ListNet (variante di H-LOSS)

KL fra `softmax(s / τ)` e l'uniforme su G. Non è il default: con annotazioni incomplete toglie massa alle frasi valide non annotate, e non dà una soglia.

## 7. Codice di riferimento

```python
import torch
import torch.nn.functional as F

def zlpr_evidence_loss(s, s0, gold, sent_mask, omega, term, mu_neg=0.3, mu_pos=0.3, rho=0.1):
    """s: [B, N] logit delle frasi; s0: [B] soglia; gold, sent_mask: [B, N] (0/1); omega: [B, N];
    term: [B] con 0 = supervised, 1 = mil_neg, 2 = mil_pos, 3 = none. Restituisce [B]."""
    ninf = torch.finfo(s.dtype).min
    d = s - s0[:, None]                                            # margine rispetto alla soglia
    is_neg = (gold == 0) & (sent_mask == 1)
    neg_logits = (d + omega.clamp_min(1e-6).log()).masked_fill(~is_neg, ninf)
    l_neg = F.softplus(torch.logsumexp(neg_logits, 1))             # log(1 + Σ ω e^{s_i − s0})
    pos_logits = (-d).masked_fill(gold == 0, ninf)
    l_pos = F.softplus(torch.logsumexp(pos_logits, 1)) * (gold.sum(1) > 0)
    supervised = l_neg + l_pos
    all_logits = d.masked_fill(sent_mask == 0, ninf)
    mil_neg = mu_neg * F.softplus(torch.logsumexp(all_logits, 1))
    lse = torch.logsumexp(all_logits, 1)                           # massimo morbido rispetto a s0
    spars = (torch.sigmoid(d) * sent_mask).sum(1) / sent_mask.sum(1)
    mil_pos = mu_pos * (F.softplus(-lse) + rho * spars)
    out = torch.stack([supervised, mil_neg, mil_pos, torch.zeros_like(lse)], 1)
    return out.gather(1, term[:, None]).squeeze(1)                 # poi media pesata con w_riga
```

(`softplus(logsumexp(x))` = `log(1 + Σ e^x)`; con `mil_neg` le frasi sono tutte nel termine negativo con ω = 1.) La loss di riferimento resta quella del piano precedente (`evidence_loss`, BCE pesata + ranking).

## Criteri per passare allo step successivo

- Funzioni testate sugli esempi numerici di questo file (§2 e la tabella di §3).
- α (riferimento) calcolato sul mix; ω per fonte fissato; μ_neg di partenza fissato.
