# Riepilogo — Architettura, layer da leggere, loss

Questo file raccoglie in un posto solo le tre scelte tecniche che definiscono la testa di evidenza per frase: com'è fatta, da quali layer del decision model legge e perché, con quale loss si addestra. I dettagli operativi sono nei passi [01](01_ambiente_e_modelli.md), [06](06_rendering_ed_estrazione.md), [08](08_testa_e_loss.md) e [09](09_training.md).

## 1. Il quadro

```text
documento ──► divisore in frasi (evidence/sentences.py) ──► frasi 0 … n−1 (intervalli di caratteri)
                                                                  │
prompt = [preambolo con il claim] + [prompt del fine-tuning binario: documento, claim, domanda]
                                                                  │
                     UN forward del decision model congelato (Qwen3.5, 32 layer)
                                                                  │
      ┌─── posizione di risposta: punteggi sì/no → P(supportato)  (il verdetto, invariato)
      │
      └─── hook sui layer 19, 23, 27, 31 e sugli 8 layer di attenzione piena:
              per frase k:  h_end[k], h_mean[k]   [4, D]   +   att[k]  [128]   +   n_tok[k]
              risposta:     q                     [4, D]
                                                                  │
                     TESTA (3,7 M parametri su KnowLine-4B, ~1 ms)
                                                                  │
               s[k] per ogni frase, s0 soglia del documento  ──►  frasi con s[k] > s0 + δ (1–5)
```

Tre principi:
1. **Un solo forward**: la testa usa vettori che il modello calcola comunque; nessun token generato, nessun forward in più.
2. **Il verdetto non cambia**: il modello è congelato; si addestra solo la testa.
3. **Le frasi si vedono fra loro**: un transformer non causale sulle frasi risolve l'evidenza sparsa, che il modello causale da solo non può risolvere (la frase 1 non vede la frase 40).

## 2. L'architettura in sintesi

| Stadio | Operazione | Forma in uscita | Parametri (KnowLine-4B) | Perché |
|---|---|---|---:|---|
| A. Miscela dei layer | `LN(Σ_l softmax(α)_l · v[l])`, stessi α per `h_end`, `h_mean`, `q` | [n, D] | 5.124 | combina i 4 layer con pesi appresi; la LN uniforma le scale |
| B. Vettore della frase | `W_h [h_end ; h_mean] + W_a [att ; log n_tok] + P[k] + T_frase` | [n, 256] | 1.475.840 | contenuto della frase (fine e media), dove ha guardato il modello, posizione |
| B'. Vettore di risposta | `W_q q + T_risposta` | [256] | 655.616 | il punto di vista del verdetto |
| C. Comunicazione | transformer non causale, 2 layer, 4 teste, FFN 1.024, su `[x0, x[0…n−1]]` | [n+1, 256] | 1.579.520 | ogni frase vede tutte le altre e la risposta: evidenza sparsa |
| D. Punteggi | `w_o · X` per frasi e per `x0` | s [n], s0 [] | 257 | un'unica scala per punteggi e soglia |
| (H-PAIR) | bilineare a rango 64 sulle coppie | [n, n] | 32.769 | solo se l'evidenza multi-frase resta difficile |
| **Totale** | | | **3.716.357** (elephant-1: 4.899.077) | |

Tempo misurato: 0,67 ms con 46 frasi, 1,55 ms con 500 (RTX 5090 Laptop). Dettagli stadio per stadio: [08](08_testa_e_loss.md) §3.

Ingressi per frase: **ultimo token** (riassume la frase letta fin lì, vede anche il contesto precedente) e **media dei token** (pesa tutta la frase, non dipende dalla punteggiatura finale). Insieme coprono i due modi in cui un'informazione può essere "dentro" una frase.

## 3. Da quali layer leggere, perché, e come sceglierli

### 3.1 Che cosa offre Qwen3.5

| Layer | Tipo | Che cosa fa |
|---|---|---|
| 0, 1, 2, 4, 5, 6, … (24 in tutto) | Gated DeltaNet | attenzione lineare, sempre causale: aggiornano un riassunto del testo letto, con memoria che decade |
| **3, 7, 11, 15, 19, 23, 27, 31** (8) | attenzione piena | ogni token guarda esplicitamente tutti gli altri (solo i precedenti nei modelli causali, tutti nei bidirezionali): è qui che parti lontane del testo si collegano |
| 31 | ultimo layer | prepara la predizione del token successivo, cioè la lettera della risposta |

### 3.2 Il punto di partenza: 19, 23, 27, 31

Gli ultimi quattro layer ad attenzione piena. Il ragionamento:
- **dopo un layer ad attenzione piena** il vettore di una frase ha appena "visto" tutto il testo che può vedere (il claim del preambolo, le altre frasi). I layer DeltaNet in mezzo portano informazione solo in avanti e con memoria che decade: leggere subito dopo un layer pieno è leggere la rappresentazione più aggiornata;
- **nella metà alta della rete**: nei primi layer i vettori rappresentano soprattutto le parole e la sintassi locale, non ancora il rapporto fra la frase e il claim; nella metà alta compaiono le rappresentazioni del contenuto e del confronto con il claim;
- **non solo l'ultimo**: l'ultimo layer è specializzato a produrre la risposta nella posizione di risposta; nelle altre posizioni può aver già "buttato via" informazione che non serve a predire il token successivo. I layer pieni precedenti (19, 23, 27) conservano di più il contenuto della frase;
- **quattro, mescolati con pesi appresi**: non si sa a priori quale layer serve di più, e può cambiare da modello a modello. La miscela lascia decidere al training, con un costo minimo (4 parametri) e uno spazio su disco proporzionale al numero di layer.

Sono ipotesi ragionevoli, non misure: vanno verificate con i due metodi seguenti prima di fissare i layer.

### 3.3 Come sceglierli davvero: la sonda per layer (prima del training)

Su un campione piccolo si estraggono i vettori delle frasi in **tutti** i 32 layer, e per ogni layer si addestra una regressione logistica (una "sonda") che deve riconoscere le frasi di evidenza dal solo vettore della frase. Un layer in cui la sonda funziona bene contiene già l'informazione che serve alla testa.

1. **Campione**: 500 righe positive del training (HoVer e SciFact) e 200 del dev. Spazio: ~12 GB su KnowLine-4B (700 righe × ~52 frasi × 32 layer × 2 vettori).
2. **Estrazione**: lo stesso forward del passo 6, con gli hook su tutti i 32 layer (`h_end`, `h_mean`); con il preambolo.
3. **Sonda**: per ogni layer, regressione logistica su `[h_end ; h_mean]` normalizzato, classi bilanciate; si misura l'AUPRC per frase sul dev.
4. **Scelta**: i 4 layer con AUPRC più alta, preferendo a parità i layer ad attenzione piena (le feature di attenzione si leggono comunque da lì). Se i migliori sono ancora 19–31, si conferma il punto di partenza.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score


def probe_layers(train, dev, layers):
    """train, dev: liste di righe con, per ogni layer l, h_end[l] e h_mean[l] ([n, D] numpy) e gold [n].
    Per ogni layer: regressione logistica su [h_end ; h_mean] (vettori normalizzati), AUPRC sul dev.
    -> {layer: AUPRC}, ordinato dal migliore."""
    def xy(rows, l):
        x = np.concatenate([np.concatenate([r["h_end"][l], r["h_mean"][l]], 1) for r in rows])
        x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-6)
        return x, np.concatenate([r["gold"] for r in rows])
    out = {}
    for l in layers:
        xt, yt = xy(train, l)
        xd, yd = xy(dev, l)
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced").fit(xt, yt)
        out[l] = float(average_precision_score(yd, clf.decision_function(xd)))
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))
```

Provato su dati finti (un layer con il segnale e uno quasi senza): la sonda separa nettamente i due (AUPRC 0,87 contro 0,08).

Varianti utili della stessa sonda:
- con `q` concatenato (la frase letta "insieme" al verdetto): dice se la posizione di risposta aggiunge informazione;
- solo `h_end` o solo `h_mean`: dice se servono entrambi (esperimento H2c);
- sui modelli bidirezionali (elephant-1): i layer migliori possono essere diversi, perché anche le frasi iniziali vedono il claim nei layer pieni.

### 3.4 Come sceglierli davvero: i pesi della miscela (dopo il training)

Dopo il primo training con 4 layer (esperimento H2a), si guardano i pesi `softmax(α)`:
- un layer con meno del 10% si toglie (metà spazio su disco se ne restano 2, meno costo negli hook);
- se un layer prende quasi tutto il peso, basta quello;
- si riaddestra con i layer rimasti e si tiene la versione più semplice se l'F1 non scende di più di 1 punto (regola degli esperimenti, [11](11_esperimenti.md)).

### 3.5 Le feature di attenzione: da quali layer e teste

L'attenzione esiste solo negli 8 layer pieni: si leggono tutti (8 × 16 = 128 numeri per frase), e il peso di ciascuno lo impara `W_a`. Per sapere quali contano, la baseline E-ATT ([07](07_baseline.md) §2) misura l'AUPRC di ogni coppia (layer, testa) sul dev: se poche coppie portano quasi tutto il segnale, si può ridurre a quelle. Il dropout per frase sulle feature di attenzione evita che la testa dipenda da una sola testa di attenzione.

### 3.6 Riepilogo della procedura

| Quando | Che cosa | Risultato |
|---|---|---|
| prima del training | sonda su tutti i 32 layer, 500 righe | i 4 layer candidati (atteso: 19, 23, 27, 31) |
| prima del training | E-ATT per layer e testa | quanto segnale c'è nell'attenzione |
| H0 → H2a | ultimo layer contro i 4 candidati | quanto guadagna la miscela |
| dopo H2a | pesi α | togliere i layer sotto il 10% |
| altri modelli | sonda veloce (200 righe) | conferma o cambio dei layer per quel modello |

## 4. La loss

### 4.1 Che cosa deve insegnare

1. le frasi di evidenza devono stare **sopra** la soglia del documento, le altre **sotto**;
2. sui claim **non supportati** nessuna frase deve stare sopra la soglia (così la testa non evidenzia una frase solo perché somiglia al claim);
3. con annotazioni incomplete, una frase valida non segnata non deve essere punita troppo;
4. il numero di frasi mostrate deve adattarsi al documento (1 per un fatto semplice, 3–4 per una catena multi-hop).

### 4.2 La loss principale: ZLPR con soglia del documento

Con `G` = frasi gold, `ω` = peso delle frasi non gold, `s0` = soglia prodotta dalla testa per quel documento:

```text
supervised (positivi con evidenza):   log(1 + Σ_{i ∉ G} ω · e^(s_i − s0))   +   log(1 + Σ_{j ∈ G} e^(s0 − s_j))
                                      └─ cresce se una frase non gold sta sopra s0 ┘   └─ cresce se una frase gold sta sotto s0 ┘
mil_neg   (negativi):                 0,3 · log(1 + Σ_i e^(s_i − s0))                 nessuna frase sopra s0
mil_pos   (positivi senza evidenza):  0,3 · [ log(1 + e^(s0 − LSE_j s_j)) + 0,1 · media σ(s_i − s0) ]     almeno una, ma poche
none:                                 0
```

| Requisito (§4.1) | Come lo soddisfa |
|---|---|
| 1. gold sopra, altre sotto | i due termini di `supervised` |
| 2. niente sui negativi | `mil_neg`: lo stesso primo termine, con tutte le frasi |
| 3. annotazioni incomplete | ω < 1 (0,5 HoVer e SciFact, 0,3 WiCE) sulle frasi non gold |
| 4. numero adattivo | la soglia `s0` è prodotta per ogni documento; la regola in uso è `s[k] > s0` |

Esempio (test): 40 frasi, gold = {1, 40}, `s0 = 0`. Con la frase 40 a −0,5 (sotto soglia) e una frase simile al claim a 0,5 (sopra), la loss è **2,526**; dopo aver portato la frase 40 a 1,5 e la frase simile a −1, scende a **1,473**.

### 4.3 La loss di riferimento (H0)

BCE pesata per frase (peso α ≈ 10 sulle frasi gold, ω sulle altre) più un termine di ordinamento (ogni gold sopra la migliore non gold di un margine 1), con una soglia fissa τ_e scelta sul dev. Serve da punto di partenza e da confronto: l'esperimento H-LOSS dice se la soglia del documento fa meglio.

### 4.4 Termine facoltativo per le coppie (H-PAIR)

Sulle righe con 2 o più frasi gold: punteggio di coppia alto fra frasi gold, basso fra una gold e le 5 non gold più vicine; peso 0,2.

### 4.5 Pesi e come si combinano

| Peso | Valore | Dove |
|---|---|---|
| ω (frasi non gold) | 0,5 HoVer e SciFact; 0,3 WiCE; 1,0 dati generati | nel primo termine di `supervised` |
| peso della riga | 1,0; 0,5 per HoVer `db_split` e per le etichette ricavate col modello | moltiplica la loss della riga |
| μ_neg | 0,3 (H-MIL prova 0, 0,3, 1) | `mil_neg` |
| μ_pos, ρ | 0,3, 0,1 | `mil_pos` (solo H-MIL) |
| λ (fase 2) | 0,3 | loss totale = loss del verdetto + λ · loss di evidenza |

La loss del batch è la somma delle loss delle righe pesate, divisa per la somma dei pesi delle righe che hanno un termine di evidenza.

### 4.6 Il rischio da sorvegliare

`x0` conosce il verdetto, quindi con `mil_neg` la testa può imparare "se il verdetto è no, alza la soglia". È coerente con l'obiettivo, ma sui positivi incerti (P appena sopra la soglia del verdetto) la testa può non mostrare nulla. Si controlla la recall in quella fascia a ogni epoca; se cala, si abbassa μ_neg. In produzione resta comunque la regola "almeno una frase se il verdetto è supportato".
