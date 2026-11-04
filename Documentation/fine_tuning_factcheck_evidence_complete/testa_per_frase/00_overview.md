# Fact checking con evidenza evidenziata — testa per frase

## 1. Che cosa si vuole ottenere

Abbiamo un **decision model** fine-tunato come fact checker: riceve un documento e un claim e risponde con P(supportato), in **un solo passaggio del modello sul testo** (un *forward*), senza generare parole. È veloce proprio perché fa un solo forward.

Vogliamo che, quando il claim è supportato, dica anche **quali frasi del documento lo dimostrano**, senza perdere velocità: sempre un solo forward, nessuna parola generata.

Esempio di risposta:

```json
{"p_supported": 0.93, "supported": true,
 "evidence": [{"sentence": 0,  "start": 0,    "end": 61,   "score": 0.91},
              {"sentence": 39, "start": 4210, "end": 4262, "score": 0.84}]}
```

- `sentence`: numero della frase nel documento (da 0); `start` / `end`: dove inizia e finisce, in caratteri, nel testo originale.
- Se il claim non è supportato, `evidence` è vuota: non si indica né cosa manca né cosa lo contraddice.

Il caso difficile è l'**evidenza sparsa**: due frasi lontane che solo **insieme** dimostrano il claim.

> Claim: "Nancy R. Howell teaches at a seminary in Kansas City."
> Frase 1: "Saint Paul School of Theology is a seminary located in Kansas City, Missouri."
> … 38 frasi che non c'entrano …
> Frase 40: "Nancy R. Howell is Professor of Theology at Saint Paul School of Theology."
>
> Nessuna delle due basta da sola. Evidenza corretta: frasi 1 e 40.

## 2. Come funziona, passo per passo

### 2.1 Che cosa fa già il decision model

Il modello legge il prompt (documento, claim, domanda). Per **ogni token** e per **ogni layer** calcola un vettore di numeri (lo *stato nascosto*): 2.560 numeri nei modelli da 4B, 4.096 in quelli da 9B. Il verdetto si legge dall'**ultimo token del prompt** (la *posizione di risposta*): lì il modello dà un punteggio alla risposta "sì" e uno alla risposta "no", da cui P(supportato). Questa parte esiste già: è il modello del fine-tuning binario, addestrato con una LoRA sulla domanda di fact checking. **Non la tocchiamo.**

### 2.2 Che cosa aggiungiamo

Il documento si divide in **frasi** con un divisore fisso (`evidence/sentences.py`). Durante il forward si leggono, per ogni frase, alcuni vettori che il modello calcola comunque, e li si dà a una **piccola rete separata**, la *testa di evidenza* (3,7 milioni di parametri contro i 4–9 miliardi del modello), che dà un punteggio a ogni frase.

```text
                 ┌──────────────────────────────── prompt ─────────────────────────────────┐
                 │ Claim under review: <claim>                         ← preambolo (nuovo)   │
                 │ <documento: frase 0, frase 1, …, frase n−1>         ← prompt del fine-    │
                 │ <claim> <domanda> <opzioni>                           tuning binario,     │
                 │ [ultimo token = posizione di risposta]                invariato           │
                 └──────────────────────────────────────────────────────────────────────────┘
                                                 │
                          UN forward del decision model (pesi congelati)
                                                 │
            ┌────────────────────────────────────┴───────────────────────────────┐
            │ per ogni frase k:                                                  │ alla posizione di risposta:
            │   h_end[k]  vettore dell'ultimo token della frase, 4 layer          │   punteggi "sì"/"no" → P(supportato)
            │   h_mean[k] media dei vettori dei token della frase, 4 layer        │   q = vettore della posizione, 4 layer
            │   att[k]    attenzione che la posizione di risposta dà alla frase:   │
            │             8 layer × 16 teste = 128 numeri                         │
            └────────────────────────────────────┬───────────────────────────────┘
                                                 │
   ┌──────────────────────────────────── TESTA DI EVIDENZA ─────────────────────────────────────┐
   │ A. miscela dei 4 layer   pesi appresi                                                        │
   │ B. vettore della frase   proiezione di [h_end, h_mean], att, posizione della frase → 256      │
   │ C. comunicazione         transformer NON causale a 2 layer su [q, frase 0 … frase n−1]:      │
   │                          ogni frase vede tutte le altre e q                                  │
   │ D. punteggi              s[k] per ogni frase; s0 da q = soglia di QUESTO documento            │
   └──────────────────────────────────────────────┬───────────────────────────────────────────────┘
                                                  │
   mostrate le frasi con s[k] > s0 (almeno 1, al massimo 5), solo se P(supportato) > soglia del verdetto
```

L'architettura completa, con forme dei tensori, parametri e costo di ogni parte, è nel passo [08](08_testa_e_loss.md). Un riepilogo di architettura, scelta dei layer e loss è in [15](15_recap_architettura_layer_loss.md).

### 2.3 Perché ogni pezzo

| Pezzo | Perché serve |
|---|---|
| vettore dell'ultimo token e media della frase | l'ultimo token riassume la frase letta fin lì; la media non dipende dalla punteggiatura finale e pesa tutta la frase |
| 4 layer, non solo l'ultimo | l'ultimo layer è specializzato a predire la risposta; i layer pieni precedenti rappresentano meglio il contenuto delle frasi |
| attenzione della posizione di risposta | dice su quali frasi il modello si è basato per decidere: un indizio forte, già calcolato |
| transformer non causale sulle frasi | nel modello la frase 1 non vede la frase 40 (viene dopo); nella testa sì, ed è ciò che risolve l'evidenza sparsa |
| soglia `s0` del documento | non c'è una soglia unica da tarare; con 3 frasi necessarie la soglia può scendere per farle passare tutte |

### 2.4 Perché il preambolo

Nei modelli **causali** (come KnowLine-4B) ogni token vede solo ciò che lo precede. Nel prompt del fine-tuning binario il claim viene **dopo** il documento: quando il modello legge le frasi non sa ancora cosa deve verificare, e i loro vettori non "parlano" del claim. Mettendo una riga `Claim under review: <claim>` **prima** del documento, ogni frase è letta sapendo cosa si verifica. Il resto del prompt resta identico, compreso il claim dopo il documento, dove il verdetto è abituato a vederlo. Dettagli nel passo [05](05_formato_prompt.md).

### 2.5 Che cosa si addestra

Solo la testa. Il modello resta congelato, quindi **il verdetto non cambia**. Le etichette sono le frasi di evidenza annotate nei dataset (SciFact, WiCE, HoVer), riportate sulle frasi del nostro divisore. Siccome il modello è congelato, i vettori delle frasi si calcolano **una volta** e si salvano su disco: poi ogni prova di training della testa dura minuti.

### 2.6 Il divisore in frasi fa parte del modello

La testa lavora su frasi: in produzione il documento va diviso **esattamente** come in training, con la stessa funzione e la stessa versione dei suoi dati (NLTK `punkt_tab`). Se le frasi fossero tagliate in modo diverso, la testa vedrebbe unità diverse da quelle su cui è stata addestrata. Per questo il divisore è versionato con il modello (hash nel manifest e in `decision.json`), e conviene includere nella repo i parametri di `punkt_tab` (passo [12](12_export_e_servizio.md) §4).

## 3. I modelli

I 10 decision model dello studio focalizzato, tutti costruiti su **Qwen3.5**: 32 layer, di cui 8 ad attenzione piena (layer 3, 7, 11, 15, 19, 23, 27, 31) e 24 Gated DeltaNet (attenzione lineare, sempre causale).

| Modello | Ruolo | d | Attenzione nei layer pieni |
|---|---|---:|---|
| **KnowLine-4B** (`PelaAI/KnowLine-4B-Gen5.5`) | tutte le prove si fanno qui | 2.560 | causale |
| **elephant-1** (`baselabs/elephant-1`) | conferma a 9B | 4.096 | bidirezionale a blocchi |
| Jeffan-4B, d3-flash, OneJev1.6-9B, Nimble-9B-v3 | estensione | 2.560 / 4.096 | Jeffan, d3: bidirezionale; OneJev, Nimble: causale |
| StartLux 9B/4B, pplx-decider-flash, Clef-Flash | estensione, più lavoro | 2.560 / 4.096 | StartLux, pplx: bidirezionale; Clef: come caricato |

Ogni modello si esegue **con il codice dei suoi autori** (prompt, maschera di attenzione, lettura della risposta), come nei benchmark della repo. Dettagli nel passo [01](01_ambiente_e_modelli.md).

## 4. Glossario

| Termine | Significato |
|---|---|
| forward | un passaggio del modello su tutto il prompt |
| stato nascosto, vettore | i numeri che il modello calcola per un token in un layer |
| posizione di risposta | il token del prompt dove si legge il verdetto (l'ultimo) |
| testa di evidenza | la piccola rete che dà un punteggio a ogni frase |
| `s[k]`, `s0` | punteggio della frase k; soglia del documento |
| frase gold | frase di evidenza annotata nel dataset, riportata sulla nostra divisione |
| insieme completo | tutte le frasi gold mostrate (per l'evidenza sparsa conta questo) |
| sufficienza | le frasi mostrate, da sole come documento, fanno ancora dire "supportato" al modello |
| fase 1a / 1b / 2 | training della testa su vettori salvati / con il modello in memoria / insieme a una LoRA del modello |

## 5. Sequenza di lavoro

| # | Passo | File | Prodotto |
|---|---|---|---|
| 1 | Modelli e hook per frase | [01](01_ambiente_e_modelli.md) | forward con cattura dei vettori e dell'attenzione per frase |
| 2 | Dati dalle fonti | [02](02_dati_fonti.md) | `dataset/evidence/*.jsonl` (già costruiti) |
| 3 | Dati generati (facoltativo per le prime prove) | [03](03_dati_generati.md) | evidenza nota per costruzione, etichette dal modello, evidenza spostata |
| 4 | Split e mix | [04](04_split_e_mix.md) | set congelati, mix di training |
| 5 | Prompt con il preambolo | [05](05_formato_prompt.md) | preambolo scelto, verdetto verificato |
| 6 | Dalle frasi ai token, estrazione | [06](06_rendering_ed_estrazione.md) | vettori delle frasi su disco |
| 7 | Baseline | [07](07_baseline.md) | numeri da battere |
| 8 | Architettura e loss | [08](08_testa_e_loss.md) | `evidence/head.py`, `evidence/losses.py` |
| 9 | Training | [09](09_training.md) | teste addestrate |
| 10 | Valutazione | [10](10_valutazione.md) | metriche e latenza |
| 11 | Esperimenti | [11](11_esperimenti.md) | configurazione finale |
| 12 | Server e rilascio | [12](12_export_e_servizio.md) | server che restituisce verdetto ed evidenza |
| — | Checklist | [13](13_checklist.md) | |
| — | Una strada da esplorare | [14](14_strada_da_esplorare.md) | |
| — | Riepilogo: architettura, layer, loss | [15](15_recap_architettura_layer_loss.md) | |

Codice già pronto nella repo: `evidence/sentences.py` (divisore), `evidence/data/` (costruzione dei dataset del passo 2), `tests/test_evidence_data.py`. Il resto è da scrivere, con i file indicati in ogni passo.

## 6. Obiettivi

| Obiettivo | Soglia |
|---|---|
| Verdetto | uguale a quello del fine-tuning binario (BAcc e AUC entro 0,3 punti) |
| F1 sulle frasi | almeno 15 punti sopra la baseline lessicale (E-LEX, [07](07_baseline.md)); indicativo ≥ 0,75 su HoVer |
| Insieme completo su HoVer | ≥ 0,65 |
| Evidenza lontana | F1 con frasi gold a ≥ 20 frasi di distanza al massimo 5 punti sotto quello con frasi vicine |
| Sufficienza | ≥ 85% |
| Latenza | al massimo +3% del solo verdetto (modelli 4B: +3% o +2 ms) |
