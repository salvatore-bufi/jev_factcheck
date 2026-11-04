# Fact checking con evidenza evidenziata — testa per token

## 1. Che cosa si vuole ottenere

Abbiamo un **decision model** fine-tunato come fact checker: riceve un documento e un claim e risponde con P(supportato), in **un solo passaggio del modello sul testo** (un *forward*), senza generare parole. È veloce perché fa un solo forward.

Vogliamo che, quando il claim è supportato, dica anche **quali parti del documento lo dimostrano**, senza perdere velocità: sempre un solo forward, nessuna parola generata.

Esempio di risposta:

```json
{"p_supported": 0.93, "supported": true,
 "evidence": [{"start": 0,    "end": 61,   "score": 0.91},
              {"start": 4210, "end": 4262, "score": 0.84}]}
```

- `start` / `end`: inizio e fine, in caratteri, di un tratto di testo del documento originale (`document[start:end]`).
- Se il claim non è supportato, `evidence` è vuota: non si indica né cosa manca né cosa lo contraddice.

Il caso difficile è l'**evidenza sparsa**: due parti lontane che solo **insieme** dimostrano il claim.

> Claim: "Nancy R. Howell teaches at a seminary in Kansas City."
> Inizio del documento: "Saint Paul School of Theology is a seminary located in Kansas City, Missouri."
> … 38 frasi che non c'entrano …
> Fine del documento: "Nancy R. Howell is Professor of Theology at Saint Paul School of Theology."
>
> Nessuna delle due basta da sola. Evidenza corretta: entrambe.

## 2. Perché "per token"

La testa dà un punteggio a **ogni token** del documento, e le parti mostrate sono i tratti di token sopra soglia. Non c'è nessuna divisione del documento in frasi, né in training né in produzione. Quindi:
- il risultato **non dipende da come il testo è diviso in frasi** (nessun NLTK, nessuna regola di punteggiatura da mantenere identica fra training e produzione);
- funziona anche su testo senza punteggiatura, elenchi, tabelle scritte come testo;
- può evidenziare **solo la parte utile** di una frase lunga.

Se all'utente si vogliono mostrare frasi intere, i tratti si allargano alla frase che li contiene **solo in visualizzazione**, con qualunque divisore: la testa non ne dipende.

## 3. Come funziona, passo per passo

### 3.1 Che cosa fa già il decision model

Il modello legge il prompt (documento, claim, domanda). Per **ogni token** e per **ogni layer** calcola un vettore di numeri (lo *stato nascosto*): 2.560 numeri nei modelli da 4B, 4.096 in quelli da 9B. Il verdetto si legge dall'**ultimo token del prompt** (la *posizione di risposta*): lì il modello dà un punteggio alla risposta "sì" e uno alla risposta "no", da cui P(supportato). Questa parte esiste già: è il modello del fine-tuning binario, addestrato con una LoRA sulla domanda di fact checking. **Non la tocchiamo.**

### 3.2 Che cosa aggiungiamo

Durante quel forward leggiamo alcuni dei vettori che il modello calcola comunque, e li diamo a una **piccola rete separata**, la *testa di evidenza* (4,5 milioni di parametri contro i 4–9 miliardi del modello), che dà un punteggio a ogni token del documento.

```text
                 ┌──────────────────────────────── prompt ─────────────────────────────────┐
                 │ Claim under review: <claim>                         ← preambolo (nuovo)   │
                 │ <documento: token 1 … token n>                      ← prompt del fine-    │
                 │ <claim> <domanda> <opzioni>                           tuning binario,     │
                 │ [ultimo token = posizione di risposta]                invariato           │
                 └──────────────────────────────────────────────────────────────────────────┘
                                                 │
                          UN forward del decision model (pesi congelati)
                                                 │
            ┌────────────────────────────────────┴───────────────────────────────┐
            │ per ogni token t del documento:                                     │ alla posizione di risposta:
            │   h[t]   vettori di 2 layer (23 e 31), ridotti a 256 numeri ciascuno │   punteggi "sì"/"no" → P(supportato)
            │   att[t] attenzione che la posizione di risposta dà al token:        │   q = vettore della posizione
            │          8 layer × 16 teste = 128 numeri                             │
            └────────────────────────────────────┬───────────────────────────────┘
                                                 │
   ┌──────────────────────────────────── TESTA DI EVIDENZA ─────────────────────────────────────┐
   │ A. vettore del token     x[t] = proiezione di [h[t], att[t], q] → 256 numeri                  │
   │ B. contesto locale       3 convoluzioni sui token vicini (finestra di 29 token)               │
   │ C. contesto globale      media per blocchi fissi di 16 token → transformer non causale        │
   │                          a 2 layer su [q, blocco 1 … blocco B] → ogni blocco vede tutto       │
   │ D. punteggi              s[t] per ogni token; s0 da q = soglia di QUESTO documento            │
   └──────────────────────────────────────────────┬───────────────────────────────────────────────┘
                                                  │
   DECODIFICA: token con s[t] > s0 → tratti consecutivi → buchi piccoli chiusi, tratti minuscoli tolti
               → al massimo 5 tratti → inizio e fine in caratteri sul documento originale
                                                  │
   mostrati solo se P(supportato) > soglia del verdetto
```

L'architettura completa, con forme dei tensori, parametri e costo di ogni parte, è nel passo [08](08_testa_e_loss.md).

### 3.3 Perché ogni pezzo

| Pezzo | Perché serve |
|---|---|
| vettori da 2 layer, non solo dall'ultimo | l'ultimo layer è specializzato a predire la risposta; un layer intermedio rappresenta meglio il contenuto dei token |
| attenzione della posizione di risposta | dice su quali token il modello si è basato per decidere: un indizio forte, già calcolato |
| `q` sommato a ogni token | dice alla testa qual è il claim e cosa ha deciso il modello |
| contesto locale | un token da solo non basta: una parola è evidenza perché sta in una certa frase |
| contesto globale a blocchi | l'evidenza sparsa richiede che parti lontane si vedano; un'attenzione piena su migliaia di token costa troppo, su ~100–400 blocchi è trascurabile. I blocchi sono fissi (16 token per posizione): nessuna dipendenza dal significato o dalla punteggiatura |
| soglia `s0` del documento | non c'è una soglia unica da tarare; in un caso multi-hop la soglia può scendere per far passare la parte "ponte" |

### 3.4 Perché il preambolo

Nei modelli **causali** (come KnowLine-4B) ogni token vede solo ciò che lo precede. Nel prompt del fine-tuning binario il claim viene **dopo** il documento: quando il modello legge il documento non sa ancora cosa deve verificare, e i vettori dei token del documento non "parlano" del claim. Mettendo una riga `Claim under review: <claim>` **prima** del documento, ogni token è letto sapendo cosa si verifica. Il resto del prompt resta identico, compreso il claim dopo il documento dove il verdetto è abituato a vederlo. Dettagli nel passo [05](05_formato_prompt.md).

### 3.5 Che cosa si addestra

Solo la testa. Il modello resta congelato, quindi **il verdetto non cambia**. Le etichette sono le frasi di evidenza annotate nei dataset (SciFact, WiCE, HoVer), trasformate in intervalli di caratteri e poi in token: un token è "evidenza" se sta dentro una di quelle frasi. La loss è pensata per il fatto che le annotazioni sono a livello di frase: chiede che ogni frase gold sia colpita e che nulla si accenda fuori, non che ogni token della frase sia acceso.

Siccome il modello è congelato, i suoi vettori si calcolano **una volta** e si salvano su disco (ridotti a 256 numeri per token); poi ogni prova di training della testa dura minuti.

## 4. I modelli

I 10 decision model dello studio focalizzato, tutti costruiti su **Qwen3.5**: 32 layer, di cui 8 ad attenzione piena (layer 3, 7, 11, 15, 19, 23, 27, 31) e 24 Gated DeltaNet (attenzione lineare, sempre causale).

| Modello | Ruolo | d | Attenzione nei layer pieni |
|---|---|---:|---|
| **KnowLine-4B** (`PelaAI/KnowLine-4B-Gen5.5`) | tutte le prove si fanno qui | 2.560 | causale |
| **elephant-1** (`baselabs/elephant-1`) | conferma a 9B | 4.096 | bidirezionale a blocchi |
| Jeffan-4B, d3-flash, OneJev1.6-9B, Nimble-9B-v3 | estensione | 2.560 / 4.096 | Jeffan, d3: bidirezionale; OneJev, Nimble: causale |
| StartLux 9B/4B, pplx-decider-flash, Clef-Flash | estensione, più lavoro | 2.560 / 4.096 | StartLux, pplx: bidirezionale; Clef: come caricato |

Ogni modello si esegue **con il codice dei suoi autori** (prompt, maschera di attenzione, lettura della risposta), come nei benchmark della repo. Dettagli nel passo [01](01_ambiente_e_modelli.md).

## 5. Glossario

| Termine | Significato |
|---|---|
| forward | un passaggio del modello su tutto il prompt |
| stato nascosto, vettore | i numeri che il modello calcola per un token in un layer |
| posizione di risposta | il token del prompt dove si legge il verdetto (l'ultimo) |
| token del documento | i token del prompt che appartengono al documento |
| testa di evidenza | la piccola rete che dà un punteggio a ogni token del documento |
| `s[t]`, `s0` | punteggio del token t; soglia del documento |
| blocco | 16 token consecutivi del documento: unità interna della testa per il contesto globale |
| tratto | sequenza di token consecutivi mostrati come evidenza, convertita in `[inizio, fine)` di caratteri |
| frase gold | frase di evidenza annotata nel dataset, come intervallo di caratteri |
| colpita | una frase gold è colpita se almeno metà dei suoi caratteri è dentro i tratti mostrati |
| insieme completo | tutte le frasi gold di una riga colpite |
| sufficienza | il documento ridotto ai soli tratti mostrati fa ancora dire "supportato" al modello |
| fase 1a / 1b / 2 | training della testa su vettori salvati / con il modello in memoria / insieme a una LoRA del modello |

## 6. Sequenza di lavoro

| # | Passo | File | Prodotto |
|---|---|---|---|
| 1 | Modelli e hook per token | [01](01_ambiente_e_modelli.md) | forward con cattura dei vettori e dell'attenzione per token |
| 2 | Dati dalle fonti | [02](02_dati_fonti.md) | `dataset/evidence/*.jsonl` con le frasi gold in caratteri |
| 3 | Dati generati (facoltativo per le prime prove) | [03](03_dati_generati.md) | evidenza esatta, etichette dal modello, evidenza spostata |
| 4 | Split e mix | [04](04_split_e_mix.md) | set di valutazione congelati, mix di training |
| 5 | Prompt con il preambolo | [05](05_formato_prompt.md) | preambolo scelto, verdetto verificato |
| 6 | Dai caratteri ai token, estrazione | [06](06_rendering_ed_estrazione.md) | etichette per token, vettori su disco |
| 7 | Baseline | [07](07_baseline.md) | numeri da battere |
| 8 | Architettura, loss, decodifica | [08](08_testa_e_loss.md) | `evidence/token_head.py`, `token_losses.py`, `decode.py` |
| 9 | Training | [09](09_training.md) | teste addestrate |
| 10 | Valutazione | [10](10_valutazione.md) | metriche e latenza |
| 11 | Esperimenti | [11](11_esperimenti.md) | configurazione finale |
| 12 | Server e rilascio | [12](12_export_e_servizio.md) | server che restituisce verdetto e tratti |
| — | Checklist | [13](13_checklist.md) | |
| — | Una strada da esplorare | [14](14_strada_da_esplorare.md) | |

Codice già pronto nella repo: `evidence/data/` (costruzione dei dataset dal passo 2) e `tests/test_evidence_data.py`. Il resto è da scrivere, con i file indicati in ogni passo.

## 7. Obiettivi

| Obiettivo | Soglia |
|---|---|
| Verdetto | uguale a quello del fine-tuning binario (BAcc e AUC entro 0,3 punti) |
| Frasi gold colpite | ≥ 0,80 su HoVer |
| Insieme completo su HoVer | ≥ 0,65 |
| F1 sui caratteri | almeno 15 punti sopra la baseline lessicale (E-LEX, [07](07_baseline.md)) |
| Evidenza lontana | frasi colpite con evidenza a ≥ 20 frasi di distanza al massimo 5 punti sotto quelle vicine |
| Documento senza punteggiatura | frasi colpite al massimo 5 punti sotto il documento normale |
| Sufficienza | ≥ 85% |
| Tratti | in media al massimo 5 per riga; nessun tratto sotto 4 token |
| Latenza | al massimo +3% del solo verdetto (modelli 4B: +3% o +3 ms) |
