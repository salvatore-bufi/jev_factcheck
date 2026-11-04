# Fact checking con evidenza evidenziata — piano operativo

## 1. Che cosa si vuole ottenere

Oggi, nel piano binario ([../fine_tuning_factcheck/](../fine_tuning_factcheck/00_overview.md)), un decision model riceve un documento e un claim e risponde con una probabilità: P(supportato). Lo fa con **un solo passaggio del modello sul testo** (un *forward*), senza generare parole: per questo è veloce.

Vogliamo che, quando il claim è supportato, il sistema dica anche **quali frasi del documento lo dimostrano**, senza perdere velocità: sempre un solo forward, nessuna parola generata.

Esempio di risposta:

```json
{"p_supported": 0.93, "supported": true,
 "evidence": [{"sentence": 0,  "start": 0,    "end": 61,   "score": 0.91},
              {"sentence": 39, "start": 4210, "end": 4262, "score": 0.84}]}
```

- `sentence`: numero della frase nel documento (da 0); `start`/`end`: dove inizia e finisce, in caratteri, nel testo originale.
- Se il claim non è supportato, `evidence` è vuota: non si mostra né cosa manca né cosa lo contraddice.

Il caso difficile è l'**evidenza sparsa**: due frasi lontane che solo **insieme** dimostrano il claim.

> Claim: "Nancy R. Howell teaches at a seminary in Kansas City."
> Frase 1: "Saint Paul School of Theology is a seminary located in Kansas City, Missouri."
> … 38 frasi che non c'entrano …
> Frase 40: "Nancy R. Howell is Professor of Theology at Saint Paul School of Theology."
>
> Nessuna delle due basta da sola. Evidenza corretta: frasi 1 e 40.

## 2. Come funziona, passo per passo

### 2.1 Che cosa fa già il decision model

Il modello legge il prompt. Per **ogni token** e per **ogni layer** calcola un vettore di numeri (lo *stato nascosto*, lungo 2.560 numeri nei modelli da 4B e 4.096 in quelli da 9B). Nei modelli causali ogni token "vede" solo i token che lo precedono; nei modelli bidirezionali, negli 8 layer ad attenzione piena, vede anche quelli dopo (§3).

Il verdetto si legge dall'**ultimo token del prompt** (la posizione di risposta, chiamata `answer_pos`): lì il modello assegna un punteggio alla risposta "sì" e uno alla risposta "no", e da questi si ricava P(supportato). Questa parte esiste già ed è stata addestrata nel piano binario. **Non la tocchiamo.**

### 2.2 Che cosa aggiungiamo

Mentre il modello fa quel forward, leggiamo alcuni dei vettori che calcola comunque, e li diamo a una **piccola rete separata** (la *testa di evidenza*, circa 3 milioni di parametri contro i 4–9 miliardi del modello). La testa dà un punteggio a ogni frase.

```text
                 ┌───────────────────────────── prompt ──────────────────────────────┐
                 │ Claim under review: <claim>                                        │  ← preambolo (nuovo)
                 │ <documento: frase 0, frase 1, …, frase n>                          │  ← prompt del piano
                 │ <claim> <domanda> <opzioni>                                        │    binario, invariato
                 │ [ultimo token = posizione di risposta]                             │
                 └────────────────────────────────────────────────────────────────────┘
                                              │
                           un forward del decision model (pesi congelati)
                                              │
              ┌───────────────────────────────┴───────────────────────────────┐
              │                                                               │
   per ogni frase k: un "riassunto"                         alla posizione di risposta:
   (vettori letti dentro la frase, §2.3)                    • punteggi di "sì"/"no" → P(supportato)
              │                                             • il vettore q (§2.3)
              └──────────────────────────────┬────────────────────────────────┘
                                             │
                               TESTA DI EVIDENZA (l'unica parte che si addestra)
                    1. riduce ogni vettore a 256 numeri
                    2. 2 layer di transformer in cui tutte le frasi e q si vedono a vicenda
                    3. un punteggio per ogni frase (s_k) e un punteggio-soglia (s_0, da q)
                                             │
                     se P(supportato) > soglia del verdetto:
                     mostra le frasi con s_k > s_0 (almeno 1, al massimo 5)
```

### 2.3 Che cosa entra nella testa

Per ogni frase k del documento si leggono, dallo stesso forward:

| Nome | Che cos'è | Perché serve |
|---|---|---|
| `h_end[k]` | il vettore dell'**ultimo token** della frase | riassume la frase letta fin lì |
| `h_mean[k]` | la **media** dei vettori di tutti i token della frase | non dipende dal punto finale; tiene conto di tutta la frase |
| `att[k]` | quanto la posizione di risposta "guarda" i token della frase (pesi di attenzione), negli 8 layer in cui il modello guarda tutto il testo | dice su quali frasi il modello si è basato per decidere |

E dalla posizione di risposta:

| Nome | Che cos'è |
|---|---|
| `q` | il vettore della posizione di risposta: ha visto tutto il prompt, quindi "sa" il verdetto e quali frasi servono |

I vettori `h_end`, `h_mean` e `q` si leggono da **4 layer** diversi (non solo dall'ultimo) e la testa impara come pesarli.

### 2.4 Che cosa fa la testa

1. **Comprime** ogni vettore a 256 numeri (una moltiplicazione per una matrice).
2. **Fa comunicare** le frasi: due layer di transformer *non causale*, cioè ogni frase vede tutte le altre e `q`, in avanti e all'indietro. Questo risolve l'evidenza sparsa: nel modello la frase 1 non può vedere la frase 40 (viene dopo: vale per i modelli causali, e per i 24 layer DeltaNet di tutti), nella testa sì.
3. **Dà i punteggi**: `s_k` per ogni frase k, e `s_0` calcolato da `q`, che fa da **soglia di quella riga**. Si mostrano le frasi con `s_k > s_0`. La soglia cambia da documento a documento: se servono 3 frasi può abbassarsi per farle passare tutte.

### 2.5 Perché il preambolo

Nei modelli causali ogni token vede solo ciò che lo precede. Nel prompt del piano binario il claim viene **dopo** il documento, quindi quando il modello legge le frasi non sa ancora che cosa deve verificare: i vettori delle frasi non dipendono dal claim. Nei modelli bidirezionali le frasi vedono il claim negli 8 layer pieni anche senza preambolo, ma non nei 24 layer DeltaNet, che restano causali: il preambolo aiuta anche lì, ma meno (lo misura l'esperimento H4).

Mettendo una riga `Claim under review: <claim>` **prima** del documento, ogni frase è letta sapendo che cosa si verifica. Il resto del prompt resta identico a quello del piano binario: il claim compare ancora dopo il documento, nel punto in cui il verdetto è stato addestrato a vederlo. Così il verdetto si può riusare (si verifica al passo [05](05_formato_prompt.md)).

### 2.6 Che cosa si addestra

Solo la testa. Il modello resta congelato, quindi **il verdetto non cambia** e non si rischia di peggiorarlo. La testa impara da documenti in cui si sa quali frasi sono l'evidenza (FEVER, HoVer e dati generati, passi [02](02_dati_fonti.md)–[03](03_dati_generati.md)), e da documenti con claim non supportati, in cui nessuna frase deve superare la soglia.

Siccome il modello è congelato, i suoi vettori si calcolano **una volta sola** e si salvano su disco ([06](06_rendering_ed_estrazione.md)); poi ogni prova di training della testa dura minuti. Solo se questo non basta si addestra anche il modello (con una LoRA), con controlli sul verdetto ([09](09_training.md) §6).

## 3. I modelli

I modelli sono quelli del piano binario ([../fine_tuning_factcheck/00_overview.md](../fine_tuning_factcheck/00_overview.md) §2), cioè i 10 decision model dello studio focalizzato. Tutti sono Qwen3.5: 32 layer, di cui 8 ad attenzione piena (3, 7, 11, 15, 19, 23, 27, 31) e 24 Gated DeltaNet, che portano avanti un riassunto del testo letto e restano sempre causali.

| Modello | Ruolo | Vettore (d) | Attenzione nei layer pieni |
|---|---|---:|---|
| **KnowLine-4B** | si fanno qui tutte le prove (4B, causale, lettura semplice) | 2.560 | causale |
| **elephant-1** | conferma a 9B; modello principale del piano binario | 4.096 | bidirezionale a blocchi |
| Jeffan-4B, d3-flash, OneJev1.6-9B, Nimble-9B-v3 | estensione (codice degli autori in transformers) | 2.560 / 4.096 | Jeffan, d3: bidirezionale; OneJev, Nimble: causale |
| StartLux 9B/4B, pplx-decider-flash, Clef-Flash | estensione, con più lavoro (StartLux gira solo con vLLM; pplx e Clef hanno una testa propria) | 2.560 / 4.096 | StartLux, pplx: bidirezionale; Clef: come caricato |

Il verdetto di ogni modello è quello del piano binario (stessa domanda, stessa soglia).

## 4. Glossario

| Termine | Significato |
|---|---|
| forward | un passaggio del modello su tutto il prompt |
| stato nascosto, vettore | i numeri che il modello calcola per un token in un layer |
| posizione di risposta (`answer_pos`) | l'ultimo token del prompt, dove si legge il verdetto |
| testa di evidenza | la piccola rete aggiunta che dà un punteggio alle frasi |
| `s_k`, `s_0` | punteggio della frase k; soglia della riga |
| gold | frasi di evidenza annotate (da persone o note per costruzione) |
| insieme completo | tutte le frasi gold mostrate (per l'evidenza sparsa conta questo) |
| sufficienza | le frasi mostrate, da sole come documento, fanno dire "supportato" al modello |
| fase 1a / 1b / 2 | training della testa su vettori salvati / con il modello in memoria / insieme a una LoRA del modello |

## 5. Sequenza di lavoro

| # | Passo | File | Prodotto |
|---|---|---|---|
| 1 | Caricare i modelli e leggere vettori e attenzione | [01](01_ambiente_e_modelli.md) | `evidence/models.py` verificato |
| 2 | Documenti divisi in frasi, con le frasi di evidenza | [02](02_dati_fonti.md) | righe da FEVER, HoVer, SciFact, WiCE |
| 3 | Altri dati con evidenza nota | [03](03_dati_generati.md) | dati generati, etichette ricavate col modello, evidenza spostata lontano |
| 4 | Split e mix di training | [04](04_split_e_mix.md) | set di valutazione congelati; mix di training |
| 5 | Prompt con il preambolo | [05](05_formato_prompt.md) | preambolo scelto; verdetto verificato |
| 6 | Calcolo e salvataggio dei vettori | [06](06_rendering_ed_estrazione.md) | vettori su disco |
| 7 | Metodi semplici di confronto | [07](07_baseline.md) | numeri da battere |
| 8 | Testa e loss | [08](08_testa_e_loss.md) | codice testato |
| 9 | Training | [09](09_training.md) | teste addestrate |
| 10 | Valutazione | [10](10_valutazione.md) | metriche e latenza |
| 11 | Esperimenti | [11](11_esperimenti.md) | configurazione finale per ogni modello |
| 12 | Server e rilascio | [12](12_export_e_servizio.md) | server che restituisce verdetto ed evidenza |
| — | Checklist e problemi frequenti | [13](13_checklist.md) | |
| — | Una strada da esplorare | [14](14_strada_da_esplorare.md) | |

Codice in `evidence/` (pacchetto Python nella radice della repo), dati in `data/evidence/`. Sul server i job GPU si lanciano così (il log va rediretto dentro il comando):

```sh
gpu-run --batch --name ev-extract-knowline -- bash -c \
  'python -m evidence.extract --model knowline-4b --mix ev_head > logs/ev-extract-knowline.out 2>&1'
```

## 6. Obiettivi

| Obiettivo | Soglia |
|---|---|
| Verdetto | uguale a quello del piano binario (BAcc e AUC entro 0,3 punti) |
| F1 sulle frasi | almeno 15 punti sopra il metodo lessicale (E-LEX, [07](07_baseline.md)); indicativo ≥ 0,75 su FEVER e HoVer |
| Insieme completo su HoVer (evidenza su più pagine) | ≥ 0,65 |
| Evidenza lontana | F1 con frasi a ≥ 20 frasi di distanza al massimo 5 punti sotto quello con frasi vicine |
| Sufficienza | ≥ 85% |
| Latenza | al massimo +3% rispetto al solo verdetto (modelli 4B: +3% o +2 ms, il maggiore dei due) |
