# Fact checking con evidenza evidenziata — testa per token

## 1. Che cosa si vuole ottenere

Come nel piano per frase ([../testa_per_frase/00_overview.md](../testa_per_frase/00_overview.md)): un decision model che, dato un documento e un claim, restituisce in **un solo forward** P(supportato) e, se il claim è supportato, **le parti del documento che lo dimostrano**.

La differenza è l'unità di uscita. Qui la testa dà un punteggio a **ogni token** del documento, non a ogni frase, e l'uscita sono **tratti di testo** (inizio e fine in caratteri):

```json
{"p_supported": 0.93, "supported": true,
 "evidence": [{"start": 0,    "end": 61,   "score": 0.91},
              {"start": 4210, "end": 4262, "score": 0.84}]}
```

Perché: così il sistema **non dipende da come il testo è diviso in frasi**. Nessun divisore (NLTK o altro) serve al modello, né in training né in produzione. Funziona anche su testo senza punteggiatura, elenchi, tabelle in testo; se una frase lunga contiene l'evidenza solo in una sua parte, si può evidenziare solo quella.

Se poi si vogliono mostrare frasi intere all'utente, i tratti si "allargano" alla frase con qualunque divisore, solo in visualizzazione: la testa non ne dipende (§2.5).

## 2. Come funziona, passo per passo

### 2.1 Il forward non cambia

Il decision model fa **un solo forward su tutto il prompt** (preambolo con il claim, documento intero, claim, domanda), esattamente come nel piano binario. Il verdetto si legge dall'ultimo token, che ha visto tutto il documento. Non c'è nessun passaggio per frase o per token.

### 2.2 Che cosa entra nella testa

Dallo stesso forward, per **ogni token t del documento**:

| Nome | Che cos'è |
|---|---|
| `h[t]` | il vettore del token in alcuni layer (ridotto a 256 numeri, §2.3) |
| `att[t]` | quanto la posizione di risposta "guarda" quel token, negli 8 layer ad attenzione piena × 16 teste = 128 numeri |

e dalla posizione di risposta:

| Nome | Che cos'è |
|---|---|
| `q` | il vettore della posizione di risposta: ha visto tutto, sa il verdetto e quali parti servono |

### 2.3 Che cosa fa la testa

```text
            per ogni token t del documento                          posizione di risposta
     h[t] (2–4 layer, ridotti a 256)   att[t] (128)                         q
                 │                         │                                │
                 └──────── somma di proiezioni ─────────┐                   │
                                                        ▼                   │
   1. CONTESTO LOCALE   3 convoluzioni 1D sui token (finestra ~30 token): capisce parole e frasi vicine
                                                        │
   2. CONTESTO GLOBALE  media a blocchi fissi di 16 token → transformer NON causale (2 layer) su [q, blocco 1 … blocco B]
                        → ogni blocco riceve il contesto di tutto il documento e di q → torna ai suoi token
                                                        │
   3. PUNTEGGI          s[t] per ogni token;  s0 dalla posizione di risposta = soglia di questo documento
                                                        │
   4. DECODIFICA        token con s[t] > s0 → tratti consecutivi → si chiudono buchi piccoli, si tolgono tratti minuscoli
                        → offset in caratteri sul testo originale
```

- **Contesto locale**: un token da solo non basta (una parola è evidenza perché sta in una certa frase); le convoluzioni guardano i token vicini.
- **Contesto globale**: l'evidenza sparsa (frase 1 e frase 40 insieme) richiede che parti lontane del documento si vedano. Un'attenzione piena su migliaia di token costa troppo; i **blocchi fissi di 16 token** (decisi solo dalla posizione, non dal significato) la rendono economica: un documento di 2.000 token diventa 125 blocchi. I blocchi sono interni alla testa: l'uscita resta per token.
- **Soglia del documento** `s0`: come nel piano per frase, si mostra ciò che supera la soglia di quel documento; nei casi multi-hop la soglia può scendere per far passare la parte "ponte".

### 2.4 Perché il preambolo

Identico al piano per frase ([../testa_per_frase/05_formato_prompt.md](../testa_per_frase/05_formato_prompt.md)): una riga `Claim under review: <claim>` prima del documento, così nei modelli causali (KnowLine) i token del documento sono letti sapendo cosa si verifica. Il resto del prompt è quello del piano binario.

### 2.5 Che cosa si mostra all'utente

La testa restituisce tratti di caratteri. Due modi di mostrarli, scelti dal client:
- **tratti esatti** (default);
- **frasi intere**: ogni tratto si allarga alla frase che lo contiene, con qualunque divisore. Il divisore sta nel client o nel server, fuori dal modello: cambiarlo non tocca la testa.

### 2.6 Che cosa si addestra

Solo la testa, a modello congelato: il verdetto non cambia. Le etichette sono le frasi di evidenza delle fonti, trasformate in intervalli di caratteri e poi in token (un token è "evidenza" se cade dentro una di quelle frasi). Le frasi delle fonti servono solo a creare le etichette: la testa impara token per token.

## 3. Differenze con il piano per frase

| | Testa per frase | Testa per token |
|---|---|---|
| Unità | frase (da un divisore) | token del modello |
| Divisore in produzione | sì, identico a quello del training | **no** |
| Uscita | indici di frase | tratti di caratteri |
| Elementi per documento | decine (mediana 46 su HoVer) | migliaia (mediana ~1.500 su HoVer) |
| Aggregatore | transformer su tutte le frasi | convoluzioni locali + transformer su blocchi fissi di token |
| Etichette | frasi gold sulla nostra divisione | frasi gold come intervalli di caratteri, poi token |
| Rumore delle etichette | basso | più alto: in una frase gold anche le parole irrilevanti sono "evidenza" (§[04](04_split_e_mix.md), loss in [08](08_testa_e_loss.md)) |
| Salvataggio dei vettori | n frasi × d | n token × 256 (ridotti con una proiezione fissa, [06](06_rendering_ed_estrazione.md)) |

## 4. I modelli

Gli stessi del piano per frase ([../testa_per_frase/01_ambiente_e_modelli.md](../testa_per_frase/01_ambiente_e_modelli.md)): i 10 decision model dello studio, tutti Qwen3.5 (32 layer, 8 ad attenzione piena ai layer 3, 7, …, 31). Si itera su **KnowLine-4B** (d = 2.560), si conferma su **elephant-1** (d = 4.096), poi gli altri.

## 5. Glossario

| Termine | Significato |
|---|---|
| token del documento | i token del prompt che appartengono al documento (per KnowLine, nelle richieste lette due volte, quelli della seconda copia) |
| tratto (span) | sequenza di token consecutivi segnati come evidenza, convertita in `[inizio, fine)` di caratteri |
| blocco | 16 token consecutivi del documento, unità interna della testa per il contesto globale |
| `s[t]`, `s0` | punteggio del token t; soglia del documento |
| frase gold | frase di evidenza indicata dalla fonte (SciFact, WiCE, HoVer), come intervallo di caratteri |
| colpita | una frase gold è "colpita" se almeno metà dei suoi caratteri è dentro i tratti mostrati |
| insieme completo | tutte le frasi gold di una riga colpite |
| sufficienza | il documento ridotto ai tratti mostrati fa ancora dire "supportato" al modello |

## 6. Sequenza di lavoro

| # | Passo | File | Prodotto |
|---|---|---|---|
| 1 | Modelli e hook per token | [01](01_ambiente_e_modelli.md) | `evidence/models.py` con cattura per token |
| 2 | Dati dalle fonti (già costruiti) + intervalli gold di caratteri | [02](02_dati_fonti.md) | `dataset/evidence/*.jsonl` con `gold_char_sets` |
| 3 | Dati generati | [03](03_dati_generati.md) | righe `construct`, `ablation`, sparse |
| 4 | Split e mix | [04](04_split_e_mix.md) | set congelati, mix `ev_tok_head` |
| 5 | Prompt con preambolo | [05](05_formato_prompt.md) | come nel piano per frase |
| 6 | Rendering, etichette per token, estrazione | [06](06_rendering_ed_estrazione.md) | vettori ridotti su disco |
| 7 | Baseline | [07](07_baseline.md) | numeri da battere |
| 8 | Testa, loss, decodifica | [08](08_testa_e_loss.md) | `evidence/token_head.py`, `token_losses.py`, `decode.py` |
| 9 | Training | [09](09_training.md) | teste addestrate |
| 10 | Valutazione | [10](10_valutazione.md) | metriche su caratteri e su frasi |
| 11 | Esperimenti | [11](11_esperimenti.md) | configurazione finale |
| 12 | Server e rilascio | [12](12_export_e_servizio.md) | server senza divisore |
| — | Checklist | [13](13_checklist.md) | |
| — | Una strada da esplorare | [14](14_strada_da_esplorare.md) | |

## 7. Obiettivi

| Obiettivo | Soglia |
|---|---|
| Verdetto | uguale al piano binario (BAcc e AUC entro 0,3 punti) |
| F1 sui caratteri (tratti mostrati contro frasi gold) | ≥ E-LEX + 15 punti |
| Frasi gold colpite (recall) | ≥ 0,80 su HoVer |
| Insieme completo su HoVer | ≥ 0,65 |
| Evidenza lontana | F1 con frasi gold a ≥ 20 frasi di distanza al massimo 5 punti sotto quello con frasi vicine |
| Sufficienza | ≥ 85% |
| Tratti mostrati per riga | in media al massimo 5; nessun tratto sotto 4 token |
| Latenza | al massimo +3% del solo verdetto (modelli 4B: +3% o +3 ms) |
