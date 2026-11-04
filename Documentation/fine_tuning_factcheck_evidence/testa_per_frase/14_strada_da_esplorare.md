# Una strada da esplorare: insegnare alla testa quanto ogni frase regge il verdetto

## Il problema

Le frasi di evidenza annotate vengono da Wikipedia, pagine web e abstract biomedici. In produzione arrivano altri testi: riassunti, risposte di sistemi RAG, notizie, documenti tecnici. Su questi abbiamo il verdetto ma non sappiamo quali frasi sono l'evidenza, quindi la testa non impara a lavorarci.

Le etichette ricavate col modello (passo [03](03_dati_generati.md) §2) aiutano solo in parte: dicono quali frasi bastano, ma non quanto conta ciascuna, e costano una ricerca per riga.

## L'idea

Su **qualunque** riga supportata, di qualunque tipo di testo, si può misurare senza annotazioni quanto ogni frase regge il verdetto: si toglie la frase dal documento e si guarda di quanto scende il punteggio del verdetto.

```text
z            = margine del verdetto (sì − no) sul documento intero
z senza k    = margine col documento senza la frase k            (un forward per frase)
importanza_k = z − z senza k                                     quanto la frase k regge il verdetto
```

La testa impara a prevedere questo ordinamento in un solo forward. Così impara, sui testi di produzione, quali frasi reggono davvero il verdetto, che è anche la metrica di sufficienza.

Se la stessa informazione è ripetuta in due frasi, togliendone una il verdetto non cambia, e entrambe sembrano poco importanti. Per questo si tiene anche il gruppo trovato con l'ablazione del passo 3, che gestisce questo caso.

## Come farlo

1. **Righe**: positivi senza evidenza annotata da testi di produzione o simili (riassunti, risposte RAG dei dati di training del fine-tuning binario), più HoVer e SciFact per confrontare con le annotazioni. Niente righe di test né pagine di LLM-AggreFact.
2. **Calcolo con KnowLine-4B** (`evidence/label_counterfactual.py`): per ogni riga, un forward per frase tolta. KnowLine è causale, quindi i prompt sono uguali fino alla frase tolta e si riusa il calcolo della parte iniziale comune (con un modello bidirezionale non si potrebbe). Documenti fino a 80 frasi; oltre, solo per le 50 frasi con punteggio più alto della testa della fase 1a.
3. **Prima verifica, senza training**: su `dev_evid_select`, si usa l'importanza come punteggio delle frasi e si confronta con le annotazioni. Se non fa meglio di E-LEX, il segnale è troppo rumoroso e ci si ferma.
4. **Accordo fra modelli**: su 300 righe si rifà il calcolo con elephant-1. Se gli ordinamenti sono molto diversi, per gli altri modelli si ricalcola col modello stesso.
5. **Training** (fase 1a, esperimento H-CF): alla loss della testa si aggiunge un termine che avvicina l'ordinamento dei punteggi della testa a quello dell'importanza:

   ```text
   L_cf = KL( softmax(importanza / T) ‖ softmax(s / T') )      sulle frasi della riga
   L    = loss ZLPR (con il gruppo dell'ablazione come evidenza, peso 0,5)  +  η · L_cf        η ∈ {0,3; 1}
   ```

   T si sceglie in modo che le 3 frasi più importanti abbiano ~80% della massa.
6. **Valutazione** su tre set:
   - `dev_evid_select` (annotato): non deve peggiorare di più di 1 punto di F1;
   - `test_evid_ood` (WiCE);
   - un nuovo piccolo set di 200 righe di riassunti e risposte RAG, con evidenza indicata da un giudice LLM e controllata a mano.

## Quando tenerlo

Rispetto alla configurazione migliore senza questa aggiunta:
- sufficienza almeno +3 punti su tutti i set;
- F1 almeno +2 punti sul set di riassunti e RAG;
- F1 su `dev_evid_select` non più di 1 punto sotto.

## Rischi

| Rischio | Cosa fare |
|---|---|
| L'importanza segue le scorciatoie del modello (per esempio frasi con parole del claim che il modello sopravvaluta) | la verifica del punto 3; tenere sempre le annotazioni dove ci sono |
| Informazione ripetuta: importanza bassa per tutte le copie | il gruppo dell'ablazione come evidenza |
| Costo: un forward per frase | KnowLine-4B, riuso della parte comune, limite di 80 frasi |
| Se il modello viene riaddestrato (fase 2), l'importanza calcolata prima non vale più | usarla solo col modello congelato, o ricalcolarla |

## Perché vale la pena

È l'unico modo, fra quelli del piano, di avere un segnale di evidenza sui testi di produzione senza annotarli a mano, ed è legato direttamente a ciò che interessa all'utente: che le frasi mostrate bastino a reggere il verdetto. Il costo è tutto prima del rilascio e riusa codice già previsto; in produzione non cambia nulla.
