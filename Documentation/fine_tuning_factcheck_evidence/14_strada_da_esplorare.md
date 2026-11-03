# Una strada da esplorare: insegnare alla testa quanto ogni frase regge il verdetto

## Il problema

Le frasi di evidenza annotate vengono quasi tutte da Wikipedia (FEVER, HoVer). In produzione arrivano altri testi: riassunti, risposte di sistemi RAG, notizie, documenti tecnici. Su questi abbiamo il verdetto ma non sappiamo quali frasi sono l'evidenza, quindi la testa non impara a lavorarci.

Le etichette ricavate col modello (passo [03](03_dati_generati.md) §2) aiutano solo in parte: dicono quali frasi bastano, ma non quanto conta ciascuna, e costano una ricerca per riga.

## L'idea

Su **qualunque** riga supportata, di qualunque tipo di testo, si può misurare senza annotazioni quanto ogni frase regge il verdetto: si toglie la frase dal documento e si guarda di quanto scende il punteggio del verdetto.

```text
z            = punteggio del verdetto sul documento intero
z senza k    = punteggio col documento senza la frase k        (un forward per frase)
importanza_k = z − z senza k                                     quanto la frase k regge il verdetto
```

La testa impara a prevedere questo ordinamento in un solo forward. Così impara, sui testi di produzione, quali frasi reggono davvero il verdetto, che è anche la metrica di sufficienza.

Se la stessa informazione è ripetuta in due frasi, togliendone una il verdetto non cambia, e entrambe sembrano poco importanti. Per questo si tiene anche il gruppo trovato con l'ablazione del passo 3, che gestisce questo caso.

## Come farlo

1. **Righe**: i positivi del training del piano binario non Wikipedia (C2D, D2C, SummEdits, AVeriTeC e le altre fonti), più FEVER e HoVer per confrontare con le annotazioni. Niente righe di test né fonti di AggreFact.
2. **Calcolo con d1-3B** (`evidence/label_counterfactual.py`): per ogni riga, un forward per frase tolta. I prompt sono uguali fino alla frase tolta, quindi si riusa il calcolo della parte comune. Documenti fino a 80 frasi; oltre, solo per le 50 frasi più promettenti.
3. **Prima verifica, senza training**: su `dev_evid_select`, si usa l'importanza come punteggio delle frasi e si confronta con le annotazioni. Se non fa meglio di E-LEX, il segnale è troppo rumoroso e ci si ferma.
4. **Accordo fra modelli**: su 300 righe si rifà il calcolo con JPT-9B. Se gli ordinamenti sono molto diversi, per JPT e Winnow si ricalcola col modello stesso.
5. **Training** (fase 1a, esperimento H-CF): alla loss della testa si aggiunge un termine che avvicina l'ordinamento dei punteggi della testa a quello dell'importanza (KL fra le due distribuzioni normalizzate sulle frasi), con peso 0,3 e 1. Il gruppo trovato con l'ablazione fa da evidenza, con peso 0,5.
6. **Valutazione** su tre set:
   - `dev_evid_select` (Wikipedia): non deve peggiorare di più di 1 punto di F1;
   - `test_evid_ood` (SciFact, WiCE);
   - un nuovo piccolo set di 200 righe di riassunti e risposte RAG dal dev del piano binario, con evidenza indicata dal giudice e controllata a mano.

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
| Costo: un forward per frase | d1-3B, riuso della parte comune, limite di 80 frasi |
| Se il modello viene riaddestrato (fase 2), l'importanza calcolata prima non vale più | usarla solo col modello congelato, o ricalcolarla |

## Perché vale la pena

È l'unico modo, fra quelli del piano, di avere un segnale di evidenza sui testi di produzione senza annotarli a mano, ed è legato direttamente a ciò che interessa all'utente: che le frasi mostrate bastino a reggere il verdetto. Il costo è tutto prima del rilascio e riusa codice già previsto; in produzione non cambia nulla.
