# Una strada da esplorare: distillare nella testa l'effetto controfattuale delle frasi

## Il problema che risolve

Le etichette di evidenza buone (gold) vengono quasi tutte da Wikipedia (FEVER, HoVer). I documenti su cui il modello lavora in produzione sono altri: riassunti, risposte RAG, notizie, documenti tecnici. Su quei domini abbiamo solo il verdetto, non l'evidenza. Le etichette per ablazione ([03](03_dati_generati.md) §3) coprono questo buco solo in parte: danno un insieme sì/no per riga, non dicono quanto conta ogni frase, e costano una ricerca greedy.

C'è un segnale che si può calcolare **su qualunque riga positiva, in qualunque dominio, senza annotazioni**: di quanto cambia il verdetto del modello togliendo ogni frase. Se la testa impara a prevederlo in un forward, impara cosa regge davvero il verdetto (la sufficienza, uno degli obiettivi) e lo impara sul dominio di produzione.

## L'idea

Per ogni riga positiva giudicata supportata, con il verdetto del modello e il prompt di produzione:

```text
z_full     = margine del verdetto sul documento intero
z_{−k}     = margine col documento senza la frase k            (n forward)
Δ_k        = z_full − z_{−k}                                    quanto la frase k regge il verdetto
```

Più l'insieme greedy dell'ablazione ([03](03_dati_generati.md) §3), che gestisce il caso in cui l'evidenza è ripetuta (togliendo una delle due copie Δ ≈ 0 per entrambe).

La testa riceve, oltre alla loss principale, un termine che allinea l'ordinamento dei suoi punteggi a quello di Δ:

```text
L_cf = KL( softmax(Δ / T) ‖ softmax(s / T') )      sulle frasi della riga, T scelto in modo che le 3 frasi con Δ più alto abbiano ~80% della massa
L    = zlpr_loss (con l'insieme greedy come gold, peso 0,5)  +  η · L_cf        η ∈ {0,3; 1}
```

## Come farlo

1. **Righe**: i positivi del mix del piano binario fuori da Wikipedia (C2D, D2C, SummEdits, AVeriTeC e le altre fonti di training del piano binario), più quelli di FEVER e HoVer per poter confrontare Δ con le gold. Esclusi tutti i dati di test e le fonti di AggreFact.
2. **Calcolo di Δ con d1-3B** (`evidence/label_counterfactual.py`): per ogni riga, i prompt col documento senza la frase k condividono il prefisso fino alla frase k−1: si ordinano per k e si riusa la cache del prefisso. Si limita a documenti ≤ 80 frasi; oltre, si calcola Δ solo per le 40 frasi con punteggio più alto della testa della fase 1a più 10 a caso.
3. **Controllo del segnale**, prima di addestrare: su `dev_evid_select`, AUPRC di Δ come punteggio di evidenza rispetto alle gold. Se Δ non supera E-LEX, il segnale è troppo rumoroso (o il modello usa scorciatoie): ci si ferma qui.
4. **Accordo fra modelli**: Δ con JPT-9B su 300 righe; correlazione di rango per riga con Δ di d1. Se bassa, per JPT e Winnow si ricalcola Δ col modello stesso.
5. **Training**: fase 1a ([09](09_training.md) §1) con un nuovo campo `delta` nel mix; esperimento H-CF con η = 0,3 e 1, sopra la configurazione vincente di [11](11_esperimenti.md).
6. **Valutazione** in tre punti:
   - `dev_evid_select` (gold Wikipedia): non deve scendere di più di 1 punto di F1;
   - `test_evid_ood` (SciFact, WiCE): F1 e insieme completo;
   - un piccolo blocco nuovo sul dominio di produzione: 200 righe positive di riassunti / RAG dal dev del piano binario, evidenza annotata dal giudice e controllata a mano; F1 e sufficienza.

## Regola di adozione

Si adotta se, rispetto alla configurazione vincente senza H-CF:
- sufficienza + ≥ 3 punti su tutti i blocchi;
- F1 sul blocco del dominio di produzione + ≥ 2 punti;
- F1 su `dev_evid_select` non più di 1 punto sotto.

## Rischi

| Rischio | Cosa fare |
|---|---|
| Δ segue le scorciatoie del modello (es. frasi con parole del claim che il verdetto sopravvaluta) | il controllo del punto 3; tenere sempre la loss principale con le gold dove esistono |
| Frasi ridondanti: Δ ≈ 0 per tutte le copie | l'insieme greedy come gold (peso 0,5) copre questo caso |
| Costo del calcolo (n forward per riga) | d1-3B, cache del prefisso, tetto di 80 frasi; un'estrazione per tutti e tre i modelli se l'accordo del punto 4 è alto |
| Δ calcolato con un verdetto che poi cambia (fase 2) | H-CF solo con il backbone congelato, oppure ricalcolare Δ dopo la fase 2 |

## Perché vale la pena

È l'unico modo, fra quelli del piano, di avere etichette di evidenza **sul dominio di produzione** senza annotarle, ed è allineato con la metrica che conta per l'utente (le frasi mostrate bastano a reggere il verdetto). Il costo è tutto offline e riusa codice già previsto (ablazione, cache del prefisso, fase 1a); in produzione non cambia nulla.
