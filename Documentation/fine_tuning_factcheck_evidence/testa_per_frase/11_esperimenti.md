# Passo 11 — Esperimenti

## In breve

Si parte da una testa semplice (H0) e si aggiunge **un pezzo alla volta**, tenendolo solo se migliora. Tutto si fa su **KnowLine-4B** con i vettori salvati (fase 1a): ogni prova è un training della sola testa. Alla fine elephant-1 e gli altri decision model ripetono la configurazione migliore.

## Regole

1. Una modifica per prova, con il suo file `configs/<id>.json`.
2. Si decide su `dev_evid_select` guardando quattro numeri: F1, insieme completo, F1 con evidenza a 10 o più frasi di distanza, F1 con evidenza oltre 1.024 token. In più, i negativi accesi su `dev_evid_neg`.
3. **Una modifica si tiene** se aumenta l'F1 di almeno 1 punto o l'insieme completo di almeno 2, senza far scendere gli altri numeri di più di 1 punto. Se aggiunge dati da leggere dal modello, deve anche restare nel limite di latenza.
4. A parità di risultato si tiene la versione più semplice.
5. Se il guadagno è sotto i 2 punti, si ripete con 2 semi in più prima di tenerlo.

## Prima degli esperimenti

| Che cosa | Dove |
|---|---|
| Testo del preambolo e verdetto da usare per ogni modello | [05](05_formato_prompt.md) |
| Metodi semplici di confronto (E-LEX, E-ATT, E-ABL, …) | [07](07_baseline.md) |

## La testa di partenza: H0

Solo l'ultimo layer, solo il vettore dell'ultimo token di ogni frase, niente attenzione, 2 layer di transformer fra le frasi, loss di partenza con soglia fissa, solo righe positive.

Prima domanda: H0 batte E-LEX ed E-ATT? Se no, si cerca un errore nelle posizioni o nei vettori ([06](06_rendering_ed_estrazione.md) §6) prima di provare altro.

## Gli esperimenti principali (in quest'ordine)

| Nome | Che cosa si cambia | Che cosa si guarda |
|---|---|---|
| **H-LOSS** | loss con soglia per riga invece di soglia fissa ([08](08_testa_e_loss.md) §3) | insieme completo; numero di frasi mostrate |
| **H-MIL** | si aggiungono i claim non supportati al training (nessuna frase sopra soglia), con peso 0,3 e 1 | negativi accesi; che non si perdano frasi nei positivi incerti |
| **H2a** | vettori da 4 layer invece che dal solo ultimo; poi solo i 2 più usati | F1; quali layer pesano |
| **H2b** | si aggiunge l'attenzione ricevuta da ogni frase | F1 con evidenza lontana |
| **H2c** | si aggiunge la media dei token della frase | F1 sulle frasi lunghe (oltre 25 parole) |

## Controlli sulla configurazione migliore

Ognuno toglie o cambia un pezzo, per capire quanto conta:

| Nome | Che cosa si cambia | Che cosa dice |
|---|---|---|
| H3a | niente transformer fra le frasi: ogni frase ha un punteggio da sola | se le frasi devono comunicare. Atteso: crollo su HoVer e sull'evidenza lontana. Se non crolla, si può semplificare |
| H3b | punteggio = somiglianza fra `x_0` e la frase | se basta una testa più semplice |
| H3c | 1 o 4 layer di transformer invece di 2 | quanta capacità serve |
| H4 | vettori calcolati senza preambolo | quanto aiuta il preambolo. Se poco, si toglie |
| H5 | niente righe con evidenza spostata nel training | se servono. Atteso: calo sull'evidenza lontana |
| H7 | peso 1 invece di 0,3 per le frasi non segnate di FEVER; per FEVER basta un gruppo di evidenza invece di tutti | come trattare annotazioni incomplete |
| H8 | solo evidenza annotata / + dati generati / + etichette ricavate col modello / + positivi senza evidenza (al posto delle etichette ricavate) | quanto valgono i dati del passo 3 |
| H9 | niente informazione sulla posizione della frase | se la testa preferisce troppo l'inizio o la fine |

## Esperimenti da fare solo se serve

| Nome | Quando | Che cosa |
|---|---|---|
| H2d | se H2c ha aiutato sulle frasi lunghe | media pesata dei token decisa dalla testa (fase 1b, [09](09_training.md) §4) |
| H-PAIR | se l'insieme completo su HoVer resta sotto 0,65, o l'F1 cala di oltre 5 punti con frasi lontane | punteggio per coppie di frasi ([08](08_testa_e_loss.md) §4) |
| H6 | se gli obiettivi non sono raggiunti dopo tutto il resto | fase 2, con peso dell'evidenza 0,1 / 0,3 / 1 |

## elephant-1 e gli altri modelli

| Nome | Che cosa |
|---|---|
| E0 | elephant-1: si salvano i vettori e si addestra la configurazione migliore di KnowLine; controllo rapido di quali layer usare (possono essere diversi) |
| H4-bi | H4 su elephant-1: con l'attenzione bidirezionale il preambolo serve ancora? Se no, per i modelli bidirezionali si toglie |
| X0 | gli altri decision model, con la configurazione migliore del loro gruppo di attenzione (causale: KnowLine; bidirezionale: elephant-1), nell'ordine del piano binario (prima Jeffan-4B, d3-flash, OneJev, Nimble; poi StartLux, pplx, Clef) |
| E6 | fase 2, solo se su KnowLine ha dato un guadagno chiaro e il modello è sotto gli obiettivi |
| L1 | latenza per ogni modello rilasciato ([10](10_valutazione.md) §5) |

## Controlli finali

| Nome | Che cosa |
|---|---|
| C1 | 3 semi della testa finale per modello: media e variabilità |
| C2 | sufficienza e necessità su tutti i set di test |
| C3 | testa contro E-ABL: quanto ci si avvicina con un forward a un metodo con centinaia di forward |
| C4 | confronto fra i modelli sulle righe giudicate supportate da tutti |

## Registro

`data/evidence/runs/registry.md`, una riga per prova (scritta in automatico da `train_head.py`, tranne la decisione):

| Nome | Modello | Fase | Modifica | F1 | Insieme completo | F1 ≥ 10 frasi | F1 > 1.024 token | Negativi accesi | Sufficienza | Δ verdetto | Δ latenza | Decisione |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| E-LEX | — | — | — | | | | | | | — | — | riferimento |
| H0 | KnowLine-4B | 1a | — | | | | | | | 0 | | |
