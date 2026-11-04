# Passo 11 — Esperimenti

## In breve

Si parte da una testa minima (H0) e si aggiunge **un pezzo alla volta**, tenendolo solo se migliora. Tutto su **KnowLine-4B** con i vettori salvati (fase 1a). Alla fine elephant-1 e gli altri modelli ripetono la configurazione migliore.

## Regole

1. Una modifica per prova, con il suo `configs/tok_<id>.json`.
2. Si decide su `dev_evid_select` con: frasi gold colpite, insieme completo, F1 sui caratteri, frasi colpite con evidenza a ≥ 10 frasi di distanza; più i negativi accesi su `dev_evid_neg`.
3. Una modifica si tiene se migliora le frasi colpite di ≥ 1 punto o l'insieme completo di ≥ 2, senza peggiorare le altre metriche di più di 1 punto (e restando nel limite di latenza).
4. A parità, la versione più semplice.
5. Guadagni sotto 2 punti si ripetono con 2 semi in più.

## La testa di partenza: H0

Un layer (31), niente attenzione, niente contesto locale né globale: punteggio per token dal suo vettore più `q`. Loss completa ([08](08_testa_e_loss.md) §3), decodifica di default.

Prima domanda: H0 batte E-LEX ed E-ATT? Se no, si controllano `tok_char` ed etichette ([06](06_rendering_ed_estrazione.md) §7).

## Esperimenti principali (in quest'ordine)

| Nome | Che cosa si aggiunge | Che cosa si guarda |
|---|---|---|
| H1 | contesto locale (3 convoluzioni) | frammentazione dei tratti, F1 sui caratteri |
| H2 | attenzione della posizione di risposta per token | frasi colpite, soprattutto le "ponte" |
| H3 | contesto globale (transformer sui blocchi) | evidenza lontana, insieme completo |
| H4 | soglia del documento `s0` contro soglia fissa sul dev | insieme completo; numero di tratti |
| H5 | termini della loss: senza `L_hit`; senza `L_out`; senza `L_tv` (uno alla volta) | quanto serve ciascuno |
| H6 | 4 layer (19, 23, 27, 31) invece di 2 | frasi colpite; spazio su disco |
| H7 | blocchi da 8 e da 32 token invece di 16 | evidenza lontana, latenza |

## Controlli sulla configurazione migliore

| Nome | Che cosa si cambia | Che cosa dice |
|---|---|---|
| H8 | vettori estratti senza preambolo | quanto vale il preambolo (anche per un modello bidirezionale: elephant-1) |
| H9 | fase 1b: vettori completi e proiezione appresa invece della PCA | se la PCA perde informazione |
| H10 | training con le righe C2D a intervalli esatti ([03](03_dati_generati.md) §1) | se la testa impara tratti più stretti di una frase |
| H11 | ω: 0,3 / 0,5 / 1 | gestione delle annotazioni incomplete |
| H12 | peso delle righe HoVer `db_split`: 0 / 0,5 / 1 | se le righe recuperate dal db aiutano o fanno rumore |
| H13 | decodifica: con e senza allargamento dei tratti alla frase (divisore qualunque, solo in visualizzazione) | quanto cambia la leggibilità rispetto alle metriche |

## Confronto con la testa per frase

| Nome | Che cosa |
|---|---|
| T-vs-F | stessa fase (1a), stessi dati e stesso modello: testa per token contro testa per frase ([../testa_per_frase/](../testa_per_frase/00_overview.md)), con le metriche per frase ([10](10_valutazione.md) §3) e con la prova senza punteggiatura |

Lettura: se la testa per token è entro ~1 punto della testa per frase sulle metriche per frase, e molto meglio senza punteggiatura, si adotta la testa per token (nessun divisore in produzione). Se è nettamente sotto, il costo dell'indipendenza dal divisore è troppo alto e si torna alla testa per frase, magari con un divisore incluso nella repo.

## elephant-1 e gli altri modelli

| Nome | Che cosa |
|---|---|
| E0 | elephant-1: PCA, estrazione, configurazione migliore di KnowLine |
| X0 | gli altri decision model, nell'ordine del piano binario |
| E2 | fase 2, solo se su KnowLine ha dato un guadagno chiaro |
| L1 | latenza per ogni modello rilasciato |

## Controlli finali

| Nome | Che cosa |
|---|---|
| C1 | 3 semi della testa finale: media e variabilità |
| C2 | sufficienza e necessità su tutti i set di test |
| C3 | testa contro E-ABL |
| C4 | confronto fra i modelli sulle righe giudicate supportate da tutti |

## Registro

`data/evidence_tok/runs/registry.md`, una riga per prova:

| Nome | Modello | Fase | Modifica | Frasi colpite | Insieme completo | F1 caratteri | Colpite ≥ 10 frasi | Tratti per riga | Negativi accesi | Sufficienza | Δ latenza | Decisione |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| E-LEX | — | — | — | | | | | | | | — | riferimento |
| H0 | KnowLine-4B | 1a | — | | | | | | | | | |
