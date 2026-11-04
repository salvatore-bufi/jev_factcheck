# Passo 11 — Esperimenti

## In breve

Si parte da una testa minima (H0) e si aggiunge **un pezzo alla volta**, tenendolo solo se migliora. Tutto si fa su **KnowLine-4B** con i vettori salvati (fase 1a): ogni prova è un training della sola testa, di pochi minuti. Alla fine elephant-1 e gli altri decision model ripetono la configurazione migliore.

## Regole

1. Una modifica per prova, con il suo `configs/tok_<id>.json` (opzioni della testa, della loss, della decodifica).
2. Si decide su `dev_evid_select` guardando: frasi gold colpite, insieme completo, F1 sui caratteri, frasi colpite con evidenza a ≥ 10 frasi di distanza; più i negativi accesi su `dev_evid_neg`.
3. **Una modifica si tiene** se aumenta le frasi colpite di almeno 1 punto o l'insieme completo di almeno 2, senza far scendere le altre metriche di più di 1 punto, e restando nel limite di latenza.
4. A parità di risultato, la versione più semplice.
5. Guadagni sotto 2 punti si ripetono con 2 semi in più prima di tenerli.

## La testa di partenza: H0

Un layer (31), niente attenzione, niente contesto locale né globale: il punteggio del token viene dal suo vettore e da `q` (`use_att=False, use_local=False, use_global=False`). Loss completa, decodifica di default.

Prima domanda: H0 batte E-LEX ed E-ATT? Se no, prima di provare altro si controllano `tok_char` ed etichette ([06](06_rendering_ed_estrazione.md) §6).

## Esperimenti principali (in quest'ordine)

| Nome | Che cosa si aggiunge | Che cosa si guarda |
|---|---|---|
| H1 | contesto locale (3 convoluzioni) | tratti meno frammentati, F1 sui caratteri |
| H2 | attenzione della posizione di risposta per token | frasi colpite, soprattutto quelle "ponte" |
| H3 | contesto globale (transformer sui blocchi) | evidenza lontana, insieme completo |
| H4 | soglia del documento `s0` contro una soglia fissa scelta sul dev | insieme completo, numero di tratti |
| H5 | termini della loss, uno alla volta: senza `L_hit`; senza `L_out`; senza `L_tv` | quanto serve ciascuno |
| H6 | 4 layer (19, 23, 27, 31) invece di 2 | frasi colpite; spazio su disco doppio |
| H7 | blocchi da 8 e da 32 token invece di 16 | evidenza lontana, latenza |

Lettura di H3: se il contesto globale non migliora l'evidenza lontana, o le parti lontane non si vedono (blocchi troppo grandi?) o i dati hanno poca evidenza lontana; HoVer ne ha molta (6.608 positivi con frasi gold a ≥ 20 frasi), quindi ci si aspetta un guadagno netto.

## Controlli sulla configurazione migliore

| Nome | Che cosa si cambia | Che cosa dice |
|---|---|---|
| H8 | vettori estratti senza preambolo | quanto vale il preambolo; si ripete su elephant-1, che è bidirezionale |
| H9 | fase 1b: vettori completi e riduzione appresa invece della PCA | se la PCA perde informazione |
| H10 | training con le righe generate a intervalli esatti (passo 3) | se la testa impara tratti più stretti di una frase |
| H11 | ω = 0,3 / 0,5 / 1 | gestione delle annotazioni incomplete |
| H12 | peso delle righe HoVer `db_split`: 0 / 0,5 / 1 | se le righe recuperate dal db aiutano o fanno rumore |
| H13 | α (peso dei token gold in `L_tok`) = 1 / 3 / 10 | quanto spingere ad accendere frasi intere |

## Testa per token contro testa per frase

| Nome | Che cosa |
|---|---|
| T-vs-F | stessa fase (1a), stessi dati, stesso modello: questa testa contro una **testa per frase** (un punteggio per frase del divisore NLTK, dallo stato dell'ultimo token e dalla media dei token di ogni frase, con un transformer sulle frasi), confrontate con le metriche per frase del passo 10 §3 e su `test_nopunct` |

Lettura: se la testa per token resta entro ~1 punto da quella per frase sulle metriche per frase, ed è nettamente meglio senza punteggiatura, si adotta la testa per token (nessun divisore in produzione). Se è nettamente sotto, il prezzo dell'indipendenza dal divisore è troppo alto: si torna a una testa per frase con il divisore incluso nella repo.

## elephant-1 e gli altri modelli

| Nome | Che cosa |
|---|---|
| E0 | elephant-1: PCA, estrazione, configurazione migliore di KnowLine |
| X0 | gli altri decision model: prima quelli con il codice degli autori in transformers (Jeffan-4B, d3-flash, OneJev, Nimble), poi StartLux (maschera da ricostruire), pplx e Clef (testa propria) |
| E2 | fase 2, solo se su KnowLine ha dato un guadagno chiaro e il modello è sotto gli obiettivi |
| L1 | latenza per ogni modello rilasciato |

## Controlli finali

| Nome | Che cosa |
|---|---|
| C1 | 3 semi della testa finale per modello: media e variabilità |
| C2 | sufficienza e necessità su tutti i set di test |
| C3 | testa contro E-ABL: quanto ci si avvicina con un forward a un metodo da centinaia di forward |
| C4 | confronto fra i modelli sulle righe giudicate supportate da tutti |

## Registro

`data/evidence_tok/runs/registry.md`, una riga per prova (scritta in automatico dallo script di training, tranne la decisione):

| Nome | Modello | Fase | Modifica | Frasi colpite | Insieme completo | F1 caratteri | Colpite ≥ 10 frasi | Tratti per riga | Negativi accesi | Sufficienza | Δ latenza | Decisione |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| E-LEX | — | — | — | | | | | | | | — | riferimento |
| H0 | KnowLine-4B | 1a | — | | | | | | | | | |
