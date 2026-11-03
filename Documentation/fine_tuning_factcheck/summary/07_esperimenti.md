# Step 7 — Piano degli esperimenti

## Obiettivo

Arrivare alla configurazione finale con poche corse, una variazione alla volta, e misurare B-ctrl. Dettagli in [../07_esperimenti.md](../07_esperimenti.md).

## Regole

Una variazione per corsa; si decide su dev_select; si adotta se la BAcc macro sale di ≥ 0,5 punti senza peggiorare l'ECE di oltre 0,02; a parità, la variante più semplice; test solo alla fine.

## Fase 0 — Baseline

B1 (10 decision model, una domanda), B2 (tre domande), B4 (backbone), B5–B7 (riferimenti esterni).

## Fase 1 — Pilota (KnowLine-4B, `fc_pilot`)

P0 (riferimento), learning rate, Brier, smoothing, ancoraggio P-KL, SFT classico (P3), varianti di domanda (P-VAR).

## Fase A — Architettura e punto di partenza

| ID | Cosa |
|---|---|
| A1 | KnowLine-4B (riferimento) |
| A2 | Qwen3.5-4B con la stessa ricetta: pilota di B-ctrl |
| A3 | Qwen3.5-4B con testa nuova |
| A4 | KnowLine-4B con testa nuova inizializzata da w_sì − w_no |
| A5 | encoder DeBERTa stile FactCG |

## Fase 2 — Dati (KnowLine-4B)

Ricetta MiniCheck da sola, ANLI difficile, D2C, C2D, CG2C, rimozione dell'evidenza, curriculum, quantità, FactKG, negazioni.

## Fase 3 — elephant-1 e lunghezza

E0 (ricetta su elephant-1), E-BB9 (stessa ricetta su Qwen3.5-9B: coppia principale di B-ctrl), KL-FIN (KnowLine con 3 seed), F3 (rank), K1 (chunk in inferenza), K2 (claim lunghi).

## Fase 4 — Contrastiva (facoltativa)

X1, X2: loss sulle coppie minime.

## Fase B — B-ctrl su tutte le ricette

Ricetta finale identica su ogni decision model e sul suo backbone (gruppo 9B con Qwen3.5-9B, gruppo 4B con Qwen3.5-4B), più una curva sui dati (0, 2k, 8k, 32k righe) sulle due coppie principali. Prima i modelli con lettura dalla testa del vocabolario e codice in transformers, poi StartLux (maschera da ricostruire) e pplx/Clef (testa propria).

## Controlli finali

C2: 3 seed della configurazione finale; C3: FactCG, MiniCheck, Bespoke sulla stessa suite.

## Registro degli esperimenti

`finetune/experiments_fc.md`, una riga per corsa: modello, mix, variazione, BAcc, AUC, BAcc a 0,5, soglia, ECE, "solo claim", "rimozione dell'evidenza", decisione.
