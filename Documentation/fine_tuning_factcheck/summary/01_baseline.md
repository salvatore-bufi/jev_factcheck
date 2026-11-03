# Step 1 — Baseline: quanto valgono i modelli prima del training

## Obiettivo

Misurare i 10 decision model, i 2 backbone e i riferimenti esterni con lo stesso protocollo del dopo-training. Dettagli e codice in [../01_baseline.md](../01_baseline.md).

## 1. Cosa si sa già (dai run dello studio, senza nuova inferenza)

I run dello studio focalizzato su AggreFact (RUN-01…05, file di domande QF1) contengono, riga per riga, le probabilità di Q1, Q2 e Q3: la domanda singola si valuta dagli `answers.jsonl` esistenti. Q4 è nei run su AF-4k (RUN-08, RUN-09). Numeri già noti con la regola a tre domande: elephant-1 76,70, KnowLine-4B 75,65.

## 2. Le baseline da misurare

| ID | Cosa |
|---|---|
| B1 | i 10 decision model, una domanda (Q1, 1 − Q2, Q4) |
| B2 | tre domande, regola `min` |
| B4 | Qwen3.5-9B e Qwen3.5-4B, prompt neutro, letture R0 e R1 |
| B5–B7 | FactCG, MiniCheck-FT5, Bespoke-MiniCheck-7B |

## 3. Procedura

1. Esportare gli split dello Step 0 come dataset locale (`dataset/fc_ours/<split>.jsonl`).
2. `benchmark.py --dataset fc_ours --split dev_select` per ogni modello, con QF1 e con la sola Q4: il benchmark usa già il codice degli autori.
3. Scala di temperatura (minimo log-loss) e soglia (BAcc macro) su dev_select; conferma su dev_confirm.
4. Test una volta.

## 4. Tabella attesa

Una colonna per modello: BAcc e AUC su dev_select per ogni domanda, BAcc a 0,5, soglia ottimale, ECE, BAcc su dev_confirm e sui test, AggreFact con soglia dal nostro dev (con e senza RAGTruth), "solo claim", costo.

## Criteri per passare allo step successivo

- B1, B2, B4, B5 misurati; domanda scelta ([02d](02d_varianti_domanda.md)); scala e soglia di partenza congelate.

## Errori comuni

- Confrontare una domanda con tre senza dirlo; riusare la soglia della baseline dopo il training; leggere un modello con un prompt diverso da quello dei suoi autori.
