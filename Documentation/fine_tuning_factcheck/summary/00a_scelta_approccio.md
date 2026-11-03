# Step 0a — Quale approccio: decision model, classificatore stile FactCG/MiniCheck, o backbone?

## Obiettivo

Decidere da dove partire e che cosa addestrare. Dettagli in [../00a_scelta_approccio.md](../00a_scelta_approccio.md).

## 1. Le opzioni

| Opzione | Pro | Contro |
|---|---|---|
| **Decision model, una domanda, lettura esistente** | parte già bene; probabilità concentrate sulle due opzioni; un forward | prompt, maschera e lettura diversi per ogni modello: vanno riprodotti |
| Backbone (Qwen3.5) con la stessa ricetta | semplice, licenza Apache 2.0 | parte più in basso: è il controllo di B-ctrl |
| Testa nuova `Linear(d → 2)` | — | stessa capacità della lettura esistente, ma parte da pesi casuali |
| Encoder stile FactCG (0,4B) | velocissimo | contesto corto, tetto ~75–76 |

## 2. Perché non una testa nuova

Con la lettura dalla testa del vocabolario, P(vero) = σ((w_vero − w_falso) · h / T): è già una regressione logistica su h, con pesi inizializzati da ciò che il modello sa. Una testa nuova non aggiunge capacità e parte al 50%. Per pplx-decider-flash e Clef-Flash, che hanno una testa propria, si tiene la loro testa e la si addestra insieme alla LoRA.

## 3. Decision model o backbone (B-ctrl)

Nessuno ha misurato se, a parità di dati e ricetta, un checker ottenuto da un decision model sia migliore di uno ottenuto dal suo backbone. Si misura sulle coppie elephant-1 / Qwen3.5-9B e KnowLine-4B / Qwen3.5-4B, poi sugli altri 8 decision model.

## 4. Raccomandazione

1. **KnowLine-4B per iterare** (causale, due token della testa del vocabolario, 24 GB bastano).
2. **elephant-1 come modello principale a 9B** (MIT).
3. Gli altri 8 decision model e i due backbone con la ricetta finale (B-ctrl).
4. Domanda scelta sul nostro dev fra Q1, Q2 e Q4 dello studio.
5. Controllo encoder (A5) e distillazione facoltativa.

## 5. Cosa si perde specializzando

Il modello non è più un decision model generale (nome `-fc`, model card); l'uscita è binaria; StartLux e Nimble restano non commerciali (CC BY-NC).

## Criteri per passare allo step successivo

- Ruoli dei modelli registrati; controlli (B-ctrl, encoder) in programma.
