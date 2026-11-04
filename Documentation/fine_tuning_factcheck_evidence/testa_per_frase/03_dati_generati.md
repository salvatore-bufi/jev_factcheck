# Passo 3 — Dati generati (NON implementato)

Il piano ([`../../fine_tuning_factcheck_evidence_complete/testa_per_frase/03_dati_generati.md`](../../fine_tuning_factcheck_evidence_complete/testa_per_frase/03_dati_generati.md)) prevede tre famiglie di dati in più:

- articoli generati da un LLM con l'evidenza in posizioni note;
- etichette ricavate col modello per ablazione;
- evidenza spostata lontano o circondata da frasi di disturbo.

Sono state lasciate fuori dal perimetro dell'implementazione, su decisione dell'utente del 2026-11-04: solo le parti obbligatorie e solo KnowLine-4B.

**Che cosa ne dipende e quindi manca:**

- il set `test_evid_sparse` ([04](04_split_e_mix.md));
- le prove di comportamento della valutazione: evidenza spostata, frasi di disturbo, frasi riscritte (doc del piano 10 §4);
- gli esperimenti H5 e H8 ([11](11_esperimenti.md)).

**Che cosa esiste già e può servire:**

- `baselines.ablation_group` ([07](07_baseline.md) §4) implementa la ricerca greedy per ablazione della doc del piano 03 §2, quindi un generatore di etichette per ablazione lo può riusare;
- `extract.rebuild(row, keep)` ([06](06_rendering_ed_estrazione.md) §6) ricostruisce un documento con un sottoinsieme di frasi e rimappa gli indici di evidenza, cioè l'operazione che serve per spostare l'evidenza;
- le righe nuove vanno scritte nel formato di [02](02_dati_fonti.md) §3, con `evidence_quality` = `construct` o `ablation`. Poi basta aggiungere una famiglia alla ricetta del mix ([04](04_split_e_mix.md) §3).
