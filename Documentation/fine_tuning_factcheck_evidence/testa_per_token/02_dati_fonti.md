# Passo 2 — Dati dalle fonti

## In breve

I dati di SciFact, WiCE e HoVer sono **già costruiti** dal codice in `evidence/data/` ([../../../evidence/README.md](../../../evidence/README.md)): `dataset/evidence/<fonte>.jsonl`. Ogni riga ha documento, claim, etichetta, frasi di evidenza e il tipo di termine di loss.

Per la testa per token serve una cosa in più: le frasi di evidenza **come le dà la fonte**, in caratteri, indipendenti dal nostro divisore. Oggi il builder le usa per calcolare `evidence`, ma non le salva. Il lavoro di questo passo è salvarle.

## 1. Che cosa c'è già

| Fonte | Righe | Positivi | Note |
|---|---:|---:|---|
| SciFact | 1.258 | 508 | abstract corti; `contra` per i CONTRADICT |
| WiCE | 5.294 | 3.100 | documenti lunghi; 801 "parzialmente supportati" senza loss di evidenza; `meta.aggrefact_overlap` segna le pagine presenti in AggreFact |
| HoVer | 22.063 | 12.974 | 2–4 hop con pagine di disturbo; `meta.gold_alignment` = `exact` (frasi HotpotQA) o `db_split` (pagina presa dal db di HoVer) |

Comando: `python -m evidence.data.build --source scifact wice hover --hover-db-fallback` (filtro di lunghezza `--max-tokens`, default 32.768 token di KnowLine-4B).

## 2. Il campo da aggiungere: `gold_char_sets`

```json
"gold_char_sets": [[[412, 497], [3120, 3188]]],
"contra_char": []
```

- `gold_char_sets`: per ogni insieme alternativo di evidenza, la lista degli intervalli `[inizio, fine)` di caratteri delle frasi gold **della fonte** (le frasi di SciFact, gli elementi di WiCE, le frasi HotpotQA di HoVer);
- `contra_char`: lo stesso per le frasi che contraddicono (SciFact).

Modifica al codice (piccola):
1. in `evidence/data/schema.py`, `assemble(...)` salva `gold_char_sets` e `contra_char` (i `gold_sets` e `contra` che riceve già) accanto a `evidence`;
2. `validate(...)` controlla che ogni intervallo stia nel documento, non sia vuoto, e che `label = 0` abbia `gold_char_sets` vuoto;
3. un test in `tests/test_evidence_data.py`: il testo di ogni intervallo è la frase gold della fonte;
4. ricostruire i dataset.

`sentences` ed `evidence` (per frase) restano nella riga: servono alla valutazione a livello di frase ([10](10_valutazione.md) §3) e al confronto con il piano per frase. La testa per token non li usa.

## 3. HoVer con pagine dal db

Per le pagine prese dal db di HoVer (`gold_alignment = db_split`) la fonte non dà le frasi: l'intervallo gold è la nostra frase i di quella pagina, ottenuta con il nostro divisore. È l'unico punto in cui il divisore entra nei dati, e solo per creare etichette (mai in produzione). Misurato: la frase così ottenuta è quella giusta nel 98% dei casi (il 2% è una frase sfasata).

Regola per il mix e la valutazione ([04](04_split_e_mix.md)):
- training: righe `db_split` tenute, con peso 0,5;
- dev e test: solo righe `exact`.

## 4. Controlli

| Controllo | Atteso |
|---|---|
| Ogni intervallo di `gold_char_sets` dentro il documento, non vuoto | tutte le righe |
| Testo degli intervalli = frase gold della fonte (a meno degli spazi ai bordi) | tutte le righe `exact` |
| 30 positivi per fonte letti a mano con gli intervalli evidenziati | le parti evidenziate dimostrano il claim, ≥ 28 su 30 |
