# Passo 10 — Valutazione

## Obiettivo

Scrivere `evidence/eval_evidence.py` ed `evidence/latency.py` e misurare, per ogni modello:
1. che il verdetto non sia cambiato;
2. che le frasi mostrate siano quelle giuste, anche lontane fra loro o dal claim;
3. che la latenza resti quella di un decision model.

## 1. Su quali righe

Le metriche di evidenza si calcolano sulle righe **supportate e giudicate supportate** dal verdetto (solo lì il sistema mostra frasi). A parte si riporta la quota di righe supportate giudicate non supportate (nessuna frase mostrata).

Per confrontare i tre modelli: anche sull'intersezione delle righe giudicate supportate da tutti e tre.

## 2. Metriche di evidenza

| Metrica | Definizione |
|---|---|
| Precisione / recall / F1 per frase | micro sulle frasi, poi media per fonte |
| Recall dell'insieme completo | quota di righe con **tutte** le frasi gold mostrate; con più insiemi gold (FEVER, WiCE) basta che uno sia contenuto nelle frasi mostrate |
| Match esatto | frasi mostrate = frasi gold (o uno degli insiemi) |
| AUPRC per frase | sul punteggio `s_k − s_0` (o `s_k`), indipendente dalla regola di selezione |
| Errore sul numero di frasi | media di \|mostrate − gold\| |
| F1 per distanza fra frasi gold | fasce: adiacenti / 2–9 / 10–19 / ≥ 20 frasi fra la prima e l'ultima gold |
| F1 per distanza dal preambolo | fasce sulla posizione in token dell'ultima frase gold nel prompt: < 1.024 / 1.024–4.096 / > 4.096 |
| Sufficienza | quota di righe in cui il documento ridotto alle frasi mostrate dà P(supportato) > soglia |
| Necessità | quota di righe in cui il documento senza le frasi mostrate dà P(supportato) < soglia (si riporta, non è un obiettivo: con documenti ridondanti può fallire anche se l'evidenza è giusta) |
| Negativi accesi | su `dev_evid_neg`: quota di righe con almeno una frase con `s_k > s_0`, e fra queste quota con una frase in `contra` |

Sufficienza e necessità:
- documento ridotto e documento senza le frasi si ricostruiscono come nell'ablazione ([03](03_dati_generati.md) §3);
- il prompt è quello di produzione (con preambolo), con lo stesso modello che ha prodotto l'evidenza;
- per JPT e Winnow, se le etichette per ablazione del mix vengono da d1, si riporta anche la sufficienza misurata con d1;
- costano 2 forward per riga: si calcolano solo in valutazione.

## 3. Metriche del verdetto

Quelle del piano binario (BAcc, AUC, ECE, soglia ottimale) su `dev_select`, test e AggreFact, con il prompt col preambolo, confrontate con il verdetto scelto al passo [05](05_formato_prompt.md). Con le fasi 1a e 1b devono coincidere (controllo che pipeline e rendering non siano cambiati).

## 4. Test comportamentali

Costruiti da `test_evid_id` con `augment_sparse.py` e controlli automatici:

| Test | Righe | Atteso |
|---|---:|---|
| Evidenza spostata (inizio / metà / fine) | 3 × 150 | stesse frasi mostrate (agli indici nuovi) in ≥ 90% delle righe |
| Evidenza oltre 1.024 e oltre 4.096 token | 2 × 150 | idem |
| Distrattori aggiunti | 200 | nessun distrattore mostrato in ≥ 95% delle righe |
| Frase di evidenza duplicata altrove | 150 | almeno una delle due copie mostrata in tutte le righe |
| Parafrasi: frase di evidenza riscritta senza parole in comune col claim (generatore + giudice) | 150 | mostrata in ≥ 80% (E-LEX fallisce qui) |
| Coppia minima C2D (positivo / negativo con una frase tolta) | 200 coppie | positivo: frasi giuste; negativo: verdetto "no" e nessuna frase sopra `s_0` |

## 5. Latenza

`evidence/latency.py`, sullo stesso server ([12](12_export_e_servizio.md)) e la stessa GPU, 1.000 richieste reali di lunghezza mista (campionate da `test_id` con la distribuzione di lunghezza di AggreFact), 50 di riscaldamento escluse.

| Configurazione | Cosa misura |
|---|---|
| A: prompt nativo, solo verdetto | riferimento del piano binario |
| B: prompt con preambolo, solo verdetto | costo del preambolo |
| C: prompt con preambolo, verdetto + testa | prodotto finale |

Si riportano p50, p95, richieste/s con batch 1 e con batch 8. Obiettivo: C ≤ B + 3% (d1: o ≤ 2 ms). Il costo di B rispetto ad A si riporta a parte.

Si misura anche il tempo della sola testa e del pooling per frase (con `torch.cuda.synchronize()`), per sapere dove intervenire se C supera il budget.

## 6. Comandi

```sh
python -m evidence.eval_evidence --model d1-3b --run data/evidence/runs/d1-3b/<id> --block dev_evid_select
# test: una volta sola, dopo il congelamento (passo 12)
gpu-run --batch --name ev-test-d1 -- bash -c \
  'python -m evidence.eval_evidence --model d1-3b --release <cartella del rilascio> \
     --block test_evid_id test_evid_ood test_evid_sparse --sufficiency > logs/ev-test-d1.out 2>&1'
gpu-run --batch --name ev-latency-d1 -- bash -c \
  'python -m evidence.latency --model d1-3b --release <cartella del rilascio> > logs/ev-latency-d1.out 2>&1'
```

`eval_evidence.py` scrive `metrics.json` (tutte le metriche di §2 per fonte e per fascia) e `predictions.jsonl` (per riga: punteggi, `s_0`, frasi mostrate, gold), così le metriche si ricalcolano senza rifare i forward.

## 7. Il report finale (uno per modello)

| Blocco | Metrica | E-LEX | E-ATT | E-ABL | Testa (fase finale) |
|---|---|---:|---:|---:|---:|
| test_evid_id | F1 / insieme completo | | | | |
| test_evid_ood | F1 / insieme completo | | | | |
| test_evid_sparse | F1 ≥ 20 frasi / F1 > 4.096 token | | | | |
| tutti | sufficienza | | | | |
| dev_evid_neg | negativi accesi | | | | |
| verdetto | BAcc (= piano binario?) | — | — | — | |
| latenza | p50 / p95 (C contro B) | | | | |

## Prodotto

- `evidence/eval_evidence.py`, `evidence/latency.py`, test comportamentali generati e congelati con gli split.
