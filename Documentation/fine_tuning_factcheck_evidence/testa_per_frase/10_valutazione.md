# Passo 10 — Valutazione

## In breve

Per ogni modello si misurano tre cose:
1. il **verdetto** non è cambiato;
2. le **frasi mostrate** sono quelle giuste, anche quando sono lontane fra loro o lontane dall'inizio del documento;
3. la **latenza** resta quella di un decision model.

Script: `evidence/eval_evidence.py` (qualità), `evidence/latency.py` (velocità).

## 1. Su quali righe

L'evidenza si misura sulle righe supportate **che il modello giudica supportate**: sono le sole in cui il sistema mostra frasi. A parte si riporta quante righe supportate il modello giudica non supportate (e quindi restano senza evidenza). Per confrontare modelli diversi si usano anche le sole righe giudicate supportate da tutti.

## 2. Metriche dell'evidenza

| Metrica | Che cosa misura |
|---|---|
| Precisione, recall, F1 per frase | quante frasi mostrate sono giuste, quante giuste sono state mostrate (rispetto all'unione degli insiemi gold) |
| **Insieme completo** | in quante righe sono state mostrate **tutte** le frasi di almeno un insieme gold. Per l'evidenza su più frasi è la metrica che conta: metà evidenza non convince |
| Corrispondenza esatta | frasi mostrate = uno degli insiemi gold |
| AUPRC per frase | quanto bene i punteggi ordinano le frasi, indipendentemente dalla soglia |
| Errore sul numero | di quante frasi in più o in meno rispetto alle frasi gold |
| F1 per distanza fra le frasi gold | vicine / 2–9 / 10–19 / ≥ 20 frasi fra la prima e l'ultima frase gold |
| F1 per distanza dall'inizio del documento | ultima frase gold a < 1.024 / 1.024–4.096 / > 4.096 token |
| **Sufficienza** | in quante righe il documento ridotto alle sole frasi mostrate fa ancora dire "supportato" al modello. Non dipende dalle annotazioni, che possono essere incomplete |
| Necessità | in quante righe il documento senza le frasi mostrate fa dire "non supportato" (si riporta solo: se l'evidenza è ripetuta nel documento può fallire anche quando è giusta) |
| Negativi accesi | su `dev_evid_neg`: in quanti claim non supportati almeno una frase supera la soglia |

Sufficienza e necessità richiedono 2 forward per riga, con lo stesso modello e lo stesso prompt di produzione: si calcolano solo in valutazione.

```python
from sklearn.metrics import average_precision_score


def select(s, s0, delta=0.0, max_shown=5, min_shown=1):
    """Frasi mostrate: s[k] > s0 + delta, al massimo max_shown (le più alte), almeno min_shown (la migliore)."""
    order = sorted(range(len(s)), key=lambda k: s[k], reverse=True)
    chosen = [k for k in order if s[k] > s0 + delta][:max_shown]
    return sorted(chosen or order[:min_shown])


def sentence_metrics(shown, evidence_sets):
    """shown: indici delle frasi mostrate; evidence_sets: insiemi alternativi di indici gold.
    -> precisione, recall, F1 rispetto all'unione; insieme completo se uno degli insiemi è tutto mostrato."""
    gold = set().union(*map(set, evidence_sets))
    shown = set(shown)
    tp = len(shown & gold)
    p = tp / len(shown) if shown else 0.0
    r = tp / len(gold) if gold else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "full_set": any(set(e) <= shown for e in evidence_sets), "exact": any(set(e) == shown for e in evidence_sets)}


def auprc(scores_per_row, gold_per_row):
    """AUPRC per frase su tutte le righe (indipendente dalla soglia)."""
    y = [int(k in g) for s, g in zip(scores_per_row, gold_per_row) for k in range(len(s))]
    x = [v for s in scores_per_row for v in s]
    return average_precision_score(y, x)
```

Esempi (eseguiti): frasi mostrate {1, 4}, insiemi gold {1, 4} e {7} → precisione 1, recall 0,67 (rispetto all'unione {1, 4, 7}), F1 0,8, insieme completo sì, esatto sì. Frase mostrata {1}, insieme gold {1, 4} → recall 0,5, F1 0,67, insieme completo no.

## 3. Metriche del verdetto

BAcc, AUC ed ECE sul dev e sui test del fine-tuning binario, con il prompt con preambolo, confrontate con il verdetto scelto al passo 5. Nelle fasi 1a e 1b devono coincidere: se no, c'è un errore nella pipeline.

## 4. Prove di comportamento

Costruite dal test con le trasformazioni del passo 3:

| Prova | Righe | Risultato atteso |
|---|---:|---|
| Evidenza spostata all'inizio / a metà / alla fine | 3 × 150 | stesse frasi mostrate in ≥ 90% delle righe |
| Evidenza spostata oltre 1.024 e oltre 4.096 token | 2 × 150 | idem |
| Frasi di disturbo aggiunte | 200 | nessuna frase di disturbo mostrata in ≥ 95% |
| Frase di evidenza ripetuta altrove | 150 | almeno una delle due copie mostrata, sempre |
| Frase di evidenza riscritta senza parole in comune col claim | 150 | mostrata in ≥ 80% (E-LEX fallisce qui) |
| Coppia quasi identica: positivo e negativo con un fatto tolto | 200 coppie | positivo: frasi giuste; negativo: "non supportato" e nessuna frase sopra soglia |

## 5. Latenza

Stesso server ([12](12_export_e_servizio.md)), stessa GPU, 1.000 richieste reali di lunghezze diverse (le prime 50 non si contano):

| | Prompt | Uscita |
|---|---|---|
| A | senza preambolo | solo verdetto |
| B | con preambolo | solo verdetto |
| C | con preambolo | verdetto + evidenza |

Obiettivo: C al massimo +3% rispetto a B (modelli 4B: +3% o +2 ms). La differenza B − A è il costo del preambolo e si riporta a parte. Si misurano anche, separatamente, la divisione in frasi (CPU), il riassunto delle frasi negli hook e la testa (0,7–1,6 ms misurati per 46–500 frasi su RTX 5090 Laptop).

## 6. Comandi

```sh
python -m evidence.eval_evidence --model knowline-4b --run data/evidence/runs/knowline-4b/<id> --block dev_evid_select
# test: una volta sola, a configurazione congelata (passo 12)
gpu-run --batch --name ev-test -- bash -c \
  'python -m evidence.eval_evidence --model knowline-4b --release <rilascio> \
     --block test_evid_id test_evid_ood test_evid_sparse --sufficiency > logs/ev-test.out 2>&1'
gpu-run --batch --name ev-latency -- bash -c \
  'python -m evidence.latency --model knowline-4b --release <rilascio> > logs/ev-latency.out 2>&1'
```

Lo script salva `metrics.json` e `predictions.jsonl` (per ogni riga: punteggi, soglia, frasi mostrate, frasi gold), così le metriche si ricalcolano senza rifare i forward.

## 7. Report finale (uno per modello)

| Set | Metrica | E-LEX | E-ATT | E-ABL | Testa |
|---|---|---:|---:|---:|---:|
| test_evid_id | F1 / insieme completo | | | | |
| test_evid_ood | F1 / insieme completo | | | | |
| test_evid_sparse | F1 a ≥ 20 frasi / oltre 4.096 token | | | | |
| tutti | sufficienza | | | | |
| dev_evid_neg | negativi accesi | | | | |
| verdetto | BAcc (uguale al fine-tuning binario?) | — | — | — | |
| latenza | p50 / p95, C contro B | | | | |
