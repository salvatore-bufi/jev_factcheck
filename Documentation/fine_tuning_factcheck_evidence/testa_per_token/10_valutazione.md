# Passo 10 — Valutazione

## In breve

Per ogni modello si misurano tre cose:
1. il **verdetto** non è cambiato;
2. i **tratti mostrati** sono quelli giusti, anche quando l'evidenza è lontana o il testo non ha punteggiatura;
3. la **latenza** resta quella di un decision model.

Script: `evidence/token_eval.py` (qualità), `evidence/latency.py` (velocità).

Le metriche di evidenza si calcolano sulle righe supportate **che il modello giudica supportate**: solo lì si mostrano tratti. A parte si riporta quante righe supportate restano senza evidenza perché il verdetto dice "no".

## 1. Metriche sulle frasi gold (le principali)

| Metrica | Definizione |
|---|---|
| **Frasi gold colpite** | quota di frasi gold con almeno metà dei caratteri (spazi esclusi) dentro i tratti mostrati |
| **Insieme completo** | quota di righe in cui tutte le frasi gold (di almeno un insieme) sono colpite: per l'evidenza multi-hop è quella che conta, metà evidenza non convince |
| Colpite per distanza fra frasi gold | fasce: vicine / 2–9 / 10–19 / ≥ 20 frasi fra la prima e l'ultima |
| Colpite per distanza dall'inizio del documento | < 1.024 / 1.024–4.096 / > 4.096 token |
| Colpite senza punteggiatura | sul set `test_nopunct` ([04](04_split_e_mix.md) §2) |
| **Sufficienza** | il documento ridotto ai soli tratti mostrati (uniti da " … ") fa ancora dire "supportato" al modello |
| Necessità | il documento senza i tratti mostrati fa dire "non supportato" (si riporta, non è un obiettivo: se l'evidenza è ripetuta può fallire anche quando è giusta) |
| Negativi accesi | su `dev_evid_neg`: righe con almeno un tratto sopra soglia |
| Tratti per riga, token per tratto | media e 95° percentile |

Sufficienza e necessità costano 2 forward per riga: si calcolano solo in valutazione.

## 2. Metriche sui caratteri

| Metrica | Definizione |
|---|---|
| Precisione | quota dei caratteri mostrati che sta dentro una frase gold |
| Recall | quota dei caratteri delle frasi gold che è mostrata |
| F1 sui caratteri | media armonica, per riga, poi media sulle righe |
| AUPRC per token | dai punteggi `s[t] − s0`, prima della decodifica: quanto bene ordinano i token |

Con più insiemi alternativi (WiCE) si prende, per riga, l'insieme con l'F1 più alto.

La recall sui caratteri penalizza una testa che evidenzia solo la parte utile di una frase gold (le etichette sono frasi intere): per questo non è la metrica principale.

```python
def _chars(spans, text):
    return {c for a, b in spans for c in range(a, b) if not text[c].isspace()}


def evidence_metrics(pred_spans, gold_sets, text, hit_share=0.5):
    """pred_spans: [(inizio, fine)] mostrati; gold_sets: insiemi alternativi di [(inizio, fine)] frasi gold.
    -> metriche della riga, con l'insieme gold che dà l'F1 più alto."""
    pred = _chars(pred_spans, text)
    best = None
    for gold in gold_sets:
        g = _chars(gold, text)
        tp = len(pred & g)
        p = tp / len(pred) if pred else 0.0
        r = tp / len(g) if g else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        hits = [len(pred & _chars([s], text)) >= hit_share * len(_chars([s], text)) for s in gold]
        m = {"precision": p, "recall": r, "f1": f1, "gold_hit": sum(hits) / len(hits), "full_set": all(hits)}
        if best is None or m["f1"] > best["f1"]:
            best = m
    return best
```

Esempio (eseguito): documento `Rose "Byrne" is an\nAustralian actress. She starred in films.`, frase gold la prima. Tratto mostrato "Rose \"Byrne\" is an": precisione 1, recall 0,48, F1 0,65, frase **non** colpita (meno di metà dei caratteri). Tratto = tutto il documento, con due insiemi alternativi: si sceglie l'insieme migliore, frase colpita, F1 0,79.

## 3. Confronto con una testa per frase

L'alternativa a questa testa è una **testa per frase**: si divide il documento in frasi con un divisore (NLTK), si riassume ogni frase in un vettore e si dà un punteggio a ogni frase. È più semplice, ma lega il modello al divisore, anche in produzione.

Per confrontarle sullo stesso terreno (solo in valutazione) i tratti si convertono in frasi: una frase della riga (`sentences`, frasi del nostro divisore già salvate nei dati) è "mostrata" se almeno metà dei suoi caratteri è dentro un tratto. Poi si calcolano, per entrambe le teste:
- precisione, recall, F1 per frase;
- insieme completo (tutte le frasi gold mostrate);
- le stesse metriche su `test_nopunct`, dove il divisore non trova le frasi e la testa per frase si degrada.

Il divisore qui serve solo a misurare: non entra nel modello.

## 4. Prove di comportamento

| Prova | Atteso |
|---|---|
| Evidenza spostata all'inizio / a metà / alla fine | stesse frasi gold colpite in ≥ 90% delle righe |
| Evidenza oltre 1.024 e oltre 4.096 token | idem |
| Frasi di disturbo aggiunte | nessun tratto dentro il disturbo in ≥ 95% |
| Documento senza punteggiatura | frasi gold colpite al massimo 5 punti sotto il documento normale |
| Frase gold riscritta senza parole in comune col claim | colpita in ≥ 80% |
| Coppia quasi identica (positivo / negativo con un fatto tolto) | positivo: tratti giusti; negativo: nessun tratto sopra soglia |

## 5. Latenza

Stesso server ([12](12_export_e_servizio.md)), stessa GPU, 1.000 richieste reali di lunghezze diverse (le prime 50 non si contano):

| | Prompt | Uscita |
|---|---|---|
| A | senza preambolo | solo verdetto |
| B | con preambolo | solo verdetto |
| C | con preambolo | verdetto + tratti |

Obiettivo: C ≤ B + 3% (modelli 4B: +3% o +3 ms). La differenza B − A è il costo del preambolo e si riporta a parte. Si misurano anche, separatamente, la proiezione negli hook, la testa (3,2 ms misurati per 6.000 token su RTX 5090 Laptop) e la decodifica.

## 6. Comandi

```sh
python -m evidence.token_eval --model knowline-4b --run data/evidence_tok/runs/knowline-4b/<id> --block dev_evid_select
# test: una volta sola, a configurazione congelata
gpu-run --batch --name evtok-test -- bash -c \
  'python -m evidence.token_eval --model knowline-4b --release <rilascio> \
     --block test_evid_id test_evid_ood test_evid_sparse test_nopunct --sufficiency > logs/evtok-test.out 2>&1'
```

Lo script salva `metrics.json` e `predictions.jsonl` (per riga: tratti, soglia, punteggi dei token compressi), così le metriche si ricalcolano senza rifare i forward.

## 7. Report finale (uno per modello)

| Set | Metrica | E-LEX | E-ATT | E-ABL | Testa per frase | Testa per token |
|---|---|---:|---:|---:|---:|---:|
| test_evid_id | frasi colpite / insieme completo / F1 caratteri | | | | | |
| test_evid_ood | idem | | | | | |
| test_evid_sparse | frasi colpite a ≥ 20 frasi di distanza | | | | | |
| test_nopunct | frasi colpite | | | | | |
| tutti | sufficienza | | | | | |
| dev_evid_neg | negativi accesi | | | | | |
| verdetto | BAcc (uguale al fine-tuning binario?) | — | — | — | | |
| latenza | p50 / p95, C contro B | | | | | |
