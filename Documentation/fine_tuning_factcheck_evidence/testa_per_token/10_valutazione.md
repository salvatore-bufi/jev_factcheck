# Passo 10 — Valutazione

## In breve

Per ogni modello si misura che il verdetto non sia cambiato, che i tratti mostrati siano quelli giusti e che la latenza resti quella di un decision model. Script: `evidence/token_eval.py`, `evidence/latency.py`.

Le metriche si calcolano sulle righe supportate **che il modello giudica supportate** (solo lì si mostrano tratti). A parte si riporta quante righe supportate restano senza evidenza perché il verdetto dice "no".

## 1. Metriche sui caratteri (indipendenti da qualunque divisione in frasi)

Si confrontano i caratteri dei tratti mostrati con quelli delle frasi gold (`gold_char_sets`), spazi esclusi:

| Metrica | Definizione |
|---|---|
| Precisione | quota dei caratteri mostrati che sta dentro una frase gold |
| Recall | quota dei caratteri delle frasi gold che è mostrata |
| F1 sui caratteri | media armonica delle due, per riga, poi media sulle righe |
| AUPRC per token | dai punteggi `s[t] − s0`, prima della decodifica: quanto bene i punteggi ordinano i token |

Con più insiemi alternativi (WiCE) si prende, per riga, l'insieme che dà l'F1 più alta.

Attenzione: la recall sui caratteri **penalizza** una testa che evidenzia solo la parte utile di una frase gold (le etichette sono frasi intere). Per questo non è la metrica principale: lo sono le due del §2.

## 2. Metriche sulle frasi gold (quelle che contano)

| Metrica | Definizione |
|---|---|
| **Frasi gold colpite** | quota di frasi gold con almeno metà dei caratteri, **oppure** almeno 5 token, dentro i tratti mostrati |
| **Insieme completo** | quota di righe in cui tutte le frasi gold (di almeno un insieme) sono colpite |
| Testo fuori | quota dei caratteri mostrati fuori da ogni frase gold (è 1 − precisione) |
| Frasi gold colpite per distanza | fasce: vicine / 2–9 / 10–19 / ≥ 20 frasi fra la prima e l'ultima frase gold |
| Frasi gold colpite per distanza dall'inizio del documento | < 1.024 / 1.024–4.096 / > 4.096 token |
| Tratti per riga, token per tratto | media e 95° percentile |
| **Sufficienza** | il documento ridotto ai soli tratti mostrati fa ancora dire "supportato" al modello |
| Necessità | il documento senza i tratti mostrati fa dire "non supportato" (si riporta, non è un obiettivo) |
| Negativi accesi | su `dev_evid_neg`: righe con almeno un tratto sopra soglia |

Sufficienza e necessità costano 2 forward per riga: solo in valutazione. Per la sufficienza i tratti si uniscono con " … " fra uno e l'altro.

## 3. Confronto con la testa per frase

Per confrontare le due teste sullo stesso terreno, i tratti si convertono in frasi (solo per la valutazione): una frase della riga (`sentences`, il nostro divisore) è "mostrata" se almeno metà dei suoi caratteri è dentro un tratto. Poi si calcolano le metriche per frase del piano per frase ([../testa_per_frase/10_valutazione.md](../testa_per_frase/10_valutazione.md) §2). Il divisore qui serve solo a misurare: non entra nel modello.

## 4. Prove di comportamento

| Prova | Atteso |
|---|---|
| Evidenza spostata all'inizio / a metà / alla fine | stesse frasi gold colpite in ≥ 90% delle righe |
| Evidenza oltre 1.024 e oltre 4.096 token | idem |
| Frasi di disturbo aggiunte | nessun tratto dentro il disturbo in ≥ 95% |
| Stesso documento, punteggiatura tolta (frasi senza punti) | frasi gold colpite al massimo 5 punti sotto: è il caso in cui la testa per frase non funziona |
| Frase gold riscritta senza parole in comune col claim | colpita in ≥ 80% |
| Coppia quasi identica (positivo / negativo con un fatto tolto) | positivo: tratti giusti; negativo: nessun tratto sopra soglia |

## 5. Latenza

Come nel piano per frase: stesso server, 1.000 richieste reali; A = solo verdetto senza preambolo, B = solo verdetto con preambolo, C = verdetto + tratti. Obiettivo: C ≤ B + 3% (modelli 4B: +3% o +3 ms).

In più rispetto al piano per frase: la proiezione dei vettori di tutti i token (n × D × 256, sulla GPU) e le convoluzioni. Si misurano a parte. Ordine di grandezza per 6.000 token su un 4B: qualche GFLOP, cioè frazioni di millisecondo su GPU.

## 6. Comandi

```sh
python -m evidence.token_eval --model knowline-4b --run data/evidence_tok/runs/knowline-4b/<id> --block dev_evid_select
# test: una volta sola, a configurazione congelata
gpu-run --batch --name evtok-test -- bash -c \
  'python -m evidence.token_eval --model knowline-4b --release <rilascio> \
     --block test_evid_id test_evid_ood test_evid_sparse --sufficiency > logs/evtok-test.out 2>&1'
```

Lo script salva `metrics.json` e `predictions.jsonl` (per riga: tratti, punteggi per token compressi, soglia), così le metriche si ricalcolano senza rifare i forward.

## 7. Report finale (uno per modello)

| Set | Metrica | E-LEX | E-ATT | E-ABL | Testa per frase | Testa per token |
|---|---|---:|---:|---:|---:|---:|
| test_evid_id | frasi gold colpite / insieme completo / F1 caratteri | | | | | |
| test_evid_ood | idem | | | | | |
| test_evid_sparse | frasi colpite a ≥ 20 frasi di distanza | | | | | |
| prova senza punteggiatura | frasi gold colpite | | | | | |
| tutti | sufficienza | | | | | |
| dev_evid_neg | negativi accesi | | | | | |
| verdetto | BAcc (uguale al piano binario?) | — | — | — | | |
| latenza | p50 / p95, C contro B | | | | | |
