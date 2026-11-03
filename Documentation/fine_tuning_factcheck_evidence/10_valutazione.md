# Passo 10 — Valutazione

## In breve

Per ogni modello si misurano tre cose:
1. il **verdetto** non è cambiato;
2. le **frasi mostrate** sono quelle giuste, anche quando sono lontane fra loro o dal claim;
3. la **latenza** resta quella di un decision model.

Script: `evidence/eval_evidence.py` (qualità) ed `evidence/latency.py` (velocità).

## 1. Su quali righe

L'evidenza si misura sulle righe supportate **che il modello giudica supportate**: sono le sole in cui il sistema mostra frasi. A parte si riporta quante righe supportate il modello giudica non supportate (e quindi restano senza evidenza).

Per confrontare i tre modelli si usano anche le sole righe giudicate supportate da tutti e tre.

## 2. Metriche dell'evidenza

| Metrica | Che cosa misura |
|---|---|
| Precisione, recall, F1 per frase | quante frasi mostrate sono giuste, quante giuste sono state mostrate |
| **Insieme completo** | in quante righe sono state mostrate **tutte** le frasi di evidenza (con più gruppi alternativi, basta uno). Per l'evidenza su più frasi è la metrica che conta: metà evidenza non convince |
| Corrispondenza esatta | frasi mostrate = frasi di evidenza |
| AUPRC | quanto bene i punteggi ordinano le frasi, indipendentemente dalla soglia |
| Errore sul numero | di quante frasi in più o in meno rispetto all'evidenza |
| F1 per distanza fra frasi | vicine / 2–9 / 10–19 / 20 o più frasi fra la prima e l'ultima frase di evidenza |
| F1 per distanza dal preambolo | ultima frase di evidenza a meno di 1.024 / 1.024–4.096 / oltre 4.096 token dal claim del preambolo |
| **Sufficienza** | in quante righe il documento ridotto alle sole frasi mostrate fa ancora dire "supportato" al modello. Non dipende dalle annotazioni, che possono essere incomplete |
| Necessità | in quante righe il documento senza le frasi mostrate fa dire "non supportato" (si riporta solo: se l'evidenza è ripetuta nel documento può fallire anche quando è giusta) |
| Negativi accesi | su `dev_evid_neg`: in quanti claim non supportati almeno una frase supera la soglia (non viene mostrata, ma indica che la testa si fa ingannare dalla somiglianza) |

Sufficienza e necessità richiedono 2 forward per riga: si calcolano solo in valutazione, con lo stesso modello e lo stesso prompt di produzione.

## 3. Metriche del verdetto

Le stesse del piano binario (BAcc, AUC, ECE) su `dev_select`, test e AggreFact, con il prompt con preambolo. Nelle fasi 1a e 1b devono coincidere con quelle del verdetto scelto al passo [05](05_formato_prompt.md): se no, c'è un errore nella pipeline.

## 4. Prove di comportamento

Costruite dal test con le trasformazioni del passo [03](03_dati_generati.md):

| Prova | Righe | Risultato atteso |
|---|---:|---|
| Evidenza spostata all'inizio / a metà / alla fine | 3 × 150 | stesse frasi mostrate in ≥ 90% delle righe |
| Evidenza spostata oltre 1.024 e oltre 4.096 token | 2 × 150 | idem |
| Frasi di disturbo aggiunte | 200 | nessuna frase di disturbo mostrata in ≥ 95% |
| Frase di evidenza ripetuta altrove | 150 | almeno una delle due copie mostrata, sempre |
| Frase di evidenza riscritta senza parole in comune col claim | 150 | mostrata in ≥ 80% |
| Coppia quasi identica: positivo e negativo con una frase tolta | 200 coppie | positivo: frasi giuste; negativo: "non supportato" e nessuna frase sopra soglia |

## 5. Latenza

Stesso server ([12](12_export_e_servizio.md)), stessa GPU, 1.000 richieste reali di lunghezze diverse (le prime 50 non si contano). Tre configurazioni:

| | Prompt | Uscita |
|---|---|---|
| A | senza preambolo | solo verdetto (come il piano binario) |
| B | con preambolo | solo verdetto |
| C | con preambolo | verdetto + evidenza |

Obiettivo: C al massimo +3% rispetto a B (d1-3B: +3% o +2 ms). La differenza fra B e A è il costo del preambolo e si riporta a parte. Si misura anche il tempo della sola testa, per sapere dove intervenire se si supera il limite.

## 6. Comandi

```sh
# durante gli esperimenti
python -m evidence.eval_evidence --model d1-3b --run data/evidence/runs/d1-3b/<id> --block dev_evid_select

# test: una volta sola, a configurazione congelata (passo 12)
gpu-run --batch --name ev-test-d1 -- bash -c \
  'python -m evidence.eval_evidence --model d1-3b --release <rilascio> \
     --block test_evid_id test_evid_ood test_evid_sparse --sufficiency > logs/ev-test-d1.out 2>&1'
gpu-run --batch --name ev-latency-d1 -- bash -c \
  'python -m evidence.latency --model d1-3b --release <rilascio> > logs/ev-latency-d1.out 2>&1'
```

Lo script salva `metrics.json` e `predictions.jsonl` (per ogni riga: punteggi, soglia, frasi mostrate, frasi di evidenza), così le metriche si ricalcolano senza rifare i forward.

## 7. Report finale (uno per modello)

| Set | Metrica | E-LEX | E-ATT | E-ABL | Testa |
|---|---|---:|---:|---:|---:|
| test_evid_id | F1 / insieme completo | | | | |
| test_evid_ood | F1 / insieme completo | | | | |
| test_evid_sparse | F1 a ≥ 20 frasi / oltre 4.096 token | | | | |
| tutti | sufficienza | | | | |
| dev_evid_neg | negativi accesi | | | | |
| verdetto | BAcc (uguale al piano binario?) | — | — | — | |
| latenza | p50 / p95, C contro B | | | | |
