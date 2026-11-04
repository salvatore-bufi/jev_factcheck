# Passo 7 — Metodi semplici da battere

## In breve

Prima di addestrare la testa si misura quanto ottengono metodi semplici, con le stesse metriche ([10](10_valutazione.md)) e la stessa decodifica in tratti ([08](08_testa_e_loss.md) §4). Script: `evidence/token_baselines.py`.

Ogni metodo dà un punteggio per token; poi stessa regola per tutti: scelta di una soglia su metà del dev (A), misura sull'altra metà (B).

## 1. I metodi

| Nome | Punteggio del token t | Costo |
|---|---|---|
| **E-RAND** | casuale, poi decodifica: tratti casuali | pavimento |
| **E-LEX** | 1 se la parola del token (minuscola, senza stopword) compare nel claim; poi media su una finestra di 15 token | nullo; riferimento principale |
| **E-EMB** | somiglianza fra l'embedding del claim e quello della finestra di 30 token centrata su t (modello piccolo, es. `BAAI/bge-small-en-v1.5`) | un encoder in più |
| **E-ATT** | attenzione della posizione di risposta verso t (`att[t]` salvata al passo 6), combinazione migliore di layer e teste scelta su A, media su una finestra di 9 token | nullo |
| **E-ABL** | gruppo di frasi trovato con l'ablazione (offline, frasi del nostro divisore), tutti i loro token a 1 | molti forward: solo riferimento |

E-LEX ed E-EMB non trovano le parti "ponte", che non condividono parole con il claim: sull'evidenza lontana ci si aspetta che crollino.

E-ATT è per natura una baseline per token: dice se il modello "guarda" già l'evidenza, e quanto serve addestrare una testa sopra.

## 2. Comandi

```sh
python -m evidence.token_baselines --method rand lex emb --block dev_evid_select
python -m evidence.token_baselines --method att --model knowline-4b --block dev_evid_select
gpu-run --batch --name evtok-eabl -- bash -c \
  'python -m evidence.token_baselines --method abl --model knowline-4b --block dev_evid_select --per-source 300 \
     > logs/evtok-eabl.out 2>&1'
```

## 3. Tabella da compilare (una per modello)

| Metodo | F1 caratteri HoVer | Frasi gold colpite HoVer | Insieme completo HoVer | F1 caratteri SciFact | F1 evidenza a ≥ 10 frasi | Tratti per riga |
|---|---:|---:|---:|---:|---:|---:|
| E-RAND | | | | | | |
| E-LEX | | | | | | |
| E-EMB | | | | | | |
| E-ATT | | | | | | |
| E-ABL | | | | | | |
