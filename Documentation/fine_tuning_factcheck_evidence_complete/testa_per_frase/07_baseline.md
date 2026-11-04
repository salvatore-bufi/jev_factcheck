# Passo 7 — Metodi semplici da battere

## In breve

Prima di addestrare la testa si misura quanto ottengono metodi semplici, sullo stesso set (`dev_evid_select`) e con le stesse metriche ([10](10_valutazione.md)). Servono a due cose: sapere se la testa vale la pena (deve batterli con margine) e capire quanto è difficile il problema sui nostri dati, soprattutto con l'evidenza lontana.

Script: `evidence/baselines.py`. Ogni metodo dà un punteggio per frase; la selezione delle frasi mostrate segue la stessa regola per tutti (§3), così il confronto è sulla qualità del punteggio.

## 1. I metodi

| Nome | Punteggio della frase k | Costo in produzione |
|---|---|---|
| **E-RAND** | casuale | nullo; è il pavimento |
| **E-LEX** | BM25 della frase rispetto al claim (`rank_bm25`, parole minuscole senza stopword, sulle frasi del documento) | nullo; è il riferimento principale |
| **E-EMB** | somiglianza (coseno) fra l'embedding del claim e quello della frase, con un modello di embedding piccolo (`BAAI/bge-small-en-v1.5`) | un encoder in più |
| **E-ATT** | attenzione della posizione di risposta sulla frase (feature `att` del passo 6), combinazione delle migliori coppie (layer, testa) | nullo: legge feature già salvate |
| **E-ABL** | gruppo di frasi trovato con l'ablazione del passo 3, con il modello stesso | centinaia di forward per riga: solo riferimento |

Che cosa dicono:
- **E-LEX** ed **E-EMB** misurano quanto si ottiene con la sola somiglianza al claim. Non possono trovare una frase "ponte" che non ha parole in comune con il claim (nell'esempio di Nancy Howell, la frase sul seminario): sull'evidenza lontana ci si aspetta che crollino. Se la testa non li batte con margine, l'utente avrebbe quasi lo stesso con un confronto di parole.
- **E-ATT** dice se il modello già "guarda" l'evidenza quando decide, e quanto aggiunge addestrare una testa.
- **E-ABL** è il riferimento "di qualità": fedele al modello per costruzione, ma costa centinaia di forward. Dice quanto ci si avvicina con uno solo.

## 2. Come si calcola E-ATT

L'attenzione è misurata in 8 layer × 16 teste: non tutte le combinazioni sono utili.
1. Si divide `dev_evid_select` in due metà, A e B, per `doc_id`.
2. Su A, per ognuna delle 128 coppie (layer, testa) e con il punteggio diviso o no per il numero di token della frase (`att / n_tok^β`, β ∈ {0; 0,5; 1}): AUPRC per frase.
3. Si tengono le 8 combinazioni migliori; il punteggio finale è la media dei loro ranghi normalizzati.
4. Si misura su B. Si riporta anche la migliore combinazione singola.

## 3. Quali frasi si mostrano

I metodi semplici non hanno una soglia propria. Per ciascuno si provano su A:
- le **k migliori**, con k = numero medio di frasi gold;
- una **soglia** sul punteggio, scelta per massimizzare l'F1 per frase (almeno 1, al massimo 5 frasi).

Si riporta la migliore delle due, misurata su B. E-ABL ha già il suo insieme.

## 4. Comandi

```sh
python -m evidence.baselines --method rand lex emb --block dev_evid_select
python -m evidence.baselines --method att --model knowline-4b --block dev_evid_select
gpu-run --batch --name ev-eabl-knowline -- bash -c \
  'python -m evidence.baselines --method abl --model knowline-4b --block dev_evid_select --per-source 300 \
     > logs/ev-eabl-knowline.out 2>&1'
```

E-ABL: 300 righe per fonte e per modello.

## 5. Tabella da compilare (una per modello)

| Metodo | F1 frasi HoVer | Insieme completo HoVer | F1 frasi SciFact | AUPRC | F1 con evidenza a ≥ 10 frasi | F1 oltre 1.024 token |
|---|---:|---:|---:|---:|---:|---:|
| E-RAND | | | | | | |
| E-LEX | | | | | | |
| E-EMB | | | | | | |
| E-ATT | | | | | | |
| E-ABL | | | | | | |

Risultati in `data/evidence/runs/baselines.json`.
