# Passo 7 — Metodi semplici da battere

## In breve

Prima di addestrare la testa si misura quanto ottengono metodi semplici, sullo stesso set (`dev_evid_select`) e con le stesse metriche ([10](10_valutazione.md)). Servono a due cose: sapere se la testa vale la pena (deve batterli con margine) e capire quanto è difficile il problema sui nostri dati (soprattutto con l'evidenza lontana).

Script: `evidence/baselines.py`.

## 1. I metodi

| Nome | Come dà un punteggio a ogni frase | Costo |
|---|---|---|
| **E-RAND** | a caso | nullo; è il pavimento |
| **E-LEX** | quante parole ha in comune col claim (BM25) | nullo; è il riferimento principale da battere |
| **E-EMB** | somiglianza di significato fra frase e claim, con un piccolo modello di embedding (es. `BAAI/bge-small-en-v1.5`) | un modello in più |
| **E-ATT** | quanta attenzione il decision model dà alla frase dalla posizione di risposta (la feature `att` già salvata al passo 6) | nullo: legge i dati salvati |
| **E-ABL** | gruppo di frasi trovato con l'ablazione del passo [03](03_dati_generati.md) §2, col modello stesso | molti forward per riga: solo come riferimento |

E-LEX ed E-EMB non possono trovare una frase ponte che non ha parole in comune col claim (come la frase 1 dell'esempio Nancy Howell): sull'evidenza lontana ci si aspetta che crollino.

E-ABL è il riferimento "di qualità": dice quanto ci si avvicina, con un solo forward, a un metodo che ne usa centinaia.

## 2. Come si calcola E-ATT

L'attenzione è misurata in 8 layer e 16–32 teste: non tutte le combinazioni sono utili.
1. Si divide `dev_evid_select` in due metà, A e B.
2. Su A si prova ogni combinazione di layer e testa, con il punteggio diviso o no per la lunghezza della frase, e si tengono le 8 migliori.
3. Il punteggio finale è la media delle 8; si misura su B.

Si calcola sia con il preambolo sia senza (per il secondo serve un'estrazione col prompt senza preambolo, solo su `dev_evid_select`): dice se il preambolo aiuta il modello a guardare le frasi giuste.

## 3. Quali frasi si mostrano

I metodi semplici non hanno una soglia propria. Per ciascuno si prova:
- le **k migliori**, con k = numero medio di frasi di evidenza;
- una **soglia** sul punteggio, scelta su A (da 1 a 5 frasi).

Si riporta la migliore delle due, misurata su B.

## 4. Comandi

```sh
python -m evidence.baselines --method rand lex emb --block dev_evid_select
python -m evidence.baselines --method att --model d1-3b --block dev_evid_select
gpu-run --batch --name ev-eabl-d1 -- bash -c \
  'python -m evidence.baselines --method abl --model d1-3b --block dev_evid_select --per-source 300 > logs/ev-eabl-d1.out 2>&1'
```

E-ABL: 300 righe per fonte e per modello.

## 5. Tabella da compilare (una per modello)

| Metodo | F1 FEVER | F1 HoVer | Insieme completo HoVer | F1 evidenza a ≥ 10 frasi | F1 evidenza oltre 1.024 token |
|---|---:|---:|---:|---:|---:|
| E-RAND | | | | | |
| E-LEX | | | | | |
| E-EMB | | | | | |
| E-ATT con preambolo | | | | | |
| E-ATT senza preambolo | | | | | |
| E-ABL | | | | | |

Risultati in `data/evidence/runs/baselines.json`.
