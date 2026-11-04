# Passo 7 — Metodi semplici da battere

## In breve

Prima di addestrare la testa si misura quanto ottengono metodi semplici, sugli stessi set, con le stesse metriche ([10](10_valutazione.md)) e la stessa decodifica in tratti ([08](08_testa_e_loss.md) §5). Servono a due cose: sapere se la testa vale la pena (deve batterli con margine) e capire quanto è difficile il problema sui nostri dati, soprattutto con l'evidenza lontana.

Script: `evidence/token_baselines.py`. Ogni metodo dà un punteggio per token; per i metodi senza soglia propria, la soglia si sceglie su metà del dev (A) e si misura sull'altra metà (B).

## 1. I metodi

| Nome | Punteggio del token t | Costo in produzione |
|---|---|---|
| **E-RAND** | casuale | nullo; è il pavimento |
| **E-LEX** | 1 se la parola del token (minuscola, senza stopword) compare nel claim, altrimenti 0; poi media mobile su 15 token | nullo; è il riferimento principale |
| **E-EMB** | somiglianza (coseno) fra l'embedding del claim e quello della finestra di 30 token centrata su t, con un modello di embedding piccolo (`BAAI/bge-small-en-v1.5`) | un encoder in più |
| **E-ATT** | attenzione della posizione di risposta verso t (feature `att` del passo 6); combinazione delle 8 coppie (layer, testa) migliori su A; media mobile su 9 token | nullo: legge feature già salvate |
| **E-ABL** | frasi trovate con l'ablazione del passo 3 (molti forward), tutti i loro token a 1 | centinaia di forward per riga: solo riferimento |

Che cosa dicono:
- **E-LEX** ed **E-EMB** misurano quanto si ottiene con la sola somiglianza al claim. Non possono trovare le parti "ponte", che non condividono parole con il claim (nell'esempio di Nancy Howell, la frase sul seminario): sull'evidenza lontana ci si aspetta che crollino. Se la testa non li batte con margine, l'utente avrebbe quasi lo stesso con un confronto di parole.
- **E-ATT** è per natura una baseline per token: dice se il modello già "guarda" l'evidenza, e quanto aggiunge addestrare una testa.
- **E-ABL** è il riferimento "di qualità": fedele al modello per costruzione, ma costa centinaia di forward. Dice quanto ci si avvicina con uno solo.

## 2. Come si sceglie la combinazione di E-ATT

1. Si divide `dev_evid_select` in A e B per `doc_id`.
2. Su A, per ognuna delle 128 coppie (layer pieno, testa): AUPRC per token dell'attenzione rispetto alle etichette.
3. Si tengono le 8 coppie migliori; il punteggio è la media dei loro ranghi normalizzati, poi media mobile su 9 token.
4. Si misura su B.

## 3. Comandi

```sh
python -m evidence.token_baselines --method rand lex emb --block dev_evid_select
python -m evidence.token_baselines --method att --model knowline-4b --block dev_evid_select
gpu-run --batch --name evtok-eabl -- bash -c \
  'python -m evidence.token_baselines --method abl --model knowline-4b --block dev_evid_select --per-source 300 \
     > logs/evtok-eabl.out 2>&1'
```

E-ABL: 300 righe per fonte.

## 4. Tabella da compilare (una per modello)

| Metodo | Frasi gold colpite HoVer | Insieme completo HoVer | F1 caratteri HoVer | Frasi colpite SciFact | Colpite con evidenza a ≥ 10 frasi | Colpite senza punteggiatura | Tratti per riga |
|---|---:|---:|---:|---:|---:|---:|---:|
| E-RAND | | | | | | | |
| E-LEX | | | | | | | |
| E-EMB | | | | | | | |
| E-ATT | | | | | | | |
| E-ABL | | | | | | | |
