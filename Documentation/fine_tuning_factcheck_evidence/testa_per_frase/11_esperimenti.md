# Passo 11 — Esperimenti

## In breve

Si parte da una testa semplice (**H0**) e si aggiunge **un pezzo alla volta**, tenendolo solo se migliora. Tutto si fa su KnowLine-4B, con le feature salvate (fase 1a): una prova è un training della sola testa, di pochi minuti.

Un esperimento sulla **testa** o sulla **loss** è un file in [`evidence/configs/`](../../../evidence/configs/). Un esperimento sui **dati** (pesi, $\omega$, negativi) è una ricetta in [`evidence/recipes/`](../../../evidence/recipes/) più un nuovo mix, da estrarre di nuovo.

## 1. Regole di decisione

1. Una modifica per prova.
2. Si decide su `dev_evid_select` con quattro numeri: F1 per frase, insieme completo, F1 con evidenza a ≥ 10 frasi, F1 con evidenza oltre 1.024 token. In più si guardano i negativi accesi su `dev_evid_neg`.
3. **Una modifica si tiene** se aumenta l'F1 di almeno 1 punto o l'insieme completo di almeno 2, senza far scendere gli altri numeri di più di 1 punto.
4. A parità di risultato, si tiene la versione più semplice.
5. Guadagni sotto i 2 punti si ripetono con altri 2 semi: `--seed 14 --tag seed14`, `--seed 15 --tag seed15`.

Il registro `data/evidence/runs/registry.md` è scritto in automatico da `train_head.append_registry`, con una riga per run: run, config, feature, F1, insieme completo, F1 ≥ 10 frasi, F1 > 1.024 token, negativi accesi, descrizione. La colonna **decision** la compila chi decide.

## 2. Le configurazioni pronte

| File | Che cosa cambia rispetto al precedente | Layer | Parti | Loss | Righe | Che cosa si guarda |
|---|---|---|---|---|---|---|
| `h0.json` | testa di partenza: niente miscela, niente media, niente attenzione | [31] | end | `ref` | supervised | batte E-LEX ed E-ATT? Se no, cercare errori nel rendering |
| `h_loss.json` | soglia del documento (ZLPR) | [31] | end | `zlpr` | supervised | insieme completo, frasi mostrate |
| `h_mil.json` | + negativi, $\mu_{neg} = 0{,}3$ | [31] | end | `zlpr` | + mil_neg | negativi accesi, `recall_uncertain` |
| `h_mil1.json` | idem, $\mu_{neg} = 1$ | [31] | end | `zlpr` | + mil_neg | idem |
| `h2a.json` | + 4 layer miscelati | 19–31 | end | `zlpr` | + mil_neg | F1; pesi $\pi$ |
| `h2b.json` | + attenzione $att_k$ | 19–31 | end | `zlpr` | + mil_neg | F1 lontano; frasi "ponte" |
| `h2c.json` | + media dei token (**testa completa**) | 19–31 | end, mean | `zlpr` | + mil_neg | F1 sulle frasi lunghe |
| `h3a.json` | testa completa **senza** transformer | 19–31 | end, mean | `zlpr` | + mil_neg | atteso un crollo su HoVer e sull'evidenza lontana |
| `h3c1.json`, `h3c4.json` | 1 o 4 layer di transformer | 19–31 | end, mean | `zlpr` | + mil_neg | quanta capacità serve |
| `h9.json` | senza vettore di posizione | 19–31 | end, mean | `zlpr` | + mil_neg | preferenza per inizio o fine |
| `h_pair.json` | + punteggio per coppie, peso 0,2 | 19–31 | end, mean | `zlpr` + pair | + mil_neg | solo se l'insieme completo su HoVer resta < 0,65 |

Le configurazioni H3a, H3c, H9 e H-PAIR partono dalla testa completa. Se gli esperimenti principali scelgono una testa diversa, vanno adattate.

## 3. Esperimenti che richiedono altre feature o un altro mix

| Nome | Come si fa |
|---|---|
| **H4** (senza preambolo) | `python -m evidence.extract --rows <mix> <dev...> --preamble none --suffix _nopre`, poi `train_head` con `--train $F/ev_head_nopre --dev $F/dev_evid_select_nopre` |
| **H7** ($\omega$ = 0,3 / 0,5 / 1) | copiare `recipes/ev_head.json`, cambiare `omega`, fare un nuovo `make_mix`. Le feature **non** dipendono da $\omega$, che sta solo nell'indice: basta ri-estrarre il nuovo mix oppure modificare `omega` in `index.jsonl` |
| **H12** (peso di `db_split` = 0 / 0,5 / 1) | come H7, con `row_weight` della famiglia `hover_db_split` |
| quota di negativi | `neg_fraction` nella ricetta |

Siccome `row_weight` e `omega` stanno in `index.jsonl` e non negli shard, un esperimento sui pesi si può fare senza ri-estrarre: basta riscrivere l'indice con i nuovi pesi.

## 4. Sequenza consigliata

```text
E-RAND, E-LEX, E-ATT (baselines.py)  →  H0  →  H-LOSS  →  H-MIL (0,3 / 1)  →  H2a  →  H2b  →  H2c
   →  controlli sulla migliore: H3a, H3c1/H3c4, H9, H4, H7, H12  →  (H-PAIR se serve)
   →  3 semi della finale  →  E-ABL su 100 righe per fonte (confronto C3)  →  rilascio  →  test una volta
```

## 5. Che cosa NON si può fare con questa implementazione

- H2d (media pesata dei token decisa dalla testa) richiede la fase 1b, con il modello in memoria;
- H6 ed E6 (fase 2, LoRA del modello) non sono implementati;
- H5 e H8 (dati generati e spostati) dipendono dal passo 03;
- E0 e X0 (elephant-1 e gli altri modelli) sono fuori perimetro ([14](14_strada_da_esplorare.md)).
