# Passo 7 — Baseline dell'evidenza

## Obiettivo

Avere, per ciascun modello, i numeri che la testa deve battere, su `dev_evid_select` e con le metriche di [10](10_valutazione.md). Script: `evidence/baselines.py`.

Le baseline producono, per ogni riga, un punteggio per frase; la selezione delle frasi mostrate segue la stessa regola per tutte (§2), così il confronto è sulla qualità del punteggio.

## 1. Le baseline

| ID | Punteggio della frase k | Serve il modello? |
|---|---|---|
| E-RAND | casuale (seme fisso) | no |
| E-LEX | BM25 della frase rispetto al claim (`rank_bm25`, token minuscoli senza stopword, sui soli documenti della riga) | no |
| E-EMB | coseno fra gli embedding di claim e frase, con un modello di embedding piccolo (es. `BAAI/bge-small-en-v1.5`, da scaricare) | un encoder |
| E-ATT | massa di attenzione di `answer_pos` sulla frase, dalla feature `att` già estratta ([06](06_rendering_ed_estrazione.md) §4), combinata come sotto | no: legge le feature |
| E-ABL | ablazione greedy con il verdetto del modello (algoritmo del passo [03](03_dati_generati.md) §3); qui si usa direttamente l'insieme scelto | molti forward |

### E-ATT nel dettaglio

1. Dividere `dev_evid_select` in due metà per `doc_id` (A e B).
2. Su A, per ogni layer pieno e testa (8 × H combinazioni), punteggio = `att[k, l, h] / n_tok[k]^β` con β ∈ {0; 0,5; 1}; misurare l'AUPRC per frase.
3. Prendere le 8 combinazioni (l, h, β) migliori su A; punteggio finale = media dei loro ranghi normalizzati.
4. Misurare su B. Riportare anche la migliore combinazione singola.

Si misura nel formato con preambolo e, per confronto, nel formato nativo (serve un'estrazione col prompt nativo solo per `dev_evid_select`).

### E-ABL nel dettaglio

300 righe per fonte (FEVER, HoVer, C2D con posizioni) di `dev_evid_select`, per ciascun modello. È il riferimento "fedele per costruzione": dice quanto si avvicina la testa in un forward a un metodo con molti forward.

```sh
gpu-run --batch --name ev-eabl-jpt -- bash -c \
  'python -m evidence.baselines --method abl --model jpt-9b --block dev_evid_select --per-source 300 \
     > logs/ev-eabl-jpt.out 2>&1'
```

## 2. Regola di selezione per le baseline

Le baseline non hanno una soglia propria. Per ciascuna:
- **top-k**: le k frasi con punteggio più alto, k = numero medio di frasi gold sul blocco A (arrotondato);
- **soglia**: soglia sul punteggio scelta su A per massimizzare l'F1 per frase, con almeno 1 e al massimo 5 frasi.

Si riporta la migliore delle due, misurata su B. E-ABL ha già il suo insieme.

## 3. Tabella da compilare (una per modello)

| Metodo | F1 frase FEVER | F1 frase HoVer | Insieme completo HoVer | AUPRC | F1 ≥ 10 frasi | F1 > 1.024 token |
|---|---:|---:|---:|---:|---:|---:|
| E-RAND | | | | | | |
| E-LEX | | | | | | |
| E-EMB | | | | | | |
| E-ATT (preambolo) | | | | | | |
| E-ATT (nativo) | | | | | | |
| E-ABL (300 per fonte) | | | | | | |

E-RAND, E-LEX ed E-EMB non dipendono dal modello: si calcolano una volta. Righe sparse (`≥ 10 frasi`) e lontane (`> 1.024 token`) sono i casi in cui E-LEX ed E-EMB dovrebbero cadere: è la misura di quanto il problema sia reale sui nostri dati.

## Prodotto

- `data/evidence/runs/baselines.json` con la tabella per i tre modelli.
- Il valore di E-LEX su `dev_evid_select` è la soglia da superare di 15 punti di F1 ([00](00_overview.md)).
