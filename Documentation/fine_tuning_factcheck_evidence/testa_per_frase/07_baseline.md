# Passo 7 — Metodi semplici da battere

## In breve

[`evidence/baselines.py`](../../../evidence/baselines.py) misura quattro metodi che danno un punteggio a ogni frase **senza** addestrare niente. Si misurano con le stesse metriche della testa ([10](10_valutazione.md)). Se la testa non li batte con margine, non vale il suo costo.

```sh
F=data/evidence/features/knowline-4b; S=dataset/evidence/splits
python -m evidence.baselines --features $F/dev_evid_select --rows $S/dev_evid_select.jsonl --method rand lex att
python -m evidence.baselines --features $F/dev_evid_select --rows $S/dev_evid_select.jsonl --method abl --per-source 100
```

Risultati in `data/evidence/runs/baselines.json`.

## 1. Protocollo

Il set si divide in due metà per documento: $u_{baseline\_half}(\textit{doc}) < 0{,}5$ dà A, il resto B. Tutto ciò che va scelto si sceglie su **A**; i numeri riportati sono misurati su **B**. Per ogni metodo si sceglie su A la migliore fra due regole di selezione (`evaluate_method`):

- **top-$k$**: le $k$ frasi con punteggio più alto, con $k = \mathrm{round}(\text{media di } \lvert G\rvert \text{ su A})$;
- **soglia** $t \in \{0{,}05; 0{,}10; \ldots; 0{,}95\}$ sul punteggio normalizzato nella riga, $\hat s_k = (s_k - \min s)/(\max s - \min s)$, con la regola standard (al massimo 5 frasi, almeno 1).

## 2. I metodi

**E-RAND** (`rand_scores`): $s_k \sim U(0,1)$, riproducibile per riga. È il pavimento.

**E-LEX** (`lex_scores`, `bm25`): BM25 della frase contro il claim, scritto a mano perché `rank_bm25` non è installato. Parole minuscole, senza stopword. Le "collezioni" sono le $n$ frasi del documento:

$$
s_k = \sum_{w \in \text{claim}} \mathrm{IDF}(w)\, \frac{\mathrm{tf}_{k}(w)\,(k_1+1)}{\mathrm{tf}_{k}(w) + k_1\big(1 - b + b\,\frac{\lvert S_k\rvert}{\overline{\lvert S\rvert}}\big)},\qquad
\mathrm{IDF}(w) = \log\Big(1 + \frac{n - \mathrm{df}(w) + 0{,}5}{\mathrm{df}(w) + 0{,}5}\Big),
$$

con $k_1 = 1{,}5$ e $b = 0{,}75$. È il riferimento principale: misura quanto si ottiene con le sole parole in comune. Non può trovare una frase "ponte" senza parole del claim, come la frase sul seminario nell'esempio di Nancy Howell.

**E-ATT** (`choose_att_combos`, `att_scores`): usa le feature $att_k$ ([01](01_modello_e_hook.md) §4), cioè quanto il modello guarda la frase quando decide. Per ognuna delle $128 \times 3$ combinazioni (colonna $c$, normalizzazione per lunghezza $\beta \in \{0; 0{,}5; 1\}$), con punteggio $att_k[c] / n_k^{\beta}$, si calcola l'AUPRC per frase sui positivi di A. Si tengono le 8 combinazioni migliori $\mathcal{C}$, e il punteggio finale è la media dei ranghi normalizzati:

$$
s_k = \frac{1}{\lvert\mathcal C\rvert}\sum_{(c,\beta)\in\mathcal C} \mathrm{rank}_k\Big(\frac{att[c]}{n^{\beta}}\Big),\qquad \mathrm{rank}_k \in [0,1].
$$

Non costa nulla in produzione, ed è il riferimento da battere per dire "la testa aggiunge qualcosa".

**E-ABL** (`ablation_group`): ricerca greedy, con il modello, del gruppo più piccolo di frasi che **da solo** fa dire "supportato". Sia $p(K)$ = P(supportato) con il documento ridotto alle frasi di $K$ (`extract.rebuild`):

1. $K \leftarrow \emptyset$; finché $\lvert K\rvert < 6$: aggiungi $k^\ast = \arg\max_{k \notin K} p(K \cup \{k\})$; fermati se $p(K) > 0{,}5$ (altrimenti la riga non ha gruppo);
2. pulizia: per ogni $k \in K$, se $p(K \setminus \{k\}) > 0{,}5$ toglilo.

Servono centinaia di forward per riga, ma su documenti ridotti e quindi corti: circa 2 s per riga. È fedele al modello per costruzione. Serve da riferimento "di qualità": quanto ci si avvicina con un forward solo a un metodo che ne usa centinaia (controllo C3).

## 3. Prima prova (300 righe di `dev_evid_select`, metà B: 146 righe valutate)

| Metodo | F1 per frase | Insieme completo | AUPRC | F1 con gold a ≥ 10 frasi |
|-----------------------|---------------------:|------------------------:|-------------:|-----------------------------:|
| E-RAND | 0,07 | 0,01 | 0,06 | 0,06 |
| E-LEX | 0,60 | 0,22 | 0,58 | 0,60 |
| E-ATT | 0,75 | 0,42 | 0,81 | 0,76 |
| E-ABL (5 righe) | 0,45 | 0,00 | — | — |

**Che cosa dicono:**

- l'attenzione del modello è già un segnale forte, molto sopra BM25;
- E-ABL trova spesso **una sola** frase sufficiente al modello, anche quando le annotazioni ne indicano più di una: KnowLine sembra dire "supportato" con evidenza parziale. Va verificato sul set completo, perché pesa anche sulla metrica di sufficienza ([10](10_valutazione.md) §2).

## 4. Codice

| Funzione | Ruolo |
|------------------------------------------|------------------------------------------|
| `words`, `bm25`, `lex_scores` | E-LEX |
| `rand_scores` | E-RAND |
| `att_columns`, `ranks`, `choose_att_combos`, `att_scores` | E-ATT |
| `ablation_group` | E-ABL (GPU) |
| `make_preds`, `evaluate_method` | regole top-$k$ / soglia, scelta su A, misura su B |
