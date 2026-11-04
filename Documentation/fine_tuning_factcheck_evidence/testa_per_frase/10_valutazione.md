# Passo 10 — Valutazione e latenza

## In breve

Per ogni testa si misurano tre cose:

1. le **frasi mostrate** sono quelle giuste, anche lontane fra loro o dall'inizio del documento;
2. il **verdetto** non è cambiato;
3. la **latenza** resta quella di un decision model.

Codice: [`evidence/metrics.py`](../../../evidence/metrics.py) (le metriche, comuni a baseline, training e valutazione), [`evidence/eval_evidence.py`](../../../evidence/eval_evidence.py) e [`evidence/latency.py`](../../../evidence/latency.py).

```sh
python -m evidence.eval_evidence --run data/evidence/runs/knowline-4b/<id> --features $F/dev_evid_select \
    --rows dataset/evidence/splits/dev_evid_select.jsonl --sufficiency 300
python -m evidence.latency --run data/evidence/runs/knowline-4b/<id> --n 1000
```

## 1. Su quali righe

Il sistema mostra frasi solo se $P(\text{supportato}) > 0{,}5$. L'evidenza si misura quindi sulle righe

$$
\mathcal{E} = \{\, b : y_b = 1,\ G_b \neq \emptyset,\ P_b > 0{,}5 \,\}.
$$

Le righe positive con $P_b \le 0{,}5$ si contano a parte (`missed_by_verdict`).

Tutte le metriche partono dallo stesso formato di **predizione**, uno per riga: `id`, `source`, `label`, `p_supported`, `scores`, `threshold`, `shown`, `evidence_sets`, `gold_tok_last`. Le producono `train_head.predictions` per la testa e `baselines.make_preds` per le baseline.

## 2. Metriche dell'evidenza (`summarize`)

Per una riga con frasi mostrate $\mathcal{S}$, insiemi gold alternativi $G^{(1)}, \ldots, G^{(m)}$ e unione $G = \bigcup_r G^{(r)}$ (`sentence_metrics`):

$$
\mathrm{P} = \frac{\lvert \mathcal{S}\cap G\rvert}{\lvert\mathcal{S}\rvert},\quad
\mathrm{R} = \frac{\lvert \mathcal{S}\cap G\rvert}{\lvert G\rvert},\quad
\mathrm{F1} = \frac{2\,\mathrm{P}\,\mathrm{R}}{\mathrm{P}+\mathrm{R}},
$$

$$
\text{completo} = \mathbb{1}\big[\exists r:\ G^{(r)} \subseteq \mathcal{S}\big],\quad
\text{esatto} = \mathbb{1}\big[\exists r:\ G^{(r)} = \mathcal{S}\big].
$$

Esempio: con $\mathcal{S} = \{1,4\}$ e insiemi gold $\{1,4\}$ e $\{7\}$ si ha P = 1, R = 2/3 e F1 = 0,8; l'insieme è completo ed esatto.

| Metrica (chiave) | Definizione | Perché |
|----------------------------------|--------------------------------------------------------|---------------------------------------------------|
| `precision`, `recall`, `f1` | media su $\mathcal{E}$ | qualità per frase |
| `full_set` | media di "completo" | **per l'evidenza multi-frase è quella che conta**: metà evidenza non convince |
| `exact` | media di "esatto" | |
| `auprc` | area sotto la curva precisione–recall su tutte le frasi di $\mathcal{E}$ | qualità dell'ordinamento, indipendente dalla soglia |
| `count_error_mean`, `shown_mean` | $\lvert\mathcal{S}\rvert - \lvert G\rvert$; $\lvert\mathcal{S}\rvert$ | troppe o troppo poche frasi |
| `f1_by_sent_distance` | F1 per bucket di $\max G - \min G$: 0–1 / 2–9 / 10–19 / ≥ 20 | evidenza sparsa |
| `f1_by_tok_distance` | F1 per bucket di `gold_tok_last` = $b_{\max G}$ − `doc_tok_start`: < 1.024 / 1.024–4.096 / > 4.096 | evidenza in fondo a contesti lunghi |
| `f1_far_10plus`, `f1_beyond_1024_tok` | F1 con gold a ≥ 10 frasi; con l'ultima gold oltre 1.024 token | i numeri del registro |
| `recall_uncertain` | recall sulle righe con $0{,}5 < P \le 0{,}65$ | effetto di `mil_neg` sui positivi incerti |
| `negatives_lit` | quota di negativi con almeno una frase sopra la propria soglia | la testa non deve accendere frasi solo perché somigliano al claim |
| `score` | $\frac12(\mathrm{F1} + \text{completo})$ | criterio del checkpoint |
| `by_source` | F1 e completo per fonte | |

### 2.1 Sufficienza e necessità (`eval_evidence.sufficiency`, GPU)

Non dipendono dalle annotazioni, che possono essere incomplete. Sia $p(\cdot)$ il verdetto del modello su un documento ridotto (`extract.rebuild`):

$$
\text{sufficienza} = \frac{1}{\lvert\mathcal{E}\rvert}\sum_{b\in\mathcal{E}} \mathbb{1}\big[p(\mathcal{S}_b) > 0{,}5\big], \qquad
\text{necessità} = \frac{1}{\lvert\mathcal{E}\rvert}\sum_{b\in\mathcal{E}} \mathbb{1}\big[p(\overline{\mathcal{S}_b}) \le 0{,}5\big].
$$

- **Sufficienza**: le sole frasi mostrate bastano a far dire "supportato".
- **Necessità**: senza di esse il modello non dice più "supportato". Si riporta soltanto, perché fallisce anche con evidenza corretta quando l'informazione è ripetuta nel documento.

Servono 2 forward per riga, e si calcola solo in valutazione. Sul campione di 20 righe: sufficienza 0,90, necessità 0,65.

## 3. Metriche del verdetto (`verdict_metrics`)

Su tutte le righe del set, con $\hat y_b = \mathbb{1}[P_b > 0{,}5]$:

$$
\mathrm{BAcc} = \frac{1}{2}\Big(\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}} + \frac{\mathrm{TN}}{\mathrm{TN}+\mathrm{FP}}\Big),\qquad
\mathrm{ECE} = \sum_{c=1}^{10} \frac{\lvert B_c\rvert}{N}\,\Big\lvert \underbrace{\tfrac{1}{\lvert B_c\rvert}\textstyle\sum_{b\in B_c} y_b}_{\text{accuratezza}} - \underbrace{\tfrac{1}{\lvert B_c\rvert}\textstyle\sum_{b\in B_c} P_b}_{\text{confidenza}} \Big\rvert ,
$$

dove $B_c$ sono le righe con $P$ nel decimo $c$. L'AUC si calcola con `roc_auc_score`.

Nella fase 1a il modello è congelato, quindi il verdetto **deve** coincidere con quello senza testa: se non coincide, c'è un errore nella pipeline. Le stesse metriche servono a `check_preamble` per confrontare i preamboli ([05](05_formato_prompt.md) §4).

## 4. Che cosa si salva

`<run>/eval/<set>/metrics.json` (`evidence`, `verdict`, `rule`, `sufficiency`) e `predictions.jsonl`: per ogni riga, punteggi, soglia, frasi mostrate e frasi gold. Le metriche si possono ricalcolare senza rifare i forward.

**I test** (`test_evid_id`, `test_evid_ood`) si misurano **una volta sola**, a configurazione congelata, sulla release ([12](12_export_e_servizio.md)). Le loro feature si estraggono a parte.

## 5. Latenza (`latency.py`)

Stesso processo e stessa GPU, con richieste reali (di default dalle righe di `test_evid_id`; le prime 50 non si contano). Tre condizioni:

| | Prompt | Uscita |
|------|-----------------------|--------------------------------------------------------|
| A | senza preambolo | solo verdetto |
| B | con preambolo | solo verdetto |
| C | con preambolo | verdetto + evidenza (hook, righe di attenzione, testa, selezione): ciò che fa il server |

Obiettivo: $C - B \le \max(0{,}03\,B,\ 2\,\text{ms})$. Si misurano a parte anche la divisione in frasi e la testa.

Misurato su RTX 5090 Laptop, 40 richieste di `dev_evid_select` (prova):

| | p50 | p95 |
|-------------------------------|-----------------------------------:|-----------------:|
| A | 322,9 ms | 480,4 ms |
| B | 337,0 ms | 484,7 ms |
| C | 338,9 ms | 490,3 ms |
| divisione in frasi | 1,3 ms | 1,8 ms |
| testa + selezione | 1,1 ms | 1,2 ms |
| **C − B** | **+1,8 ms (+0,5%)**, entro l'obiettivo | |
| B − A (costo del preambolo) | +14,1 ms | |

## 6. Codice

| Funzione | File | Ruolo |
|-------------------------------------|------------------------|-----------------------------------|
| `select`, `sentence_metrics`, `auprc` | `metrics.py` | selezione e metriche per riga |
| `sent_bucket`, `tok_bucket`, `gold_tok_last` | `metrics.py` | bucket di distanza |
| `summarize` | `metrics.py` | tutte le metriche di un set |
| `verdict_metrics`, `sufficiency`, `main` | `eval_evidence.py` | verdetto, sufficienza, salvataggio |
| `percentiles`, `main` | `latency.py` | condizioni A/B/C |

Non implementato: le prove di comportamento della doc del piano 10 §4 (evidenza spostata, frasi di disturbo, frasi riscritte), che dipendono dal passo 03 ([03](03_dati_generati.md)).
