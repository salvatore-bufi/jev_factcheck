# Step 1 — Baseline: misurare tutto prima di addestrare

## Obiettivo

Sapere quanto valgono i modelli **prima** del fine-tuning, con lo stesso protocollo che si userà dopo (split dello Step 0, soglia e temperatura dal nostro dev). Ogni guadagno successivo si misura contro questi numeri.

## 1. Cosa si sa già, dai run dello studio focalizzato

Lo studio focalizzato ([../paper_idea/15_focused_10_tdm_experiment_guide.md](../paper_idea/15_focused_10_tdm_experiment_guide.md) §7) esegue i 10 decision model e i due backbone su AggreFact con il file di domande QF1 (Q1 `support_simple`, Q2 `unsupported_detail`, Q3 `contradicted_detail`, in una richiesta). I risultati sono in `results/lytang--LLM-AggreFact/<split>/<modello>_<hash>_551bb722/answers.jsonl` (`551bb722` è l'hash di QF1): ogni riga contiene le probabilità delle tre domande, quindi la domanda singola si valuta **senza nuova inferenza**.

| Run dello studio | Split | Cosa dà a questo piano |
|---|---|---|
| RUN-01, RUN-02 | AF-dev (4.520 righe) | soglia e temperatura su AggreFact (solo come riferimento: in questo piano si fittano sul nostro dev) |
| RUN-03, RUN-04 | AF test | AUC e BAcc di partenza dei 10 modelli e dei backbone, Q1 e Q2 lette da sole |
| RUN-08 … RUN-11 | AF-4k | Q4 (`support_every_part`) e Q5 (`any_unsupported`), una per richiesta |

Numeri già noti (regola a tre domande di Jev, soglia 0,30): elephant-1 76,70, KnowLine-4B 75,65. Gli altri si compilano qui dai run quando sono finiti.

Script che ricava le colonne della tabella di §6 dai run (da salvare come `finetune/single_question_from_results.py`):

```python
import json, glob, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.metrics import metrics

def scores(answers):
    s = answers["support_simple"]["probs"]["true"]            # P(sì): la chiave "true" vale per ogni modello
    u = 1 - answers["unsupported_detail"]["probs"]["true"]
    c = 1 - answers["contradicted_detail"]["probs"]["true"]
    return {"Q1": s, "1-Q2": u, "min3": min(s, u, c)}

for run in sorted(glob.glob(str(ROOT / "results/lytang--LLM-AggreFact/test/*_551bb722"))):
    rows, sc = [], {"Q1": [], "1-Q2": [], "min3": []}
    with open(f"{run}/answers.jsonl") as f:
        next(f)                                               # intestazione {"identity": ...}
        for line in f:
            r = json.loads(line)
            rows.append({"dataset": r["source"], "label": r["label"]})
            for k, v in scores(r["answers"]).items():
                sc[k].append(v)
    for k, v in sc.items():
        at05 = metrics(rows, v, 0.5)
        auc = np.mean([x["roc_auc"] for x in at05["by_source"].values()])
        print(Path(run).name, k, round(auc * 100, 2), round(at05["macro_balanced_accuracy"] * 100, 2))
```

I backbone salvano anche le due disposizioni delle opzioni (`orders`): la loro lettura calibrata (R1 dello studio, §5.7 della guida) si ricava dagli stessi file.

## 2. Le baseline da misurare

| ID | Modello | Regola | Perché |
|---|---|---|---|
| B1 | i 10 decision model | una domanda: Q1, 1 − Q2, Q4 | **punto di partenza vero** per la domanda che si sceglierà ([02d](02d_varianti_domanda.md)) |
| B2 | idem | tre domande (QF1, regola `min`) | quanto deve recuperare la domanda singola |
| B4 | Qwen3.5-9B e Qwen3.5-4B (adapter `llm-backbone`, prompt neutro, letture R0 e R1) | una domanda | punto di partenza dei controlli di B-ctrl |
| B5 | FactCG-DeBERTa-v3-L | chunk 550 parole, `max`, soglia 0,5 e fittata | riferimento encoder |
| B6 | MiniCheck-Flan-T5-L | idem | riferimento MiniCheck |
| B7 | Bespoke-MiniCheck-7B | procedura ufficiale (adapter `minicheck-7b`) | riferimento LLM con ricetta MiniCheck |

Su: `dev_select`, `dev_confirm`, e (una volta) test_id, test_ood, test_long. Su AggreFact B1 (Q1, Q2) e B2 sono già nei run dello studio (§1).

## 3. Lo scoring sul nostro dev: `benchmark.py`

Non serve uno scorer nuovo: `benchmark.py` esegue già ogni modello con il codice dei suoi autori e legge dataset locali (`--dataset NOME --split SPLIT` legge `dataset/NOME/SPLIT.jsonl`, formato in [../../dataset/README.md](../../dataset/README.md)).

1. Esportare gli split dello Step 0 nel formato del benchmark (`id`, `source`, `document`, `claim`, `label`): `dataset/fc_ours/dev_select.jsonl`, `dev_confirm.jsonl`, `test_id.jsonl`, `test_ood.jsonl`, `test_long.jsonl`.
2. Lanciare ogni modello con QF1 (`config/questions/claim_support.json`) e con la domanda Q4 da sola (`config/questions/q4_only.json`):

```sh
gpu-run --batch --name bl-knowline-dev -- bash -c \
  'python benchmark.py --model knowline-4b --gpu 0 --dataset fc_ours --split dev_select --no-profile > logs/bl-knowline-dev.out 2>&1'
gpu-run --batch --name bl-knowline-dev-q4 -- bash -c \
  'python benchmark.py --model knowline-4b --gpu 0 --dataset fc_ours --split dev_select \
     --questions config/questions/q4_only.json --no-profile > logs/bl-knowline-dev-q4.out 2>&1'
```

(Stesso comando per gli altri modelli; per i backbone `--model llm-backbone --model-config config/models/llm-backbone-qwen3.5-9b.json` o `-4b.json`.)

3. Le probabilità si leggono dagli `answers.jsonl` dei run, come in §1.

## 4. Calibrazione: temperatura e soglia — `finetune/calibrate.py`

### 4.1 Temperatura

Dal punteggio p si ricava il margine m = ln(p / (1 − p)). Una nuova temperatura relativa a scala il margine: p' = σ(m / a). Si sceglie a che **minimizza il log-loss** sul dev:

```python
import numpy as np
def fit_scale(p, y, grid=np.arange(0.3, 5.01, 0.01)):
    p = np.clip(np.asarray(p), 1e-6, 1 - 1e-6); y = np.asarray(y)
    m = np.log(p / (1 - p))
    def nll(a):
        q = 1 / (1 + np.exp(-m / a))
        return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
    return min(grid, key=nll)
```

Esempio: se il modello è troppo sicuro (molti p = 0,99 sbagliati), il minimo cade a a > 1 (es. 1,6): le probabilità si "allargano". La temperatura assoluta è T_nuova = a · T_modello, dove T_modello è quella degli autori (KnowLine: 1, già incorporata nei pesi; elephant-1: 1; StartLux-9B: 1,3396 per le domande sì/no; pplx-decider-flash: 1,554; gli altri: 1). Gli adapter salvano i logit e la temperatura, quindi la scala si applica senza nuova inferenza.

### 4.2 Soglia

Si usa `core.metrics.tune_threshold` della repo: massimizza la BAcc macro (media sulle fonti) su una griglia 0,01–0,99; a parità sceglie la più vicina a 0,5. `core/metrics.py` si aspetta righe con i campi `dataset` e `label`:

```python
from core.metrics import tune_threshold, metrics
rows = [{"dataset": r["source"], "label": r["label"]} for r in recs]
tuned = tune_threshold(rows, [p for p in calibrated])
print(tuned["threshold"], tuned["macro_balanced_accuracy"])
```

### 4.3 ECE

```python
def ece(p, y, bins=10):
    p, y = np.asarray(p), np.asarray(y)
    conf = np.maximum(p, 1 - p); correct = ((p > 0.5) == (y == 1))
    edges = np.linspace(0.5, 1.0, bins + 1); total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            total += mask.mean() * abs(conf[mask].mean() - correct[mask].mean())
    return total
```

### 4.4 Procedura

1. Scala a su `dev_select` (log-loss).
2. Soglia su `dev_select` con le probabilità ricalibrate.
3. Su `dev_confirm`: BAcc macro con quella soglia. Accettabile se il calo rispetto a dev_select è ≤ 1 punto.
4. Si registrano: a (e T assoluta), soglia, BAcc e AUC su select e confirm, BAcc a 0,5, ECE.

## 5. I riferimenti esterni

### 5.1 FactCG (B5)

Esiste già `FactCG/benchmark.py` per AggreFact. Sul nostro dev si riusano `FactCG/factcg_client.py` e `FactCG/chunking.py`:

```python
sys.path.insert(0, str(ROOT / "FactCG"))
import config
from factcg_client import FactCGModel
from chunking import chunk_document
model = FactCGModel(config.MODEL_ID, config.CACHE_DIR, config.MAX_LENGTH)
def score(doc, claim):
    chunks = chunk_document(doc, 550) or [""]
    probs, _ = model.score_pairs(chunks, [claim] * len(chunks))
    return max(probs)                                  # massimo sui chunk
```

### 5.2 MiniCheck (B6, B7)

Pacchetto ufficiale (da installare in un ambiente separato se le dipendenze confliggono):

```sh
pip install "minicheck @ git+https://github.com/Liyan06/MiniCheck.git@main"
```

```python
from minicheck.minicheck import MiniCheck
scorer = MiniCheck(model_name="flan-t5-large", cache_dir="./ckpts")           # B6
# scorer = MiniCheck(model_name="Bespoke-MiniCheck-7B", enable_prefix_caching=False, cache_dir="./ckpts")  # B7, usa vLLM
pred_label, raw_prob, _, _ = scorer.score(docs=[doc1, doc2], claims=[claim1, claim2])
```

`raw_prob` è P(supportato) dopo chunking e `max` interni. Verificare i nomi dei parametri sul README del pacchetto installato: l'interfaccia indicata è quella documentata dagli autori al momento della scrittura.

## 6. Tabella da compilare

Una colonna per modello (i 10 decision model, i 2 backbone in lettura R1, i 3 riferimenti esterni). Per ogni modello, la domanda migliore e le tre domande:

| Blocco | Metrica | KnowLine-4B | elephant-1 | … altri 8 … | BB4 (R1) | BB9 (R1) | FactCG | MiniCheck-FT5 | Bespoke-7B |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dev_select | BAcc / AUC (Q1, 1 − Q2, Q4, min3) | | | | | | | | |
| dev_select | BAcc a 0,5 | | | | | | | | |
| dev_select | soglia ottimale / ECE | | | | | | | | |
| dev_confirm | BAcc | | | | | | | | |
| test_id / test_ood / test_long | BAcc | | | | | | | | |
| AggreFact (soglia dal nostro dev) | BAcc / AUC, con e senza RAGTruth | | | | | | | | |
| Solo claim | P media | | | | | | | | |
| Costo | ms per claim | | | | | | | | |

Per AggreFact con soglia dal nostro dev non serve nuova inferenza se la domanda scelta è Q1 o Q2: le risposte sono nei run dello studio; si applicano scala e soglia. Per Q4 servono i run su AF test (lo studio la esegue solo su AF-4k).

## Criteri per passare allo step successivo

- B1, B2, B4, B5 misurati su dev_select e dev_confirm per tutti i modelli; test una volta.
- Domanda scelta e scritta in `finetune/question.json` ([02d](02d_varianti_domanda.md)).
- Scala e soglia di partenza congelate per ogni modello.

## Errori comuni

- Confrontare il fine-tunato a una domanda con la baseline a tre domande senza dirlo.
- Usare la soglia della baseline sul modello fine-tunato: va rifittata.
- Leggere un decision model con un prompt diverso da quello dei suoi autori.
- Rieseguire i test della baseline mentre si sperimenta.
