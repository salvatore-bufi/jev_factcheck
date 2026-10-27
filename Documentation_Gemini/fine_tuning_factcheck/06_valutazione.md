# Step 6 — Valutazione: metriche, test comportamentali, report

## Obiettivo

Misurare se il modello fine-tunato è un fact checker migliore della baseline e dei riferimenti esterni: sulle fonti viste, su domini nuovi, su documenti lunghi, su LLM-AggreFact zero-shot, con probabilità utilizzabili a soglia fissa e un costo noto.

## 1. La suite

| Blocco | Contenuto | Quando | Serve a decidere? |
|---|---|---|---|
| dev_select | ~60% del dev delle fonti di training | a ogni esperimento | **sì** |
| dev_confirm | ~40% del dev | una volta per fase | solo conferma |
| test_id | test delle fonti di training | baseline e fine | no |
| test_ood | SciFact, Climate-FEVER, HealthVer, COVID-Fact, HaluBench, HalluMix, LLM-Oasis | baseline e fine | no |
| test_long | righe lunghe di test_id e test_ood | baseline e fine | no |
| AggreFact test | 11 fonti, 29.320 righe | baseline e fine | no |
| Comportamentali | generati da dev_select (§4) | a ogni esperimento | come vincolo |
| Costo | 1.000 righe miste | modello finale | no |

## 2. Metriche e codice

Tutte calcolate su P(supportato) già ricalibrato (scala e soglia da dev_select, [01](01_baseline.md) §4).

| Metrica | Definizione | Codice |
|---|---|---|
| BAcc per fonte | ½ (TPR + TNR) | `metrics.metrics(rows, scores, thr)["by_source"]` |
| BAcc macro | media delle BAcc per fonte | `metrics.metrics(...)["macro_balanced_accuracy"]` |
| AUC per fonte e macro | area sotto la curva ROC | `by_source[s]["roc_auc"]`, media |
| BAcc a 0,5 | soglia fissa | `metrics.metrics(rows, scores, 0.5)` |
| ECE | 10 fasce di confidenza | `ece()` di [01](01_baseline.md) §4.3 |
| Brier | media di (p − y)² | `np.mean((p - y) ** 2)` |
| Soglia ottimale | da `tune_threshold` su dev_select | segnale del bias "sì" |
| Recall per classe | TPR e TNR separati | `supported_recall`, `unsupported_recall` |
| BAcc per tipo di negativo | TNR sui negativi `contradicted` e su quelli `not_established` | filtro su `neg_type` |
| Costo | latenza p50/p95 per claim, claim al secondo | misura con vLLM |

`metrics.py` vuole righe con `dataset` (la fonte) e `label`; per il bootstrap anche `doc`:

```python
rows = [{"dataset": r["source"], "label": r["label"], "doc": r["document"]} for r in split_rows]
```

**Esempio di ECE.** 1.000 previsioni con confidenza fra 0,8 e 0,9 (media 0,85), di cui 780 giuste: contributo = (1.000 / N) · |0,85 − 0,78|. Se tutte le fasce hanno scarti simili, ECE ≈ 0,07: modello un po' sovra-confidente.

**Incertezza.** Intervalli al 95% con `metrics.document_cluster_bootstrap(rows, scores, baseline_scores, threshold, baseline_threshold)`: ricampiona i documenti dentro ogni fonte, con le stesse estrazioni per i due modelli, e dà l'intervallo della **differenza** fra modello e baseline. Differenze di BAcc media sotto ~0,5 punti si considerano non conclusive.

## 3. Aggregazione

1. BAcc per fonte.
2. Media non pesata per blocco: test_id, test_ood (HaluBench senza la parte RAGTruth), AggreFact (11 fonti).
3. **Numero unico** = media dei tre blocchi.
4. AggreFact anche da solo, formato leaderboard (colonne CNN, XSum, MediaS, MeetB, WiCE, REVEAL, ClaimVerify, FactCheck, ExpertQA, LFQA, RAGTruth).

## 4. Test comportamentali — `finetune/behavioral.py`

Si generano da dev_select con trasformazioni automatiche, una volta, e si salvano in `data/finetune/splits/behavioral/`.

| Test | Costruzione | Atteso | Metrica | Obiettivo |
|---|---|---|---|---|
| Solo claim | documento sostituito da stringa vuota, su 300 righe | non supportato | P media | < 0,15 |
| Rimozione dell'evidenza | righe supportate di FEVER e HoVer dev con evidenza nota: si tolgono le frasi gold dal documento | P scende sotto soglia | % sotto soglia | > 80% |
| Distrattori | si aggiungono 3 paragrafi presi da documenti di altre righe di dev (stessa fonte) prima e dopo | decisione invariata | % invariata | > 95% |
| Posizione | il documento di una riga supportata breve si inserisce all'inizio / a metà / alla fine di 3.000 token di distrattori | decisione invariata | BAcc per posizione | differenza < 2 punti |
| Coppie minime | righe `contradict_claim` costruite **dal dev** solo per questo test (mai in training) e coppie VitaminC di dev | la decisione si inverte | % di coppie con p_pos > p_neg | più alta della baseline |
| Claim parziale | claim supportato + una frase inventata plausibile aggiunta in coda (es. "..., and the event was held in March.") generata da un LLM, controllata a mano su 50 casi | non supportato | % sotto soglia | > 80% |

Codice per i primi due:

```python
def claim_only(rows):
    return [{**r, "id": r["id"] + "#claim_only", "document": "", "label": 0} for r in rows]

def remove_gold(row, gold_sentences):
    doc = row["document"]
    for s in gold_sentences:                     # testo delle frasi gold (FEVER: righe; HoVer: supporting_facts)
        doc = doc.replace(s, "")
    return {**row, "id": row["id"] + "#no_evidence", "document": " ".join(doc.split()), "label": 0}
```

Per "rimozione dell'evidenza" servono le frasi gold: FEVER dal campo `lines` delle pagine (gli indici dell'evidenza), HoVer da `supporting_facts` (titolo, indice di frase) con le frasi della pagina ottenute dallo splitter di [03](03_rendering_formato.md) §6; se l'indice non corrisponde (il testo del db HoVer è senza confini di frase originali) si scarta la riga.

Esempio:

> Documento (HoVer): pagina "Nancy R. Howell" + pagina "United Methodist Church of the Resurrection" + ...
> Claim supportato: "The city of the college that Nancy R. Howell is a Professor in and United Methodist Church of the Resurrection have in common is Kansas."
> Variante: si toglie "She is Professor of Theology and Philosophy of Religion at Saint Paul School of Theology in Kansas City, Missouri." → manca il collegamento Howell → Kansas City: P deve scendere sotto soglia.

## 5. Il protocollo

1. Ogni esperimento: dev_select + comportamentali. Si decide solo qui.
2. A fine fase: dev_confirm (una volta).
3. Test (test_id, test_ood, test_long, AggreFact): **una volta** per la baseline, **una volta** per il modello finale congelato ([08](08_merge_export.md)). Nessuna scelta dopo averli visti.
4. Prima del test finale: `frozen.json` con gli hash ([08](08_merge_export.md) §5).
5. Baseline e modello fine-tunato sempre con la stessa procedura (stessa domanda, stessa regola di soglia).
6. Mai addestrare su AggreFact né su test_ood.

## 6. Come si esegue

```sh
# probabilità del modello fuso su ogni blocco
for s in dev_select dev_confirm test_id test_ood; do
  conda run -n factcheck python finetune/score.py --model d1 --model-dir finetune/release/d1-3b-fc-v1 \
      --question finetune/question.json --split $s
done
# AggreFact con la stessa domanda (nuova inferenza: le cache esistenti sono dei modelli di partenza)
conda run -n factcheck python finetune/score.py --model d1 --model-dir finetune/release/d1-3b-fc-v1 \
    --question finetune/question.json --split aggrefact_test
# calibrazione su dev_select, applicazione a tutti i blocchi, tabelle e intervalli
conda run -n factcheck python finetune/report.py --run d1-3b-fc-v1 --baseline d1
```

Stesso schema per `--model jpt` e `--model winnow`. In alternativa, il modello fuso si può valutare su AggreFact con il nuovo runner del progetto (`benchmark.py`, che scrive in `results/<dataset>/<split>/...` nel formato di [README_RESULTS.md](../../README_RESULTS.md)), aggiungendo una configurazione di modello che punta ai pesi fusi; così i risultati finiscono accanto a quelli di ottobre e `grid_search.py` li confronta direttamente.

(`score.py` va esteso per leggere AggreFact: le righe di `lytang___llm-aggre_fact` hanno `doc`, `claim`, `label`, `dataset`; si mappano su `document`, `claim`, `label`, `source`.)

## 7. Il report finale — formato

| Blocco | Metrica | Baseline 1D | Baseline 3D | Fine-tunato | Δ vs 1D | IC 95% di Δ |
|---|---|---:|---:|---:|---:|---|
| test_id | BAcc macro | | | | | |
| test_ood | BAcc macro | | | | | |
| test_long | BAcc macro | | | | | |
| AggreFact (zero-shot) | BAcc / AUC | | | | | |
| **Media dei blocchi** | BAcc | | | | | |
| AggreFact a 0,5 | BAcc | | | | | |
| AggreFact (secondaria) | BAcc | 76,69 / 76,20 | 77,42 / 76,71 | | | |
| Calibrazione | ECE / soglia ottimale | | | | | |
| Comportamentali | solo claim / rimozione / coppie minime | | | | | — |
| Costo | ms per claim | | | | | — |

Più: tabelle per fonte di ogni blocco, tabella AggreFact nel formato della leaderboard, e accanto i riferimenti con la stessa procedura (modello base fine-tunato, encoder fine-tunato, FactCG, MiniCheck-FT5, Bespoke-MiniCheck-7B).

## Criteri di completamento

- Report con tutti i blocchi, intervalli di confidenza e comportamentali.
- Test eseguiti una sola volta, dopo `frozen.json`.
