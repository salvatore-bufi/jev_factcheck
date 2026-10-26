# Step 1 — Baseline: quanto valgono i candidati prima del training

## Obiettivo

Misurare, con lo stesso protocollo ([00b](00b_split_e_protocollo.md)), tutti i punti di partenza fra cui si sceglie in [00a](00a_scelta_approccio.md): decision model con una e con tre domande, modelli base, classificatori stile FactCG/MiniCheck. Ogni guadagno successivo si misura contro queste righe.

## 1. Cosa si sa già (dalle cache, senza nuova inferenza)

Calcolato in questa sessione da `results/runs/<modello>{,-dev}/answers.jsonl`: soglia fittata sui 4.520 righe del dev di AggreFact (`dev_split.py`), applicata al test. Sono numeri **secondari** (dev-tuned), utili per orientarsi.

| Modello | Domande | Soglia | BAcc dev | BAcc test | AUC test |
|---|---|---:|---:|---:|---:|
| Winnow-12B-BF16 | 3 (`min`) | 0,22 | 77,77 | 77,42 | 85,02 |
| Winnow-12B-BF16 | solo `support_simple` | 0,95 | 76,76 | **76,69** | 84,03 |
| JPT-9B (T = 1,087) | 3 (`min`) | 0,72 | 77,76 | 76,71 | 84,57 |
| JPT-9B (T = 1,087) | solo `support_simple` | 0,90 | 76,87 | **76,20** | 84,07 |
| FactCG-DeBERTa-v3-L | — | 0,5 | | 75,64 | |

Script usato (fuori dalla repo, da rendere permanente in `finetune/`): per Winnow `P = σ((z_B − z_A)/T)` dai logit di `support_simple`; per JPT la probabilità già in cache; stessa `tune_threshold` e stessa `metrics` di [metrics.py](../../metrics.py). La regola a 3 domande riproduce esattamente i numeri di `results/runs/dev_threshold_summary.md`, quindi il calcolo è coerente.

Lettura:
- **Costo della domanda singola**: −0,7 (Winnow) e −0,5 (JPT) di BAcc, −1,0 e −0,5 di AUC. Piccolo, e il training lavora proprio su questa domanda.
- **Soglia 0,90–0,95**: zero-shot `support_simple` dice "sì" troppo facilmente. Le domande 2 e 3 della regola di Jev servivano proprio a correggere questo. Il fine-tuning deve riportare la soglia ottimale verso il centro.

## 2. Le baseline da misurare

| ID | Modello | Regola | Perché |
|---|---|---|---|
| B1 | Winnow-12B-BF16, JPT-9B | 1 domanda (`support_simple`) | **punto di partenza vero** del piano |
| B2 | idem | 3 domande | per sapere quanto la domanda singola deve recuperare |
| B3 | idem | 1 domanda, formulazione alternativa (quella di `packs/binary_support.json`: "Does the document establish the entire claim? ...") | la domanda migliore zero-shot si sceglie qui, una volta, sul dev |
| B4 | Gemma-4-12B, Qwen3.5-9B **base** | 1 domanda, stesso prompt | punto di partenza del controllo C1 (opzione L2 di [00a](00a_scelta_approccio.md)) |
| B5 | FactCG-DeBERTa-v3-L (`FactCG/benchmark.py`) | chunk 550, `max`, soglia 0,5 e fittata | riferimento encoder (opzione E1) |
| B6 | MiniCheck-Flan-T5-L, MiniCheck-DeBERTa-L | idem | riferimento della ricetta MiniCheck |
| B7 | Bespoke-MiniCheck-7B | prompt e aggregazione ufficiali | riferimento LLM con la stessa ricetta di dati |

B5–B7 servono a confrontare **sulla nostra suite** (test_id, test_ood, test_long), non solo su AggreFact dove i numeri sono già pubblici.

## 3. Procedura

1. Inferenza su `dev_select` e `dev_confirm` per B1–B7 (B2 è gratuita se si fa B1 con tutte e tre le domande: stessa cache).
2. Per ciascuno, su `dev_select`: temperatura (minimo log-loss), poi soglia (BAcc macro, griglia 0,01–0,99). Riportare anche la BAcc a soglia 0,5.
3. Conferma su `dev_confirm` (calo ≤ ~1 punto).
4. Una sola esecuzione su test_id, test_ood, test_long; AggreFact dalle cache dove esistono.
5. Test comportamentali ([06](06_valutazione.md), §4).

**Scelta della domanda (B1 contro B3).** Si sceglie la formulazione con AUC più alta su `dev_select`. Da lì in poi è **la** domanda del modello: training, valutazione e produzione la usano identica.

## 4. Tabella attesa

| Blocco | Metrica | Winnow 1D | JPT 1D | Gemma-4-12B base | Qwen3.5-9B base | FactCG-DBT | Bespoke-7B |
|---|---|---:|---:|---:|---:|---:|---:|
| dev_select | BAcc / AUC | | | | | | |
| dev_select | BAcc a 0,5 | | | | | | |
| test_id | BAcc macro | | | | | | |
| test_ood | BAcc macro | | | | | | |
| test_long | BAcc macro | | | | | | |
| AggreFact (soglia dal nostro dev) | BAcc / AUC | | / 84,03 | | | | |
| AggreFact (secondaria) | BAcc | 76,69 | 76,20 | | | 75,64 (0,5) | 77,4 |
| Solo claim | P media | | | | | | |
| Costo | ms per claim, GPU | | | | | | |

## Criteri per passare allo step successivo

- B1, B4, B5 misurati su dev e test; domanda scelta (B1 o B3).
- Configurazione di partenza (domanda, temperatura, soglia) congelata per JPT e Winnow.

## Errori comuni

- Confrontare il modello fine-tunato a una domanda con la baseline a tre domande senza dirlo: la differenza di regola vale ~0,5–0,7 punti da sola.
- Usare la soglia della baseline (0,90–0,95) sul modello fine-tunato: va sempre rifittata.
