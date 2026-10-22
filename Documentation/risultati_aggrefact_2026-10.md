# Risultati su LLM-AggreFact (test) — run di ottobre 2026

Lettura dei risultati in `results/lytang--LLM-AggreFact/test/`, prodotti dal nuovo runner (`benchmark.py` / `report.py` / `grid_search.py`, descritti in [README_RESULTS.md](../README_RESULTS.md)). Sostituiscono la vecchia cartella `results/runs/`, che non è più nella repo: i numeri "storici" citati altrove (soglie fittate sul dev di Jev) vengono da file che oggi non esistono più.

## 1. Cosa c'è

| Run | Modello | Taglia | Righe | Stato | GPU |
|---|---|---:|---:|---|---|
| `jpt-9b_5147a4ec_551bb722` | JPT-9B (Qwen3.5-9B, decision model) | 9B | 29.320 | complete | L40S |
| `winnow-12b-bf16_2bbfeae9_551bb722` | Winnow-12B-BF16 (Gemma-4-12B, decision model) | 12B | 29.320 | complete | L40S |
| `rune-26b_825febc6_551bb722` | Rune-26B-A4B-v3 (MoE, GGUF, 17 GB su CPU) | 26B-A4B | 29.320 | complete | L40S |
| `d1-3b_e3eff318_551bb722` | **d1-3B** (LiquidAI, nuovo) | 3B | 29.320 | complete | L40S |
| `clm-8b_2fe74d82_551bb722` | **CLM-v0.1-8B** (Contrastive-LM, nuovo) | 8B | 29.320 | complete | L40S |

- Tutte le domande sono il set `551bb722` = `claim_support` (le stesse tre domande di `packs/claim_support.json`).
- I registri `hash_factcg.json` e `hash_winnow-12b.json` (NVFP4) esistono ma **non c'è la cartella del run**: FactCG e Winnow-NVFP4 non sono in questi risultati.
- Solo lo split **test**: non ci sono run sul dev. Ogni soglia "migliore" qui sotto è scelta sul test stesso ed è quindi **ottimistica**; l'AUC invece non dipende dalla soglia.
- Ambiente dei run: torch 2.13, transformers 5.18, vLLM 0.30, llm2jev 0.6.1.

## 2. Regola fissa di Jev (3 domande, `min`, soglia 0,30) — come la leaderboard

| Modello | Average (BAcc macro) | AUC macro | CNN | XSum | MediaS | MeetB | WiCE | REVEAL | ClaimVerify | FactCheck | ExpertQA | LFQA | RAGTruth |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Winnow-12B-BF16 | **76,95** | **84,99** | 72,2 | 72,8 | 73,4 | 80,3 | 77,4 | 90,1 | 75,7 | 73,9 | 59,0 | 84,7 | 87,1 |
| Rune-26B-A4B | 76,24 | 84,64 | 66,0 | 74,2 | 70,8 | 76,5 | 83,7 | 87,1 | 73,1 | 78,4 | 59,7 | 83,1 | 86,0 |
| JPT-9B | 76,04 | 84,48 | 62,4 | 75,8 | 67,6 | 76,9 | 83,8 | 87,9 | 72,8 | 81,1 | 58,8 | 84,8 | 84,6 |
| d1-3B | 71,84 | 83,89 | 54,9 | 68,8 | 68,0 | 70,9 | 81,6 | 84,3 | 67,2 | 76,4 | 59,9 | 79,7 | 78,6 |
| CLM-v0.1-8B | 49,98 | 46,79 | 50,0 | 50,0 | 50,0 | 50,0 | 49,8 | 50,0 | 50,0 | 50,0 | 50,0 | 50,0 | 50,0 |

I numeri di JPT (76,04), Winnow-BF16 (76,95 contro 76,99 del run precedente) e Rune (76,24 contro 76,32) coincidono con quelli già noti entro le variazioni dovute alle nuove versioni di vLLM/transformers: il nuovo runner riproduce i vecchi risultati.

Riferimenti esterni (leaderboard): Bespoke-MiniCheck-7B 77,4; FactCG-DeBERTa-L 75,6; MiniCheck-Flan-T5-L 75,0.

## 3. Una domanda sola — il dato che conta per il fine-tuning

Calcolato dagli `answers.jsonl` (nessuna nuova inferenza). "1 − unsupported" = la domanda `unsupported_detail` usata da sola, presa come 1 − P(sì). La colonna "migliore" è la soglia migliore su una griglia 0,01–0,99 scelta **sul test** (ottimistica, solo per esplorare).

| Modello | Domanda | AUC macro | BAcc @0,5 | BAcc migliore (soglia) | P(sì) media |
|---|---|---:|---:|---:|---:|
| Winnow-12B | support_simple | 84,03 | 74,91 | 76,85 (0,93) | 0,784 |
| Winnow-12B | **1 − unsupported_detail** | **85,04** | **76,77** | 77,37 (0,26) | 0,651 |
| Winnow-12B | 3 domande (min) | 84,99 | 76,38 | 77,71 (0,14) | 0,632 |
| JPT-9B | support_simple | 84,15 | 74,32 | 76,43 (0,86) | 0,765 |
| JPT-9B | **1 − unsupported_detail** | 84,30 | **76,74** | 76,94 (0,63) | 0,681 |
| JPT-9B | 3 domande (min) | 84,48 | 77,11 | 77,21 (0,61) | 0,670 |
| Rune-26B | **support_simple** | **84,91** | **76,65** | **77,89** (0,75) | 0,763 |
| Rune-26B | 1 − unsupported_detail | 84,80 | 76,14 | 76,53 (0,59) | 0,634 |
| Rune-26B | 3 domande (min) | 84,64 | 76,56 | 76,79 (0,48) | 0,620 |
| d1-3B | support_simple | 83,51 | 72,48 | 74,97 (0,73) | 0,751 |
| d1-3B | **1 − unsupported_detail** | **84,41** | **74,30** | **75,59** (0,67) | 0,698 |
| d1-3B | 3 domande (min) | 83,89 | 74,37 | 74,76 (0,65) | 0,664 |
| CLM-8B | support_simple | 65,11 | 58,78 | 58,78 (0,50) | 0,550 |

Letture:

1. **`unsupported_detail` da sola vale quanto (o più di) tre domande**, al costo di una. Per Winnow, JPT e d1 ha AUC più alta di `support_simple` e, soprattutto, una BAcc a soglia 0,5 di ~2 punti più alta: è meglio calibrata attorno a 0,5. È la domanda con le descrizioni delle opzioni ("At least one factual detail is absent from, contradicted by, or more specific ..."), che definiscono esattamente l'etichetta. Rune fa eccezione (lì `support_simple` è la migliore).
2. **`support_simple` dice "sì" troppo facilmente**: soglie migliori 0,73–0,93, P(sì) media 0,75–0,78 contro una quota di positivi molto più bassa. Conferma quanto già visto sul dev di Jev (soglie 0,90–0,95 per JPT e Winnow).
3. **d1-3B è vicinissimo ai modelli grandi in AUC**: 84,41 con `1 − unsupported_detail`, contro 84,30 di JPT e 85,04 di Winnow. La differenza in BAcc a soglia fissa (72–74 contro 76–77) è soprattutto di calibrazione e di soglia, cioè proprio ciò che il fine-tuning corregge più facilmente.
4. **CLM-8B non è adatto**: con la regola a 3 domande dice sempre "non supportato" (BAcc 50 su tutte le fonti, P(sì) di `unsupported_detail` media 0,94); con la sola `support_simple` l'AUC è 65. È un modello a punteggi coseno (stato contro testo dell'opzione), non un decision model a lettere.

## 4. Velocità e risorse (stessa GPU: NVIDIA L40S)

| Modello | Latenza p50 / p95 (1 riga, 3 domande) | Righe/s (passaggio principale) | Memoria GPU di picco | Energia per 1.000 righe | Caricamento |
|---|---:|---:|---:|---:|---:|
| **d1-3B** | **38 ms / 103 ms** | **24,4** | **11,6 GB** | **12,8 kJ** | 6–9 s |
| CLM-8B | 101 ms / 269 ms | 30,1 | 41,6 GB | 9,9 kJ | 60 s |
| JPT-9B | 172 ms / 336 ms | 9,4 | 43,2 GB* | 35,7 kJ | 148 s |
| Winnow-12B-BF16 | 169 ms / 744 ms | 9,8 | 43,5 GB* | 34,9 kJ | 133 s |
| Rune-26B-A4B | 1.450 ms / 4.620 ms | 1,7 | 43,5 GB* (+17 GB su CPU) | 70,5 kJ | 257 s |

\* vLLM riserva il 92% della memoria in anticipo: è memoria riservata, non usata davvero. d1-3B non usa vLLM ma la sua API (`system_one_batch`, richieste impacchettate senza padding): il suo throughput non cresce con la dimensione del batch (24 righe/s a batch 1, 8 e 32), quello di JPT e Winnow sì (da 5,5/4,0 a 7,4/4,6).

Per lunghezza del documento la latenza p50 di d1-3B resta sotto 100 ms fino a 16.000 caratteri (206 ms oltre); JPT 0,3–0,6 s, Winnow 0,7–1,4 s.

## 5. Accuratezza per lunghezza del documento (BAcc aggregata, regola a 3 domande)

| Caratteri del documento | Righe | d1-3B | JPT-9B | Winnow-12B | Rune-26B |
|---|---:|---:|---:|---:|---:|
| 0–1.000 | 6.966 | 72,7 | 74,1 | 73,7 | 73,2 |
| 1.000–2.000 | 7.928 | 76,5 | 82,3 | 85,5 | 83,0 |
| 2.000–4.000 | 9.130 | 78,8 | 83,4 | 86,5 | 84,5 |
| 4.000–8.000 | 4.279 | 71,6 | 75,5 | 78,5 | 77,3 |
| 8.000–16.000 | 664 | 70,6 | 75,5 | 78,7 | 75,9 |
| ≥ 16.000 | 353 | 66,2 | 71,1 | 70,9 | 68,4 |

Tutti i modelli peggiorano sopra ~4.000 caratteri e molto sopra 16.000: conferma che documenti lunghi (e il loro chunking) vanno curati nei dati di training e nella valutazione (test_long).

## 6. d1-3B in breve (dalla model card e dai file del repository, revisione `051bcc4`)

| Aspetto | Valore |
|---|---|
| Base | LFM2.5-VL-3B (Liquid AI), multimodale; 3,12B parametri, di cui ~400M nel codificatore visivo |
| Backbone testuale | `lfm2`: 30 layer ibridi (22 a convoluzione corta, 8 ad attenzione piena: layer 2, 5, 9, 13, 17, 21, 24, 27), hidden 2.048, vocabolario 128.000, embedding legate, contesto 32.768 |
| Interfaccia | System One (stesso schema `noul` / `choice` / `score` di JPT e Winnow), una passata, zero token generati |
| Prompt | stato come JSON indentato, poi `QUESTION:`, la domanda, per una `noul` con criteri le righe `Yes: ...` / `No: ...`, poi `Reply with yes or no only.` e l'intestazione dell'assistente |
| Lettura | softmax su due gruppi di token, ciascuno con il massimo: sì = max(`yes`, `Yes`), no = max(`no`, `No`, `NO`); nessuna temperatura |
| Licenza | **LFM Open License v1.0**: uso commerciale consentito solo a entità con fatturato annuo sotto 10 milioni di dollari; vale anche per i derivati |

## 7. Cosa cambia per il piano

- **d1-3B entra come candidato** per il fact checker binario: AUC quasi pari a JPT/Winnow, latenza 4–5 volte più bassa, memoria ~4 volte più bassa, LoRA completa su una GPU da 24 GB. Vedi [fine_tuning_factcheck/00a_scelta_approccio.md](fine_tuning_factcheck/00a_scelta_approccio.md).
- **La domanda singola da preferire è probabilmente `unsupported_detail` (letta come 1 − P(sì))**, non `support_simple`. Va confermato sul nostro dev, perché questi numeri sono sul test: vedi [fine_tuning_factcheck/02d_varianti_domanda.md](fine_tuning_factcheck/02d_varianti_domanda.md).
- **CLM-8B esce dai candidati.**
- I percorsi `results/runs/...` citati nei documenti precedenti non esistono più: i risultati sono in `results/<dataset>/<split>/<modello>_<hash>_<hash>/`.

## 8. Come rifare i calcoli di §3

```python
import json, glob, sys
import numpy as np
sys.path.insert(0, ".")
from metrics import metrics
for d in sorted(glob.glob("results/lytang--LLM-AggreFact/test/*_551bb722")):
    rows, scores = [], {"support_simple": [], "1-unsupported": [], "min3": []}
    with open(f"{d}/answers.jsonl") as f:
        next(f)                                            # riga 1 = intestazione {"identity": ...}
        for line in f:
            r = json.loads(line); a = r["answers"]
            rows.append({"dataset": r["source"], "label": r["label"]})
            s = a["support_simple"]["probs"]["true"]
            u = 1 - a["unsupported_detail"]["probs"]["true"]
            c = 1 - a["contradicted_detail"]["probs"]["true"]
            scores["support_simple"].append(s); scores["1-unsupported"].append(u); scores["min3"].append(min(s, u, c))
    for name, sc in scores.items():
        m = metrics(rows, sc, 0.5)
        auc = np.mean([v["roc_auc"] for v in m["by_source"].values()])
        print(d.split("/")[-1], name, round(auc * 100, 2), round(m["macro_balanced_accuracy"] * 100, 2))
```
