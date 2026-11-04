# Passo 9 — Training della testa (fase 1a)

## In breve

[`evidence/train_head.py`](../../../evidence/train_head.py) addestra **solo la testa**, sulle feature salvate ([06](06_rendering_ed_estrazione.md)). Il decision model non è in memoria, quindi il verdetto non può cambiare. Una prova dura minuti. Ogni esperimento è un file in [`evidence/configs/`](../../../evidence/configs/) ([11](11_esperimenti.md)).

```sh
F=data/evidence/features/knowline-4b
python -m evidence.train_head --config evidence/configs/h0.json --train $F/ev_head --overfit 64   # controllo
python -m evidence.train_head --config evidence/configs/h2c.json --train $F/ev_head \
    --dev $F/dev_evid_select --confirm $F/dev_evid_confirm --dev-neg $F/dev_evid_neg [--seed 14 --tag seed14]
```

Le fasi 1b (modello in memoria, media pesata dei token) e 2 (LoRA del modello insieme alla testa) del piano **non sono implementate** ([14](14_strada_da_esplorare.md)).

## 1. La configurazione di un esperimento

```json
{"id": "h2c", "description": "H2c: H2b + token mean of the sentence (full head)",
 "layers_read": [19, 23, 27, 31], "parts": ["end", "mean"],
 "head":  {"d": 256, "layers": 2, "heads": 4, "use_mix": true, "use_mean": true, "use_att": true,
           "use_pos": true, "use_pair": false, "att_dropout": 0.1, "dropout": 0.1},
 "loss":  {"name": "zlpr", "mu_neg": 0.3, "mu_pos": 0.3, "rho": 0.1},
 "train": {"epochs": 10, "batch": 64, "lr": 0.001, "weight_decay": 0.01, "patience": 2, "seed": 13,
           "terms": ["supervised", "mil_neg"]}}
```

- `layers_read` e `parts` decidono **che cosa si legge dal disco** (`load_rows`), quindi anche quanta memoria serve: H0 legge un tensore su otto.
- `terms` decide quali righe del mix entrano: H0 e H-LOSS usano solo `supervised`, da H-MIL in poi anche `mil_neg`.

## 2. Il ciclo (`train`)

1. **Righe in RAM.** Si leggono le righe del mix con i soli layer e parti della configurazione (`load_rows`).
2. **Batch per lunghezza** (`length_batches`). Le righe si ordinano per numero di frasi e si raggruppano a 64, così il riempimento è minimo. L'ordine dei gruppi è casuale e cambia a ogni epoca (seme $13 + \text{epoca}$).
3. **Riempimento** (`collate`): ogni batch si porta alla riga con più frasi; la maschera segna le frasi vere.
4. **Loss** (`batch_loss`): `ref_loss` o `zlpr_loss` per riga, moltiplicata per $w_b$, più $0{,}2 \cdot$ `pair_loss` con H-PAIR ([08](08_testa_e_loss.md) §5).
5. **Ottimizzazione**: AdamW (weight decay 0,01), clipping della norma del gradiente a 1. Il learning rate al passo $\kappa$ su $K$ passi totali, con riscaldamento sui primi $W = \max(1, \lfloor 0{,}05K \rfloor)$ passi, è

   $$
   \eta_\kappa = \eta_0 \cdot \min\Big(1, \frac{\kappa+1}{W}\Big) \cdot \frac{1}{2}\Big(1 + \cos\frac{\pi \min(\kappa, K)}{K}\Big), \qquad \eta_0 = 10^{-3}.
   $$

6. **A fine epoca**: i punteggi sul dev (`scores_of`), poi la scelta della regola sul dev (`choose_rule`, §3), poi le metriche. Il criterio del checkpoint è

   $$
   \text{score} = \tfrac{1}{2}\big(\mathrm{F1}_{frase} + \text{insieme completo}\big) \quad \text{su } \texttt{dev\_evid\_select}.
   $$

   Si tiene lo stato migliore e ci si ferma dopo 2 epoche senza miglioramento (`patience`).

| Iperparametro | Valore |
|---|---|
| learning rate | $10^{-3}$, riscaldamento 5%, coseno |
| weight decay | 0,01 |
| batch | 64 righe, raggruppate per lunghezza |
| epoche | fino a 10, arresto dopo 2 senza miglioramento |
| dropout | 0,1 nel transformer; 0,1 per frase sulle feature di attenzione |
| seme | 13 (per la configurazione finale anche 14 e 15, con `--seed` e `--tag`) |

## 3. La regola di selezione (`choose_rule`)

Si fissa **dopo** il training, su `dev_evid_select`. Si valutano solo i positivi che il verdetto giudica supportati ([10](10_valutazione.md) §1).

| Loss | Regola | Griglia | Scelta |
|---|---|---|---|
| `ref` | $\sigma(s_k) > \tau_e$, cioè soglia $\mathrm{logit}(\tau_e)$ | $\tau_e \in \{0{,}05, \ldots, 0{,}95\}$ | F1 migliore |
| `zlpr` | $s_k > s_\star + \delta$ | $\delta \in \{-1{,}0; -0{,}9; \ldots; 1{,}0\}$ | F1 migliore, ma $\delta \neq 0$ solo se guadagna almeno 1 punto |

Sempre almeno 1 frase e al massimo 5. La regola va in `selection.json`, la confermano le metriche su `dev_evid_confirm`, e la usano valutazione e server.

## 4. Che cosa scrive una run

`data/evidence/runs/knowline-4b/<id>[_<tag>]/`:

| File | Contenuto |
|---|---|
| `config.json` | la configurazione, le cartelle di feature, il loro `meta.json`, $D$ |
| `head.safetensors` | i pesi dell'epoca migliore |
| `selection.json` | la regola scelta |
| `log.jsonl` | per epoca: loss, F1, insieme completo, score, regola, secondi, pesi $\pi$ dei layer |
| `dev_metrics.json` | metriche complete su select, confirm e neg ([10](10_valutazione.md)) |

E una riga in `data/evidence/runs/registry.md` ([11](11_esperimenti.md)).

Il training rifiuta dev e train estratti con impostazioni diverse (`same_features`: modello, revisione, preambolo, layer, domanda, hash del codice).

## 5. Che cosa guardare

| Cosa | Dove | Problema | Cosa fare |
|---|---|---|---|
| F1 e insieme completo | `log.jsonl` | non superano E-LEX dopo 2 epoche | controllare il rendering ([06](06_rendering_ed_estrazione.md) §5) prima della testa |
| F1 per distanza in frasi | `dev_metrics.json`: `f1_by_sent_distance` | crollo oltre 10 frasi | H3c (4 layer), H-PAIR |
| F1 per distanza in token | `f1_by_tok_distance` | crollo oltre 1.024 o 4.096 | H2b (attenzione) |
| frasi mostrate | `shown_mean`, `count_error_mean` | molte più o molte meno delle gold | $\delta$, $\omega$ |
| negativi accesi | `dev_evid_neg` → `negatives_lit` | oltre 0,30 | $\mu_{neg}$ più alto (`h_mil1`) |
| recall sui positivi incerti | `recall_uncertain` | molto sotto la media | $\mu_{neg}$ più basso |
| pesi $\pi$ dei layer | `log.jsonl` → `layer_weights` | un layer sotto il 10% | toglierlo (meno disco, meno costo) |

## 6. Controlli prima di ogni nuova configurazione

| Controllo | Atteso | Misurato (2026-11-04) |
|---|---|---|
| `--overfit 64`: la testa impara a memoria 64 righe (30 epoche, batch 16) | score ≈ 1 | 1,000 con H0 e con H2c |
| righe `none` | loss e gradiente 0 | sì (test) |
| riempimento | stessi punteggi da sola o in batch | differenza < $10^{-4}$ (test) |

Prova end-to-end su 300 righe del mix (circa 190 positivi) e 300 di `dev_evid_select`: F1 0,66 (H0), 0,68 (H-MIL), 0,78 (H2c); negativi accesi 0,95, 0,40, 0,35. Sono numeri da campione minuscolo: indicano che la pipeline impara, non quanto è buona la testa.

## 7. Codice

| Funzione | Ruolo |
|---|---|
| `collate`, `length_batches` | batch riempiti e raggruppati per lunghezza |
| `run_head`, `batch_loss` | forward e loss del batch |
| `scores_of`, `predictions` | punteggi → predizioni nel formato delle metriche |
| `choose_rule` | $\tau_e$ o $\delta$ sul dev |
| `load_rows`, `train` | il ciclo con arresto anticipato |
| `append_registry`, `main` | registro e salvataggio della run |
