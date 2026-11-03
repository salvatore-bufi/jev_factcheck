# Fine-tuning per un solo compito: "il documento supporta il claim?" — panoramica

Versione riassuntiva del piano: decisioni e numeri. La guida operativa completa, con comandi e codice, è nei file con lo stesso numero nella cartella superiore ([../00_overview.md](../00_overview.md)).

## 1. La domanda e la risposta in breve

**Domanda**: come si specializza un decision model in un fact checker binario documento–claim, migliore dei checker esistenti (Bespoke-MiniCheck-7B: 77,4 su LLM-AggreFact)?

**Risposta**: si parte dai decision model dello studio focalizzato ([../../paper_idea/15_focused_10_tdm_experiment_guide.md](../../paper_idea/15_focused_10_tdm_experiment_guide.md)), si fa **una sola domanda sì/no**, si addestra una LoRA con la cross-entropy **sui soli punteggi delle due opzioni** che il modello legge già, con dati pubblici più dati generati alla MiniCheck/FactCG, in due stadi. Ogni modello si addestra e si serve con il codice dei suoi autori (prompt, maschera di attenzione, lettura).

## 2. Da dove partiamo

| Modello | Base | Ruolo |
|---|---|---|
| **KnowLine-4B** | Qwen3.5-4B, causale, lettura dalla testa del vocabolario | si itera qui (entra in 24 GB anche per il training) |
| **elephant-1** | Qwen3.5-9B, bidirezionale a blocchi | modello principale a 9B; coppia principale di B-ctrl |
| StartLux-9B, d3-flash, OneJev1.6-9B, Nimble-9B-v3, Clef-Flash | Qwen3.5-9B | ricetta finale, per B-ctrl |
| StartLux-4B, pplx-decider-flash, Jeffan-4B | Qwen3.5-4B | ricetta finale, per B-ctrl |
| Qwen3.5-9B, Qwen3.5-4B | — | controlli: stessa ricetta sul backbone (B-ctrl) |

Tutti hanno la stessa architettura (32 layer, 24 Gated DeltaNet + 8 ad attenzione piena): stessi moduli LoRA per tutti. Cambiano prompt, maschera (causale o bidirezionale sul prompt) e lettura (due righe della testa del vocabolario, o una testa propria per pplx e Clef).

Numeri già noti (regola a tre domande, soglia 0,30): elephant-1 76,70, KnowLine-4B 75,65. I numeri a una domanda dei 10 modelli arrivano dai run dello studio (RUN-01…05).

## 3. Il percorso in 10 step

| Step | Cosa | File |
|---|---|---|
| 0 | split nostri, congelati | [00b](00b_split_e_protocollo.md) |
| 0a | perché questo approccio | [00a](00a_scelta_approccio.md) |
| 1 | baseline sul nostro dev, con `benchmark.py` | [01](01_baseline.md) |
| 2a–2e | dati: fonti, generazione, niente replay, domanda, mix | [02a](02a_dataset_fonti_esistenti.md) … [02e](02e_mix_finale.md) |
| 3 | rendering con il codice degli autori | [03](03_rendering_formato.md) |
| 4 | loss sulle due opzioni | [04](04_loss.md) |
| 5 | training LoRA | [05](05_training.md) |
| 6 | valutazione | [06](06_valutazione.md) |
| 7 | esperimenti, poi B-ctrl su tutte le ricette | [07](07_esperimenti.md) |
| 8 | fusione, soglia, servizio con il runtime degli autori | [08](08_merge_export.md) |

## 4. Le decisioni chiave

1. Decision model, **una domanda**, lettura che il modello già usa (nessuna testa nuova).
2. **Si itera su KnowLine-4B**, si conferma su elephant-1, poi tutti gli altri e i backbone (B-ctrl).
3. **Prompt, maschera e lettura degli autori**, verificati con un test di identità contro l'adapter del benchmark prima di ogni training.
4. Dati MiniCheck/FactCG + fonti pubbliche, curriculum in due stadi; niente replay.
5. Tutte le scelte sul nostro dev; AggreFact solo come test finale, zero-shot, con e senza RAGTruth.

## 5. Obiettivi misurabili

| Obiettivo | Soglia |
|---|---|
| LLM-AggreFact zero-shot, BAcc macro, una domanda | > 77,4; puntare a ≥ 78,5 |
| AUC macro su AggreFact | > 85,0 |
| Soglia ottimale sul nostro dev | fra 0,3 e 0,7 |
| ECE e test_ood | non peggiori della baseline |
| "Solo claim" / "rimozione dell'evidenza" | P media < 0,15 / scende sotto soglia in > 80% |
| B-ctrl | decision model e backbone fine-tunati con la stessa ricetta, sulle due coppie principali e poi sulle altre |

## 6. Relazione con gli altri documenti

- [../../paper_idea/04_parte_B_finetuning.md](../../paper_idea/04_parte_B_finetuning.md): l'esperimento B-ctrl del paper usa questo piano.
- [../../fine_tuning_factcheck_evidence/](../../fine_tuning_factcheck_evidence/00_overview.md): estensione con l'evidenza evidenziata, sullo stesso verdetto.
