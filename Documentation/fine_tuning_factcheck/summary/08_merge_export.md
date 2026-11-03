# Step 8 — Merge, export, soglia, (opzionale) distillazione

## Obiettivo

Pesi fusi caricabili dal runtime degli autori, verificati, con soglia e temperatura finali; congelamento; un solo test; pacchetto. Dettagli in [../08_merge_export.md](../08_merge_export.md).

## Input e output

- Input: adapter scelto, cartella originale del modello.
- Output: `finetune/release/<modello>-fc-<versione>/`.

## 1. Fusione e verifica

La cartella fusa è una **copia della cartella originale** in cui cambiano solo i tensori: i runtime degli autori leggono anche file propri (`modeling_overrides` di elephant-1, `decision_config.json`, `readout.safetensors` di pplx, codice di d3-flash e Clef, front end di KnowLine). Fusione sempre su pesi bf16. Verifica con l'adapter del benchmark sulla cartella fusa: differenza media di P < 0,005 su 200 righe.

## 2. Una sola domanda in produzione

Quella di `question.json`, servita dal runtime degli autori (motore `qev`, `startlux-decide`, Flide, `NimbleEngine`, `jeffan_engine`, `pplx_decider`, …), come negli adapter.

## 3. Soglia e temperatura finali

Scala e soglia rifittate su dev_select, confermate su dev_confirm; se la BAcc a 0,5 è entro 0,5 punti, in produzione si usa 0,5. Soglia secondaria su AF-dev, solo per confronto con lo studio.

## 4. Opzione: estrarre la testa binaria

Per la lettura dalla testa del vocabolario: u = (w_sì − w_no) / T come testa lineare equivalente, se il runtime accetta un classificatore.

## 5. Opzione: distillazione in un encoder

Etichette dell'insegnante sui chunk che vedrà lo studente; KL + CE; obiettivo ≥ 75,6 su AggreFact.

## 6. Congelamento e test

`frozen.json` con hash di pesi, decisione, domanda, split, mix, configurazione e codice di valutazione; poi una sola valutazione dei test.

## 7. Pacchetto finale e licenze

Pesi fusi, `decision.json`, `frozen.json`, report, model card. Il derivato eredita la licenza dei pesi: MIT (elephant-1), Apache 2.0 (KnowLine, d3-flash, OneJev, Clef, pplx, Jeffan, Qwen3.5), **CC BY-NC (StartLux, Nimble)**; in più i vincoli dei dati (ANLI, FactKG, AVeriTeC non commerciali).

## Criteri di completamento

- Fusione verificata con l'adapter del benchmark; scala e soglia confermate; `frozen.json` prima del test; report e model card.
