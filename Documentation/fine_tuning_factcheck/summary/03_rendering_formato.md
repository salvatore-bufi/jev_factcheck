# Step 3 — Rendering: una domanda, il prompt degli autori, i documenti lunghi

## Obiettivo

Trasformare ogni riga del mix in un esempio pronto per il training di **un** modello, identico a ciò che il modello vede in produzione. Dettagli e codice in [../03_rendering_formato.md](../03_rendering_formato.md).

## Input e output

- Input: mix (`fc_pilot`, `fc_stage1`, `fc_stage2`), `finetune/question.json`.
- Output: `data/finetune/rendered/<modello>/<mix>/train.jsonl` con `input_ids`, posizione di lettura, opzioni lette, maschera, temperatura, `take`, `label`.

## 1. Il principio

Non si scrive nessun prompt: si chiamano le funzioni degli autori, le stesse che gli adapter della repo (`models/<modello>/adapter.py`) usano e hanno verificato. Anche la **maschera** conta: i modelli bidirezionali caricati come modelli causali rispondono diversamente senza dare errori.

## 2. I modelli per tipo di lettura

| Gruppo | Modelli | Note per il training |
|---|---|---|
| Testa del vocabolario, causale | KnowLine-4B, OneJev1.6-9B, Nimble-9B-v3, backbone | maschera standard; KnowLine scrive due volte le richieste fino a 3.000 token |
| Testa del vocabolario, bidirezionale sul prompt | elephant-1 (a blocchi), StartLux 9B/4B, d3-flash, Jeffan-4B | maschera nei soli 8 layer pieni; per StartLux va ricostruita in transformers (gli autori usano vLLM con un plugin) |
| Testa propria | pplx-decider-flash (255 vie), Clef-Flash (congiunta) | si addestra anche la loro testa |

I backbone usano il prompt neutro dello studio, con l'ordine delle opzioni A/B scelto a caso per ogni esempio.

## 3. Budget di token

8.192 token in training (16.384 per i 4B se la memoria basta); 32.768 è il limite dei runtime degli autori.

## 4. Documenti lunghi

- **In inferenza**: se il prompt sta in C token, documento intero; altrimenti chunk di frasi ≤ C e `max` sui chunk (C scelto in K1).
- **In training**: negativi oltre il budget → un chunk a caso con etichetta 0; positivi oltre il budget → scartati.

## 5. Test obbligatori

| Test | Atteso |
|---|---|
| Identità con il benchmark: P(supportato) dal forward di training contro l'adapter, 50 righe | differenza media < 0,005 |
| Maschera: per i modelli bidirezionali, con maschera causale le probabilità devono cambiare | differenza > 0,01 |
| Opzioni e `take` | identiche al codice degli autori; target coerenti |
| Frasi e budget | ricopertura; positivi oltre il budget scartati |

## Criteri per passare allo step successivo

- Test superati per il modello da addestrare; statistiche di lunghezza registrate.
