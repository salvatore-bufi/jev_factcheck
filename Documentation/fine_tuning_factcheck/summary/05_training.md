# Step 5 — Il training (LoRA sul decision model; traccia encoder per i controlli)

## Obiettivo

Addestrare una LoRA su KnowLine-4B (prove), poi elephant-1, poi gli altri decision model e i due backbone (B-ctrl), con la loss sulle due opzioni e il curriculum in due stadi. Dettagli e codice in [../05_training.md](../05_training.md).

## Input e output

- Input: esempi renderizzati ([03](03_rendering_formato.md)), loss ([04](04_loss.md)).
- Output: adapter in `finetune/outputs/<esperimento>/`, log, configurazione.

## 1. Configurazione di partenza

| Parametro | Valore |
|---|---|
| LoRA | r 16, α 32, dropout 0,05, sugli stessi moduli per tutti (Qwen3.5: `self_attn` dei layer pieni, `linear_attn` dei DeltaNet, MLP) |
| Teste proprie (pplx, Clef) | addestrate per intero |
| Nimble-9B-v3 | prima si fonde la sua LoRA, poi si aggiunge la nostra |
| Learning rate | pilota 2e-5; stadio 1 5e-5; stadio 2 1e-5 (ottimizzatore nuovo) |
| Epoche, batch | 1 per stadio; batch effettivo 32 |
| Lunghezza | 8.192 token; bf16; gradient checkpointing |

Stessa configurazione per tutti i modelli, decision model e backbone: è la condizione di B-ctrl.

## 2. Hardware

| | Gruppo 4B | Gruppo 9B |
|---|---|---|
| RTX 5090, 24 GB | LoRA in bf16 | solo QLoRA per prove |
| L40S, 45 GB | LoRA in bf16, sequenze più lunghe | LoRA in bf16 (~28–32 GB a 8k token) |

Servono `peft`, `flash-linear-attention` e `causal-conv1d` (layer DeltaNet). Sul server: `gpu-run --batch`.

## 3. Il ciclo

Per ogni micro-batch: forward con la maschera del modello, stato finale nella posizione di lettura, punteggi delle sole due opzioni (o la testa propria), margine, cross-entropy binaria (eventualmente + β·KL). Mai i logit di tutto il vocabolario.

## 4. Cosa controllare durante il training

Loss; BAcc macro su dev_select; soglia ottimale (deve scendere verso 0,4–0,6); BAcc a 0,5; ECE; "solo claim".

## 5. Test prima del primo training (per ogni modello)

Identità con il benchmark; passo zero = baseline; gradienti con la maschera (modelli bidirezionali); overfit di 64 esempi; parametri addestrabili > 0.

## 6. Scelta del checkpoint

BAcc macro più alta su dev_select (a parità, ECE più bassa); conferma su dev_confirm sui pesi fusi; AggreFact non entra.

## 7. Problemi tipici

Opzioni o `take` invertiti; maschera persa in training; lr troppo alto; memoria (batch 1, checkpointing, solo le due opzioni); training lento (kernel DeltaNet mancanti); soglia bloccata in alto (pochi negativi "non stabilito").

## 8. Appendice: traccia encoder stile FactCG (controllo A5, studente)

DeBERTa-v3-large, testa nuova a 2 classi, template FLAN di FactCG, chunk di 550 parole, 2 stadi, lr 1e-5.

## 9. Riproducibilità

`config.json` con argomenti, hash di dati e mix, modello e revisione, `question.json`, versioni delle librerie, seed.

## Criteri per passare allo step successivo

- Test superati; un checkpoint che batte la baseline a una domanda su dev_select, confermato su dev_confirm, con soglia fra 0,3 e 0,7.
