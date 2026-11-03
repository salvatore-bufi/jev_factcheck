# Passo 5 — Il prompt con il preambolo

## In breve

Si aggiunge una riga con il claim **prima** del documento, lasciando il resto del prompt identico a quello del piano binario. Poi si controlla che il verdetto del piano binario funzioni ancora con questa riga in più. Se sì, si riusa; se no, si riaddestra con la stessa ricetta del piano binario, includendo la riga.

Perché serve: il modello legge da sinistra a destra e ogni token vede solo quello che lo precede. Senza preambolo, quando il modello legge il documento non sa ancora che cosa deve verificare, e i vettori delle frasi non "parlano" del claim.

## 1. Esempio (d1-3B)

Prompt del piano binario:

```text
<|startoftext|><|im_start|>user
{
  "document": "The meeting was moved to Tuesday at 10:00. It will be held in Rome.",
  "claim": "The meeting is on Tuesday."
}


QUESTION:
<domanda del piano binario>
Reply with yes or no only.<|im_end|>
<|im_start|>assistant
```

Prompt con il preambolo:

```text
<|startoftext|><|im_start|>user
Claim under review: The meeting is on Tuesday.

{
  "document": "The meeting was moved to Tuesday at 10:00. It will be held in Rome.",
  "claim": "The meeting is on Tuesday."
}


QUESTION:
<domanda del piano binario>
Reply with yes or no only.<|im_end|>
<|im_start|>assistant
```

Unica differenza: la riga `Claim under review: …` e una riga vuota, all'inizio del messaggio. Il verdetto si legge sempre all'ultimo token.

## 2. Dove va il preambolo negli altri modelli

| Modello | Prompt del piano binario | Dove si aggiunge |
|---|---|---|
| d1-3B | JSON indentato con `document` e `claim`, poi `QUESTION:` | prima del JSON (esempio sopra); il claim scritto con lo stesso escaping del JSON |
| JPT-9B | righe `document: …` e `claim: …` nel primo messaggio | prima di `document:`; eventuali a capo nel claim diventano spazi |
| Winnow-12B | `State:` + JSON compatto (con `<` scritto `<`) | prima di `State:`: `Claim under review: ` + `{"claim": …}` con lo stesso escaping |

`evidence/render.py` aggiunge un'opzione `preamble=` ai renderer del piano binario. Senza opzione il prompt è identico token per token a quello del piano binario (si verifica con un test).

## 3. Scegliere il testo del preambolo

Due candidati:
- **P-A**: `Claim under review: <claim>`
- **P-B**: `The following document will be checked against this claim: <claim>`

Sul dev del piano binario (`dev_select`), per ciascun modello, si misurano BAcc, AUC ed ECE (calibrazione) in sei condizioni:

| Condizione | Modello del verdetto | Prompt |
|---|---|---|
| B-1D | di partenza | senza preambolo |
| B-SAND-A, B-SAND-B | di partenza | P-A, P-B |
| B-FT | addestrato nel piano binario | senza preambolo |
| B-FT-SAND-A, B-FT-SAND-B | addestrato nel piano binario | P-A, P-B |

Si sceglie il testo con cui B-FT-SAND è più vicino a B-FT in AUC; a parità (entro 0,2 punti), il più corto. Lo stesso testo per tutti e tre i modelli.

## 4. Riusare il verdetto o riaddestrarlo

Per ciascun modello:

| Risultato su `dev_select` | Cosa fare |
|---|---|
| B-FT-SAND entro 0,3 punti di BAcc e AUC da B-FT, ECE entro 0,01 | si usa il verdetto del piano binario così com'è (si ricalcola solo la soglia, se si sposta) |
| peggio di così | si riaddestra il verdetto con la ricetta del piano binario e il preambolo nel prompt |

## 5. Dove si scrive la scelta

In `question.json` del piano binario si aggiungono due campi, letti da tutti gli script:

```json
"preamble": {"template": "Claim under review: {claim}", "chosen_on": "dev_select"},
"verdict":  {"d1": "binary_plan", "jpt": "binary_plan", "winnow": "retrained_with_preamble"}
```

(Valori di esempio.)
