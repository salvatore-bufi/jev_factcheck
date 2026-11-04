# Passo 5 — Il prompt con il preambolo

## In breve

Si aggiunge una riga con il claim **prima** del documento, lasciando il resto del prompt identico a quello del piano binario. Poi si controlla che il verdetto del piano binario funzioni ancora con questa riga in più. Se sì, si riusa; se no, si riaddestra con la stessa ricetta del piano binario, includendo la riga.

Perché serve: nei modelli causali (KnowLine, OneJev, Nimble) ogni token vede solo quello che lo precede. Senza preambolo, quando il modello legge il documento non sa ancora che cosa deve verificare, e i vettori delle frasi non "parlano" del claim. Nei modelli bidirezionali le frasi vedono il claim negli 8 layer pieni, ma non nei 24 layer DeltaNet, che restano causali: il preambolo serve meno, e l'esperimento H4 ([11](11_esperimenti.md)) misura quanto.

## 1. Esempio (KnowLine-4B, schema)

Prompt del piano binario (schema: il testo esatto lo produce il front end degli autori, `knowline_layout.py`):

```text
[turno utente]       document: The meeting was moved to Tuesday at 10:00. It will be held in Rome.
                     claim: The meeting is on Tuesday.
[turno utente]       <istruzione> Question: <domanda del piano binario> Options: A. Yes: … B. No: …
[turno assistente]   Answer:
```

Prompt con il preambolo:

```text
[turno utente]       Claim under review: The meeting is on Tuesday.

                     document: The meeting was moved to Tuesday at 10:00. It will be held in Rome.
                     claim: The meeting is on Tuesday.
[turno utente]       <istruzione> Question: <domanda del piano binario> Options: A. Yes: … B. No: …
[turno assistente]   Answer:
```

Unica differenza: la riga `Claim under review: …` e una riga vuota, all'inizio del turno con lo stato. Il verdetto si legge sempre nello stesso punto. Nelle richieste che KnowLine scrive due volte (fino a 3.000 token), il preambolo compare in entrambe le copie, perché fa parte dello stato.

## 2. Dove va il preambolo negli altri modelli

Regola unica: **nello stesso turno dello stato, subito prima dello stato**, con lo stesso escaping usato per lo stato. Si aggiunge come opzione `preamble=` del renderer del piano binario, che chiama il codice degli autori ([../fine_tuning_factcheck/03_rendering_formato.md](../fine_tuning_factcheck/03_rendering_formato.md) §2).

| Modello | Lo stato nel prompt | Dove va il preambolo |
|---|---|---|
| KnowLine-4B | testo indentato in un turno utente | prima dello stato (esempio sopra) |
| elephant-1 | `<state>` JSON indentato `</state>` nel prefisso comune | prima di `<state>`: resta nel prefisso, quindi è visto da tutte le domande |
| StartLux 9B/4B | `Evidence:` + JSON compatto | prima di `Evidence:` |
| d3-flash, Jeffan-4B | `State:` + JSON | prima di `State:` |
| OneJev1.6-9B | blocco dello stato nel prefisso comune | prima del blocco dello stato |
| Nimble-9B-v3 | turno utente JSON con contesto e schema | come campo testuale aggiuntivo **prima** del contesto, se il motore accetta un prefisso nel turno; altrimenti si salta (Nimble esce dagli esperimenti sul preambolo) |
| pplx-decider-flash | sequenza unica con lo stato e le domande | prima dello stato |
| Clef-Flash | `STATE:` + JSON compatto | prima di `STATE:` |

Test: con `preamble=None`, `input_ids` identici token per token a quelli del piano binario; con il preambolo, i token dopo il preambolo identici a quelli del prompt originale (a meno di un token al confine).

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

Si sceglie il testo con cui B-FT-SAND è più vicino a B-FT in AUC; a parità (entro 0,2 punti), il più corto. Lo stesso testo per tutti i modelli.

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
"verdict":  {"knowline-4b": "binary_plan", "elephant-1": "retrained_with_preamble"}
```

(Valori di esempio.)
