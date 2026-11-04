# Passo 5 — Il prompt con il preambolo

## In breve

Si aggiunge una riga con il claim **prima** dello stato (documento e claim), lasciando il resto del prompt identico a quello del fine-tuning binario. Poi si controlla che il verdetto funzioni ancora con questa riga in più: se sì, si riusa così com'è; se no, si riaddestra con la stessa ricetta del fine-tuning binario, includendo la riga.

## 1. Perché serve

Nei modelli causali (KnowLine-4B, OneJev, Nimble) ogni token vede solo i token che lo precedono. Nel prompt del fine-tuning binario lo stato è "documento, poi claim": quando il modello legge il documento non sa ancora che cosa deve verificare, e i vettori delle frasi (quelli che la testa usa) non dipendono dal claim. Con il claim scritto anche prima, ogni frase del documento è letta "sapendo cosa si cerca".

Nei modelli bidirezionali (elephant-1, StartLux, d3-flash, pplx, Jeffan) le frasi del documento vedono il claim negli 8 layer ad attenzione piena anche senza preambolo, ma non nei 24 layer DeltaNet, che restano causali. Il preambolo serve meno; l'esperimento H8 ([11](11_esperimenti.md)) misura quanto.

Il claim resta anche nella posizione originale, dopo il documento: lì il verdetto è stato addestrato a vederlo.

## 2. Esempio (KnowLine-4B, schema)

Il testo esatto lo produce il front end degli autori (`knowline_layout.py`). Schema:

```text
senza preambolo
[turno utente]       document: The meeting was moved to Tuesday at 10:00. It will be held in Rome.
                     claim: The meeting is on Tuesday.
[turno utente]       <istruzione> Question: <domanda> Options: A. Yes: … B. No: …
[turno assistente]   Answer:

con preambolo
[turno utente]       Claim under review: The meeting is on Tuesday.

                     document: The meeting was moved to Tuesday at 10:00. It will be held in Rome.
                     claim: The meeting is on Tuesday.
[turno utente]       <istruzione> Question: <domanda> Options: A. Yes: … B. No: …
[turno assistente]   Answer:
```

KnowLine scrive **due volte** le richieste fino a 3.000 token ("Read the same input again before answering:"). Il preambolo fa parte dello stato e compare in entrambe le copie. Le frasi per la testa si leggono dalla **seconda** copia: ha già visto tutto il documento e il claim della prima, quindi è la lettura più informata.

## 3. Dove va il preambolo negli altri modelli

Regola unica: **nello stesso turno dello stato, subito prima dello stato**, con lo stesso escaping usato per lo stato (per gli stati in JSON, il claim scritto come stringa JSON).

| Modello | Lo stato nel prompt | Dove va il preambolo |
|---|---|---|
| KnowLine-4B | testo indentato in un turno utente | prima dello stato |
| elephant-1 | `<state>` JSON indentato `</state>` nel prefisso comune | prima di `<state>`: resta nel prefisso, visto da tutte le domande |
| StartLux 9B/4B | `Evidence:` + JSON compatto | prima di `Evidence:` |
| d3-flash, Jeffan-4B | `State:` + JSON | prima di `State:` |
| OneJev1.6-9B | blocco dello stato nel prefisso comune | prima del blocco |
| Nimble-9B-v3 | turno utente JSON con contesto e schema | campo testuale prima del contesto, se il motore lo accetta; altrimenti Nimble esce dagli esperimenti sul preambolo |
| pplx-decider-flash | sequenza unica con stato e domande | prima dello stato |
| Clef-Flash | `STATE:` + JSON compatto | prima di `STATE:` |

Si implementa come opzione `preamble=` del renderer di ciascun modello (`evidence/render.py`), che chiama il codice degli autori. Test: con `preamble=None` il prompt è identico token per token a quello del fine-tuning binario; con il preambolo, i token dopo il preambolo sono identici (al più un token di differenza al confine).

## 4. Scegliere il testo del preambolo

Due candidati:
- **P-A**: `Claim under review: <claim>`
- **P-B**: `The following document will be checked against this claim: <claim>`

Sul dev del fine-tuning binario (`dev_select`), per ciascun modello, si misurano BAcc, AUC ed ECE in sei condizioni:

| Condizione | Modello del verdetto | Prompt |
|---|---|---|
| B-1D | di partenza (prima del fine-tuning binario) | senza preambolo |
| B-SAND-A, B-SAND-B | di partenza | P-A, P-B |
| B-FT | fine-tunato (fine-tuning binario) | senza preambolo |
| B-FT-SAND-A, B-FT-SAND-B | fine-tunato | P-A, P-B |

Si sceglie il testo con cui B-FT-SAND è più vicino a B-FT in AUC; a parità (entro 0,2 punti) il più corto. Lo stesso testo per tutti i modelli.

## 5. Riusare il verdetto o riaddestrarlo

| Risultato su `dev_select` | Cosa fare |
|---|---|
| B-FT-SAND entro 0,3 punti di BAcc e AUC da B-FT, ECE entro 0,01 | il verdetto fine-tunato si usa così com'è; si ricalcola solo la soglia, se si sposta |
| peggio di così | si riaddestra la LoRA del verdetto con la ricetta del fine-tuning binario e il preambolo nel prompt |

## 6. Dove si scrive la scelta

In `question.json` (il file della domanda congelata del fine-tuning binario) si aggiungono due campi, letti da tutti gli script:

```json
"preamble": {"template": "Claim under review: {claim}", "chosen_on": "dev_select"},
"verdict":  {"knowline-4b": "binary_plan", "elephant-1": "retrained_with_preamble"}
```

(Valori di esempio.)
