# Passo 5 — Il formato del prompt: preambolo + prompt nativo

## Obiettivo

Fissare il prompt di produzione per i tre modelli e verificare che il verdetto del piano binario funzioni con esso. Il prompt è:

```text
[inizio del turno, come nel prompt nativo]
Claim under review: <claim>

<stato nativo: documento, poi claim>        ← identico al piano binario
<domanda nativa e opzioni>                  ← identico al piano binario
Answer:
```

Il claim compare due volte: nel preambolo (così le frasi del documento sono lette sapendo cosa si verifica) e nello stato, dove il verdetto lo ha sempre visto.

## 1. Dove va il preambolo, per modello

Il preambolo sta nello stesso turno dello stato, subito prima, e usa lo stesso escaping dello stato.

| Modello | Stato nativo | Testo inserito prima dello stato |
|---|---|---|
| d1-3B | JSON indentato (`json.dumps(stato, ensure_ascii=False, indent=2)`), poi due righe vuote e `QUESTION:` | `Claim under review: <claim>\n\n` con il claim passato per `json.dumps(claim, ensure_ascii=False)[1:-1]` |
| JPT-9B | righe `document: …` / `claim: …` nel primo turno utente | `Claim under review: <claim>\n\n` (gli a capo dentro il claim si sostituiscono con uno spazio, così il preambolo resta una riga) |
| Winnow-12B | `State:` + JSON compatto con `safe_data` | `Claim under review: ` + `safe_data({"claim": claim})` + `\n\n` |

`evidence/render.py` implementa questo come un'opzione `preamble=` dei renderer del piano binario, che restano invariati quando l'opzione è spenta.

Test: con `preamble=None`, `input_ids` identici token per token a quelli del piano binario; con il preambolo, i token dopo il preambolo identici a quelli del prompt nativo (a meno della tokenizzazione al confine: si controlla che la differenza sia al più di un token).

## 2. Scelta del testo del preambolo

Due candidati:
- **P-A**: `Claim under review: <claim>`
- **P-B**: `The following document will be checked against this claim: <claim>`

Su `dev_select` del piano binario, per ciascun modello:

| ID | Modello del verdetto | Prompt |
|---|---|---|
| B-1D | di partenza | nativo |
| B-SAND-A / B-SAND-B | di partenza | P-A / P-B |
| B-FT | del piano binario (fine-tunato) | nativo |
| B-FT-SAND-A / B-FT-SAND-B | del piano binario | P-A / P-B |

Metriche: BAcc alla soglia del piano binario, AUC, ECE (stesso script di valutazione del verdetto del piano binario, con `--preamble`).

Scelta del testo: quello con B-FT-SAND più vicino a B-FT in AUC; a parità (entro 0,2 punti), il più corto. Un solo testo per i tre modelli.

## 3. Decisione sul verdetto, per modello

| Esito su `dev_select` | Cosa fare |
|---|---|
| B-FT-SAND entro 0,3 punti di BAcc e AUC da B-FT, ECE entro 0,01 | si usa il verdetto del piano binario **così com'è**; si ricalcola solo la soglia su `dev_select` col preambolo, se cambia |
| oltre questi limiti | si **riaddestra** il verdetto con la ricetta del piano binario e il preambolo attivo nel renderer (stesso mix, stessi iperparametri) |

La decisione si scrive in `question.json`.

## 4. `question.json`

Il file del piano binario, con due campi in più:

```json
{
  "question_id": "unsupported_detail",
  "take": "no",
  "state_fields": ["document", "claim"],
  "preamble": {"template": "Claim under review: {claim}", "chosen_on": "dev_select",
               "candidates": {"P-A": {"d1": 0.0, "jpt": 0.0, "winnow": 0.0},
                              "P-B": {"d1": 0.0, "jpt": 0.0, "winnow": 0.0}}},
  "verdict": {"d1": "binary_plan", "jpt": "binary_plan", "winnow": "retrained_with_preamble"}
}
```

(Valori di esempio.) Ogni script (render, extract, eval, serve) legge il preambolo da qui.

## 5. Offset delle frasi con il preambolo

Il preambolo sposta l'inizio del documento nel prompt di una quantità che dipende dal claim e dall'escaping. Non si stima: `render.py` costruisce il prompt a pezzi e registra l'offset di inizio del documento ([06](06_rendering_ed_estrazione.md) §2).

## Prodotto

- Testo del preambolo scelto; `question.json` aggiornato.
- Per ogni modello: verdetto da usare (piano binario o riaddestrato con il preambolo) e la sua soglia col preambolo.
