# Step 2d — Formato del prompt: il sandwich

## Obiettivo

Fissare il prompt di produzione. Come nel piano binario ([../fine_tuning_factcheck/02d_varianti_domanda.md](../fine_tuning_factcheck/02d_varianti_domanda.md)) la domanda è unica e congelata in `finetune/question.json`, e l'ordine dei campi dello stato resta quello **nativo** (`document`, poi `claim`). Qui si aggiunge una cosa sola: un **preambolo** con il claim prima dello stato.

## 1. Il formato

```text
[inizio del turno, come nel prompt nativo]
<preambolo: il claim>
<stato nativo: documento, poi claim>
<domanda nativa, opzioni>
Answer:
```

- Le frasi del documento vedono il claim del preambolo: i loro stati sono calcolati "sapendo cosa si verifica".
- `Answer:` ha il claim dello stato a poca distanza, nell'ordine in cui il modello lo ha sempre visto.
- Tutto ciò che segue il preambolo è **identico** al prompt del piano binario: il verdetto si può riusare ([01](01_baseline.md) §1).

Perché non "claim → documento" (il piano precedente): cambia il prompt che il verdetto ha visto in training e allontana il claim dalla posizione di lettura. Perché non il claim ripetuto nella domanda (la forma di [idea.md](idea.md)): cambia la domanda, che il piano binario ha scelto e congelato con una misura. Confronto in [00a](00a_scelta_approccio.md) §4.

## 2. Il preambolo per modello

Il preambolo sta **dentro lo stesso turno dello stato**, subito prima, e usa lo stesso escaping dello stato (il claim è testo da valutare, non istruzioni).

| Modello | Stato nativo | Preambolo |
|---|---|---|
| d1-3B | JSON indentato (`json.dumps(…, ensure_ascii=False, indent=2)`), poi due righe vuote e `QUESTION:` | una riga `Claim under review: <claim>` e una riga vuota, prima del JSON |
| JPT-9B | righe `document: …` / `claim: …` nel turno utente | una riga `Claim under review: <claim>` e una riga vuota, prima di `document:` |
| Winnow-12B | `State:` + JSON compatto con `<` escapato | una riga `Claim under review: ` + JSON compatto `{"claim": …}` (stesso escaping), prima di `State:` |

Due formulazioni del preambolo si provano in B-SAND ([01](01_baseline.md)), con la stessa regola di scelta:
- P-A: `Claim under review: <claim>`;
- P-B: `The following document will be checked against this claim: <claim>`.

Si sceglie quella con il verdetto più vicino a B-1D; a parità, la più corta. Il testo esatto, carattere per carattere, va in `question.json` (§3).

Su Winnow il preambolo sta fuori dal JSON dello stato. Se B-SAND mostra un calo, si prova la variante con il claim come primo campo di un oggetto separato (`State:` preceduto da `Claim: {"claim": …}`), sempre lasciando lo stato nativo intatto.

## 3. `question.json`

Il file del piano binario si estende con due campi:

```json
{
  "question_id": "unsupported_detail",
  "take": "no",
  "state_fields": ["document", "claim"],
  "preamble": {"enabled": true, "template": "Claim under review: {claim}", "chosen_on": "dev_select"},
  "verdict_reused": {"d1": true, "jpt": true, "winnow": true}
}
```

(Valori di esempio.) `verdict_reused` dice, per modello, se il verdetto è quello del piano binario usato col preambolo o un modello riaddestrato con il preambolo ([02c](02c_replay_e_forgetting.md)).

## 4. Nessun marcatore nel documento

Il documento resta testo normale. Le frasi si individuano dagli offset ([03](03_rendering_formato.md)), non con token inseriti come `[S1]`. Vantaggi: nessun cambio di formato oltre al preambolo; il testo che vede il modello è quello che vede l'utente.

## 5. Varianti

Nessuna variante di domanda né di nomi di campo: il chiamante è il nostro codice. Se in futuro l'input arriva con altri nomi (`context`, `answer`), si normalizzano prima del modello.

## 6. Il documento dentro un JSON (d1-3B, Winnow-12B)

In d1-3B e Winnow il documento è una stringa JSON: virgolette, a capo e (per Winnow) `<` vengono **escapati** (`\"`, `\n`, `<`). Gli offset delle frasi sul testo originale non coincidono con le posizioni nel prompt: il rendering deve mappare ogni frase sul testo escapato ([03](03_rendering_formato.md) §3). Con JPT (righe `campo: valore`) il problema è minore ma va gestito per gli a capo. Il preambolo sposta tutti gli offset di una quantità fissa: si calcola, non si stima.

## Criteri per passare allo step successivo

- Preambolo scelto (P-A o P-B) e scritto in `question.json`.
- B-SAND e B-FT-SAND misurati; `verdict_reused` compilato per i tre modelli.
