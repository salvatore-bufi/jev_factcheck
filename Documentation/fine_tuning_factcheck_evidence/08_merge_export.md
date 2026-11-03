# Step 8 — Export e servizio: verdetto ed evidenza dallo stesso forward

## Obiettivo

Mettere in produzione, per ciascun modello, il verdetto più la testa di evidenza in modo che **un solo forward** restituisca entrambe le cose, con la latenza di un decision model.

## Input e output

| | |
|---|---|
| Input | pesi del verdetto (piano binario, nel formato scelto), eventuale adapter della fase 2, pesi della testa |
| Output | `finetune/release/<modello>-fc-ev-<versione>/`: pesi fusi, `evidence_head.safetensors`, `decision.json`, `frozen.json`, report, model card |

Un rilascio per modello (d1-3B, JPT-9B, Winnow-12B): la testa dipende dagli stati del modello e non si trasferisce.

## 1. Fusione

- Fasi 1 (backbone congelato): i pesi del verdetto sono quelli già rilasciati dal piano binario (o quelli riaddestrati con il preambolo); si aggiunge solo `evidence_head.safetensors` (~10–20 MB).
- Fase 2: si fonde l'adapter congiunto come nel piano binario e si verifica il verdetto (PEFT contro fuso su 200 righe).

## 2. Che cosa deve restituire il server

Da **un** forward sul prompt (formato sandwich):
1. i logit delle risposte in `answer_pos` (o direttamente P(supportato));
2. gli stati dei layer del mix in `answer_pos` (`q`);
3. per ogni frase, gli stati dei layer del mix nell'ultimo token e la media sui suoi token (pooling sulla GPU);
4. se adottate (H2b), le masse di attenzione di `answer_pos` sulle frasi negli 8 layer pieni.

Poi, sulla stessa GPU, l'aggregatore.

| Modello | Server | Note |
|---|---|---|
| d1-3B | **server proprio con transformers** | il benchmark usa già transformers; hook sui layer del mix e sugli 8 layer di attenzione |
| JPT-9B | server proprio con transformers + kernel `flash-linear-attention` / `causal-conv1d` | senza kernel i layer DeltaNet sono lenti: verificarlo nella settimana 1. vLLM solo se si scrive un modello registrato che restituisca stati in posizioni scelte, righe di attenzione e logit delle lettere: non è il primo tentativo |
| Winnow-12B | server proprio con transformers sui pesi BF16 | il server llama.cpp degli autori (NVFP4) **non** si usa per l'evidenza. Una quantizzazione (es. FP8) si valida rifacendo §5 |

Regole:
- **non** trasferire dalla GPU gli stati di tutti i token (decine di MB per richiesta): il pooling per frase si fa sulla GPU;
- la riga di attenzione si ricalcola da query e chiavi con un hook ([03](03_rendering_formato.md) §4), senza passare all'implementazione "eager";
- con il batching, posizioni e intervalli delle frasi sono per richiesta: attenzione al padding a sinistra.

## 3. Logica di risposta

```python
p = p_supported(answer_logits, temperature)
supported = p > threshold
if not supported:
    return {"p_supported": p, "supported": False, "evidence": []}
s, s0 = evidence_head(q, h_end, h_mean, att, att_len, sent_mask)
order = argsort_desc(s - s0)
chosen = [k for k in order if s[k] > s0 + delta][:max_evidence] or [order[0]]
return {"p_supported": p, "supported": True,
        "evidence": [{"sentence": k, "start": spans[k][0], "end": spans[k][1],
                      "score": sigmoid(s[k] - s0)} for k in sorted(chosen)]}
```

(Con la loss di riferimento la condizione è `sigmoid(s[k]) > tau_e`.) Gli offset `start`/`end` si riferiscono al **testo originale** del documento (non al JSON escapato né al prompt con il preambolo), così il client evidenzia direttamente.

## 4. `decision.json`

```json
{"model": "d1-3b-fc-ev-v1",
 "question": "unsupported_detail", "take": "no",
 "state_order": ["document", "claim"],
 "preamble": "Claim under review: {claim}",
 "temperature": 1.0, "threshold": 0.50,
 "evidence": {"layers": [21, 24, 27, 29], "features": ["end", "mean", "att"],
              "attention_layers": [2, 5, 9, 13, 17, 21, 24, 27],
              "rule": "s_k > s_0 + delta", "delta": 0.0, "max_evidence": 5, "min_evidence": 1,
              "segmenter": "finetune/sentences.py@<hash>"},
 "max_document_tokens": 32768}
```

(Valori di esempio.) Il segmentatore, il preambolo e gli indici dei layer sono parte del modello: cambiarli invalida la testa.

## 5. Verifiche prima del rilascio

| Verifica | Soglia |
|---|---|
| Verdetto del server = verdetto in valutazione | differenza media di P < 0,005 su 200 righe |
| Evidenza del server = evidenza in valutazione | stesse frasi su ≥ 199 di 200 righe |
| Feature del server = feature estratte in fase 1a | differenza < 1e-3 (stesso codice di pooling) |
| Latenza con testa vs senza | ≤ +3% p50 e p95 (d1: o ≤ 2 ms) |
| Documenti con virgolette, a capo, `<`, testo non latino | offset corretti sul testo originale |
| Batch con documenti di lunghezze diverse | stesse frasi che con batch 1 |

## 6. Congelamento, test, pacchetto

`frozen.json` con hash di pesi, testa, segmentatore, `question.json` (con il preambolo), `decision.json`, mix e codice di valutazione; poi una sola esecuzione dei test ([06](06_valutazione.md)).

Model card: verifica binaria documento–claim in inglese; **evidenzia solo quando il claim è supportato**; non indica cosa manca né cosa contraddice; segmentazione a livello di frase; il prompt contiene un preambolo con il claim.

Licenze: come nel piano binario (d1-3B LFM Open 1.0; JPT-9B CC BY-NC 4.0, non commerciale; Winnow-12B Apache 2.0). La testa eredita i vincoli dei dati di evidenza (FEVER e HoVer CC BY-SA; SciFact e WiCE usati solo in test).

## Criteri di completamento

- Verifiche di §5 superate per ciascun modello rilasciato.
- `frozen.json` prima del test; test una volta; report e model card pubblicati.
