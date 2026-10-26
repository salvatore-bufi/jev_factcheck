# Step 8 — Export e servizio: verdetto ed evidenza dallo stesso forward

## Obiettivo

Mettere in produzione il modello del verdetto più la testa di evidenza in modo che **un solo forward** restituisca entrambe le cose, con la latenza di un decision model.

## Input e output

| | |
|---|---|
| Input | pesi fusi del verdetto (piano binario, claim → documento), eventuale adapter della fase 2, pesi della testa |
| Output | `finetune/release/<modello>-fc-ev-<versione>/`: pesi fusi, `evidence_head.safetensors`, `decision.json`, `frozen.json`, report, model card |

## 1. Fusione

- Fase 1 (backbone congelato): i pesi del verdetto sono quelli già rilasciati dal piano binario; si aggiunge solo `evidence_head.safetensors` (~5–10 MB).
- Fase 2: si fonde l'adapter congiunto come nel piano generale ([../fine_tuning/08_merge_export.md](../fine_tuning/08_merge_export.md) §1–2) e si verifica il verdetto (PEFT contro fuso su 200 righe).

## 2. Che cosa deve restituire il server

Da **un** forward sul prompt:
1. i logit delle due lettere in `Answer:` (o direttamente P(supportato));
2. lo stato nascosto in `Answer:` (`q`);
3. gli stati nascosti nelle posizioni `sent_end_pos` (`h_k`), dal layer scelto in H2.

Poi, sulla stessa macchina, l'aggregatore (< 1 ms).

| Server | Come | Note |
|---|---|---|
| **Server proprio con transformers** (più semplice) | forward con hook sul layer scelto; si prendono solo le posizioni necessarie; lettere dalle due righe della testa | pieno controllo; throughput inferiore a vLLM con molte richieste parallele |
| **vLLM** | le modalità di pooling restituiscono uscite per token; serve un modello registrato che restituisca stati in posizioni scelte e i logit delle lettere | velocità migliore; supporto per `gemma4_text` e per i layer DeltaNet di Qwen3.5 da verificare |
| **llama.cpp (GGUF di Winnow)** | embedding per token senza pooling | va verificato che la versione del fork di Winnow restituisca stati e logit dallo stesso passaggio; altrimenti non si usa per l'evidenza |

Regola: **non** trasferire dalla GPU gli stati di tutti i token (decine di MB per richiesta), ma solo `q` e gli `h_k`.

## 3. Logica di risposta

```python
p = p_supported(letter_logits, temperature)
supported = p > threshold
if not supported:
    return {"p_supported": p, "supported": False, "evidence": []}
scores = sigmoid(evidence_head(q, h, sent_mask))
chosen = [k for k in argsort_desc(scores) if scores[k] > tau_e][:max_evidence] or [argmax(scores)]
return {"p_supported": p, "supported": True,
        "evidence": [{"sentence": k, "start": spans[k][0], "end": spans[k][1], "score": scores[k]}
                     for k in sorted(chosen)]}
```

Gli offset `start`/`end` si riferiscono al **testo originale** del documento (non al JSON escapato), così il client evidenzia direttamente.

## 4. `decision.json`

```json
{"model": "winnow-12b-fc-ev-v1",
 "question": "support_simple", "state_order": ["claim", "document"],
 "read": "P(B)", "temperature": 1.0, "threshold": 0.50,
 "evidence": {"layer": "last", "tau_e": 0.42, "max_evidence": 5, "min_evidence": 1,
              "segmenter": "finetune/sentences.py@<hash>"},
 "max_document_tokens": 32768}
```

(Valori di esempio.) Il segmentatore è parte del modello: cambiarlo cambia le posizioni e invalida la testa.

## 5. Verifiche prima del rilascio

| Verifica | Soglia |
|---|---|
| Verdetto del server = verdetto in valutazione | differenza media di P < 0,005 su 200 righe |
| Evidenza del server = evidenza in valutazione | stesse frasi su ≥ 199 di 200 righe |
| Latenza con testa vs senza | ≤ +3% p50 e p95 |
| Documenti con virgolette, a capo, `<`, testo non latino | offset corretti sul testo originale |

## 6. Congelamento, test, pacchetto

`frozen.json` con hash di pesi, testa, segmentatore, `question.json`, `decision.json`, mix e codice di valutazione; poi una sola esecuzione dei test ([06](06_valutazione.md)).

Model card: verifica binaria documento–claim in inglese; **evidenzia solo quando il claim è supportato**; non indica cosa manca né cosa contraddice; segmentazione a livello di frase.

Licenze: come nel piano binario; la testa eredita i vincoli dei dati di evidenza (FEVER e HoVer CC BY-SA; SciFact e WiCE usati solo in test).

## Criteri di completamento

- Verifiche di §5 superate.
- `frozen.json` prima del test; test una volta; report e model card pubblicati.
