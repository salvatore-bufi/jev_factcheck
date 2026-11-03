# Passo 12 — Export e servizio

## Obiettivo

Per ciascun modello, un rilascio che risponde con verdetto ed evidenza da **un** forward, e un server (`evidence/serve.py`) che lo esegue.

## 1. Il pacchetto di rilascio

`releases/<modello>-fc-ev-<versione>/`:

| File | Contenuto |
|---|---|
| pesi del verdetto | quelli del piano binario (o riaddestrati col preambolo); se c'è stata la fase 2, l'adapter congiunto fuso |
| `evidence_head.safetensors` | pesi della testa (~10–20 MB) |
| `decision.json` | tutto ciò che serve per riprodurre l'uscita (§3) |
| `frozen.json` | hash di pesi, testa, `sentences.py`, `render.py`, `question.json`, `decision.json`, mix, split, codice di valutazione |
| `report.md` | tabelle di [10](10_valutazione.md) §7 |
| `MODEL_CARD.md` | §6 |

Fusione nella fase 2: come nel piano binario; poi controllo PEFT contro fuso su 200 righe (differenza media di P < 0,005, stesse frasi su ≥ 199 righe).

## 2. Il server

Server proprio con transformers per tutti e tre i modelli, con lo stesso codice di `models.py`, `render.py` ed `extract.py`:

```text
richiesta {document, claim}
  → sentences.split(document)
  → render (preambolo + prompt nativo) → input_ids, answer_pos, sent_tok_spans
  → forward unico con gli hook: logit delle risposte, stati per frase (fine e media, layer del mix), righe di attenzione (8 layer pieni)
  → P(supportato); se ≤ soglia: risposta senza evidenza
  → testa → s, s_0 → frasi scelte → offset sul documento originale
```

| Modello | Note di servizio |
|---|---|
| d1-3B | transformers con `trust_remote_code` |
| JPT-9B | transformers con i kernel `flash-linear-attention` e `causal-conv1d` installati (senza, è molto più lento) |
| Winnow-12B | transformers sui pesi BF16 convertiti; il server llama.cpp degli autori non espone gli stati e non si usa |

Regole:
- il pooling per frase si fa sulla GPU dentro gli hook; dalla GPU escono solo `[n, L, D]` (o direttamente le uscite della testa, se la testa gira sulla stessa GPU, come da default);
- batching: le richieste si raggruppano per lunghezza; con padding a sinistra, `answer_pos` e `sent_tok_spans` si spostano per riga;
- documenti oltre 32.768 token: risposta di errore esplicita (`document_too_long`), mai troncamento silenzioso;
- la testa e il verdetto si caricano una volta all'avvio; `decision.json` si verifica contro `frozen.json` all'avvio (hash).

## 3. Logica di risposta

```python
def answer(document, claim):
    spans = sentences.split(document)
    ex = render(document, claim, spans, preamble=DECISION["preamble"])
    logits, feats = model.forward(ex)
    p = p_supported(logits, DECISION["temperature"])
    if p <= DECISION["threshold"]:
        return {"p_supported": p, "supported": False, "evidence": []}
    s, s0, _ = head(**feats)
    margin = s - s0 - DECISION["evidence"]["delta"]            # con ref_loss: sigmoid(s) - tau_e
    order = margin.argsort(descending=True).tolist()
    chosen = [k for k in order if margin[k] > 0][: DECISION["evidence"]["max_evidence"]] or order[:1]
    return {"p_supported": p, "supported": True,
            "evidence": [{"sentence": k, "start": spans[k][0], "end": spans[k][1],
                          "score": float(torch.sigmoid(s[k] - s0))} for k in sorted(chosen)]}
```

`start` / `end` sono offset sul documento **originale** (non sul JSON escapato, non sul prompt).

## 4. `decision.json`

```json
{"model": "d1-3b-fc-ev-v1",
 "question_id": "unsupported_detail", "take": "no",
 "state_order": ["document", "claim"],
 "preamble": "Claim under review: {claim}",
 "temperature": 1.0, "threshold": 0.50,
 "evidence": {"mix_layers": [21, 24, 27, 29], "attention_layers": [2, 5, 9, 13, 17, 21, 24, 27],
              "features": ["end", "mean", "att"], "head_config": "configs/<id>.json",
              "rule": "s_k > s_0 + delta", "delta": 0.0, "max_evidence": 5, "min_evidence": 1},
 "segmenter": "evidence/sentences.py@<sha256>",
 "max_document_tokens": 32768}
```

(Valori di esempio.) Segmentatore, preambolo, layer e feature fanno parte del modello: cambiarne uno invalida la testa.

## 5. Verifiche prima del rilascio

| Verifica | Soglia |
|---|---|
| Verdetto del server = verdetto in valutazione | differenza media di P < 0,005 su 200 righe |
| Evidenza del server = evidenza in valutazione | stesse frasi su ≥ 199 di 200 righe |
| Feature del server = feature di `extract.py` | differenza < 1e-3 |
| Batch misti (lunghezze diverse) | stesse frasi che con batch 1 |
| Documenti con `"`, `\`, a capo, `<`, emoji, testo non latino | offset corretti sul testo originale (il testo tra `start` ed `end` è la frase) |
| Latenza | C ≤ B + 3% (d1: o ≤ 2 ms), p50 e p95 ([10](10_valutazione.md) §5) |

## 6. Congelamento, test, model card

1. Scrivere `frozen.json`.
2. Eseguire **una volta** i blocchi di test ([10](10_valutazione.md) §6) e la latenza; scrivere `report.md`.
3. Model card: verifica binaria documento–claim in inglese; evidenzia **solo** quando il claim è supportato; non indica cosa manca né cosa contraddice; evidenza a livello di frase; il prompt contiene un preambolo con il claim; lunghezza massima 32.768 token.
4. Licenze: d1-3B LFM Open 1.0; JPT-9B CC BY-NC 4.0 (non commerciale); Winnow-12B Apache 2.0. Dati della testa: FEVER e HoVer CC BY-SA; SciFact e WiCE solo in test.

## Prodotto

- `evidence/serve.py`; un rilascio per modello con le verifiche di §5 superate e il report di test.
