# Passo 12 — Server e rilascio

## In breve

Per ogni modello si prepara una cartella di rilascio e un server (`evidence/serve.py`) che, per ogni richiesta, fa **un forward** e restituisce verdetto ed evidenza. Il server usa lo stesso codice della valutazione (divisione in frasi, prompt, hook): così quello che si è misurato è quello che gira.

## 1. Che cosa fa il server per ogni richiesta

```text
richiesta {document, claim}
  1. divide il documento in frasi (sentences.py)
  2. costruisce il prompt con il preambolo e trova i token di ogni frase (render.py)
  3. un forward del modello con gli hook → punteggi "sì"/"no", vettori delle frasi, attenzione
  4. P(supportato); se non supera la soglia → risposta senza evidenza
  5. testa di evidenza → punteggi delle frasi e soglia della riga
  6. frasi sopra la soglia (almeno 1, al massimo 5) → posizioni nel documento originale
```

```python
def answer(document, claim):
    spans = sentences.split(document)
    ex = render(document, claim, spans, preamble=DECISION["preamble"])
    logits, feats = model.forward(ex)
    p = p_supported(logits, DECISION["temperature"])
    if p <= DECISION["threshold"]:
        return {"p_supported": p, "supported": False, "evidence": []}
    s, s0, _ = head(**feats)
    margin = s - s0 - DECISION["evidence"]["delta"]
    order = margin.argsort(descending=True).tolist()
    chosen = [k for k in order if margin[k] > 0][: DECISION["evidence"]["max_evidence"]] or order[:1]
    return {"p_supported": p, "supported": True,
            "evidence": [{"sentence": k, "start": spans[k][0], "end": spans[k][1],
                          "score": float(torch.sigmoid(s[k] - s0))} for k in sorted(chosen)]}
```

`start`/`end` si riferiscono al documento come l'ha inviato il client, non al prompt.

## 2. Come gira ogni modello

Tutti e tre con transformers:

| Modello | Note |
|---|---|
| d1-3B | con `trust_remote_code`, come nel benchmark |
| JPT-9B | servono i pacchetti `flash-linear-attention` e `causal-conv1d`, altrimenti è molto più lento |
| Winnow-12B | sui pesi BF16 convertiti. Il server degli autori (llama.cpp, NVFP4) non dà accesso ai vettori interni, quindi non si può usare per l'evidenza |

Regole:
- dalla GPU escono solo i vettori delle frasi (o direttamente i punteggi), mai quelli di tutti i token;
- richieste in batch raggruppate per lunghezza simile;
- documenti oltre 32.768 token: errore esplicito (`document_too_long`), mai tagliati in silenzio;
- all'avvio il server controlla che i file caricati abbiano gli hash registrati in `frozen.json`.

## 3. La cartella di rilascio

`releases/<modello>-fc-ev-<versione>/`:

| File | Contenuto |
|---|---|
| pesi del verdetto | quelli del piano binario (o riaddestrati col preambolo, o con la fase 2) |
| `evidence_head.safetensors` | pesi della testa (~10–20 MB) |
| `decision.json` | tutte le impostazioni (sotto) |
| `frozen.json` | hash di pesi, testa, codice di divisione in frasi e di rendering, `question.json`, `decision.json`, mix, split |
| `report.md` | risultati di test e latenza ([10](10_valutazione.md) §7) |
| `MODEL_CARD.md` | descrizione per chi usa il modello |

```json
{"model": "d1-3b-fc-ev-v1",
 "question_id": "unsupported_detail", "take": "no",
 "preamble": "Claim under review: {claim}",
 "temperature": 1.0, "threshold": 0.50,
 "evidence": {"mix_layers": [21, 24, 27, 29], "attention_layers": [2, 5, 9, 13, 17, 21, 24, 27],
              "features": ["end", "mean", "att"], "head_config": "configs/<id>.json",
              "delta": 0.0, "max_evidence": 5, "min_evidence": 1},
 "segmenter": "evidence/sentences.py@<sha256>",
 "max_document_tokens": 32768}
```

(Valori di esempio.) Se cambia la divisione in frasi, il preambolo o i layer, la testa va riaddestrata.

## 4. Controlli prima del rilascio

| Controllo | Deve dare |
|---|---|
| Verdetto del server = verdetto in valutazione (200 righe) | differenza media di P < 0,005 |
| Evidenza del server = evidenza in valutazione (200 righe) | stesse frasi in almeno 199 |
| Richieste in batch con lunghezze diverse | stesse frasi che una alla volta |
| Documenti con `"`, `\`, a capo, `<`, emoji, testo non latino | il testo fra `start` e `end` è la frase giusta |
| Latenza | entro il limite ([10](10_valutazione.md) §5) |

## 5. Congelamento e test

1. Scrivere `frozen.json`.
2. Eseguire **una sola volta** i set di test e la misura di latenza; scrivere `report.md`.
3. Model card: verifica documento–claim in inglese; evidenzia frasi **solo** quando il claim è supportato; non dice cosa manca né cosa contraddice; il prompt contiene una riga col claim prima del documento; documenti fino a 32.768 token.
4. Licenze: d1-3B LFM Open 1.0; JPT-9B CC BY-NC 4.0 (non commerciale); Winnow-12B Apache 2.0. Dati usati per la testa: FEVER e HoVer CC BY-SA.
