# Passo 12 — Server e rilascio

## In breve

Un server (`evidence/serve_tokens.py`) che per ogni richiesta fa **un forward** e restituisce verdetto e tratti di evidenza. **Nessun divisore in frasi**: il documento entra così com'è, l'uscita sono intervalli di caratteri. Usa lo stesso codice di rendering, estrazione e decodifica della valutazione.

## 1. Che cosa fa il server per ogni richiesta

```text
richiesta {document, claim, [display: "spans" | "sentences"]}
  1. prompt con il preambolo (codice degli autori) → input_ids; token del documento e loro caratteri (tok_char)
  2. un forward con gli hook: punteggi delle opzioni, vettori dei token del documento (proiettati), attenzione
  3. P(supportato); se non supera la soglia → risposta senza evidenza
  4. testa → s[t], s0 → decodifica → tratti in caratteri sul documento originale
  5. (solo se display = "sentences") ogni tratto si allarga alla frase che lo contiene, con il divisore del server
```

```python
def answer(document, claim, display="spans"):
    ex = render_tokens(document, claim, preamble=DECISION["preamble"])          # input_ids, doc_tok, tok_char
    logits, feats = model.forward(ex)
    p = p_supported(logits, DECISION["temperature"])
    if p <= DECISION["threshold"]:
        return {"p_supported": p, "supported": False, "evidence": []}
    s, s0 = head(**feats)
    spans = decode(s.tolist(), float(s0), ex["tok_char"], document, **DECISION["decode"])
    if display == "sentences":
        spans = expand_to_sentences(spans, document)                            # solo visualizzazione
    return {"p_supported": p, "supported": True, "evidence": spans}
```

Il passo 5 è l'unico punto in cui un divisore può comparire, e solo per mostrare. Cambiarlo (NLTK, regex, quello del client) non tocca la testa né i punteggi.

## 2. Come gira ogni modello

Come nel piano per frase ([../testa_per_frase/12_export_e_servizio.md](../testa_per_frase/12_export_e_servizio.md) §2): transformers con il codice degli autori e gli hook. Per OneJev, Nimble e StartLux il forward è in transformers con il prompt degli autori (i loro runtime non espongono gli stati), con il verdetto verificato contro il loro server.

Regole:
- la proiezione dei vettori dei token a 256 si fa sulla GPU nell'hook: escono n × 256 numeri, non n × D;
- per KnowLine, nelle richieste lette due volte, si usano i token della seconda copia;
- documenti oltre il limite di token: errore esplicito (`document_too_long`), mai troncati.

## 3. La cartella di rilascio

`releases/<modello>-fc-evtok-<versione>/`:

| File | Contenuto |
|---|---|
| pesi del verdetto | quelli del piano binario (o riaddestrati col preambolo, o con la fase 2) |
| `token_head.safetensors` | pesi della testa (~18 MB) |
| `projection/` | matrici di proiezione per layer (PCA, o la proiezione appresa della fase 1b) |
| `decision.json` | impostazioni (sotto) |
| `frozen.json` | hash di tutto: pesi, testa, proiezioni, `token_render.py`, `escape.py`, `decode.py`, `question.json`, split, mix |
| `report.md`, `MODEL_CARD.md` | risultati e descrizione |

```json
{"model": "knowline-4b-fc-evtok-v1",
 "question_id": "…", "take": "…",
 "preamble": "Claim under review: {claim}",
 "temperature": 1.0, "threshold": 0.50,
 "evidence": {"layers": [23, 31], "attention_layers": [3, 7, 11, 15, 19, 23, 27, 31], "block": 16,
              "head_config": "configs/tok_<id>.json"},
 "decode": {"delta": 0.0, "max_gap": 3, "min_len": 4, "max_spans": 5, "force_one": true},
 "max_prompt_tokens": 32768}
```

(Valori di esempio.) Non c'è nessun campo per il divisore: il modello non ne ha bisogno.

## 4. Controlli prima del rilascio

| Controllo | Deve dare |
|---|---|
| Verdetto del server = verdetto in valutazione (200 righe) | differenza media di P < 0,005 |
| Tratti del server = tratti in valutazione (200 righe) | identici in almeno 199 righe |
| Richieste in batch con lunghezze diverse | stessi tratti che una alla volta |
| Documenti con `"`, `\`, a capo, `<`, emoji, testo non latino | `document[start:end]` è il testo atteso |
| Documento senza punteggiatura | risposta valida, tratti dentro il testo |
| Latenza | entro il limite ([10](10_valutazione.md) §5) |

## 5. Congelamento e test

`frozen.json` prima dei test; test eseguiti una sola volta; `report.md`. Model card: verifica documento–claim in inglese; evidenzia **solo** quando il claim è supportato; restituisce tratti di testo, non frasi; i tratti sono stati addestrati da annotazioni a livello di frase, quindi tendono a coprire frasi intere o loro parti significative. Licenze: quelle dei pesi di partenza (StartLux e Nimble CC BY-NC).
