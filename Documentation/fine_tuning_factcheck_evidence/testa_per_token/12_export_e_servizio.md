# Passo 12 — Server e rilascio

## In breve

Un server (`evidence/serve_tokens.py`) che per ogni richiesta fa **un forward** e restituisce verdetto e tratti di evidenza. **Nessun divisore in frasi**: il documento entra così com'è e l'uscita sono intervalli di caratteri. Usa lo stesso codice di rendering ([06](06_rendering_ed_estrazione.md)), hook ([01](01_ambiente_e_modelli.md)), testa e decodifica ([08](08_testa_e_loss.md)) usato in valutazione: quello che si è misurato è quello che gira.

## 1. Che cosa fa il server per ogni richiesta

```text
richiesta {document, claim, [display: "spans" | "sentences"]}
  1. prompt con il preambolo (codice degli autori del modello) → input_ids; token del documento e loro caratteri (tok_char)
  2. UN forward con gli hook: punteggi delle opzioni, vettori dei token del documento proiettati, attenzione, q
  3. P(supportato); se non supera la soglia → risposta senza evidenza
  4. testa → s[t], s0 → decodifica → tratti in caratteri sul documento originale
  5. solo se display = "sentences": ogni tratto si allarga alla frase che lo contiene, con il divisore del server
```

```python
def answer(document, claim, display="spans"):
    ex = render_tokens(document, claim, preamble=DECISION["preamble"])          # input_ids, answer_pos, doc_tok, tok_char
    logits, feats = model.forward_tokens(ex)                                    # un forward con gli hook
    p = p_supported(logits, DECISION["temperature"], DECISION["take"])
    if p <= DECISION["threshold"]:
        return {"p_supported": p, "supported": False, "evidence": []}
    s, s0 = head(**feats)
    spans = decode(s[0].tolist(), float(s0[0]), ex["tok_char"], document, **DECISION["decode"])
    if display == "sentences":
        spans = expand_to_sentences(spans, document)                            # solo visualizzazione
    return {"p_supported": p, "supported": True, "evidence": spans}
```

Il passo 5 è l'unico punto in cui può comparire un divisore, e serve solo a mostrare: cambiarlo (NLTK, regex, quello del client) non tocca la testa né i punteggi.

## 2. Come gira ogni modello

Il server ha bisogno degli **stati interni** del forward, non solo della risposta: il forward si fa in transformers, con il codice degli autori e gli hook.

| Modello | Forward nel server |
|---|---|
| KnowLine-4B | front end degli autori (`knowline_server.py`) con backend transformers, più gli hook |
| elephant-1 | motore `qev` in-process (legge la maschera a blocchi da `modeling_overrides`, solo da una cartella locale) |
| Jeffan-4B, d3-flash, pplx-decider-flash, Clef-Flash | il codice degli autori in transformers |
| OneJev1.6-9B, Nimble-9B-v3 | transformers con il prompt degli autori: i loro runtime (Flide, motore Nimble) non espongono gli stati; il verdetto si verifica contro il loro runtime |
| StartLux 9B/4B | transformers con la maschera bidirezionale ricostruita: il server degli autori è vLLM e non espone gli stati; il verdetto si verifica contro il loro server |

Regole:
- servono `flash-linear-attention` e `causal-conv1d` (layer DeltaNet), altrimenti il forward è molto più lento;
- la proiezione dei vettori dei token si fa sulla GPU nell'hook: escono n × 256 numeri per layer;
- per KnowLine, nelle richieste lette due volte, si usano i token della seconda copia;
- richieste raggruppate per lunghezza simile; la testa è invariante al riempimento;
- documenti oltre il limite di token: errore esplicito (`document_too_long`), mai troncati.

## 3. La cartella di rilascio

`releases/<modello>-fc-evtok-<versione>/`:

| File | Contenuto |
|---|---|
| pesi del verdetto | quelli del fine-tuning binario (o riaddestrati con il preambolo, o con la fase 2) |
| `token_head.safetensors` | pesi della testa (18 MB per KnowLine-4B, 19,6 MB per elephant-1) |
| `projection/` | media e proiezione per layer (PCA, o la riduzione appresa nella fase 1b) |
| `decision.json` | tutte le impostazioni (sotto) |
| `frozen.json` | hash di pesi, testa, proiezioni, `token_render.py`, `escape.py`, `decode.py`, `question.json`, split e mix |
| `report.md` | risultati dei test e latenza ([10](10_valutazione.md) §7) |
| `MODEL_CARD.md` | descrizione per chi usa il modello |

```json
{"model": "knowline-4b-fc-evtok-v1",
 "question_id": "…", "take": "…",
 "preamble": "Claim under review: {claim}",
 "temperature": 1.0, "threshold": 0.50,
 "evidence": {"layers": [23, 31], "attention_layers": [3, 7, 11, 15, 19, 23, 27, 31],
              "block": 16, "head_config": "configs/tok_<id>.json"},
 "decode": {"delta": 0.0, "max_gap": 3, "min_len": 4, "max_spans": 5, "force_one": true},
 "max_prompt_tokens": 32768}
```

(Valori di esempio.) Non c'è nessun campo per un divisore in frasi: il modello non ne ha bisogno. Se cambiano preambolo, layer o proiezione, la testa va riaddestrata.

## 4. Controlli prima del rilascio

| Controllo | Deve dare |
|---|---|
| Verdetto del server = verdetto in valutazione (200 righe) | differenza media di P < 0,005 |
| Tratti del server = tratti in valutazione (200 righe) | identici in almeno 199 righe |
| Richieste in batch con lunghezze diverse | stessi tratti che una alla volta |
| Documenti con `"`, `\`, a capo, `<`, emoji, testo non latino | `document[start:end]` è il testo atteso |
| Documento senza punteggiatura | risposta valida, tratti dentro il testo |
| Latenza | entro il limite ([10](10_valutazione.md) §5) |

## 5. Congelamento, test, model card

1. Scrivere `frozen.json`.
2. Eseguire **una sola volta** i set di test e la latenza; scrivere `report.md`.
3. Model card: verifica documento–claim in inglese; evidenzia **solo** quando il claim è supportato; non dice cosa manca né cosa contraddice; restituisce tratti di testo, non frasi; i tratti sono stati appresi da annotazioni a livello di frase, quindi tendono a coprire frasi intere o loro parti significative; prompt con una riga del claim prima del documento.
4. Licenze: quelle dei pesi di partenza (MIT per elephant-1; Apache 2.0 per KnowLine-4B, Jeffan-4B, d3-flash, OneJev, pplx-decider-flash, Clef-Flash; **CC BY-NC per StartLux e Nimble**, quindi derivati non commerciali). Dati della testa: HoVer CC BY-SA; SciFact e WiCE con le loro licenze.
