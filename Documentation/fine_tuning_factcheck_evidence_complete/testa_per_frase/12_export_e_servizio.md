# Passo 12 — Server e rilascio

## In breve

Un server (`evidence/serve.py`) che per ogni richiesta fa **un forward** e restituisce verdetto ed evidenza. Usa lo stesso codice della valutazione (divisione in frasi, rendering, hook, testa, selezione): quello che si è misurato è quello che gira.

## 1. Che cosa fa il server per ogni richiesta

```text
richiesta {document, claim}
  1. divide il documento in frasi (evidence/sentences.py, lo stesso divisore del training)
  2. prompt con il preambolo (codice degli autori) e token di ogni frase (evidence/render.py)
  3. UN forward con gli hook: punteggi "sì"/"no", vettori delle frasi (fine e media, 4 layer), attenzione, q
  4. P(supportato); se non supera la soglia → risposta senza evidenza
  5. testa → s[k], s0 → frasi sopra la soglia (almeno 1, al massimo 5) → posizioni nel documento originale
```

```python
def answer(document, claim):
    spans = sentences.split(document)
    ex = render(document, claim, spans, preamble=DECISION["preamble"])          # input_ids, answer_pos, sent_tok_spans
    logits, feats = model.forward_sentences(ex)                                 # un forward con gli hook
    p = p_supported(logits, DECISION["temperature"], DECISION["take"])
    if p <= DECISION["threshold"]:
        return {"p_supported": p, "supported": False, "evidence": []}
    s, s0, _ = head(**feats)
    shown = select(s[0].tolist(), float(s0[0]), **DECISION["selection"])
    return {"p_supported": p, "supported": True,
            "evidence": [{"sentence": k, "start": spans[k][0], "end": spans[k][1],
                          "score": float(torch.sigmoid(s[0, k] - s0[0]))} for k in shown]}
```

`start` / `end` si riferiscono al documento come l'ha inviato il client, non al prompt.

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
- il riassunto delle frasi si fa sulla GPU negli hook: escono n × 4 × D numeri, mai i vettori di tutti i token;
- per KnowLine, nelle richieste lette due volte, si usano le frasi della seconda copia;
- richieste raggruppate per lunghezza simile;
- documenti oltre il limite di token: errore esplicito (`document_too_long`), mai troncati.

## 3. La cartella di rilascio

`releases/<modello>-fc-ev-<versione>/`:

| File | Contenuto |
|---|---|
| pesi del verdetto | quelli del fine-tuning binario (o riaddestrati con il preambolo, o con la fase 2) |
| `evidence_head.safetensors` | pesi della testa (15 MB per KnowLine-4B, 19,6 MB per elephant-1) |
| `sentences/` | il divisore: `sentences.py` e i parametri inglesi di `punkt_tab` (§4) |
| `decision.json` | tutte le impostazioni (sotto) |
| `frozen.json` | hash di pesi, testa, `sentences.py`, `punkt_tab`, `render.py`, `escape.py`, `question.json`, split e mix |
| `report.md` | risultati dei test e latenza ([10](10_valutazione.md) §7) |
| `MODEL_CARD.md` | descrizione per chi usa il modello |

```json
{"model": "knowline-4b-fc-ev-v1",
 "question_id": "…", "take": "…",
 "preamble": "Claim under review: {claim}",
 "temperature": 1.0, "threshold": 0.50,
 "evidence": {"mix_layers": [19, 23, 27, 31], "attention_layers": [3, 7, 11, 15, 19, 23, 27, 31],
              "features": ["end", "mean", "att"], "head_config": "configs/<id>.json"},
 "selection": {"delta": 0.0, "max_shown": 5, "min_shown": 1},
 "segmenter": {"file": "sentences/sentences.py", "sha256": "<…>", "punkt_tab_sha256": "<…>", "nltk": "3.10.4"},
 "max_prompt_tokens": 32768}
```

(Valori di esempio.) Se cambia il divisore, il preambolo o i layer, la testa va riaddestrata.

## 4. Il divisore in frasi va rilasciato con il modello

La testa lavora su frasi: in produzione il documento va diviso **esattamente** come in training. Il risultato del divisore dipende da tre cose fuori dal nostro codice: la versione di NLTK, i file `punkt_tab` (in `~/nltk_data`, con il problema dei permessi) e la macchina. Per non dipenderne:
1. copiare nella cartella di rilascio (e nella repo, in `evidence/punkt/`) i parametri inglesi di `punkt_tab` (pochi file, licenza Apache 2.0) e caricarli da lì;
2. fissare la versione di NLTK nei requisiti;
3. registrare gli hash in `frozen.json` e verificarli all'avvio del server.

Così il divisore è identico ovunque, senza `nltk.download` e senza permessi da sistemare. La divisione costa qualche millisecondo di CPU, trascurabile rispetto al forward.

## 5. Controlli prima del rilascio

| Controllo | Deve dare |
|---|---|
| Verdetto del server = verdetto in valutazione (200 righe) | differenza media di P < 0,005 |
| Evidenza del server = evidenza in valutazione (200 righe) | stesse frasi in almeno 199 righe |
| Frasi del server = frasi del training (stessi documenti) | identiche, su 1.000 documenti |
| Richieste in batch con lunghezze diverse | stesse frasi che una alla volta |
| Documenti con `"`, `\`, a capo, `<`, emoji, testo non latino | il testo fra `start` ed `end` è la frase giusta |
| Latenza | entro il limite ([10](10_valutazione.md) §5) |

## 6. Congelamento, test, model card

1. Scrivere `frozen.json`.
2. Eseguire **una sola volta** i set di test e la latenza; scrivere `report.md`.
3. Model card: verifica documento–claim in inglese; evidenzia frasi **solo** quando il claim è supportato; non dice cosa manca né cosa contraddice; il prompt contiene una riga col claim prima del documento; documenti fino a 32.768 token; l'evidenza è a livello di frase, con il divisore incluso nel rilascio.
4. Licenze: quelle dei pesi di partenza (MIT per elephant-1; Apache 2.0 per KnowLine-4B, Jeffan-4B, d3-flash, OneJev, pplx-decider-flash, Clef-Flash; **CC BY-NC per StartLux e Nimble**, quindi derivati non commerciali). Dati della testa: HoVer CC BY-SA; SciFact e WiCE con le loro licenze.
