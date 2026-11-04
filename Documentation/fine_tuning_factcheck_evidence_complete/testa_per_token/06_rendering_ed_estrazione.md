# Passo 6 — Dai caratteri ai token, e salvataggio dei vettori

## In breve

1. **Rendering** (`evidence/token_render.py`): per ogni riga si costruisce il prompt con il preambolo (codice degli autori) e si calcola, per **ogni token del documento**, a quali caratteri del documento originale corrisponde (`tok_char`). Da qui l'etichetta di ogni token (dentro o fuori da una frase gold) e, in produzione, la conversione dei tratti in caratteri.
2. **Estrazione** (`evidence/token_extract.py`): un forward per riga con gli hook del passo [01](01_ambiente_e_modelli.md); si salvano su disco i vettori ridotti di ogni token, la sua attenzione e `q`. Il modello è congelato: si calcolano una volta sola.

## 1. L'esempio renderizzato

```json
{"id": "hover:train:…", "model": "knowline-4b",
 "input_ids": [248045, 846, "…"], "answer_pos": 2371,
 "doc_tok": [1103, 2290],
 "tok_char": [[0, 7], [7, 11], [11, 19], "…"],
 "tok_label": [0, 0, 1, 1, 1, "…"],
 "gold_tok_spans": [[37, 61], [612, 640]],
 "label": 1, "evidence_term": "supervised", "row_weight": 1.0, "omega": 0.5}
```

| Campo | Significato |
|---|---|
| `doc_tok` | primo e ultimo token del documento nel prompt (KnowLine letto due volte: della seconda copia) |
| `tok_char[i]` | caratteri `[inizio, fine)` del documento **originale** coperti dal token `doc_tok[0] + i` |
| `tok_label[i]` | 1 se il token sta dentro una frase gold (§3) |
| `gold_tok_spans` | per ogni frase gold dell'insieme usato in training, il primo e l'ultimo token con etichetta 1 (servono a `L_hit`, [08](08_testa_e_loss.md) §3) |

Con più insiemi alternativi (WiCE), in training si usa l'**unione** delle frasi gold: tutte sono evidenza valida.

## 2. Dai token ai caratteri del documento originale

Nel prompt il documento può essere scritto diversamente dall'originale. Per i modelli con lo stato in JSON (elephant-1, StartLux, d3-flash, Jeffan, Nimble, pplx, Clef) virgolette e a capo diventano sequenze di due caratteri (`"` → `\"`, a capo → `\n`). Per KnowLine lo stato è testo indentato. Serve quindi, per ogni carattere del documento **come appare nel prompt**, il carattere originale da cui viene.

Procedura:
1. costruire il prompt in tre pezzi: prima del documento, documento come appare nel prompt, dopo il documento (si conosce così `doc_start`, dove inizia il documento nel prompt);
2. per il pezzo del documento, `escape_with_map` restituisce il testo trasformato e, per ogni suo carattere, l'indice del carattere originale (`origin`);
3. tokenizzare il prompt con `return_offsets_mapping=True`;
4. `token_char_spans`: per ogni token che cade nel pezzo del documento, `tok_char` = dal primo all'ultimo carattere originale che copre.

```python
ESC = {'"': '\\"', "\\": "\\\\", "\n": "\\n", "\r": "\\r", "\t": "\\t", "\b": "\\b", "\f": "\\f"}   # come json.dumps


def escape_with_map(text, escape_lt=False):
    """Il testo come appare in una stringa JSON e, per ogni carattere del testo trasformato, il carattere originale da cui viene."""
    out, origin = [], []
    for i, ch in enumerate(text):
        if ch in ESC:
            piece = ESC[ch]
        elif escape_lt and ch == "<":
            piece = "\\u003c"
        elif ord(ch) < 0x20:
            piece = f"\\u{ord(ch):04x}"
        else:
            piece = ch
        out.append(piece)
        origin.extend([i] * len(piece))
    return "".join(out), origin


def token_char_spans(offsets, doc_start, doc_text_in_prompt_len, origin):
    """offsets: (inizio, fine) di ogni token nel prompt (return_offsets_mapping); doc_start: dove inizia il documento nel prompt;
    origin: per ogni carattere del documento nel prompt, il carattere originale. -> (indici dei token del documento, tok_char)."""
    doc_end = doc_start + doc_text_in_prompt_len
    idx, spans = [], []
    for k, (a, b) in enumerate(offsets):
        a, b = max(a, doc_start), min(b, doc_end)
        if a >= b:
            continue
        idx.append(k)
        spans.append((origin[a - doc_start], origin[b - 1 - doc_start] + 1))
    return idx, spans


def token_labels(tok_char, gold_char_spans, text):
    """1 se almeno metà dei caratteri non spazio del token sta dentro un intervallo gold."""
    labels = []
    for a, b in tok_char:
        chars = [c for c in range(a, b) if not text[c].isspace()]
        inside = sum(any(g0 <= c < g1 for g0, g1 in gold_char_spans) for c in chars)
        labels.append(int(bool(chars) and inside * 2 >= len(chars)))
    return labels
```

Il codice è stato eseguito su un documento con virgolette e a capo dentro un JSON: il testo trasformato è identico a `json.dumps`, `\"` torna a `"`, `\n` torna all'a capo, e le etichette dei token seguono la frase gold. Esempio: documento `Rose "Byrne" is an\nAustralian actress. She starred in films.`, frase gold la prima; i token `Rose " Byrne " is an Australian actress .` hanno etichetta 1, `\n` e `She starred in films .` 0.

Casi da gestire:
- **KnowLine** non usa JSON: il documento compare quasi così com'è (con l'indentazione delle righe). `escape_with_map` si sostituisce con la trasformazione del suo `knowline_layout`, registrando allo stesso modo `origin`. Nelle richieste lette due volte si prende la seconda copia del documento (`doc_start` della seconda).
- **Prefisso e domanda tokenizzati separatamente** (elephant-1, OneJev, Nimble): gli offset si calcolano sui pezzi e si sommano.
- **Token a cavallo** fra il documento e il testo intorno (rari, ai bordi): vengono tagliati all'intervallo del documento.

`escape_with_map` sostituisce quella usata dai dati del passo 2 per lo stesso scopo: un solo file, `evidence/escape.py`.

## 3. L'etichetta di un token

```text
tok_label[i] = 1  se almeno metà dei caratteri non spazio del token sta dentro un intervallo di gold_char_sets
             = 0  altrimenti (e sempre 0 se la riga ha label 0)
```

Le fonti annotano frasi intere: in una frase gold anche le parole irrilevanti hanno etichetta 1. La loss tiene conto di questo rumore ([08](08_testa_e_loss.md) §2).

## 4. Che cosa si salva (fase 1a)

| Feature | Forma | Tipo | Byte per token |
|---|---|---|---:|
| `h` | n × 2 × 256 | fp16, proiettato con la PCA (layer 23 e 31) | 1.024 |
| `att` | n × 128 | fp16 | 256 |
| `tok_char` | n × 2 | int32 | 8 |
| `tok_label` | n | int8 | 1 |
| `q` | 2 × D | fp32 | — (uno per riga) |
| `answer_logits` | 2 | fp32 | — |

Per una riga HoVer (~1.500 token) circa 2 MB; per le ~28.600 righe delle tre fonti circa 60 GB.

Formato: `data/evidence_tok/features/<modello>/<set>/shard-00000.safetensors`, 500 righe per file (tensori concatenati lungo i token) più `index.jsonl` (`id`, shard, primo token, n, `gold_tok_spans`, `evidence_term`, pesi). Ogni cartella ha un `meta.json` con modello, revisione dei pesi, preambolo, layer, file della PCA, hash di `token_render.py` ed `escape.py`: il training rifiuta vettori calcolati con impostazioni diverse.

## 5. Comando

```sh
gpu-run --batch --name evtok-extract-knowline -- bash -c \
  'python -m evidence.token_extract --model knowline-4b \
     --rows dataset/evidence/mixes/ev_tok_head.jsonl dataset/evidence/splits/dev_evid_select.jsonl \
            dataset/evidence/splits/dev_evid_confirm.jsonl dataset/evidence/splits/dev_evid_neg.jsonl \
     --out data/evidence_tok/features/knowline-4b > logs/evtok-extract-knowline.out 2>&1'
```

Riprende dagli shard completi; ordina le righe per lunghezza (meno riempimento); batch 1 sopra i 4.096 token.

## 6. Test

| Test | Deve dare |
|---|---|
| Testo ricostruito dai `tok_char` = documento originale (spazi a parte) | 200 righe su 200, per ogni modello |
| Documenti con `"`, `\`, a capo, tab, `<`, emoji, testo non latino | `tok_char` giusti |
| Per ogni frase gold, i token con etichetta 1 la coprono (bordi a parte) | tutte le righe |
| KnowLine letto due volte: `doc_tok` cade nella seconda copia | sempre |
| Senza preambolo, `input_ids` identici al fine-tuning binario | sempre |
| P(supportato) dai logit salvati = verdetto scelto al passo 5 | differenza < 0,005 |
