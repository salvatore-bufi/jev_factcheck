# Passo 6 — Dalle frasi ai token, e salvataggio dei vettori

## In breve

1. **Rendering** (`evidence/render.py`): per ogni riga si costruisce il prompt con il preambolo (codice degli autori) e si trova, per ogni frase del documento, **quali token del prompt le appartengono** (primo e ultimo). È il dato da cui dipende tutto: se le posizioni sono sbagliate anche di un token, la testa legge i vettori sbagliati.
2. **Estrazione** (`evidence/extract.py`): un forward per riga con gli hook del passo [01](01_ambiente_e_modelli.md); si salvano su disco i vettori di ogni frase. Il modello è congelato: si calcolano una volta e la testa si addestra su di essi in pochi minuti.

## 1. L'esempio renderizzato

```json
{"id": "hover:train:…", "model": "knowline-4b",
 "input_ids": [248045, 846, "…"], "answer_pos": 2371,
 "sent_tok_spans": [[1103, 1117], [1118, 1150], "…"],
 "sent_char_spans": [[0, 61], [62, 140], "…"],
 "label": 1, "gold": [0, 1, 0, "…"],
 "evidence_term": "supervised", "row_weight": 1.0, "omega": 0.5}
```

| Campo | Significato |
|---|---|
| `sent_tok_spans[k]` | primo e ultimo token della frase k nel prompt (KnowLine letto due volte: nella seconda copia) |
| `sent_char_spans[k]` | inizio e fine della frase k nel documento originale (le `sentences` della riga): servono al server per evidenziare |
| `answer_pos` | la posizione di risposta |
| `gold[k]` | 1 se la frase k è di evidenza (unione degli insiemi alternativi) |

## 2. Il problema: il documento nel prompt non è scritto come l'originale

Per i modelli con lo stato in JSON (elephant-1, StartLux, d3-flash, Jeffan, Nimble, pplx, Clef) virgolette e a capo diventano sequenze di due caratteri (`"` → `\"`, a capo → `\n`). Per KnowLine lo stato è testo indentato. Le posizioni delle frasi nel testo originale non coincidono con quelle nel prompt: serve una mappa.

## 3. Dai token alle frasi

Procedura:
1. costruire il prompt in tre pezzi: prima del documento, documento come appare nel prompt, dopo il documento (si conosce così `doc_start`, dove inizia il documento nel prompt);
2. per il pezzo del documento, `escape_with_map` restituisce il testo trasformato e, per ogni suo carattere, l'indice del carattere originale (`origin`);
3. tokenizzare il prompt con `return_offsets_mapping=True`;
4. `token_char_spans`: per ogni token del documento, i caratteri originali che copre (`tok_char`);
5. `sentence_token_spans`: ogni token va alla frase che contiene il suo ultimo carattere non spazio; per ogni frase, primo e ultimo token.

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


import bisect


def sentence_token_spans(tok_char, sentences, text):
    """tok_char: [(inizio, fine)] in caratteri originali dei token del documento (in ordine);
    sentences: [(inizio, fine)] delle frasi (evidence.sentences.split). Un token appartiene alla frase che contiene
    il suo ultimo carattere non spazio. -> [(primo token, ultimo token)] per frase, indici relativi al documento."""
    starts = [s for s, _ in sentences]
    first, last = [None] * len(sentences), [None] * len(sentences)
    for i, (a, b) in enumerate(tok_char):
        c = b - 1
        while c >= a and text[c].isspace():
            c -= 1
        if c < a:
            continue                                        # token di soli spazi: nessuna frase
        k = bisect.bisect_right(starts, c) - 1
        if k < 0 or c >= sentences[k][1]:
            continue                                        # carattere fra due frasi
        first[k] = i if first[k] is None else first[k]
        last[k] = i
    missing = [k for k, f in enumerate(first) if f is None]
    if missing:
        raise ValueError(f"frasi senza token propri: {missing}")
    return list(zip(first, last))
```

Gli indici restituiti sono relativi ai token del documento: si sommano all'indice del primo token del documento nel prompt per avere `sent_tok_spans`.

Provato su un documento con virgolette e a capo dentro un JSON (`Rose "Byrne" is an\nAustralian actress. She starred in films.`): il testo trasformato è identico a `json.dumps`, `\"` torna a `"`, `\n` all'a capo, e le due frasi ricevono i token 0–9 e 10–14.

Casi da gestire:
- **KnowLine** non usa JSON: il documento compare quasi così com'è (con l'indentazione delle righe). `escape_with_map` si sostituisce con la trasformazione del suo `knowline_layout`, registrando allo stesso modo `origin`. Nelle richieste lette due volte si prende la seconda copia del documento.
- **Prefisso e domanda tokenizzati separatamente** (elephant-1, OneJev, Nimble): gli offset si calcolano sui pezzi e si sommano.
- **Token a cavallo di due frasi** (es. `."` in un solo token): va alla frase che contiene il suo ultimo carattere non spazio.
- **Frase senza token propri** (rarissimo: una frase di un carattere fusa in un token): la funzione la segnala; si unisce alla precedente nel rendering.

## 4. Documenti lunghi

Il filtro di lunghezza del build ([02](02_dati_fonti.md) §3) ha già tolto le righe oltre il contesto. Limiti per fase:

| Uso | Lunghezza massima del prompt |
|---|---|
| Estrazione (fase 1a) | modelli 4B 16.384 token; modelli 9B 8.192 (16.384 se la memoria basta) |
| Training con il modello in memoria (fasi 1b, 2) | 8.192 token |
| Server | 32.768 token |

Righe di training oltre il limite: per i positivi si tolgono frasi **non** di evidenza, partendo da quelle più lontane dall'evidenza (le etichette restano valide); per i negativi si taglia la fine. Le righe di dev e test non si tagliano: se sono troppo lunghe si escludono e si contano.

## 5. Che cosa si salva (fase 1a)

| Feature | Forma | Tipo |
|---|---|---|
| `q` | 4 × D | fp32 |
| `h_end` | n × 4 × D | bf16 |
| `h_mean` | n × 4 × D | bf16 |
| `att` | n × 128 | fp16 |
| `n_tok` | n | int16 |
| `answer_logits` | 2 | fp32 |

Spazio: 82 KB per frase su KnowLine-4B (4 layer); ~120 GB per le ~28.600 righe delle tre fonti. Se lo spazio non basta, si salvano solo i layer con peso dopo il primo training (o i migliori secondo la sonda, [15](15_recap_architettura_layer_loss.md) §3).

Formato: `data/evidence/features/<modello>/<set>/shard-00000.safetensors` (1.000 righe per file, tensori concatenati lungo le frasi) più `index.jsonl` (`id`, shard, offset della prima frase, n, `gold`, `evidence_term`, pesi). Ogni cartella ha un `meta.json` con modello, revisione dei pesi, preambolo, layer, hash di `sentences.py`, `render.py`, `escape.py`: il training rifiuta vettori calcolati con impostazioni diverse.

## 6. Comando

```sh
gpu-run --batch --name ev-extract-knowline -- bash -c \
  'python -m evidence.extract --model knowline-4b \
     --rows dataset/evidence/mixes/ev_head.jsonl dataset/evidence/splits/dev_evid_select.jsonl \
            dataset/evidence/splits/dev_evid_confirm.jsonl dataset/evidence/splits/dev_evid_neg.jsonl \
     --out data/evidence/features/knowline-4b > logs/ev-extract-knowline.out 2>&1'
```

Riprende dagli shard completi; ordina le righe per lunghezza (meno riempimento); batch 1 sopra i 4.096 token.

## 7. Test

| Test | Deve dare |
|---|---|
| Il testo dei token di ogni frase, decodificato e de-escapato, è la frase (spazi ai bordi a parte) | 200 righe su 200, per ogni modello |
| Documenti con `"`, `\`, a capo, tab, `<`, emoji, testo non latino | posizioni giuste |
| Senza preambolo, `input_ids` identici al fine-tuning binario | sempre |
| KnowLine letto due volte: le frasi cadono nella seconda copia | sempre |
| P(supportato) dai logit salvati = verdetto scelto al passo 5 | differenza < 0,005 |
| Stessa riga estratta due volte | vettori identici |
