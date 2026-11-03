# Passo 6 — Rendering ed estrazione delle feature

## Obiettivo

1. `evidence/render.py`: da una riga a un esempio con i token del prompt (con preambolo), `answer_pos` e l'**intervallo di token di ogni frase**.
2. `evidence/extract.py`: un forward per riga con gli hook di [01](01_ambiente_e_modelli.md); salva su disco le feature per frase. Su queste feature si addestra la testa nella fase 1a.

La stessa funzione di rendering serve in training, in valutazione e nel server.

## 1. L'esempio renderizzato

```json
{"id": "hover:train:...", "model": "d1-3b",
 "input_ids": [1, 2, "..."], "answer_pos": 1287,
 "doc_char_offset": 61,
 "sent_tok_spans": [[198, 212], [213, 248], "…", [1201, 1240]],
 "sent_char_spans": [[0, 112], [113, 260], "…"],
 "label": 1, "evidence_target": [1, 0, 0, "…", 1],
 "evidence_term": "supervised", "evidence_weight": 1.0, "omega": 0.3, "truncated": false}
```

- `sent_tok_spans[k] = [primo, ultimo]`: token (estremi inclusi) della frase k nel prompt. Lo stato di fine frase si legge in `ultimo`.
- `sent_char_spans`: intervalli sul documento originale (da `sentences`): li usa il server per evidenziare.
- `answer_pos = len(input_ids) − 1` per i tre modelli.

## 2. Dalle frasi ai token

```text
1. render del prompt a pezzi: [prefisso fino al documento] [documento codificato] [resto]
   (il "documento codificato" è il valore così come appare nel prompt: escapato per d1 e Winnow)
2. mappa carattere per carattere: offset nel documento originale → offset nel prompt
3. tokenizza il prompt con return_offsets_mapping=True
4. sent_tok_spans[k] = [primo token che contiene un carattere della frase k, ultimo token che ne contiene uno]
```

La mappa del punto 2 si costruisce mentre si fa l'escaping, non si stima:

```python
ESC = {'"': '\\"', "\\": "\\\\", "\n": "\\n", "\r": "\\r", "\t": "\\t", "\b": "\\b", "\f": "\\f"}   # come json.dumps

def escape_with_map(text, escape_lt=False):
    """Il testo come appare dentro una stringa JSON e, per ogni carattere originale (più la fine), il suo offset nel testo escapato."""
    out, pos, at = [], [], 0
    for ch in text:
        pos.append(at)
        if ch in ESC:
            piece = ESC[ch]
        elif escape_lt and ch == "<":
            piece = "\\u003c"
        elif ord(ch) < 0x20:
            piece = f"\\u{ord(ch):04x}"
        else:
            piece = ch
        out.append(piece)
        at += len(piece)
    pos.append(at)
    return "".join(out), pos
```

Prima di usarla: per 1.000 documenti, `escape_with_map(doc)[0]` deve essere identico alla sottostringa che `json.dumps` (d1) o `safe_data` (Winnow) producono per il documento. Se `json.dumps` escapa altri caratteri (es. ` `), la tabella si allinea a quello che fa davvero.

Casi da gestire:

| Caso | Regola |
|---|---|
| Token a cavallo di due frasi (es. `."` in un token) | appartiene alla frase che contiene il suo ultimo carattere non spazio |
| Token che contiene solo spazi fra due frasi | non appartiene a nessuna frase |
| Frase senza token propri (rarissimo: frase di un carattere fusa in un token) | si unisce alla frase precedente nel rendering e si segnala |
| Winnow: prefisso e suffisso tokenizzati separatamente | gli offset si calcolano sui pezzi tokenizzati, poi si sommano le lunghezze |
| JPT: righe `campo: valore` | il documento è testo semplice nel prompt; si mappa solo lo spostamento del prefisso |

## 3. Documenti lunghi

| Uso | Lunghezza massima del prompt |
|---|---|
| Estrazione (fase 1a) | d1: 16.384 token; JPT e Winnow: 8.192, 16.384 se la memoria lo consente |
| Fasi 1b e 2 | 8.192 token |
| Server | 32.768 token |

Oltre il limite, per le righe di training:
- positivi con evidenza: si tolgono frasi **non** di evidenza partendo da quelle più lontane da ogni frase di evidenza, finché si rientra; le etichette restano valide; `truncated = true`;
- negativi: si toglie la coda del documento.

Le righe di dev e test non si tagliano: se superano il limite del server si escludono e si contano.

## 4. Le feature salvate (fase 1a)

Un forward senza gradiente per riga, con gli hook di stati e attenzione ([01](01_ambiente_e_modelli.md) §4–5):

| Feature | Forma | Tipo | Contenuto |
|---|---|---|---|
| `q` | [L, D] | bf16 | stato in `answer_pos` per i L layer del mix |
| `h_end` | [n, L, D] | bf16 | stato nell'ultimo token di ogni frase |
| `h_mean` | [n, L, D] | bf16 | media degli stati dei token della frase |
| `att` | [n, 8, H] | fp16 | per layer pieno e testa: somma dei pesi di attenzione di `answer_pos` sui token della frase |
| `n_tok` | [n] | int16 | numero di token della frase |
| `answer_logits` | [2] | fp32 | logit (o log-probabilità di gruppo per d1) di sì e no: per ricalcolare P(supportato) |
| `att_pair` (facoltativo) | [n, n, 8] | fp16 | massa di attenzione del token di fine frase k sui token della frase j (j < k), media sulle teste; solo se si fa H-PAIR ([11](11_esperimenti.md)) |

`att_pair` richiede le righe di attenzione di tutti i token di fine frase, non solo di `answer_pos`: lo stesso hook di [01](01_ambiente_e_modelli.md) §5 con le query in `sent_tok_spans[:, 1]` invece dell'ultima posizione. Si estrae in un secondo passaggio, solo se serve.

Spazio stimato per 26k righe, ~60 frasi, 4 layer (`h_end` + `h_mean`): ~50 GB per d1, ~100 GB per JPT e Winnow. Se lo spazio non basta, per JPT e Winnow si estraggono solo i layer che hanno avuto peso nel mix di d1 (risultato dell'esperimento H2a).

Formato: `data/evidence/features/<modello>/<mix o blocco>/shard-00000.safetensors` (1.000 righe per file, tensori concatenati lungo le frasi) + `index.jsonl` (`id`, shard, offset della prima frase, n). Il loader legge una riga come fetta dei tensori del suo shard.

## 5. Estrazione

```sh
gpu-run --batch --name ev-extract-d1 -- bash -c \
  'python -m evidence.extract --model d1-3b --rows data/evidence/mixes/ev_head.jsonl \
     --rows data/evidence/splits/dev_evid_select data/evidence/splits/dev_evid_confirm data/evidence/splits/dev_evid_neg \
     --out data/evidence/features/d1-3b > logs/ev-extract-d1.out 2>&1'
```

- Riprendibile: salta gli shard completi.
- Ordina le righe per lunghezza prima di fare i batch (meno padding); batch 1 sopra i 4.096 token.
- Scrive `meta.json`: modello, revisione, hash dei pesi del verdetto, testo del preambolo, layer del mix, layer pieni, hash di `sentences.py` e `render.py`. Il training rifiuta feature con un `meta.json` diverso da quello atteso.

## 6. Test

| Test | Come | Soglia |
|---|---|---|
| Ricopertura | `sent_char_spans` coprono il documento senza sovrapposizioni | tutte le righe |
| Posizioni | 200 righe per modello: il testo decodificato da `sent_tok_spans[k]`, de-escapato, coincide con la frase k (a meno di spazi ai bordi) | 200 su 200 |
| Escaping | documenti sintetici con `"`, `\`, a capo, tab, `<`, emoji, cinese, arabo | posizioni corrette su d1 e Winnow |
| Identità del prompt | senza preambolo, `input_ids` = piano binario | identici |
| Verdetto | P(supportato) da `answer_logits` salvati = P del piano binario col preambolo (passo 5) | differenza < 0,005 |
| Feature riproducibili | due estrazioni della stessa riga | identiche |
| Coerenza con il server | `serve.py` e `extract.py` sulla stessa riga | differenza < 1e-3 sulle feature |

## Prodotto

- `evidence/render.py`, `evidence/extract.py` con i test di §6.
- Feature di d1-3B per `ev_head` e per i blocchi dev; statistiche: frasi per documento (p50/p95), token per documento per tokenizer, righe tagliate per fonte, spazio occupato.
