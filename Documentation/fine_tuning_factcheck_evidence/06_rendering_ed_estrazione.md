# Passo 6 — Dalle frasi ai token, e salvataggio dei vettori

## In breve

1. **Rendering** (`evidence/render.py`): per ogni riga si costruisce il prompt con il preambolo e si trova, per ogni frase, **quali token del prompt le appartengono** (primo e ultimo). È il dato da cui dipende tutto: se le posizioni sono sbagliate anche di un token, la testa legge i vettori sbagliati.
2. **Estrazione** (`evidence/extract.py`): un forward per riga con gli hook del passo [01](01_ambiente_e_modelli.md); si salvano su disco i vettori di ogni frase. Il modello è congelato, quindi questi vettori non cambiano mai: si calcolano una volta e la testa si addestra su di essi in pochi minuti.

## 1. L'esempio pronto per il modello

```json
{"id": "hover:train:...", "model": "d1-3b",
 "input_ids": [1, 2, "..."], "answer_pos": 1287,
 "sent_tok_spans": [[198, 212], [213, 248], "…", [1201, 1240]],
 "sent_char_spans": [[0, 112], [113, 260], "…"],
 "label": 1, "evidence_target": [1, 0, 0, "…", 1],
 "evidence_term": "supervised", "evidence_weight": 1.0, "omega": 0.3}
```

- `sent_tok_spans[k]`: primo e ultimo token della frase k nel prompt.
- `sent_char_spans[k]`: inizio e fine della frase k nel documento originale (servono al server per evidenziare).
- `answer_pos`: l'ultimo token del prompt.
- `evidence_target`: 1 per le frasi di evidenza, 0 per le altre.

## 2. Trovare i token di ogni frase

Il problema: in d1-3B e Winnow il documento è dentro un JSON, quindi nel prompt non è scritto come nell'originale. Virgolette, a capo e (per Winnow) `<` diventano sequenze più lunghe: `"` → `\"`, a capo → `\n`, `<` → `<`. Le posizioni delle frasi nel testo originale non coincidono con quelle nel prompt.

Procedura:
1. costruire il prompt in tre pezzi: prima del documento, documento come appare nel prompt, dopo il documento;
2. mentre si trasforma il documento, registrare dove finisce ogni carattere originale (funzione sotto);
3. tokenizzare il prompt chiedendo al tokenizer la posizione in caratteri di ogni token (`return_offsets_mapping=True`);
4. per ogni frase, prendere il primo e l'ultimo token che contengono suoi caratteri.

```python
ESC = {'"': '\\"', "\\": "\\\\", "\n": "\\n", "\r": "\\r", "\t": "\\t", "\b": "\\b", "\f": "\\f"}   # come json.dumps

def escape_with_map(text, escape_lt=False):
    """Il testo come appare in una stringa JSON e, per ogni carattere originale (più la fine), la sua posizione nel testo trasformato."""
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

Verificata contro `json.dumps(…, ensure_ascii=False)` su 20.000 stringhe con caratteri difficili: identica. Con `escape_lt=True` riproduce `safe_data` di Winnow.

Casi particolari:
- un token a cavallo di due frasi (es. `."`) va alla frase che contiene il suo ultimo carattere non spazio;
- Winnow tokenizza separatamente la parte prima e dopo la domanda: le posizioni si calcolano sui pezzi e si sommano;
- JPT: il documento è scritto così com'è, basta lo spostamento dovuto al testo che lo precede.

## 3. Documenti troppo lunghi

| Uso | Lunghezza massima del prompt |
|---|---|
| Estrazione | d1-3B 16.384 token; JPT e Winnow 8.192 (16.384 se la memoria basta) |
| Training con il modello in memoria | 8.192 |
| Server | 32.768 |

Righe di training oltre il limite: per i positivi si tolgono frasi **non** di evidenza, a partire dalle più lontane dall'evidenza (le etichette restano valide); per i negativi si taglia la fine. Le righe di dev e test non si tagliano: se sono troppo lunghe si escludono e si contano.

## 4. Che cosa si salva

Per ogni riga, dal forward:

| Nome | Forma | Che cos'è |
|---|---|---|
| `q` | 4 layer × d | vettore della posizione di risposta |
| `h_end` | n frasi × 4 layer × d | vettore dell'ultimo token di ogni frase |
| `h_mean` | n frasi × 4 layer × d | media dei vettori dei token di ogni frase |
| `att` | n frasi × 8 layer × teste | quanta attenzione la posizione di risposta dà ai token di ogni frase |
| `n_tok` | n frasi | quanti token ha ogni frase |
| `answer_logits` | 2 | punteggi di "sì" e "no", per ricalcolare P(supportato) |

Spazio per 26.000 righe con ~60 frasi: ~50 GB per d1-3B, ~100 GB per JPT e Winnow. Se lo spazio non basta, per JPT e Winnow si salvano solo i layer che su d1 sono risultati utili (esperimento H2a).

Formato: `data/evidence/features/<modello>/<set>/shard-00000.safetensors` (1.000 righe per file) più un indice `index.jsonl`. Ogni cartella ha un `meta.json` con modello, pesi, preambolo, layer e versione del codice di divisione in frasi e di rendering: il training rifiuta vettori calcolati con impostazioni diverse.

## 5. Comando

```sh
gpu-run --batch --name ev-extract-d1 -- bash -c \
  'python -m evidence.extract --model d1-3b \
     --rows data/evidence/mixes/ev_head.jsonl data/evidence/splits/dev_evid_select data/evidence/splits/dev_evid_confirm data/evidence/splits/dev_evid_neg \
     --out data/evidence/features/d1-3b > logs/ev-extract-d1.out 2>&1'
```

Se si interrompe, riparte dai file già completati.

## 6. Test

| Test | Deve dare |
|---|---|
| I token di ogni frase, ritrasformati in testo, sono la frase (200 righe per modello) | 200 su 200 |
| Documenti con `"`, `\`, a capo, tab, `<`, emoji, cinese, arabo | posizioni giuste su d1 e Winnow |
| Senza preambolo, il prompt è identico a quello del piano binario | identico |
| P(supportato) dai punteggi salvati = P del verdetto scelto al passo 5 | differenza < 0,005 |
| Stessa riga estratta due volte | vettori identici |
