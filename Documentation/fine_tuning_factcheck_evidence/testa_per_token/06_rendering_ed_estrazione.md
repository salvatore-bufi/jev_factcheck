# Passo 6 — Rendering, etichette per token, estrazione dei vettori

## In breve

1. **Rendering** (`evidence/token_render.py`): per ogni riga si costruisce il prompt con il preambolo (codice degli autori) e si calcola, per **ogni token del documento**, a quali caratteri del documento originale corrisponde. Da qui l'etichetta del token (dentro o fuori da una frase gold) e, in produzione, la conversione dei tratti in caratteri.
2. **Estrazione** (`evidence/token_extract.py`): un forward per riga con gli hook del passo 1; si salvano su disco i vettori ridotti di ogni token, la sua attenzione, `q`. Il modello è congelato: si calcolano una volta.

## 1. L'esempio renderizzato

```json
{"id": "hover:train:…", "model": "knowline-4b",
 "input_ids": [248045, 846, "…"], "answer_pos": 2371,
 "doc_tok": [1103, 2290],
 "tok_char": [[0, 7], [7, 11], [11, 19], "…"],
 "tok_label": [0, 0, 1, 1, 1, "…"],
 "gold_tok_spans": [[[37, 61], [612, 640]]],
 "label": 1, "evidence_term": "supervised", "row_weight": 1.0, "omega": 0.5}
```

| Campo | Significato |
|---|---|
| `doc_tok` | primo e ultimo token del documento nel prompt (KnowLine letto due volte: della seconda copia) |
| `tok_char[i]` | caratteri `[inizio, fine)` del documento **originale** coperti dal token `doc_tok[0] + i` |
| `tok_label[i]` | 1 se il token sta dentro una frase gold (§3) |
| `gold_tok_spans` | per ogni insieme alternativo, gli intervalli di token di ogni frase gold (servono alla loss per frase gold, [08](08_testa_e_loss.md) §3) |

## 2. Dai caratteri del documento ai token

Il documento nel prompt può essere scritto diversamente dall'originale: dentro un JSON (`"` → `\"`, a capo → `\n`) per elephant-1, StartLux, d3-flash, Jeffan, Nimble, pplx e Clef; come testo indentato per KnowLine. La procedura del piano per frase ([../testa_per_frase/06_rendering_ed_estrazione.md](../testa_per_frase/06_rendering_ed_estrazione.md) §2) va qui al contrario: serve, per ogni token, la sua posizione nel testo **originale**.

1. Costruire il prompt in tre pezzi (prima del documento, documento come appare nel prompt, dopo); per il pezzo del documento, `escape_with_map` dà per ogni carattere originale la sua posizione nel testo trasformato.
2. Invertire la mappa: per ogni posizione del testo trasformato, il carattere originale da cui viene (un `\"` di due caratteri torna a un solo `"`).
3. Tokenizzare il prompt con `return_offsets_mapping=True`.
4. Per ogni token nel pezzo del documento: `tok_char` = primo e ultimo carattere originale che copre. I token fatti solo di escape o spazi ereditano l'intervallo vuoto `[c, c]`.

`escape_with_map` è già scritta e verificata (identica a `json.dumps` su 20.000 stringhe difficili): si sposta in `evidence/escape.py` e si usa sia qui sia nel piano per frase.

## 3. L'etichetta di un token

```text
tok_label[i] = 1  se almeno metà dei caratteri non spazio del token sta dentro un intervallo di gold_char_sets
             = 0  altrimenti (e sempre 0 se la riga ha label 0)
```

Esempio: la frase gold "Rose Byrne is an Australian actress." diventa ~8 token tutti con etichetta 1; i token di "Rose Byrne" come titolo della pagina, fuori dalla frase, restano 0.

`gold_tok_spans`: per ogni frase gold, il primo e l'ultimo token con etichetta 1 al suo interno.

Rumore da sapere: in una frase gold anche le parole irrilevanti hanno etichetta 1 (le fonti annotano frasi, non parole). La loss ne tiene conto ([08](08_testa_e_loss.md) §3): chiede soprattutto che **ogni frase gold sia colpita** e che **fuori dalle frasi gold non si accenda nulla**, non che ogni token di una frase gold sia sopra soglia.

## 4. Documenti lunghi

Il filtro di lunghezza del build ([02](02_dati_fonti.md)) ha già tolto le righe oltre il contesto. Per la fase 1b (modello in memoria) vale il budget di 8.192 token: le righe oltre si escludono da quella fase (oggi 75 righe di WiCE, nessuna di HoVer e SciFact).

## 5. Che cosa si salva (fase 1a)

| Feature | Forma | Tipo |
|---|---|---|
| `h` | n_token × L × 256 | fp16, proiettato con la PCA del passo 1 |
| `att` | n_token × 128 | fp16 |
| `q` | L × d | bf16 |
| `tok_char`, `tok_label` | n_token × 2, n_token | int32, int8 |
| `answer_logits` | 2 | fp32 |

Formato: `data/evidence_tok/features/<modello>/<set>/shard-00000.safetensors`, 500 righe per file (le righe hanno lunghezze molto diverse), più `index.jsonl` con `id`, shard, offset del primo token e n. `meta.json` con modello, pesi, preambolo, layer, file PCA, hash di `token_render.py`: il training rifiuta vettori calcolati con impostazioni diverse.

## 6. Comando

```sh
gpu-run --batch --name evtok-extract-knowline -- bash -c \
  'python -m evidence.token_extract --model knowline-4b \
     --rows dataset/evidence/mixes/ev_tok_head.jsonl dataset/evidence/splits/dev_evid_select.jsonl \
            dataset/evidence/splits/dev_evid_confirm.jsonl dataset/evidence/splits/dev_evid_neg.jsonl \
     --out data/evidence_tok/features/knowline-4b > logs/evtok-extract-knowline.out 2>&1'
```

Riprende dagli shard completi; ordina le righe per lunghezza; batch 1 sopra i 4.096 token.

## 7. Test

| Test | Deve dare |
|---|---|
| Testo ricostruito dai `tok_char` dei token del documento = documento originale (a meno degli spazi) | 200 righe su 200, per ogni modello |
| Documenti con `"`, `\`, a capo, tab, `<`, emoji, testo non latino | `tok_char` giusti |
| Per ogni frase gold, i token con etichetta 1 coprono la frase (a meno dei bordi) | tutte le righe |
| KnowLine letto due volte: `doc_tok` cade nella seconda copia | sempre |
| Senza preambolo, `input_ids` = piano binario | identici |
| P(supportato) dai logit salvati = verdetto scelto al passo 5 | differenza < 0,005 |
