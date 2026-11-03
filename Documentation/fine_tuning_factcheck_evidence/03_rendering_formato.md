# Step 3 — Rendering ed estrazione: frasi, posizioni, stati, attenzione

## Obiettivo

Trasformare ogni riga in un esempio con:
- i token del prompt nel formato sandwich ([02d](02d_varianti_domanda.md));
- la posizione `answer_pos` da cui si leggono le lettere (come nel piano binario);
- per ogni frase, l'**intervallo dei suoi token** nel prompt (`sent_tok_spans`): serve per lo stato di fine frase, per la media della frase e per la massa di attenzione;
- i target del verdetto e dell'evidenza.

E, per la fase 1a, salvare su disco le feature lette dal forward ([05](05_training.md) §2).

La stessa funzione deve servire in training e in produzione: una differenza nella segmentazione o nelle posizioni rende la testa inutilizzabile.

## 1. Formato di un esempio renderizzato

```json
{"id": "hover:train:...", "model": "d1-3b",
 "input_ids": ["<token del prompt>"], "answer_pos": 1287,
 "yes_ids": [11683, 12447], "no_ids": [2243, 4547, 19598], "target": [0.0, 1.0],
 "preamble_tok_span": [3, 21],
 "sent_tok_spans": [[198, 212], [213, 248], "…", [1201, 1240]],
 "sent_char_spans": [[0, 112], [113, 260], "…"],
 "label": 1, "evidence_target": [1, 0, 0, "…", 1],
 "evidence_term": "supervised", "evidence_weight": 1.0, "neg_weight": 0.3}
```

- `sent_tok_spans[k] = [primo, ultimo]`: indici (inclusi) dei token della frase k nel prompt. Lo stato di fine frase si legge in `ultimo`.
- `evidence_term`: `supervised` (positivo con evidenza), `mil_neg` (negativo), `mil_pos` (positivo senza evidenza, solo H-MIL), `none`.
- `neg_weight`: peso delle frasi non gold (ω, [04](04_loss.md) §4), per fonte.
- `sent_char_spans`: servono in produzione per evidenziare sul testo originale.

## 2. Segmentazione in frasi

Una sola funzione, `finetune/sentences.py`, usata da builder ([02a](02a_dataset_fonti_esistenti.md)), rendering, valutazione e produzione, identica per i tre modelli.

| Regola | Perché |
|---|---|
| Splitter: NLTK `sent_tokenize` (come MiniCheck/FactCG), poi correzioni | standard, veloce |
| `\n\n` e righe di elenco sono sempre confini | titoli, elenchi, Q/R di AVeriTeC |
| Righe `Title: …` sono una "frase" a sé | documenti FEVER/HoVer |
| Frasi > 60 parole spezzate su `;`, `:` o, in ultima istanza, a metà su una virgola | evidenziare meno testo irrilevante |
| Frasi < 3 parole unite alla successiva | niente evidenze di una parola |
| Massimo 512 frasi per documento | memoria dell'aggregatore; oltre si segnala |

Per le fonti con frasi già date (FEVER `lines`, SciFact `abstract`) si usano quelle; la funzione deve comunque produrre le stesse `sent_char_spans` sul `document` ricostruito.

## 3. Dalle frasi alle posizioni dei token

```text
1. rendering del prompt con il renderer del piano binario + preambolo (02d §2)
2. nel testo del prompt, trova l'intervallo in cui è scritto il documento
3. mappa ogni sent_char_span del documento sull'intervallo del prompt
     (d1, Winnow: il documento è una stringa JSON escapata → mappa offset originale → offset escapato)
4. tokenizza con return_offsets_mapping=True
5. sent_tok_spans[k] = [primo token con un carattere nella frase k, ultimo token con un carattere nella frase k]
```

Dettagli che rompono tutto se sbagliati:

| Dettaglio | Modelli | Cosa fare |
|---|---|---|
| **Escaping JSON** | d1 (`json.dumps(…, ensure_ascii=False, indent=2)`: `"` → `\"`, a capo → `\n`), Winnow (in più `<` → `<`, `safe_data` in [winnow-12b-bf16/adapter.py](../../models/winnow-12b-bf16/adapter.py)) | la mappa degli offset si costruisce carattere per carattere durante l'escaping, non si stima |
| **Token a cavallo** di due frasi (es. `."` unito in un token) | tutti | il token si assegna alla frase che contiene il suo ultimo carattere non spazio |
| **Tokenizzazione separata** di prefisso e suffisso | Winnow | gli offset vanno ricalcolati sui pezzi effettivamente tokenizzati |
| **Preambolo** | tutti | sposta gli offset di una quantità fissa: calcolata dal rendering |
| **Indentazione di d1** | d1 | il documento è il valore di una chiave del JSON indentato: la prima frase inizia dopo `"document": "` |

## 4. Le feature da salvare (fase 1a)

Un forward del modello del verdetto, senza gradiente, con hook ([00c](00c_modelli_e_architetture.md) §2 per i layer):

| Feature | Forma per riga | Come |
|---|---|---|
| `q` | [L_mix, d] | stato in `answer_pos` per ciascun layer del mix |
| `h_end` | [n, L_mix, d] | stato nell'ultimo token di ogni frase |
| `h_mean` | [n, L_mix, d] | media degli stati dei token di ogni frase |
| `att` | [n, 8, H] | per ogni layer pieno e testa: somma delle probabilità di attenzione di `answer_pos` sui token della frase |
| `att_len` | [n] | numero di token della frase (la massa cresce con la lunghezza: la testa deve poterla normalizzare) |
| `att_pair` (facoltativo, H-PAIR) | [n, n, 8] | massa di attenzione del token di fine frase k sui token della frase j (j < k), media sulle teste |

Calcolo della riga di attenzione, senza implementazione "eager":

```python
def attn_row_hook(layer_idx, store):
    """Sul modulo di attenzione di un layer pieno: salva la query dell'ultima posizione e le chiavi (dopo RoPE)."""
    def hook(module, args, kwargs, output):
        q_last, k_all = module._last_q, module._last_k      # [H, d_h], [H_kv, T, d_h], esposti con un piccolo patch
        k = k_all.repeat_interleave(q_last.size(0) // k_all.size(0), dim=0)
        scores = (q_last[:, None, :] @ k.transpose(1, 2)).squeeze(1) * module.scaling   # [H, T]
        store[layer_idx] = scores.softmax(-1)              # una riga per testa: costo trascurabile
    return hook
```

Il patch che espone `_last_q` e `_last_k` dipende dalla classe di attenzione di ciascun modello (LFM2, Qwen3.5, Gemma 4): va scritto tre volte, e verificato contro l'implementazione "eager" ([00c](00c_modelli_e_architetture.md), criteri). Su Gemma 4 lo `scaling` e l'eventuale softcap dei logit di attenzione vanno presi dal modulo, non assunti.

Il pooling per frase (`h_mean`, `att`) si fa sulla GPU subito, nel forward: su disco vanno solo n vettori per riga, mai gli stati di tutti i token.

Formato su disco: un file `safetensors` per blocco di 1.000 righe, con un indice (`id` → offset). bf16 per gli stati, fp16 per le attenzioni.

## 5. Documenti lunghi

Niente chunking ([00a](00a_scelta_approccio.md) §5): il documento passa intero.

| Fase | Budget | Cosa fare oltre il budget |
|---|---|---|
| Fase 1a (estrazione) | 16.384 token per d1; 8.192–16.384 per JPT e Winnow secondo la memoria | righe positive con evidenza: si tagliano frasi **non evidenza** lontane dall'evidenza finché si rientra (le etichette restano valide); negative: si taglia la coda; contare i casi |
| Fasi 1b, 2 | 8.192 token (16k se la memoria lo permette con batch 1) | come sopra |
| Produzione | 32.768 token | oltre: fuori perimetro, si segnala |

## 6. Test obbligatori

| Test | Cosa verifica |
|---|---|
| Ricopertura | le `sent_char_spans` coprono il documento senza buchi (a parte spazi) né sovrapposizioni |
| Posizioni | per 200 righe per modello, il testo decodificato da `sent_tok_spans[k]` è la frase k (a meno dell'escaping) |
| Escaping | documenti con virgolette, a capo, `<`, emoji, testo non latino: posizioni corrette su d1 e Winnow |
| Identità del prompt | togliendo il preambolo, prompt identico token per token a quello del piano binario |
| Attenzione | la riga ricalcolata coincide con l'implementazione "eager" entro 1e-3 |
| Coerenza training/produzione | la funzione di produzione e quella di estrazione danno le stesse feature sugli stessi input (differenza < 1e-3) |

## Criteri per passare allo step successivo

- Test di §6 superati per i tre modelli.
- Statistiche: frasi per documento (p50/p95), token per documento per tokenizer, righe tagliate per fonte, spazio su disco delle feature.
