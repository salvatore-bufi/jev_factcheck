# Passo 3 — Etichette generate, ablazione offline, righe sparse

## Obiettivo

Aggiungere righe con evidenza nota fuori da Wikipedia e righe con evidenza lontana. Tre script:

| Script | Produce | `evidence_quality` |
|---|---|---|
| `evidence/gen_marked.py` | documenti C2D e CG2C in cui si sa quali frasi stabiliscono il claim | `construct` |
| `evidence/label_ablation.py` | evidenza per righe supportate che non ne hanno (C2D/D2C rilasciati, HoVer non riallineate) | `ablation` |
| `evidence/augment_sparse.py` | versioni delle righe con evidenza spostata lontano o separata da distrattori | come la riga di origine |

Tutto si genera **solo dalle righe di train** ([04](04_split_e_mix.md) §1 per gli split). Generatore e giudice LLM sono quelli del piano binario ([../fine_tuning_factcheck/02b_dataset_generazione_llm.md](../fine_tuning_factcheck/02b_dataset_generazione_llm.md)).

## 1. C2D con posizioni note

Si parte dagli insiemi di frasi atomiche della ricetta C2D del piano binario: per ogni claim, un insieme `s = {s_1, s_2, …}` che insieme lo implica.

Prompt al generatore:

```text
Write a coherent news-style article of 15 to 40 sentences that includes each of the
following sentences word for word, each wrapped in its tag: <e1>…</e1>, <e2>…</e2>, …
Add other context sentences without tags. Do not restate the information of the
tagged sentences anywhere else in the article. Place the tagged sentences far apart.

<e1>{s_1}</e1>
<e2>{s_2}</e2>
…
```

Post-processing:
1. ogni `<ei>…</ei>` presente esattamente una volta e parola per parola, altrimenti la riga si scarta;
2. si tolgono i tag e si annotano gli intervalli di caratteri delle frasi marcate;
3. `sentences.py` sul testo pulito; evidenza = frasi nostre che coprono gli intervalli marcati (mappatura di [02](02_dati_fonti.md) §2); se una frase marcata finisce in più di 2 frasi nostre, la riga si scarta;
4. giudice: "Do the following sentences, taken alone, establish the claim? Answer yes or no." con le sole frasi di evidenza; si tengono le righe con "yes".

Varianti dallo stesso documento:
- **negativo**: si toglie dal documento la frase marcata di un fatto necessario → `label = 0`;
- **sotto-claim supportato**: claim riscritto con un sottoinsieme dei fatti → `label = 1`, evidenza = solo le frasi di quei fatti. Insegna a evidenziare un sottoinsieme preciso.

## 2. CG2C con posizioni note

Nell'estrazione del grafo del piano binario, salvare per ogni tripla l'indice della frase da cui viene. Claim positivo = cammino sul sottografo → evidenza = frasi delle triple del cammino (riportate su `sentences.py`). È la fonte di evidenza multi-hop fuori da Wikipedia. Stesso giudice di §1.

## 3. Ablazione offline con d1-3B

Per le righe supportate con `evidence_quality = none`. Si usa il verdetto **d1-3B** del piano binario nel formato con preambolo ([05](05_formato_prompt.md)).

```python
def ablation_label(row, p_supported, thr, max_k=6):
    """p_supported(sentence_ids) = P(supportato) col documento ridotto a quelle frasi, nell'ordine originale."""
    n, chosen = len(row["sentences"]), []
    while len(chosen) < max_k:
        best = max((k for k in range(n) if k not in chosen),
                   key=lambda k: p_supported(sorted(chosen + [k])))
        chosen.append(best)
        if p_supported(sorted(chosen)) > thr:
            break
    else:
        return None                                        # nessun insieme ≤ 6 frasi basta: riga scartata
    for k in list(chosen):                                 # pulizia
        rest = [j for j in chosen if j != k]
        if rest and p_supported(sorted(rest)) > thr:
            chosen = rest
    complement = [k for k in range(n) if k not in chosen]
    redundant = p_supported(complement) > thr              # l'evidenza è ripetuta altrove nel documento
    return sorted(chosen), ("ablation_partial" if redundant else "ablation")
```

Dettagli:
- il documento ridotto si ricostruisce unendo le frasi scelte con uno spazio (le frasi su righe proprie restano su righe proprie);
- i prompt di un passo greedy si mandano in un solo batch; con transformers si riusa la cache del prefisso comune (preambolo e inizio del turno);
- `thr` = soglia del verdetto d1 del piano binario;
- le righe `ablation_partial` si tengono con peso più basso ([04](04_split_e_mix.md) §3).

Filtri:
1. **giudice** (come §1) sulle frasi scelte; si tengono le righe confermate. Tasso atteso ≥ 80%: sotto, rivedere la procedura prima di continuare;
2. **accordo con JPT-9B** su 300 righe (stessa procedura con il verdetto JPT): Jaccard medio fra gli insiemi ≥ 0,7. Sotto, le etichette sono troppo legate a d1: per JPT e Winnow si usano solo le righe in cui i due insiemi coincidono.

Lancio sul server:

```sh
gpu-run --batch --name ev-ablation-d1 -- bash -c \
  'python -m evidence.label_ablation --model d1-3b --pairs c2d_train d2c_train hover_train_unaligned \
     --out data/evidence/pairs/ablation_d1.jsonl > logs/ev-ablation-d1.out 2>&1'
```

Lo script è riprendibile: salta le righe già scritte nel file di uscita.

## 4. Righe sparse

Da righe con evidenza `gold` o `construct` e almeno 2 frasi di evidenza:

| Trasformazione | Come | Etichette |
|---|---|---|
| **Estremi** | prima frase di evidenza in testa al documento, ultima in coda | stesso verdetto, indici aggiornati |
| **Distrattori** | 5–40 frasi da altri documenti di train della stessa fonte inserite fra le frasi di evidenza | idem |
| **Lontano dal preambolo** | blocco di frasi distrattrici in testa al documento finché la prima frase di evidenza è oltre 1.024 token (fascia media) o oltre 4.096 (fascia lunga), misurato con il tokenizer del modello di destinazione | idem |
| **Negativi sparsi** | le stesse operazioni su negativi FEVER REFUTES e HoVer NOT_SUPPORTED | `label = 0` |

Controlli per ogni riga trasformata:
- una frase spostata che inizia con un riferimento alla precedente (`He`, `She`, `It`, `They`, `This`, `The company`, `The film`, …, o un pronome nelle prime 3 parole) non si sposta da sola: si sposta insieme alla frase precedente, o la riga si scarta;
- giudice sul documento trasformato: "Does the document support the claim?" deve restare uguale all'etichetta;
- i distrattori vengono da documenti di **train** diversi da quello di origine (mai dev/test).

```sh
python -m evidence.augment_sparse --pairs fever_train hover_train construct --n 10000 --seed 0
```

## 5. Volumi

| Famiglia | Pilota | Corsa piena |
|---|---:|---:|
| C2D con posizioni | 2k | 10k |
| CG2C con posizioni | — | 5–10k |
| Ablazione (C2D, D2C, HoVer non riallineate) | 2k | 10–20k |
| Righe sparse (positive) | 4k | 10k |
| Negativi sparsi | 1k | 3k |

## 6. Controlli prima di andare avanti

| Controllo | Atteso |
|---|---|
| 30 righe per famiglia lette a mano: le frasi di evidenza stabiliscono il claim, nessuna frase necessaria manca | ≥ 27 su 30 |
| Conferma del giudice per l'ablazione | ≥ 80% |
| Accordo d1–JPT sull'ablazione | Jaccard ≥ 0,7 |
| Righe sparse con riferimenti rotti (lettura di 30) | 0 |
