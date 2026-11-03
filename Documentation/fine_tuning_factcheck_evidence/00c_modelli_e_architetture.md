# Step 0c — I tre modelli: cosa si legge, cosa si può mascherare, come si servono

## Obiettivo

Fissare, per ciascun modello del piano, i fatti di architettura da cui dipendono rendering, estrazione degli stati, testa e server. I numeri vengono dai `config.json` delle revisioni usate nei benchmark (d1-3B `051bcc4`, JPT-9B `b447cc7`, Winnow-12B convertito da `gguf/Winnow-12B-BF16.gguf`).

## 1. Architetture

| | d1-3B | JPT-9B | Winnow-12B |
|---|---|---|---|
| Base | LFM2.5-VL-3B (`lfm2`) | Qwen3.5-9B (`qwen3_5_text`), LoRA fusa | Gemma 4 12B (`gemma4_text`) |
| Layer | 30 | 32 | 48 |
| Tipi di layer | 22 convoluzioni corte (cache di 3) + 8 attenzione piena | 24 Gated DeltaNet + 8 attenzione piena (1 ogni 4) | 40 a finestra scorrevole (1.024) + 8 globali (1 ogni 6) |
| Layer di attenzione piena (0-indicizzati) | 2, 5, 9, 13, 17, 21, 24, 27 | 3, 7, 11, 15, 19, 23, 27, 31 | 5, 11, 17, 23, 29, 35, 41, 47 |
| Teste di query (layer pieni) | 32 (8 KV) | 16 (4 KV), head_dim 256 | 16 (1 KV globale), head_dim globale 512 |
| d (hidden) | 2.048 | 4.096 | 3.840 |
| Contesto usato | 32.768 | 32.768 (`max_model_len`) | 32.768 |
| Risposta letta | parole sì/no (gruppi di token) | lettere A/B | lettere A/B |
| Posizione di lettura (`answer_pos`) | a capo dopo `assistant` | `:` di `Answer:` | a capo dopo `Answer:` |
| Licenza | LFM Open 1.0 | CC BY-NC 4.0 | Apache 2.0 |

Le posizioni di lettura e i token delle risposte sono quelli del piano binario ([../fine_tuning_factcheck/03_rendering_formato.md](../fine_tuning_factcheck/03_rendering_formato.md)).

## 2. Cosa si legge per la testa

| Feature | Da dove | d1-3B | JPT-9B | Winnow-12B |
|---|---|---|---|---|
| Stati per il mix di layer (fine frase, media frase, `q`) | uscita del layer, dopo gli ultimi globali e alla fine | 21, 24, 27, 29 | 23, 27, 31 + 19 | 35, 41, 47 + 29 |
| Attenzione di `Answer:` sulle frasi | gli 8 layer pieni, per testa | 8 × 32 = 256 | 8 × 16 = 128 | 8 × 16 = 128 |
| Attenzione fra fine frase (prior per le coppie, facoltativo) | gli 8 layer pieni, mediata sulle teste | n × n × 8 | n × n × 8 | n × n × 8 |

Per d1-3B l'ultimo layer pieno è il 27, seguito da due convoluzioni: lo stato del layer 29 è l'uscita finale. Indici esatti e numero di layer del mix si scelgono in H2a ([07](07_esperimenti.md)); quelli sopra sono il punto di partenza.

Spazio su disco per la fase 1a (20k righe, ~60 frasi, bf16, per layer e per tipo di stato): d1 ~4,9 GB; JPT ~9,8 GB; Winnow ~9,2 GB. Con 4 layer × (fine frase + media) si moltiplica per 8: ~40 GB per d1, ~80 GB per JPT e Winnow. Se non c'è spazio, per JPT e Winnow si riducono i layer a 2 dopo H2a su d1.

## 3. Come viaggia il claim (e perché conta)

| Modello | Una frase a ~3.000 token dal preambolo vede il claim tramite |
|---|---|
| d1-3B | 8 layer di attenzione; le convoluzioni sono locali |
| JPT-9B | 8 layer di attenzione + stato ricorrente dei 24 DeltaNet (che non cancella del tutto, ma decade) |
| Winnow-12B | 8 layer globali; i 40 a finestra vedono solo 1.024 token |

Conseguenza: nei documenti lunghi la parte "condizionata sul claim" di `h_k` è portata da pochi layer, e i layer intermedi del mix possono non averla. Si misura con l'F1 per distanza in token dal preambolo ([06](06_valutazione.md) §2).

## 4. Cosa si può mascherare per frase (H-FAITH, [00a](00a_scelta_approccio.md) §8)

| Modello | Maschera con bias sulle chiavi | Fuga di informazione |
|---|---|---|
| Winnow-12B | in tutti i 48 layer (sono tutti di attenzione) | nessuna |
| d1-3B | negli 8 layer di attenzione | convoluzioni ai confini fra frasi (3 token) |
| JPT-9B | negli 8 layer pieni | i 24 layer DeltaNet: andrebbero modulati scalando chiavi e valori della frase dentro la ricorrenza; non previsto |

H-FAITH si fa su d1-3B (economico), poi eventualmente su Winnow (esatto). Su JPT no.

## 5. Implementazione: training ed estrazione

| | d1-3B | JPT-9B | Winnow-12B |
|---|---|---|---|
| Libreria | transformers, `trust_remote_code` (codice del repository del modello, già letto nel piano binario) | transformers; per velocità servono i kernel di `flash-linear-attention` e `causal-conv1d` (senza, i layer DeltaNet usano un'implementazione lenta in torch) | transformers sui pesi convertiti (`models/winnow-12b-bf16/model/`) |
| Prompt | renderer del piano binario (`D1Renderer`, da `prompt.py` del modello) | renderer del piano binario (testo di `llm2jev`) | renderer del piano binario (`Scorer.render`, JSON con escaping) |
| Attenzione per le feature | hook sui layer `full_attention`: query dell'ultima posizione + chiavi | idem (gate di output di Qwen3.5 dopo l'attenzione: irrilevante per i pesi) | idem; nei layer globali 1 testa KV condivisa da 16 query |
| GPU per la fase 1a (estrazione) | 24 GB | 24 GB (~18 GB di pesi) | 48 GB (22 GB di pesi; con 24 GB serve offload, lento) |
| GPU per la fase 1b (backbone congelato in linea) | 24 GB | 48 GB | 48 GB |
| GPU per la fase 2 (LoRA + testa) | 24 GB | 48 GB (o QLoRA su 24 GB) | 48 GB |

Prima di estrarre: sul modello con il prompt del piano binario, P(supportato) calcolata dal nostro forward deve coincidere con quella del benchmark entro 0,01 su 50 righe (lo stesso test di identità del piano binario).

## 6. Servizio

| Modello | Server per verdetto + evidenza | Note |
|---|---|---|
| d1-3B | **server proprio con transformers** (lo stesso codice della valutazione) | il benchmark lo usa già così; nessun motivo di cambiare |
| JPT-9B | server proprio con transformers + kernel DeltaNet | vLLM restituisce stati solo in modalità pooling: un modello registrato per stati in posizioni scelte, attenzione e logit delle lettere va scritto e verificato; non è il primo tentativo |
| Winnow-12B | server proprio con transformers sui pesi BF16 | il server llama.cpp degli autori (NVFP4) non restituisce gli stati: per l'evidenza **non si usa**. Un'eventuale quantizzazione si rifà sui pesi BF16 (es. FP8) e si rivalida |

## 7. Fuori perimetro

| Modello | Perché |
|---|---|
| CLM-8B | non ha una posizione di lettura con logit di risposta: confronta embedding (pooling sull'ultimo token) dello stato e delle opzioni. Una testa di evidenza sugli stati di Qwen3-8B sarebbe possibile, ma il verdetto non verrebbe dallo stesso meccanismo e il piano binario non lo addestra |
| Rune-26B-A4B | 51,6 GB di pesi: fuori dal budget di una GPU per il fine-tuning; nel piano binario è solo un confronto |
| Winnow-12B NVFP4 (llama.cpp) | nessun accesso agli stati (§6) |
| FactCG | baseline del piano binario, non un decision model |

## Criteri per passare allo step successivo

- Indici dei layer verificati sul modello caricato (non solo dal config) con un forward e `output_hidden_states`.
- Hook delle feature di attenzione verificati: la riga ricalcolata coincide con i pesi di attenzione dell'implementazione "eager" entro 1e-3 su 5 prompt.
- Test di identità del verdetto superato per i tre modelli.
