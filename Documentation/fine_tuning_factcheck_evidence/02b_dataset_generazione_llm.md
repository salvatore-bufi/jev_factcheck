# Step 2b — Etichette di evidenza: generate per costruzione o ricavate offline

## Obiettivo

Le fonti con evidenza gold sono soprattutto Wikipedia (FEVER, HoVer). Per evidenziare bene anche su notizie, riassunti e documenti lunghi servono etichette di evidenza su altri tipi di testo. Tre strade:

1. **per costruzione**: generare dati in cui si sa dove sta l'evidenza (C2D e CG2C con posizioni note);
2. **per ablazione offline**: etichettare righe supportate esistenti con il modello del verdetto e molti forward (consentito: è un costo di preparazione, non di produzione);
3. **evidenza sparsa costruita**: spostare l'evidenza nota lontano, con distrattori.

Regole invariate: si parte solo dal train; le etichette di verdetto vengono dall'operazione e da un giudice ([../fine_tuning_factcheck/02b_dataset_generazione_llm.md](../fine_tuning_factcheck/02b_dataset_generazione_llm.md)).

## 1. C2D con posizioni note

Nel C2D di MiniCheck il documento si genera da un insieme di frasi `s = {s_1,1, s_1,2, …}` che insieme implicano il claim. Il paper fa scrivere all'LLM un passaggio che "menziona" quelle frasi, e nel dataset rilasciato le posizioni si perdono.

Modifica: chiedere all'LLM un documento in cui ogni frase dell'insieme compare **marcata**, e poi togliere i marcatori.

```text
Scrivi un breve articolo coerente che includa le seguenti frasi, ciascuna racchiusa
fra <e1>…</e1>, <e2>…</e2>, …, riportate parola per parola. Aggiungi altre frasi di
contesto non marcate. Non ripetere le informazioni delle frasi marcate altrove.
```

Controlli automatici:
- ogni frase marcata compare parola per parola;
- dopo la segmentazione ([03](03_rendering_formato.md)), ogni frase marcata corrisponde a una frase (o al massimo due) del documento;
- l'evidenza = indici delle frasi che contengono i marcatori.

Le varianti negative di C2D (una frase tolta) restano negativi senza evidenza. I sotto-claim del power set che sono ancora supportati ereditano come evidenza **solo** le frasi dei loro fatti: un esempio con documento "bucato" e claim supportato è utilissimo, perché l'evidenza è un sottoinsieme preciso.

## 2. CG2C con posizioni note

In CG2C ogni arco del sottografo viene da una frase del documento. Il claim positivo attraversa tutti gli archi, quindi l'evidenza = frasi da cui vengono gli archi del sottografo. Serve solo salvare, nell'estrazione del grafo, l'indice della frase per ogni tripla.

È la fonte naturale di **evidenza multi-hop** fuori da Wikipedia.

## 3. Ablazione offline (etichettatore)

Per righe supportate senza evidenza (MiniCheck rilasciato, D2C, SummEdits, AVeriTeC, HoVer non riallineabile):

```text
E = {}
ripeti:   aggiungi a E la frase k che massimizza P(supportato | solo frasi E ∪ {k}, claim)
fino a:   P(supportato | E) > soglia del verdetto      (o al massimo 6 frasi)
pulizia:  per ogni k in E, se P(supportato | E \ {k}) resta sopra soglia, togli k
controllo di necessità: P(supportato | documento senza E) deve scendere sotto soglia,
                        altrimenti l'evidenza è ridondante nel documento → evidence_quality = ablation_partial
```

| Aspetto | Valore |
|---|---|
| Modello | il verdetto fine-tunato del piano binario (con claim prima del documento), o il modello di partenza se non ancora pronto |
| Costo | ~n·k forward per riga (n frasi, k ≤ 6); per documenti da 30 frasi ~100–180 forward brevi. Con vLLM e prefix caching del claim, ~1–2 giorni di L40S per 20k righe |
| Qualità | buona sui casi semplici; rischia di scegliere frasi "plausibili" che il modello sopravvaluta |
| Filtro | un giudice LLM diverso conferma: "queste frasi, da sole, stabiliscono il claim?" Si tengono le righe confermate |

Le righe etichettate così hanno `evidence_quality = ablation` e peso più basso nella loss ([04](04_loss.md)).

## 4. Evidenza sparsa costruita (augmentation)

Da righe con evidenza nota (gold o per costruzione) e almeno 2 frasi di evidenza:

| Trasformazione | Esempio | Etichette |
|---|---|---|
| **Estremi** | prima frase di evidenza all'inizio, ultima alla fine | verdetto 1; evidenza con i nuovi indici |
| **Distrattori in mezzo** | 5–40 frasi da altri documenti dello stesso split (stesso dominio) inserite fra le frasi di evidenza | idem |
| **Ponte senza parole del claim** | (da HoVer e CG2C) si sceglie apposta una catena in cui la prima frase non condivide entità con il claim | idem |

Controlli: le frasi spostate non devono contenere pronomi o riferimenti alla frase che le precedeva ("He", "The company", "This"); se li contengono, o si sposta il blocco intero o si scarta. Un giudice LLM conferma che il documento modificato supporta ancora il claim.

## 5. Volumi

| Famiglia | Pilota | Corsa piena |
|---|---:|---:|
| FEVER + HoVer gold | 8k | 30k |
| C2D con posizioni | 2k | 10k |
| CG2C con posizioni | 0 | 5–10k |
| Ablazione offline (MiniCheck, D2C, SummEdits) | 2k | 10–20k |
| Evidenza sparsa costruita | 2k | 10k |

## 6. Codice da scrivere

- in `llm_gen/tasks.py`: `c2d_passage_marked`, l'opzione per salvare l'indice di frase in `cg2c_graph`, `verify_evidence` (giudice);
- `finetune/label_evidence.py`: ablazione offline con vLLM;
- `finetune/augment_sparse.py`: trasformazioni di §4.

## Criteri per passare allo step successivo

- Per ogni famiglia, 30 esempi letti a mano: le frasi marcate stabiliscono il claim e nessuna frase necessaria manca.
- Tasso di conferma del giudice per l'ablazione ≥ 80%; sotto, rivedere la procedura.
