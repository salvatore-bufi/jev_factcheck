# Step 2a — Fonti esistenti con evidenza a livello di frase

## Obiettivo

Estendere le coppie di `data_prep/build_pairs.py` con due campi:
- `sentences`: il documento come **lista di frasi** (con offset di carattere), così la segmentazione usata per le etichette è la stessa usata in training e in produzione;
- `evidence`: indici delle frasi che supportano il claim (solo per righe supportate con evidenza nota).

Verificato in questa sessione sui dati locali: cosa c'è già e cosa va ricostruito.

## 1. Il formato

```json
{"id": "fever:train:75397", "source": "fever", "split": "train", "label": 1,
 "claim": "Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.",
 "document": "Nikolaj Coster-Waldau (born 27 July 1970) is a Danish actor ... He then played ... on the Fox television series New Amsterdam ...",
 "sentences": [[0, 112], [113, 260], [261, 402]],
 "evidence": [2], "evidence_sets": [[2], [2, 0]],
 "evidence_quality": "gold", "doc_id": "..."}
```

(Offset e indici illustrativi: mostrano il formato, non una riga reale.)

| Campo | Significato |
|---|---|
| `sentences` | coppie `[inizio, fine)` in caratteri sul `document` |
| `evidence` | unione degli insiemi gold (frasi da evidenziare) |
| `evidence_sets` | insiemi alternativi, quando la fonte li dà (FEVER) |
| `evidence_quality` | `gold` (annotazione umana), `construct` (nota per costruzione, [02b](02b_dataset_generazione_llm.md)), `ablation` (ricavata offline con il modello), `none` |

Le righe con `label = 0` hanno `evidence = []` e `evidence_quality = none`: la loss per frase le maschera ([04](04_loss.md)).

## 2. Fonte per fonte

| Fonte | Evidenza disponibile | Cosa fare | Difficoltà |
|---|---|---|---|
| **FEVER** | insiemi di (pagina, numero di riga); le wiki-pages hanno il campo `lines` con frasi già numerate | costruire il documento **dalle righe** (oggi si concatena il testo): `sentences` esatte, indici diretti | bassa |
| **HoVer** | `supporting_facts` = (titolo, indice di frase) | il db `wiki_wo_links.db` contiene il testo introduttivo **unito**, senza confini di frase (verificato): va risegmentato e l'indice riallineato (§3) | media |
| **SciFact** | `abstract` è già una lista di frasi, `evidence` gli indici (verificato in `allenai/scifact_entailment`) | diretto; solo test OOD | bassa |
| **WiCE** | frasi di evidenza annotate | diretto; solo test OOD (è fonte di AggreFact) | bassa |
| **MiniCheck C2D/D2C** | il dataset rilasciato ha solo `claim`, `doc`, `label` (verificato): **nessuna evidenza** | ricavarla per ablazione offline o rigenerare C2D con posizioni note ([02b](02b_dataset_generazione_llm.md)) | media |
| ANLI | premesse di 1–3 frasi, nessuna evidenza per frase | `evidence_quality = none` (solo verdetto); eventualmente ablazione | — |
| SummEdits | nessuna | solo verdetto, oppure ablazione offline | — |
| AVeriTeC | evidenza in forma domanda–risposta | ogni coppia Q/R come "frase"; etichette per ablazione | media |
| VitaminC | il documento è già quasi solo evidenza (1–2 frasi) | inutile per l'evidenza (tutto è evidenza); solo verdetto | — |

## 3. HoVer: riallineare gli indici

HoVer deriva da HotpotQA, dove le pagine sono liste di frasi; l'indice di `supporting_facts` si riferisce a quella segmentazione. Nel db locale il testo è unito.

Procedura:
1. segmentare il testo con lo stesso splitter di produzione ([03](03_rendering_formato.md) §2);
2. controllo: per ogni pagina, l'indice gold deve essere < numero di frasi; per un campione di 100 pagine confrontare con la segmentazione di HotpotQA (`hotpot_qa`, config `distractor`, campo `context`, che contiene le frasi per titolo quando la pagina compare in una domanda HotpotQA);
3. se il numero di frasi diverge, usare la segmentazione di HotpotQA quando disponibile; altrimenti la riga va in `evidence_quality = ablation` (etichette ricavate dal modello) o si scarta.

Ricordare che il builder **mischia l'ordine delle pagine** (per non rivelare l'ordine degli hop): gli indici di frase vanno calcolati **dopo** il mescolamento.

## 4. Annotazioni incomplete

FEVER in particolare: gli annotatori hanno indicato **un** insieme sufficiente (a volte alcuni), non tutte le frasi che supportano il claim. Esempio illustrativo:

> Claim: "Nikolaj Coster-Waldau worked with the Fox Broadcasting Company."
> Gold: la frase su *New Amsterdam* (serie Fox).
> Se più avanti nella pagina un'altra frase ripetesse lo stesso fatto, sarebbe anch'essa evidenza valida ma non annotata.

Conseguenza: una frase "non gold" non è sicuramente "non evidenza". La loss tratta i negativi per frase con un peso più basso o con una loss di ranking ([04](04_loss.md) §3), e la valutazione riporta anche la sufficienza, che non dipende dalle annotazioni ([06](06_valutazione.md)).

## 5. Quantità attese (righe supportate con evidenza)

| Fonte | Train (stima) | Note |
|---|---:|---|
| FEVER (SUPPORTS) | ~80.000 claim | con tetto, come nel mix |
| HoVer (SUPPORTED) | ~6.000–7.000 | dopo il riallineamento |
| MiniCheck positivi | ~7.000 | solo con etichette per ablazione |
| SciFact, WiCE | — | solo test |

## 6. Codice

Estensioni di `build_pairs.py` (**da scrivere**): opzione `--with-evidence` per FEVER (documento dalle `lines`), HoVer (risegmentazione e allineamento), SciFact e WiCE (test); verifica `sentences` ↔ `document` (le frasi devono ricoprire il documento senza sovrapposizioni).

## Criteri per passare allo step successivo

- `sentences` ed `evidence` presenti per FEVER, HoVer, SciFact, WiCE.
- Riallineamento HoVer verificato su 100 pagine; quota di righe scartate registrata.
- 30 righe per fonte lette a mano: le frasi marcate stabiliscono davvero il claim.
