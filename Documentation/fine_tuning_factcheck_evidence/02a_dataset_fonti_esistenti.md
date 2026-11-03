# Step 2a — Fonti esistenti con evidenza a livello di frase

## Obiettivo

Estendere le coppie di `data_prep/build_pairs.py` con due campi:
- `sentences`: il documento come **lista di frasi** (con offset di carattere), così la segmentazione usata per le etichette è la stessa usata in training e in produzione;
- `evidence`: indici delle frasi che supportano il claim (solo per righe supportate con evidenza nota).

Le righe sono le stesse per i tre modelli: tokenizer e prompt entrano solo nel rendering ([03](03_rendering_formato.md)).

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

Come la loss usa le righe ([04](04_loss.md)):

| `label` | `evidence_quality` | Termine di evidenza |
|---|---|---|
| 1 | gold, construct, ablation | supervisionato (frasi gold sopra la soglia dell'istanza, le altre sotto) |
| 1 | none | facoltativo: MIL positivo (almeno una frase sopra la soglia), esperimento H-MIL |
| 0 | — | MIL negativo: **nessuna** frase sopra la soglia |

Nel piano precedente le righe negative erano mascherate; ora entrano nella loss di evidenza.

## 2. Fonte per fonte

| Fonte | Evidenza disponibile | Cosa fare | Difficoltà |
|---|---|---|---|
| **FEVER** | insiemi di (pagina, numero di riga); le wiki-pages hanno il campo `lines` con frasi già numerate | costruire il documento **dalle righe** (oggi si concatena il testo): `sentences` esatte, indici diretti | bassa |
| **HoVer** | `supporting_facts` = (titolo, indice di frase) | il db `wiki_wo_links.db` contiene il testo introduttivo **unito**, senza confini di frase: va risegmentato e l'indice riallineato (§3) | media |
| **SciFact** | `abstract` è già una lista di frasi, `evidence` gli indici (`allenai/scifact_entailment`) | diretto; solo test OOD | bassa |
| **WiCE** | frasi di evidenza annotate | diretto; solo test OOD (è fonte di AggreFact) | bassa |
| **MiniCheck C2D/D2C** | il dataset rilasciato ha solo `claim`, `doc`, `label`: **nessuna evidenza** | ricavarla per ablazione offline o rigenerare C2D con posizioni note ([02b](02b_dataset_generazione_llm.md)) | media |
| ANLI | premesse di 1–3 frasi, nessuna evidenza per frase | `evidence_quality = none` (solo verdetto) | — |
| SummEdits | nessuna | solo verdetto, oppure ablazione offline | — |
| AVeriTeC | evidenza in forma domanda–risposta | ogni coppia Q/R come "frase"; etichette per ablazione | media |
| VitaminC | il documento è già quasi solo evidenza (1–2 frasi) | inutile per l'evidenza; solo verdetto | — |

## 3. Negativi utili per la testa

Il termine MIL negativo ([04](04_loss.md) §3) serve soprattutto con documenti **lunghi** che contengono frasi simili al claim:

| Fonte | Righe | Perché sono utili |
|---|---|---|
| FEVER REFUTES, con la pagina intera | la frase che confuta somiglia al claim | insegna a non evidenziare una frase solo perché ne condivide le parole |
| FEVER NOT ENOUGH INFO, con la pagina recuperata | frasi sull'entità giusta, senza il fatto | insegna che "parla dello stesso argomento" non basta |
| HoVer NOT_SUPPORTED | catene in cui manca un anello | il caso multi-hop: S1 c'è, S40 no |
| C2D negativi (una frase tolta) | documento quasi identico al positivo | coppia minima: stessa testa, verdetto opposto |

VitaminC e ANLI hanno documenti troppo corti per dare segnale a livello di frase.

## 4. HoVer: riallineare gli indici

HoVer deriva da HotpotQA, dove le pagine sono liste di frasi; l'indice di `supporting_facts` si riferisce a quella segmentazione. Nel db locale il testo è unito.

Procedura:
1. segmentare il testo con lo stesso splitter di produzione ([03](03_rendering_formato.md) §2);
2. controllo: per ogni pagina, l'indice gold deve essere < numero di frasi; per un campione di 100 pagine confrontare con la segmentazione di HotpotQA (`hotpot_qa`, config `distractor`, campo `context`, che contiene le frasi per titolo quando la pagina compare in una domanda HotpotQA);
3. se il numero di frasi diverge, usare la segmentazione di HotpotQA quando disponibile; altrimenti la riga va in `evidence_quality = ablation` (etichette ricavate dal modello) o si scarta.

Il builder **mischia l'ordine delle pagine** (per non rivelare l'ordine degli hop): gli indici di frase vanno calcolati **dopo** il mescolamento.

## 5. Annotazioni incomplete

FEVER in particolare: gli annotatori hanno indicato **un** insieme sufficiente (a volte alcuni), non tutte le frasi che supportano il claim. Esempio illustrativo:

> Claim: "Nikolaj Coster-Waldau worked with the Fox Broadcasting Company."
> Gold: la frase su *New Amsterdam* (serie Fox).
> Se più avanti nella pagina un'altra frase ripetesse lo stesso fatto, sarebbe anch'essa evidenza valida ma non annotata.

Conseguenza: una frase "non gold" non è sicuramente "non evidenza". La loss pesa meno le frasi non gold delle fonti incomplete ([04](04_loss.md) §4), e la valutazione riporta anche la sufficienza, che non dipende dalle annotazioni ([06](06_valutazione.md)).

## 6. Quantità attese

| Fonte | Train (stima) | Note |
|---|---:|---|
| FEVER SUPPORTS | ~80.000 claim | con tetto, come nel mix |
| FEVER REFUTES / NEI con pagina | decine di migliaia | se ne prende una quota ([02e](02e_mix_finale.md)) |
| HoVer SUPPORTED | ~6.000–7.000 | dopo il riallineamento |
| HoVer NOT_SUPPORTED | ~9.000 | |
| MiniCheck positivi | ~7.000 | solo con etichette per ablazione |
| SciFact, WiCE | — | solo test |

## 7. Codice

Estensioni di `build_pairs.py` (**da scrivere**): opzione `--with-evidence` per FEVER (documento dalle `lines`, anche per REFUTES e NEI), HoVer (risegmentazione e allineamento, anche per NOT_SUPPORTED), SciFact e WiCE (test); verifica `sentences` ↔ `document` (le frasi devono ricoprire il documento senza sovrapposizioni).

## Criteri per passare allo step successivo

- `sentences` ed `evidence` presenti per FEVER, HoVer, SciFact, WiCE; `sentences` anche per i negativi lunghi di FEVER e HoVer.
- Riallineamento HoVer verificato su 100 pagine; quota di righe scartate registrata.
- 30 righe per fonte lette a mano: le frasi marcate stabiliscono davvero il claim.
