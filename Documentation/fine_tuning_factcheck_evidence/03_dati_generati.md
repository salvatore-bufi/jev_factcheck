# Passo 3 — Altri dati con evidenza nota

## In breve

Le righe del passo 2 hanno due limiti: l'evidenza annotata viene quasi solo da Wikipedia, e le frasi di evidenza sono spesso vicine fra loro. Per insegnare alla testa a lavorare su altri tipi di testo e a trovare frasi lontane servono tre tipi di dati in più:

| Tipo | Idea | Script | `evidence_quality` |
|---|---|---|---|
| **Generati con posizioni note** | un LLM scrive un articolo che contiene frasi date; si sa dove sono | `evidence/gen_marked.py` | `construct` |
| **Etichettati col modello** | per documenti senza evidenza, si cerca col modello il gruppo più piccolo di frasi che basta a fargli dire "supportato" | `evidence/label_ablation.py` | `ablation` |
| **Evidenza spostata** | si prendono righe con evidenza nota e si allontanano le frasi di evidenza, o si inseriscono frasi di disturbo | `evidence/augment_sparse.py` | come la riga di partenza |

Tutto si genera **solo da righe di training**. Generatore e giudice LLM sono quelli del piano binario ([../fine_tuning_factcheck/02b_dataset_generazione_llm.md](../fine_tuning_factcheck/02b_dataset_generazione_llm.md)).

## 1. Documenti generati con posizioni note (C2D e CG2C)

**C2D.** Nella ricetta C2D del piano binario ogni claim viene scomposto in poche frasi semplici che insieme lo dimostrano. Si chiede al generatore un articolo che le contenga **marcate**:

```text
Write a coherent news-style article of 15 to 40 sentences that includes each of the
following sentences word for word, each wrapped in its tag: <e1>…</e1>, <e2>…</e2>, …
Add other context sentences without tags. Do not restate the information of the
tagged sentences anywhere else in the article. Place the tagged sentences far apart.

<e1>{frase 1}</e1>
<e2>{frase 2}</e2>
```

Poi:
1. si controlla che ogni frase marcata ci sia, una volta sola, parola per parola (altrimenti si scarta);
2. si tolgono i tag ricordando dove stavano;
3. si divide il testo con `sentences.py`; evidenza = le nostre frasi che contengono i pezzi marcati;
4. un giudice LLM conferma: "queste frasi, da sole, dimostrano il claim?". Si tengono solo le righe confermate.

Dallo stesso articolo si ricavano due righe in più:
- **negativo**: si toglie una frase marcata necessaria → il claim non è più supportato;
- **claim parziale**: si riscrive il claim usando solo alcune frasi → supportato, e l'evidenza sono solo quelle frasi.

**CG2C.** Nella ricetta CG2C del piano binario si estrae un grafo di fatti dal documento e si costruisce il claim seguendo un percorso nel grafo. Basta salvare, per ogni fatto, la frase da cui viene: l'evidenza sono le frasi dei fatti del percorso. Danno evidenza su più frasi fuori da Wikipedia.

## 2. Etichette ricavate col modello (ablazione)

Per righe supportate senza evidenza (C2D e D2C come rilasciati, righe HoVer non allineabili) si usa il verdetto di **KnowLine-4B** del piano binario, con il prompt con preambolo ([05](05_formato_prompt.md)). Si cerca il gruppo più piccolo di frasi che, da solo, fa dire "supportato":

```text
1. si parte da un gruppo vuoto
2. si aggiunge la frase che, insieme al gruppo, dà la P(supportato) più alta
3. si ripete finché P(supportato) supera la soglia (al massimo 6 frasi; se non basta, la riga si scarta)
4. pulizia: si toglie ogni frase del gruppo senza la quale P resta sopra soglia
5. controllo: se anche il documento SENZA il gruppo resta "supportato", l'evidenza è ripetuta altrove
   → la riga si tiene con peso più basso (ablation_partial)
```

Ogni passo richiede un forward per frase candidata: si mandano tutti insieme e si riusa il calcolo della parte iniziale comune del prompt.

Filtri:
- un giudice LLM conferma le frasi trovate; atteso almeno l'80% di conferme, altrimenti si rivede la procedura;
- su 300 righe si rifà la ricerca con elephant-1: i gruppi trovati devono coincidere in buona parte (Jaccard medio ≥ 0,7). Se no, le etichette dipendono troppo da KnowLine, e per gli altri modelli si usano solo le righe su cui i due concordano.

```sh
gpu-run --batch --name ev-ablation-knowline -- bash -c \
  'python -m evidence.label_ablation --model knowline-4b --pairs c2d_train d2c_train hover_train_unaligned \
     --out data/evidence/pairs/ablation_knowline.jsonl > logs/ev-ablation-knowline.out 2>&1'
```

Lo script riprende da dove si era fermato se interrotto.

## 3. Evidenza spostata (righe sparse)

Da righe con almeno 2 frasi di evidenza (annotate o generate):

| Trasformazione | Come |
|---|---|
| **Agli estremi** | la prima frase di evidenza va all'inizio del documento, l'ultima alla fine |
| **Frasi di disturbo** | si inseriscono 5–40 frasi prese da altri documenti di training della stessa fonte fra le frasi di evidenza |
| **Lontano dal claim del preambolo** | si aggiungono frasi di disturbo all'inizio finché la prima frase di evidenza è oltre 1.024 token (o oltre 4.096) dall'inizio del documento |
| **Negativi spostati** | le stesse operazioni su righe non supportate |

Controlli:
- una frase che comincia riferendosi alla precedente ("He", "It", "This", "The company"…) non si sposta da sola: si sposta insieme alla precedente, o si scarta la riga;
- il giudice conferma che l'etichetta del documento modificato non è cambiata.

```sh
python -m evidence.augment_sparse --pairs fever_train hover_train construct --n 10000 --seed 0
```

## 4. Quante righe

| Tipo | Prima prova | Versione completa |
|---|---:|---:|
| C2D con posizioni | 2k | 10k |
| CG2C con posizioni | — | 5–10k |
| Etichettate col modello | 2k | 10–20k |
| Evidenza spostata (positivi) | 4k | 10k |
| Negativi spostati | 1k | 3k |

## 5. Controlli

| Controllo | Atteso |
|---|---|
| 30 righe per tipo lette a mano: le frasi di evidenza dimostrano il claim e non ne manca nessuna necessaria | ≥ 27 su 30 |
| Conferme del giudice sull'ablazione | ≥ 80% |
| Accordo KnowLine–elephant-1 sull'ablazione | Jaccard ≥ 0,7 |
| Righe spostate con riferimenti rotti (30 lette) | 0 |
