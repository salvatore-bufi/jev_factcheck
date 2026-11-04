# Passo 3 — Dati generati, etichette ricavate col modello, evidenza spostata

## In breve

Le tre fonti del passo 2 hanno due limiti: l'evidenza annotata è a livello di frase intera, e i testi sono quasi solo Wikipedia, pagine web e abstract biomedici. Tre famiglie di dati in più li correggono. Per le prime prove si possono saltare; servono quando la testa funziona e si vuole migliorarla.

| Famiglia | Idea | Che cosa porta | Script |
|---|---|---|---|
| **Generati con posizioni note** | un LLM scrive un articolo che contiene fatti dati; si sa esattamente dove sono | etichette **esatte**, anche dentro una frase; testi tipo notizia | `evidence/data/gen_marked.py` |
| **Etichettati col modello** | per documenti senza evidenza annotata, si cerca col modello il gruppo più piccolo di frasi che basta a fargli dire "supportato" | evidenza su testi senza annotazioni | `evidence/data/label_ablation.py` |
| **Evidenza spostata** | si allontanano le frasi gold, o si inseriscono frasi di disturbo | più casi di evidenza lontana | `evidence/data/augment_sparse.py` |

Tutte scrivono righe nel formato del passo 2, con `gold_char_sets`, validate come le altre. Si generano **solo da righe di training**.

## 1. Documenti generati con posizioni note

Si parte da claim scomposti in fatti semplici (ricetta C2D di MiniCheck: ogni claim è riscritto come 2–4 frasi atomiche che insieme lo implicano). Al generatore (un LLM aperto in locale) si chiede:

```text
Write a coherent news-style article of 15 to 40 sentences that includes each of the
following sentences word for word, each wrapped in its tag: <e1>…</e1>, <e2>…</e2>, …
You may place a tagged sentence inside a longer sentence of yours.
Add other context sentences without tags. Do not restate the information of the
tagged sentences anywhere else in the article. Place the tagged sentences far apart.

<e1>{fatto 1}</e1>
<e2>{fatto 2}</e2>
```

Poi:
1. ogni `<ei>…</ei>` deve comparire una volta sola e parola per parola, altrimenti la riga si scarta;
2. si tolgono i tag, registrando l'intervallo di caratteri di ogni testo marcato: quello è l'intervallo gold, **esatto**, anche se il generatore l'ha messo dentro una frase più lunga;
3. un giudice LLM diverso conferma: "do these passages, taken alone, establish the claim? yes/no"; si tengono solo i "yes".

Varianti dallo stesso articolo:
- **negativo**: si toglie un testo marcato necessario (e la sua frase) → il claim non è più supportato, `evidence_term = mil_neg`;
- **claim parziale**: il claim si riscrive usando solo alcuni fatti → supportato, con gold = solo quei testi marcati.

Queste sono le uniche righe in cui l'etichetta dice esattamente quale parte di una frase conta: sono il segnale migliore per insegnare alla testa tratti più stretti di una frase (esperimento H10, [11](11_esperimenti.md)). `evidence_quality = construct`, peso 1.

## 2. Etichette ricavate col modello (ablazione)

Per righe supportate senza evidenza annotata (per esempio i dati C2D e D2C di MiniCheck come rilasciati). Si usa il verdetto di KnowLine-4B con il prompt con preambolo. Togliere pezzi di documento richiede di dividerlo in unità: si usano le frasi del nostro divisore. È un lavoro offline sui dati di training, che non entra in produzione.

```python
def ablation_label(sentences, p_supported, thr, max_k=6):
    """sentences: intervalli di caratteri delle frasi del documento;
    p_supported(indici) = P(supportato) con il documento ridotto a quelle frasi (nell'ordine originale).
    -> (indici delle frasi di evidenza, qualità) oppure None."""
    n, chosen = len(sentences), []
    while len(chosen) < max_k:
        best = max((k for k in range(n) if k not in chosen), key=lambda k: p_supported(sorted(chosen + [k])))
        chosen.append(best)
        if p_supported(sorted(chosen)) > thr:
            break
    else:
        return None                                        # nessun gruppo di al massimo 6 frasi basta: riga scartata
    for k in list(chosen):                                 # pulizia: via le frasi che non servono
        rest = [j for j in chosen if j != k]
        if rest and p_supported(sorted(rest)) > thr:
            chosen = rest
    complement = [k for k in range(n) if k not in chosen]
    redundant = p_supported(complement) > thr              # l'evidenza è ripetuta altrove nel documento
    return sorted(chosen), ("ablation_partial" if redundant else "ablation")
```

- Ogni passo greedy manda tutti i documenti ridotti in un batch; KnowLine è causale, quindi si riusa il calcolo della parte iniziale comune del prompt.
- `thr` = soglia del verdetto.
- Filtri: un giudice LLM conferma le frasi trovate (atteso almeno l'80% di conferme); su 300 righe si rifà la ricerca con elephant-1, e i gruppi devono coincidere in buona parte (Jaccard medio ≥ 0,7).
- Le frasi trovate diventano `gold_char_sets`; peso della riga 0,5 (`ablation_partial`: 0,3).

## 3. Evidenza spostata

Da righe con almeno 2 frasi gold:

| Trasformazione | Come |
|---|---|
| Agli estremi | la prima frase gold va all'inizio del documento, l'ultima alla fine |
| Frasi di disturbo | 5–40 frasi di altri documenti di training della stessa fonte inserite fra le frasi gold |
| Lontano dall'inizio | testo di disturbo aggiunto in testa finché la prima frase gold è oltre 1.024 token (o oltre 4.096) |
| Negativi spostati | le stesse operazioni su righe non supportate |

Si spostano frasi intere della fonte (gli intervalli gold), non token: il documento nuovo si ricostruisce pezzo per pezzo con `DocBuilder` (`evidence/data/schema.py`), che restituisce i nuovi intervalli di caratteri.

Controlli:
- una frase gold che inizia riferendosi alla precedente ("He", "It", "This", "The company"…) non si sposta da sola: si sposta con la precedente, o la riga si scarta;
- il giudice conferma che l'etichetta non è cambiata;
- le frasi di disturbo vengono solo da documenti di training.

## 4. Quante righe

| Famiglia | Prima prova | Versione completa |
|---|---:|---:|
| Generati con posizioni esatte | 2.000 | 10.000 |
| Etichettati col modello | 2.000 | 10.000 |
| Evidenza spostata (positivi) | 4.000 | 10.000 |
| Negativi spostati | 1.000 | 3.000 |

## 5. Controlli

| Controllo | Atteso |
|---|---|
| 30 righe per famiglia lette a mano: le parti gold dimostrano il claim e non ne manca nessuna necessaria | almeno 27 su 30 |
| Conferme del giudice sull'ablazione | ≥ 80% |
| Accordo KnowLine–elephant-1 sull'ablazione | Jaccard ≥ 0,7 |
| Righe spostate con riferimenti rotti (30 lette) | 0 |
