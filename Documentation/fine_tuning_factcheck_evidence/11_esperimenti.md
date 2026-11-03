# Passo 11 — Esperimenti

## Obiettivo

Arrivare alla configurazione finale della testa per d1-3B, JPT-9B e Winnow-12B. Gli esperimenti si fanno su **d1-3B in fase 1a** (feature salvate: ogni prova è un training della sola testa); JPT e Winnow ripetono la configurazione vincente.

## Regole

1. **Una variazione per esperimento**, registrata con il suo `configs/<id>.json`.
2. Gli esperimenti della scala principale si fanno in ordine; una variazione adottata entra nel riferimento dei successivi.
3. Metriche di decisione, su `dev_evid_select`: F1 per frase, recall dell'insieme completo, F1 ≥ 10 frasi, F1 > 1.024 token; più i negativi accesi su `dev_evid_neg`.
4. **Si adotta** una variazione se migliora l'F1 di ≥ 1 punto o l'insieme completo di ≥ 2 punti, e non peggiora nessuna delle altre metriche di più di 1 punto. Se aggiunge feature lette dal modello (attenzione, media, layer), deve anche stare nel budget di latenza.
5. A parità, si tiene la variante più semplice (meno layer, meno feature, meno spazio su disco).
6. Ogni esperimento con 1 seme; le variazioni adottate per un margine < 2 punti si ripetono con 2 semi in più prima di adottarle.

## Fase 0 — Prerequisiti

| ID | Cosa | Passo |
|---|---|---|
| B-SAND, B-FT-SAND | scelta del preambolo e del verdetto per modello | [05](05_formato_prompt.md) |
| E-RAND, E-LEX, E-EMB, E-ATT, E-ABL | baseline | [07](07_baseline.md) |

## Fase H — d1-3B

### H0 — riferimento

Ultimo layer del mix, solo stato di fine frase, niente attenzione, embedding di posizione, aggregatore 2 layer, `ref_loss` (γ = 0,5), soglia `τ_e`, solo righe `supervised` di `ev_head`.

Prima domanda: H0 batte E-LEX ed E-ATT? Se no, prima di tutto si ricontrollano posizioni e feature ([06](06_rendering_ed_estrazione.md) §6).

### Scala principale (in quest'ordine)

| ID | Variazione rispetto al riferimento corrente | Cosa guardare |
|---|---|---|
| H-LOSS | `zlpr_loss` con soglia `s_0` invece di `ref_loss` e `τ_e` | insieme completo, errore sul numero di frasi |
| H-LOSS-b | variante: KL fra softmax(s) e l'uniforme sulle gold | solo se H-LOSS non è adottata |
| H-MIL | righe `mil_neg` nel mix, μ_neg = 0,3 e 1 | negativi accesi; recall sui positivi incerti (fascia soglia … soglia + 0,15) |
| H2a | mix di layer: 4 layer contro l'ultimo; poi 2 layer (quelli con più peso) | F1 globale; pesi del mix |
| H2b | feature di attenzione degli 8 layer pieni | F1 ≥ 10 frasi, F1 > 1.024 token |
| H2c | media dei token della frase accanto allo stato di fine frase | F1 sulle frasi lunghe (> 25 parole) |

### Ablazioni della configurazione vincente

| ID | Variazione | Cosa dice |
|---|---|---|
| H3a | testa lineare per frase, senza aggregatore | quanto serve l'aggregatore: atteso un calo forte su HoVer e sulle righe sparse |
| H3b | solo `x_0ᵀ W x_k` (pointer) | se una testa più semplice basta |
| H3c | aggregatore a 1 e a 4 layer | capacità necessaria |
| H4 | feature estratte col prompt nativo (senza preambolo) | quanto vale il preambolo per l'evidenza; se vale poco, il preambolo si toglie |
| H5 | mix senza righe sparse | atteso un calo oltre 10 frasi e oltre 1.024 token |
| H7 | ω FEVER = 1 invece di 0,3; insieme FEVER "almeno uno" invece dell'unione | gestione delle annotazioni incomplete |
| H8 | etichette: solo gold / + construct / + ablation / gold + `mil_pos` al posto di ablation | quanto valgono le etichette generate; se il MIL sui positivi sostituisce l'ablazione |
| H9 | senza embedding di posizione | bias verso inizio o fine |

Lettura di H3a: se la testa lineare è vicina su FEVER (una frase) ma molto sotto su HoVer e sulle sparse, l'aggregatore fa il suo lavoro; se è vicina anche sulle sparse, si può semplificare.

Lettura di H4: se il sandwich vince solo sotto i 1.024 token, il claim del preambolo non arriva alle frasi lontane; la fascia > 4.096 lo mostra.

### Esperimenti condizionati

Si fanno solo se si presenta la condizione.

| ID | Condizione | Variazione | Fase |
|---|---|---|---|
| H2d | H2c adottata, con guadagno sulle frasi lunghe | pooling attentivo guidato da `[c ; q]` ([09](09_training.md) §4) | 1b |
| H-PAIR | insieme completo HoVer < 0,65, o calo fra adiacenti e ≥ 20 frasi > 5 punti | `use_pair=True` + `pair_loss` (peso 0,2); variante con `att_pair` sommata a `S` come prior | 1a |
| H-FAITH | sufficienza < 85% | `L_faith` ([09](09_training.md) §5); si adotta solo se la sufficienza sale **e** l'F1 non scende di più di 1 punto | 1b |
| H6 | obiettivi non raggiunti dopo tutto il resto | fase 2, λ = 0,1 / 0,3 / 1 | 2 |

## Fase W — JPT-9B e Winnow-12B

| ID | Cosa |
|---|---|
| J0 / W0 | estrazione delle feature (con i layer che hanno avuto peso su d1, riportati alla stessa profondità relativa) e configurazione vincente di d1 |
| J0-mix / W0-mix | controllo rapido di H2a: i layer del mix possono essere diversi |
| W-FAITH | H-FAITH su Winnow (maschera esatta), solo se su d1 è stata adottata o è risultata vicina all'adozione |
| J6 / W6 | fase 2, solo se su d1 la fase 2 ha dato un guadagno chiaro e il modello è sotto gli obiettivi |
| L1 | latenza A / B / C ([10](10_valutazione.md) §5) per i tre modelli |

## Controlli finali

| ID | Cosa |
|---|---|
| C1 | 3 semi della testa finale per modello: media e deviazione standard su `dev_evid_select` |
| C2 | sufficienza e necessità su tutti i blocchi di test |
| C3 | testa contro E-ABL: distanza fra un forward e molti forward |
| C4 | confronto fra i modelli sull'intersezione delle righe giudicate supportate |

## Registro

`data/evidence/runs/registry.md`, una riga per corsa:

| ID | Modello | Fase | Variazione | F1 | Insieme completo | F1 ≥ 10 frasi | F1 > 1.024 tok | Neg. accesi | Sufficienza | Δ verdetto | Δ latenza | Decisione |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| E-LEX | — | — | — | | | | | | | — | — | baseline |
| H0 | d1-3B | 1a | — | | | | | | | 0 | | |

`evidence/train_head.py` aggiunge la riga automaticamente a fine corsa (tranne la decisione).
