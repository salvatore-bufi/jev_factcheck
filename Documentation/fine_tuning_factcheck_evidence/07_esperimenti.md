# Step 7 — Piano degli esperimenti

## Obiettivo

Arrivare alla testa di evidenza finale per d1-3B, JPT-9B e Winnow-12B con il minimo di corse. Quasi tutti gli esperimenti si fanno su **d1-3B in fase 1a** (stati salvati): durano minuti. JPT e Winnow ripetono solo la configurazione vincente.

## Regole

1. Una variazione per esperimento, registrata. Gli esperimenti H sono una **scala**: una variazione adottata entra nel riferimento degli esperimenti successivi.
2. Si decide su `dev_evid_select` (F1 per frase, recall dell'insieme completo, F1 oltre 10 frasi, F1 oltre 1.024 token), con un occhio a `dev_evid_neg`; per la fase 2, anche sul verdetto in `dev_select`.
3. Si adotta una variazione se migliora l'F1 di ≥ 1 punto o la recall dell'insieme completo di ≥ 2 punti, senza peggiorare il verdetto né le altre due metriche di più di 1 punto. Con un costo di latenza (feature in più da leggere), serve il guadagno **e** il rispetto del budget.
4. Fra due varianti equivalenti si tiene la più semplice (meno layer, meno feature, meno spazio).
5. Prima d1-3B, poi JPT-9B, poi Winnow-12B.

## Fase 0 — Baseline (tre modelli)

| ID | Cosa |
|---|---|
| B-1D, B-ORD, B-SAND (P-A, P-B), B-FT, B-FT-SAND | verdetto nei tre formati; scelta del formato e del preambolo ([01](01_baseline.md) §1) |
| E-RAND, E-LEX, E-EMB | evidenza senza modello |
| E-ATT | attenzione da `Answer:` negli 8 layer pieni (teste scelte su metà del dev); formato nativo e sandwich |
| E-ABL | ablazione greedy offline, su campione |

## Fase H — Testa di evidenza su d1-3B (fase 1a salvo dove indicato)

Riferimento **H0**: ultimo layer, solo stato di fine frase, niente feature di attenzione, aggregatore 2 layer, BCE pesata + ranking (γ = 0,5), soglia fissa `τ_e`, solo righe positive di `ev_head`. È la testa del piano precedente.

### Scala principale (in quest'ordine)

| ID | Variazione | Domanda | Idea |
|---|---|---|---|
| **H0** | — | la testa batte E-LEX ed E-ATT? Se no, controllare rendering e posizioni prima di tutto | |
| H-LOSS | ZLPR con `s_0` (solo righe supervised) invece di BCE + ranking; variante ListNet | la soglia dell'istanza fa meglio di `τ_e`, soprattutto sull'insieme completo? | 6 |
| H-MIL | termine sui negativi: μ_neg = 0 / 0,3 / 1 | i negativi riducono i falsi positivi lessicali senza spegnere i positivi incerti? | 6 |
| H2a | mix di layer: ultimo / ultimo + 1 globale / 4 layer ([00c](00c_modelli_e_architetture.md) §2) | dove si localizza meglio l'evidenza; dove vanno i pesi del mix | 1 |
| H2b | feature di attenzione degli 8 layer pieni: no / sì | l'attenzione già calcolata aiuta? (guardare l'evidenza sparsa) | 1 |
| H2c | media della frase accanto allo stato di fine frase: no / sì | aiuta sulle frasi lunghe (> 25 parole)? | 2 |

### Ablazioni della configurazione vincente

| ID | Variazione | Domanda |
|---|---|---|
| H3a | testa lineare per frase (senza aggregatore) | quanto serve l'aggregatore (atteso: crollo sull'evidenza sparsa) |
| H3b | solo pointer `x_0ᵀ W x_k` | alternativa più semplice |
| H3c | aggregatore 1 / 2 / 4 layer | capacità necessaria |
| H4 | formato nativo e "claim prima" invece del sandwich (richiede una nuova estrazione) | quanto vale il preambolo per l'evidenza |
| H5 | senza righe sparse costruite | l'augmentation serve? (atteso: calo oltre 10 frasi e oltre 1.024 token) |
| H7 | ω FEVER 0,3 / 1; unione vs "almeno un insieme"; esclusione della non gold più alta | gestione delle annotazioni incomplete |
| H8 | solo gold / + construct / + ablation / gold + MIL positivo al posto dell'ablation | valore delle etichette generate e ricavate; il MIL basta? |
| H9 | senza embedding della posizione della frase | la posizione aiuta o crea un bias verso inizio/fine? |

**Come leggere H3a contro la configurazione vincente.** Se la testa lineare è vicina all'aggregatore su FEVER (evidenza di una frase) ma molto sotto su HoVer e sulle righe sparse, l'aggregatore sta facendo il lavoro previsto. Se è vicina anche sulle sparse, il preambolo basta e l'aggregatore si può semplificare.

**Come leggere H4.** Se il formato nativo fa quasi come il sandwich sull'evidenza, il preambolo si può togliere (un costo in meno). Se il sandwich vince solo sotto i 1.024 token, il claim non arriva alle frasi lontane: lo si vede anche nell'F1 per distanza dal preambolo.

### Esperimenti condizionati (solo se il motivo si presenta)

| ID | Condizione | Variazione | Fase | Idea |
|---|---|---|---|---|
| H2d | H2c guadagna sulle frasi lunghe | pooling attentivo guidato da `[claim; q]` | 1b | 2 |
| H-PAIR | recall dell'insieme completo HoVer < 0,65 o calo oltre 20 frasi > 5 punti | termine bilineare a coppie + perdita sulle coppie gold; variante con `att_pair` come prior | 1a | 3 |
| H-FAITH | sufficienza < 85% | perdita di fedeltà con maschera per frase (d1-3B; poi Winnow) | 1b | 4 |
| H1 / H6 | obiettivi non raggiunti dopo la scala | fase 2 (LoRA + testa); λ = 0,1 / 0,3 / 1 | 2 | |

H-FAITH si adotta solo se la sufficienza sale **e** l'F1 rispetto alle gold non scende di più di 1 punto ([00a](00a_scelta_approccio.md) §8).

## Fase W — JPT-9B, Winnow-12B, conferma

| ID | Cosa |
|---|---|
| J0 | configurazione vincente di d1 su JPT-9B (fase 1a); controllo rapido di H2a (i layer del mix possono cambiare) |
| W0 | idem su Winnow-12B |
| J1 / W1 | fase 2, solo se su d1 la fase 2 ha dato un guadagno chiaro e il modello è sotto gli obiettivi |
| W-FAITH | H-FAITH su Winnow (maschera esatta), solo se su d1 è stato adottato |
| L1 | misura di latenza con e senza testa (e con e senza preambolo) sul server di produzione, per i tre modelli |

## Controlli finali

| ID | Cosa |
|---|---|
| C1 | 3 seed della testa finale per modello (fase 1: costo minimo) |
| C2 | sufficienza e necessità su tutti i test di evidenza |
| C3 | confronto con E-ABL: quanto la testa in un forward si avvicina al metodo da n forward |
| C4 | confronto fra modelli sull'intersezione delle righe giudicate supportate da tutti e tre |

## Calendario indicativo

| Settimana | Lavoro |
|---|---|
| 1 | builder con evidenza ([02a](02a_dataset_fonti_esistenti.md)), segmentazione, rendering sandwich ed estrazione per d1 ([03](03_rendering_formato.md)); Fase 0 |
| 2 | etichette construct/ablation e righe sparse ([02b](02b_dataset_generazione_llm.md)); estrazione su d1; H0 → H2c; ablazioni |
| 3 | esperimenti condizionati su d1; estrazione e J0, W0; latenza |
| 4 | fase 2 se serve; controlli, congelamento, test, export |

Prerequisito: il modello del verdetto del piano binario ([../fine_tuning_factcheck/](../fine_tuning_factcheck/00_overview.md)); può procedere in parallelo, la fase H si rifà sul modello finale (per d1 l'estrazione costa meno di un'ora).

## Registro

`finetune/experiments_evidence.md`:

| ID | Modello | Fase | Variazione | F1 frase | Insieme completo | F1 ≥ 10 frasi | F1 > 1.024 token | Neg. accesi | Sufficienza | Δ verdetto | Latenza | Decisione |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| E-LEX | — | — | — | | | | | | | — | | baseline |
| H0 | d1 | 1a | — | | | | | | | 0 | | |
