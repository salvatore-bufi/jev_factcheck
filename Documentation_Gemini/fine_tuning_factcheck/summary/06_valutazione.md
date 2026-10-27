# Step 6 — Valutazione di un fact checker binario specializzato

## Obiettivo

Misurare se il modello fine-tunato è un fact checker binario migliore della baseline e dei riferimenti esterni (FactCG, MiniCheck, Bespoke), su domini visti e non visti, su documenti lunghi, con probabilità utilizzabili con una soglia fissa e a un costo noto. Rispetto al piano generale ([../fine_tuning/06_valutazione.md](../fine_tuning/06_valutazione.md)) sparisce l'oblio e aumentano calibrazione, documenti lunghi e costo.

## 1. La suite

| Blocco | Contenuto | Cosa misura | Quando | Usato per decidere? |
|---|---|---|---|---|
| dev_select | ~60% del dev delle fonti di training | soglia, temperatura, checkpoint, esperimenti, dimensione dei chunk | sempre | **sì** |
| dev_confirm | ~40% del dev | conferma | una volta per fase | solo conferma |
| test_id | test delle fonti di training | apprendimento | baseline e fine | no |
| test_ood | SciFact, Climate-FEVER, HealthVer/COVID-Fact, HaluBench, HalluMix, LLM-Oasis | generalizzazione | baseline e fine | no |
| test_long | righe di test_id e test_ood > 4.000 token | chunking, documenti lunghi | baseline e fine | no |
| AggreFact test | 11 fonti | confronto con la leaderboard, zero-shot | baseline e fine | no |
| Comportamentali | generati da `dev_select` | il modello guarda il documento? | ogni esperimento | come vincolo |
| Costo | 1.000 righe miste | latenza e throughput | modello finale | no |

## 2. Metriche

| Metrica | Perché |
|---|---|
| BAcc per fonte, media per blocco, media dei blocchi | metrica principale, come la leaderboard |
| AUC | ordinamento, indipendente dalla soglia |
| **BAcc a soglia 0,5** | se è vicina a quella a soglia fittata, il modello è usabile senza dev (obiettivo MiniCheck) |
| ECE (10 intervalli), Brier | calibrazione |
| Soglia ottimale su `dev_select` | segnale diretto del bias: baseline 0,90–0,95 |
| Recall dei supportati e dei non supportati, separatamente | dove sbaglia: "sì" troppo facili o "no" troppo facili |
| BAcc per tipo di negativo (`contradicted` / `not_established`) | i "non stabilito" sono il punto debole atteso |
| Latenza p50/p95 per claim, claim/s | costo d'uso |

Incertezza: bootstrap per documento (`document_cluster_bootstrap` in [metrics.py](../../metrics.py)); differenze sotto ~0,5 punti di media non conclusive.

## 3. Soglia

1. Temperatura (minimo log-loss) e soglia (BAcc macro, griglia 0,01–0,99) su `dev_select`; conferma su `dev_confirm`; applicate a tutti i test.
2. Riportare **anche** la soglia fissa 0,5.
3. Soglia secondaria, dichiarata, fittata sul dev di AggreFact: solo per confronto con i numeri esistenti.
4. Niente soglie per fonte.

## 4. Test comportamentali

| Test | Trasformazione | Atteso | Obiettivo |
|---|---|---|---|
| Solo claim | documento vuoto | non supportato | P media < 0,15 |
| Rimozione dell'evidenza | righe supportate con evidenza nota (FEVER, HoVer, C2D): tolte le frasi gold | la probabilità scende sotto soglia | > 80% delle righe (baseline da misurare: con soglia 0,95 il modello di partenza dice spesso "sì" comunque) |
| Claim parziale | claim supportato + un fatto in più non presente nel documento (da C2D: power set) | non supportato | > 80% |
| Distrattori | paragrafi irrilevanti aggiunti | decisione invariata | > 95% |
| Posizione | evidenza all'inizio, a metà, alla fine di un documento lungo | decisione invariata | differenza < 2 punti |
| Evidenza divisa fra chunk | documento lungo con le due frasi necessarie in chunk diversi | supportato se C è abbastanza grande | misura del limite della regola `max` |
| Coppie minime | numero/data/entità cambiati | decisione invertita | migliore della baseline |

"Claim parziale" ed "evidenza divisa" sono nuovi rispetto al piano generale: misurano proprio ciò che C2D/CG2C dovrebbero insegnare.

## 5. Confronto con i riferimenti

Tutti con lo stesso protocollo (soglia dal nostro dev, più soglia 0,5):

| Modello | Taglia | Ruolo |
|---|---:|---|
| Baseline a una domanda (Winnow, JPT) | 12B / 9B | punto di partenza |
| Baseline a tre domande | 12B / 9B | quanto si è recuperato della regola di Jev |
| Modello base fine-tunato (C1) | 12B / 9B | valore dell'inizializzazione |
| Encoder E1 fine-tunato sugli stessi dati | 0,4B | valore della taglia |
| FactCG-DeBERTa-v3-L, MiniCheck-FT5 | 0,4B / 0,8B | riferimenti pubblici |
| Bespoke-MiniCheck-7B | 7B | miglior modello aperto con la stessa ricetta di dati |

## 6. Report finale (formato)

| Blocco | Metrica | Baseline 1D | Baseline 3D | Fine-tunato | Δ vs 1D | IC 95% |
|---|---|---:|---:|---:|---:|---|
| test_id | BAcc | | | | | |
| test_ood | BAcc | | | | | |
| test_long | BAcc | | | | | |
| AggreFact (zero-shot) | BAcc / AUC | | | | | |
| **Media dei blocchi** | BAcc | | | | | |
| AggreFact, soglia 0,5 | BAcc | | | | | |
| AggreFact (secondaria) | BAcc | 76,69 / 76,20 | 77,42 / 76,71 | | | |
| Calibrazione | ECE, soglia ottimale | | | | | |
| Comportamentali | solo claim / rimozione / parziale | | | | | |
| Costo | ms per claim | | | | | |

Più le tabelle per fonte e la tabella AggreFact nel formato della leaderboard.

## 7. Codice

Esiste: benchmark dei modelli, `calibrate_transfer.py`, `metrics.py`, `FactCG/benchmark.py`. Da scrivere: lettura degli split dello Step 0 e opzione `--model-dir` per i pesi fusi; supporto della domanda singola e di `question.json` nei benchmark; chunking per i decision model (oggi i benchmark troncano a `max_doc_chars`); `finetune/behavioral.py` con i test di §4; script `support_simple`-only (il calcolo di [01](01_baseline.md) §1, oggi fuori dalla repo).
