# Step 7 — Il piano degli esperimenti e come decidere

## Obiettivo

Arrivare alla configurazione finale con poche corse, **una variazione alla volta**, decidendo con criteri fissati prima. Si sperimenta su **KnowLine-4B** (4B, causale, lettura dalla testa del vocabolario: corse brevi, anche sul portatile in bf16); ciò che vince si ripete su elephant-1, poi su tutti gli altri decision model e sui due backbone (fase B, l'esperimento B-ctrl del paper).

## 1. Regole

1. Ogni esperimento ha un **ID**, una **configurazione di riferimento** e **una sola variazione**.
2. Si giudica su dev_select: BAcc macro, AUC macro, BAcc a 0,5, soglia ottimale, ECE; più i test comportamentali "solo claim" e "rimozione dell'evidenza".
3. **Si adotta** una variazione se migliora la BAcc macro su dev_select di **≥ 0,5 punti** e non peggiora l'ECE di più di 0,02. Sotto 0,5 punti è rumore fra seed.
4. A parità, si tiene la variante più semplice o più economica.
5. Ogni corsa si registra (§9).
6. I test si guardano solo alla fine.

## 2. Fase 0 — Baseline (nessun training)

| ID | Cosa | Uscita |
|---|---|---|
| B1 | i 10 decision model, domande Q1, 1 − Q2, Q4 (su AggreFact: dai run dello studio; sul nostro dev: `benchmark.py`) | punto di partenza |
| B2 | tre domande (QF1), regola `min` | quanto recuperare |
| B4 | Qwen3.5-9B e Qwen3.5-4B, prompt neutro, letture R0 e R1 | partenza dei controlli B-ctrl |
| B5–B7 | FactCG, MiniCheck-FT5, Bespoke-7B | riferimenti |

Procedura: [01_baseline.md](01_baseline.md); scelta della domanda: [02d](02d_varianti_domanda.md).

## 3. Fase 1 — Pilota (KnowLine-4B, `fc_pilot`, uno stadio)

Riferimento **P0**: KnowLine-4B, LoRA r16, lr 2e-5, 1 epoca, CE sulle due opzioni, 8.192 token, domanda scelta in [02d](02d_varianti_domanda.md).

| ID | Variazione rispetto a P0 | Domanda | Come si legge |
|---|---|---|---|
| **P0** | — | il fine-tuning funziona? | se non batte B1 su dev_select di ≥ 0,5: **fermarsi** e ricontrollare rendering, opzioni, target (test di [03](03_rendering_formato.md) e [05](05_training.md)) |
| P1a / P1b | lr 5e-5 / 1e-5 | sensibilità | tenere il migliore |
| P2 | `--loss brier` | calibrazione con dati rumorosi | adottare se ECE migliora senza perdere BAcc |
| P-LS | `--smoothing 0.05` (solo fonti generate e `anli_hard`) | rumore delle etichette | idem |
| P-KL | `--beta 0.05` | serve l'ancoraggio? | adottare solo se P0 peggiora ECE o fonti piccole di dev_select |
| P3 | SFT classico: CE su tutto il vocabolario al token della risposta | quanto vale la loss sulle sole opzioni | atteso peggiore in ECE; conferma della scelta |
| P-VAR | 10% di varianti di domanda ([02d](02d_varianti_domanda.md) §5) | regolarizzazione | tenere solo se migliora |

## 4. Fase A — Architettura e punto di partenza (stessi dati: `fc_pilot`)

| ID | Partenza | Uscita | Domanda |
|---|---|---|---|
| A1 | KnowLine-4B | opzioni A/B | riferimento (= P0 o migliore della fase 1) |
| A2 | Qwen3.5-4B (backbone) | lettere A/B, prompt neutro, ordine casuale | **pilota di B-ctrl**: vale partire da un decision model? |
| A3 | Qwen3.5-4B | testa nuova `Linear(2560 → 2)` sullo stato nella posizione di lettura (lr della testa 1e-4) | cambiare la testa aiuta? |
| A4 | KnowLine-4B | testa nuova **inizializzata** con w_sì − w_no | verifica pratica dell'equivalenza con A1 |
| A5 | DeBERTa-v3-large stile FactCG | testa nuova, chunk 550 parole | quanto vale la taglia |

| Esito | Conclusione |
|---|---|
| A1 > A2 di ≥ 0,5 | il decision model aiuta (prima evidenza per B-ctrl) |
| A1 ≈ A2 | il vantaggio è nei dati: coerente con l'audit della letteratura; si conferma sulle altre coppie (fase B) |
| A3 < A2 | non cambiare la testa |
| A4 ≈ A1 | equivalenza confermata; la testa estratta si può usare per servire ([08](08_merge_export.md) §4) |
| A5 ≥ A1 − 1 | l'encoder è quasi equivalente: considerare seriamente la distillazione |

## 5. Fase 2 — Dati (KnowLine-4B, configurazione migliore delle fasi 1 e A)

| ID | Variazione | Domanda |
|---|---|---|
| D-MC | solo MiniCheck + ANLI difficile (≈ ricetta di Bespoke) | quanto vale la ricetta pubblica da sola |
| D-HARD | `anli_hard` contro ANLI casuale (stesso numero) | la selezione degli esempi difficili serve? |
| D-D2C | + D2C esteso | migliorano i riassunti (CNN, XSum, MediaS, MeetB nel test, SummEdits nel dev)? |
| D-C2D | + C2D esteso | migliora il "claim parziale"? |
| D-CG | + CG2C-MHQA + CG2C-Doc | migliora il multi-hop (HoVer, rimozione dell'evidenza)? |
| D-RE | + `remove_evidence` su documenti lunghi | la soglia ottimale scende? |
| D-CUR | due stadi contro uno stadio con gli stessi dati | il curriculum serve? |
| D-SCALE | 25k / 50k / 100k righe | conviene generare di più? |
| D-FKG | + FactKG trasformato in prosa | aiuta? |
| D-NEG | FEVER senza claim con negazioni esplicite | scorciatoie |

Ordine consigliato: D-MC, D-D2C, D-HARD, D-CUR, poi gli altri. D-SCALE si fa sul mix migliore.

**Come leggere un esperimento sui dati**: confrontare la BAcc per fonte, non solo la media. Se aggiungere D2C fa salire SummEdits di 3 punti e scendere VitaminC di 1, è un buon segno (il dominio debole migliora).

## 6. Fase 3 — elephant-1 e lunghezza

| ID | Variazione | Domanda |
|---|---|---|
| E0 | configurazione migliore di KnowLine-4B su elephant-1 | il risultato si trasferisce a 9B? quanto guadagna il modello più grande? |
| E-BB9 | la stessa ricetta su Qwen3.5-9B | **coppia principale di B-ctrl** |
| KL-FIN | KnowLine-4B con la ricetta finale e 3 seed | se arriva entro ~1 punto da elephant-1 su dev_select e test_ood, è il candidato da rilasciare (costo minore) |
| F3 | rank 8 / 32 / 64 | capacità dell'adapter |
| K1 | chunk in inferenza C = 8.192 / 16.384 / 32.768 token | scelta di C su dev_select (righe lunghe) |
| K2 | claim lunghi divisi in frasi: `min` sulle frasi del `max` sui chunk | risposte lunghe |

## 7. Fase 4 — Facoltativa: loss contrastiva

| ID | Variazione |
|---|---|
| X1 | + `L_pair` sulle coppie C2D / CG2C / VitaminC (μ = 0,3) |
| X2 | μ = 0,1 e coppie `remove_evidence` / `contradict_claim` |

## 8. Fase B — B-ctrl su tutte le ricette

La ricetta finale (dati, loss, iperparametri), **identica**, su ogni decision model e sul suo backbone. Domanda del paper: a parità di dati, partire da un decision model è meglio che partire dal backbone? E il vantaggio dipende dalla ricetta?

| Gruppo | Decision model | Backbone |
|---|---|---|
| 9B | elephant-1 (E0), StartLux-9B, d3-flash, OneJev1.6-9B, Nimble-9B-v3, Clef-Flash | Qwen3.5-9B (E-BB9) |
| 4B | KnowLine-4B (KL-FIN), StartLux-4B, pplx-decider-flash, Jeffan-4B | Qwen3.5-4B (A2 con la ricetta finale) |

Per ogni modello: test di identità e passo zero prima di partire ([05](05_training.md) §8), poi stadi 1 e 2.

**Curva sui dati** (sulle due coppie principali, elephant-1/Qwen3.5-9B e KnowLine-4B/Qwen3.5-4B): stadio 1 con 0, 2k, 8k, 32k righe. Il vantaggio del decision model, se c'è, dovrebbe essere massimo con pochi dati.

| Esito | Lettura |
|---|---|
| decision model ≫ backbone, soprattutto con pochi dati | l'inizializzazione da decision model aiuta: consiglio pratico a chi costruisce checker |
| decision model ≈ backbone | il valore è nei dati |
| decision model < backbone | il training da decision model irrigidisce il modello: risultato da controllare con un secondo seed e su un'altra coppia |

Ordine consigliato: prima i modelli con lettura dalla testa del vocabolario e codice in transformers (Jeffan-4B, d3-flash, OneJev, Nimble), poi quelli con maschera da ricostruire (StartLux) o testa propria (pplx, Clef).

## 9. Controlli finali (obbligatori)

| ID | Cosa |
|---|---|
| C2 | 3 seed (13, 14, 15) della configurazione finale su KnowLine-4B ed elephant-1: media e deviazione standard |
| C3 | FactCG, MiniCheck-FT5, Bespoke-MiniCheck-7B sulla stessa suite |

## 10. Registro: `finetune/experiments_fc.md`

Una riga per corsa:

| ID | Modello | Mix | Variazione | BAcc dev | AUC dev | BAcc@0,5 | Soglia ott. | ECE | Solo claim | Rimozione evid. | Decisione |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| B1 | KnowLine-4B | — | domanda scelta | | | | | | | | baseline |
| P0 | KnowLine-4B | fc_pilot | — | | | | | | | | |

## Criteri di completamento

- Configurazione finale scelta con le regole di §1, registrata, confermata su dev_confirm.
- Fase B eseguita almeno sulle due coppie principali; C2 e C3 eseguiti.
