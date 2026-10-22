# Step 7 — Il piano degli esperimenti e come decidere

## Obiettivo

Arrivare alla configurazione finale con poche corse, **una variazione alla volta**, decidendo con criteri fissati prima. Si sperimenta su **d1-3B** (3B: corse brevi, anche sul portatile in bf16); ciò che vince si ripete su Winnow-12B, e su JPT-9B come confronto. Fino all'arrivo dei run di ottobre il modello per iterare era JPT; i risultati ([../risultati_aggrefact_2026-10.md](../risultati_aggrefact_2026-10.md)) mostrano che d1-3B ha quasi la stessa AUC a un quarto del costo.

## 1. Regole

1. Ogni esperimento ha un **ID**, una **configurazione di riferimento** e **una sola variazione**.
2. Si giudica su dev_select: BAcc macro, AUC macro, BAcc a 0,5, soglia ottimale, ECE; più i test comportamentali "solo claim" e "rimozione dell'evidenza".
3. **Si adotta** una variazione se migliora la BAcc macro su dev_select di **≥ 0,5 punti** e non peggiora l'ECE di più di 0,02. Sotto 0,5 punti è rumore fra seed.
4. A parità, si tiene la variante più semplice o più economica.
5. Ogni corsa si registra (§9).
6. I test si guardano solo alla fine.

## 2. Fase 0 — Baseline (nessun training)

| ID | Cosa | Costo | Uscita |
|---|---|---|---|
| B1 | d1-3B, JPT e Winnow, domande Q1 e Q4 | 0 su AggreFact (già nei run di ottobre); ~1 h sul nostro dev | punto di partenza |
| B2 | tre domande (stessa cache di B1 se si renderizzano tutte e tre) | 0 | quanto recuperare |
| B3 | domande Q2, Q3 | ~2 h | **domanda di produzione** |
| B4 | Qwen3.5-9B base, Gemma-4-12B-it | ~2 h | partenza del controllo |
| B5–B7 | FactCG, MiniCheck-FT5, Bespoke-7B | ~3 h | riferimenti |

Procedura: [01_baseline.md](01_baseline.md).

## 3. Fase 1 — Pilota (d1-3B, `fc_pilot`, uno stadio)

Riferimento **P0**: d1-3B, LoRA r16, lr 2e-5, 1 epoca, CE sulle due risposte, 8.192 token, domanda scelta in B3 (favorita: Q4).

| ID | Variazione rispetto a P0 | Domanda | Come si legge |
|---|---|---|---|
| **P0** | — | il fine-tuning funziona? | se non batte B1 su dev_select di ≥ 0,5: **fermarsi** e ricontrollare rendering, lettere, target (test di [03](03_rendering_formato.md) e [05](05_training.md)) |
| P1a / P1b | lr 5e-5 / 1e-5 | sensibilità | tenere il migliore |
| P2 | `--loss brier` | calibrazione con dati rumorosi | adottare se ECE migliora senza perdere BAcc |
| P-LS | `--smoothing 0.05` (solo fonti generate e `anli_hard`) | rumore delle etichette | idem |
| P-KL | `--beta 0.05` | serve l'ancoraggio? | adottare solo se P0 peggiora ECE o fonti piccole di dev_select |
| P3 | SFT classico: CE su tutto il vocabolario al token della risposta | quanto vale la loss sulle sole risposte | atteso peggiore in ECE; conferma della scelta |
| P-VAR | 10% di varianti di domanda ([02d](02d_varianti_domanda.md) §5) | regolarizzazione | tenere solo se migliora |

Costo: ~7 corse × ~1–1,5 h (L40S) o ~2–3 h (portatile) con d1-3B; con JPT sarebbero ~4 h e ~8 h.

## 4. Fase A — Architettura (stessi dati: `fc_pilot`)

È la fase che conferma con numeri nostri la scelta di [00a](00a_scelta_approccio.md).

| ID | Partenza | Uscita | Domanda |
|---|---|---|---|
| A1 | d1-3B | yes/no | riferimento (= P0 o migliore della fase 1) |
| A1b | JPT-9B | lettere | stesso training su un decision model più grande |
| A2 | Qwen3.5-9B base | lettere, stesso prompt | vale partire da un decision model? |
| A3 | Qwen3.5-9B base | testa nuova `Linear(4096 → 2)` sullo stato in `answer_pos` (addestrata insieme alla LoRA, lr della testa 1e-4) | cambiare la testa aiuta? |
| A4 | d1-3B | testa nuova **inizializzata** con w_sì − w_no | verifica pratica dell'equivalenza con A1 |
| A5 | DeBERTa-v3-large stile FactCG | testa nuova, chunk 550 parole | quanto vale la taglia |

| Esito | Conclusione |
|---|---|
| A1 > A2 di ≥ 0,5 | il decision model aiuta: si prosegue con d1-3B / Winnow |
| A1 ≈ A2 | il vantaggio è nei dati; il backbone si sceglie per licenza e costo (Qwen3.5 base è Apache 2.0, JPT è CC BY-NC, d1-3B e LFM2.5 hanno la LFM Open License) |
| A3 < A2 | non cambiare la testa |
| A4 ≈ A1 | equivalenza confermata; la testa estratta si può usare per servire ([08](08_merge_export.md) §4) |
| A5 ≥ A1 − 1 | l'encoder è quasi equivalente: considerare seriamente la distillazione |

## 5. Fase 2 — Dati (d1-3B, configurazione migliore delle fasi 1 e A)

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

## 6. Fase 3 — Winnow e lunghezza

| ID | Variazione | Domanda |
|---|---|---|
| W0 | configurazione migliore di d1-3B su Winnow | il risultato si trasferisce? quanto guadagna un modello 4 volte più grande? |
| J0 | idem su JPT-9B | terzo punto di confronto |
| D1-FIN | d1-3B con la ricetta finale e 3 seed | se arriva entro ~1 punto da Winnow su dev_select e test_ood, è il candidato da rilasciare (costo 4–5 volte più basso) |
| F2 | 8.192 token in training invece di 4.096 | documenti lunghi |
| F3 | rank 8 / 32 / 64 | capacità dell'adapter |
| K1 | chunk in inferenza C = 8.192 / 16.384 / 32.768 token | scelta di C su dev_select (righe lunghe) |
| K2 | claim lunghi divisi in frasi: `min` sulle frasi del `max` sui chunk | risposte lunghe |

## 7. Fase 4 — Facoltativa: loss contrastiva

| ID | Variazione |
|---|---|
| X1 | + `L_pair` sulle coppie C2D / CG2C / VitaminC (μ = 0,3) |
| X2 | μ = 0,1 e coppie `remove_evidence` / `contradict_claim` |

## 8. Controlli finali (obbligatori)

| ID | Cosa |
|---|---|
| C1 | la ricetta finale su LFM2.5-VL-3B (controllo di d1-3B), Gemma-4-12B-it (controllo di Winnow) e Qwen3.5-9B base (controllo di JPT) |
| C2 | 3 seed (13, 14, 15) della configurazione finale: media e deviazione standard |
| C3 | FactCG, MiniCheck-FT5, Bespoke-MiniCheck-7B sulla stessa suite |

## 9. Registro: `finetune/experiments_fc.md`

Una riga per corsa:

| ID | Modello | Mix | Variazione | BAcc dev | AUC dev | BAcc@0,5 | Soglia ott. | ECE | Solo claim | Rimozione evid. | Durata | Decisione |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| B1 | d1-3B | — | domanda Q4 | | | | | | | | 0 | baseline |
| P0 | d1-3B | fc_pilot | — | | | | | | | | | |

## 10. Calendario indicativo

| Settimana | Lavoro |
|---|---|
| 1 | Step 0 (split), fase 0, `render.py`, `losses.py`, `train.py`, ANLI difficile, `fc_pilot`; avvio della generazione D2C su un'altra GPU |
| 2 | fase 1 e fase A su d1-3B (corse da poche ore: si possono fare più esperimenti al giorno) |
| 3 | fine generazione (C2D, CG2C), fase 2 |
| 4 | fase 3 su Winnow e JPT; D1-FIN |
| 5 | C1–C3, congelamento, test, export |

## Criteri di completamento

- Configurazione finale scelta con le regole di §1, registrata, confermata su dev_confirm.
- C1–C3 eseguiti.
