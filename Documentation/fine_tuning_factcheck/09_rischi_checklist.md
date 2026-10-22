# Rischi, rimedi e checklist completa

## 1. Rischi principali

| Rischio | Come si riconosce | Prevenzione | Rimedio |
|---|---|---|---|
| **Prompt diverso da quello di produzione** | il passo zero non riproduce la baseline; la loss non scende | rendering con le funzioni dei benchmark; test di identità ([03](03_rendering_formato.md) §8) | correggere il rendering prima di addestrare |
| **Risposte o `take` invertiti** | la loss sale o resta a ~0,69 | test delle risposte (Winnow A = falso; JPT A = sì; d1-3B gruppi yes/no) e di `take` (Q4 vuole "no" per un claim supportato) | correggere `yes_ids`/`no_ids` o `take` |
| **d1-3B: due implementazioni diverse** | il passo zero del training (classe standard di transformers) non riproduce le P(sì) dell'API `system_one` | test del passo zero ([05](05_training.md) §8) | addestrare con `trust_remote_code=True` |
| **Licenza di d1-3B** | uso commerciale da parte di un'entità sopra 10 M$ di fatturato | LFM Open License v1.0 letta prima di scegliere d1-3B per la produzione | rilasciare il derivato di Winnow (Apache 2.0) |
| **Documenti lunghi** | BAcc in calo sopra ~4.000 caratteri (tutti i modelli; d1-3B 66 sopra 16.000) | stadio 2 con documenti lunghi; test_long; chunking scelto in K1 | più dati lunghi; chunk più piccoli per d1-3B |
| **Margine ridotto ≠ forward completo** | probabilità del training diverse da quelle dell'inferenza | test [04](04_loss.md) §9, softcapping di Winnow incluso | applicare la stessa trasformazione dei logit |
| **Bias verso "sì" non corretto** | soglia ottimale ancora > 0,85; "rimozione dell'evidenza" < 80% | negativi `not_established` ≥ 40% dei negativi; domanda Q3 | più `remove_evidence` e C2D |
| **Sovra-confidenza** | ECE in aumento; probabilità tutte 0/1 | loss sulle lettere, 1 epoca per stadio | smoothing, Brier, β·KL, scala rifittata |
| **Stile sintetico imparato** | ottimo su dati generati, nessun guadagno su test_ood e AggreFact | stadio 2 su testi reali | più D2C reale; meno sintetico nello stadio 1 |
| **Contaminazione** | guadagno forte su una sola fonte di AggreFact | decontaminazione a 13-grammi (Step 0 e famiglie generate); RAGTruth e WiCE esclusi | togliere la fonte e riportare con e senza |
| **Etichette generate sbagliate** | campione letto a mano con errori; giudice che scarta > 40% | etichetta dall'operazione + giudice diverso | rivedere i prompt; smoothing o Brier |
| **ANLI difficile rumoroso** | molti "errori sicuri" con etichette discutibili | lettura di 50 righe | smoothing su `anli_hard`; tetto più basso |
| **Positivi troncati** | evidenza tagliata → etichetta 1 sbagliata | positivi oltre il budget scartati ([03](03_rendering_formato.md) §7) | budget più alto |
| **Scorciatoie** | P bassa anche senza documento per claim con "not"; test "solo claim" alto | fonti diverse; tetti su FEVER | D-NEG; più negativi per cancellazione |
| **Overfitting del dev** | ottimo su dev_select, peggio su dev_confirm | criterio di adozione ≥ 0,5 punti fissato prima | conferma su dev_confirm, meno varianti provate |
| **Fughe fra split** | dev o test con documenti visti in training | split congelati prima di tutto; doc_id disgiunti; generazione solo dal train | rifare gli split e rigenerare |
| **AggreFact usato per decidere** | scelte fatte guardando il suo dev o test | protocollo dello Step 0 | dichiararlo: il risultato non è più zero-shot |
| **Test guardato troppo presto** | — | test solo dopo `frozen.json` | dichiararlo nel report |
| **Conclusione sbagliata sull'approccio** | si sceglie il decision model senza sapere se un base farebbe uguale | fase A e controllo C1 obbligatori | — |
| **Specializzazione non dichiarata** | qualcuno usa il modello per altre domande | nome `-fc`, model card | — |
| **Versioni delle librerie** | peft incompatibile con transformers 5.14 | ambiente di training separato se serve | clonare l'ambiente |
| **Licenze** | — | [08](08_merge_export.md) §9 | ricetta senza dati non commerciali |

## 2. Checklist per step

### Step 0 — Split
- [ ] `build_pairs.py` eseguito con `--splits train dev test` dove serve.
- [ ] Builder di test_ood scritti dopo aver controllato `ds.features`.
- [ ] `make_splits.py` eseguito; scarti per motivo registrati.
- [ ] Verifica doc_id disgiunti superata.
- [ ] test_long con ≥ 300 righe.
- [ ] `manifest.json` scritto.

### Step 1 — Baseline
- [ ] `score.py` e `calibrate.py` scritti; tabella dalle cache riprodotta.
- [ ] B1–B5 su dev_select e dev_confirm; test una volta.
- [ ] Configurazione di partenza (scala, soglia) registrata.

### Step 2a — Fonti
- [ ] `meta.neg_type` aggiunto dove ricavabile.
- [ ] `anli_hard.jsonl` prodotto solo dal train; 50 righe `wrong` lette.
- [ ] Seed HotpotQA/MuSiQue prodotti e decontaminati.

### Step 2b — Generazione
- [ ] Server del generatore e del giudice avviati; client locale aggiunto.
- [ ] Per ogni famiglia: ≥ 200 esempi con tasso di accettazione misurato; 30 letti a mano con < 3 errori.
- [ ] Documenti D2C decontaminati contro AggreFact.
- [ ] `collect.py` eseguito; `data/finetune/raw/<famiglia>.jsonl` con `doc_id` ereditato.

### Step 2c — Replay
- [ ] Confermato che non serve; p0 calcolata solo se si fa P-KL.

### Step 2d — Domanda
- [ ] Q1–Q4 misurate su d1-3B, JPT e Winnow (dev_select); `finetune/question.json` scritto, con `take`.

### Step 2e — Mix
- [ ] `make_mix.py` esteso (`--gen`, `--no-val`, meta conservato).
- [ ] `fc_pilot`, `fc_stage1`, `fc_stage2` prodotti; controlli di §5 superati.

### Step 3 — Rendering
- [ ] Test di identità (per d1-3B: contro l'API `system_one`), risposte, `take`, frasi, budget superati.
- [ ] Statistiche di lunghezza registrate.

### Step 4 — Loss
- [ ] `test_losses.py` passa; margine ridotto = forward completo.

### Step 5 — Training
- [ ] peft installato; test passo zero e overfit superati.
- [ ] Configurazione e seed salvati.
- [ ] Monitoraggio a ogni checkpoint (BAcc, soglia ottimale, BAcc a 0,5, ECE, solo claim).
- [ ] Checkpoint scelto con il criterio di [05](05_training.md) §10.

### Step 6 — Valutazione
- [ ] Comportamentali generati e salvati.
- [ ] Report con intervalli di confidenza e riferimenti esterni.

### Step 7 — Esperimenti
- [ ] Fasi 0, 1, A, 2, 3 registrate; C1–C3 eseguiti.

### Step 8 — Export
- [ ] Fusione su bf16 verificata.
- [ ] `decision.json` e `frozen.json` scritti prima del test.
- [ ] Test eseguito una volta; report e model card.
