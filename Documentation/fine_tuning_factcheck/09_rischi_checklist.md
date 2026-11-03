# Rischi, rimedi e checklist completa

## 1. Rischi principali

| Rischio | Come si riconosce | Prevenzione | Rimedio |
|---|---|---|---|
| **Prompt diverso da quello di produzione** | il passo zero non riproduce la baseline; la loss non scende | rendering con il codice degli autori, come negli adapter; test di identità ([03](03_rendering_formato.md) §8) | correggere il rendering prima di addestrare |
| **Maschera bidirezionale non applicata in training** | solo per elephant-1, StartLux, d3-flash, pplx, Jeffan: il passo zero non riproduce la baseline, e non dà errori | test della maschera ([03](03_rendering_formato.md) §8) | usare il forward degli autori o ricostruire la maschera nei soli layer pieni |
| **Risposte o `take` invertiti** | la loss sale o resta a ~0,69 | test delle opzioni (KnowLine, elephant-1: A = sì; d3-flash, Jeffan, pplx: A = no) e di `take` (Q2 vuole "no" per un claim supportato) | correggere `yes_ids`/`no_ids` o `take` |
| **StartLux: runtime solo vLLM** | gli autori lo eseguono solo con vLLM e un plugin: in training la maschera va ricostruita in transformers | test di identità contro il server degli autori | ricostruire la maschera; se non coincide, StartLux esce da B-ctrl e lo si dichiara |
| **Pesi fusi non caricabili dal runtime degli autori** | l'adapter del benchmark fallisce o dà probabilità diverse sulla cartella fusa | cartella fusa = copia dell'originale con i soli tensori cambiati ([08](08_merge_export.md) §1) | controllare config e file propri (`modeling_overrides`, `decision_config.json`, teste) |
| **Licenze non commerciali** | StartLux 9B/4B e Nimble sono CC BY-NC 4.0: anche i derivati | scegliere il modello da rilasciare fra quelli MIT/Apache 2.0 | rilasciare il derivato di elephant-1 (MIT) o KnowLine-4B (Apache 2.0) |
| **Documenti lunghi** | BAcc in calo sui documenti lunghi (misurato nello studio con l'esperimento E4f) | stadio 2 con documenti lunghi; test_long; chunking scelto in K1 | più dati lunghi; chunk più piccoli |
| **Margine ridotto ≠ forward completo** | probabilità del training diverse da quelle dell'inferenza | test [04](04_loss.md) §9 (per pplx e Clef: la loro testa) | applicare la stessa trasformazione dei logit |
| **Bias verso "sì" non corretto** | soglia ottimale ancora > 0,85; "rimozione dell'evidenza" < 80% | negativi `not_established` ≥ 40% dei negativi; domanda Q3 | più `remove_evidence` e C2D |
| **Sovra-confidenza** | ECE in aumento; probabilità tutte 0/1 | loss sulle lettere, 1 epoca per stadio | smoothing, Brier, β·KL, scala rifittata |
| **Stile sintetico imparato** | ottimo su dati generati, nessun guadagno su test_ood e AggreFact | stadio 2 su testi reali | più D2C reale; meno sintetico nello stadio 1 |
| **Contaminazione** | guadagno forte su una sola fonte di AggreFact | decontaminazione a 13-grammi (Step 0 e famiglie generate); RAGTruth e WiCE esclusi; alcuni decision model potrebbero aver visto RAGTruth (Decision Index pubblico) | riportare AggreFact con e senza RAGTruth |
| **Etichette generate sbagliate** | campione letto a mano con errori; giudice che scarta > 40% | etichetta dall'operazione + giudice diverso | rivedere i prompt; smoothing o Brier |
| **ANLI difficile rumoroso** | molti "errori sicuri" con etichette discutibili | lettura di 50 righe | smoothing su `anli_hard`; tetto più basso |
| **Positivi troncati** | evidenza tagliata → etichetta 1 sbagliata | positivi oltre il budget scartati ([03](03_rendering_formato.md) §7) | budget più alto |
| **Scorciatoie** | P bassa anche senza documento per claim con "not"; test "solo claim" alto | fonti diverse; tetti su FEVER | D-NEG; più negativi per cancellazione |
| **Overfitting del dev** | ottimo su dev_select, peggio su dev_confirm | criterio di adozione ≥ 0,5 punti fissato prima | conferma su dev_confirm, meno varianti provate |
| **Fughe fra split** | dev o test con documenti visti in training | split congelati prima di tutto; doc_id disgiunti; generazione solo dal train | rifare gli split e rigenerare |
| **AggreFact usato per decidere** | scelte fatte guardando il suo dev o test | protocollo dello Step 0 | dichiararlo: il risultato non è più zero-shot |
| **Test guardato troppo presto** | — | test solo dopo `frozen.json` | dichiararlo nel report |
| **Conclusione sbagliata sull'approccio** | si sceglie il decision model senza sapere se il backbone farebbe uguale | fase A e fase B (B-ctrl) | — |
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
- [ ] Split esportati come dataset locale per `benchmark.py`; `calibrate.py` scritto; tabella dai run dello studio riprodotta.
- [ ] B1, B2, B4, B5 su dev_select e dev_confirm per tutti i modelli; test una volta.
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
- [ ] Q1, Q2, Q4 misurate sui 10 decision model (dev_select); `finetune/question.json` scritto, con `take`.

### Step 2e — Mix
- [ ] `make_mix.py` esteso (`--gen`, `--no-val`, meta conservato).
- [ ] `fc_pilot`, `fc_stage1`, `fc_stage2` prodotti; controlli di §5 superati.

### Step 3 — Rendering
- [ ] Test di identità con l'adapter del benchmark, maschera (modelli bidirezionali), opzioni, `take`, frasi, budget superati.
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
- [ ] Fasi 0, 1, A, 2, 3 registrate; fase B (B-ctrl) almeno sulle due coppie principali; C2–C3 eseguiti.

### Step 8 — Export
- [ ] Fusione su bf16 verificata con l'adapter del benchmark sulla cartella fusa.
- [ ] `decision.json` e `frozen.json` scritti prima del test.
- [ ] Test eseguito una volta; report e model card.
