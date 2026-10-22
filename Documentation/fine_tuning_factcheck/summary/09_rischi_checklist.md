# Rischi e checklist

## 1. Rischi principali

| Rischio | Come si riconosce | Prevenzione | Rimedio |
|---|---|---|---|
| **Bias verso "sì" non corretto** | soglia ottimale ancora > 0,85; "rimozione dell'evidenza" < 80% | negativi "non stabilito" ≥ 40% dei negativi; C2D, `remove_evidence` | più negativi per cancellazione; domanda Q3 con descrizioni esplicite |
| **Sovra-confidenza** | ECE in aumento; probabilità tutte 0/1 | loss sulle lettere, 1 epoca per stadio | Brier, smoothing, β·KL, temperatura rifittata |
| **Stile sintetico imparato** | ottimo su C2D/CG2C, nessun guadagno su test_ood/AggreFact | stadio 2 su testi reali (D2C, SummEdits) | più D2C reale; meno sintetico nello stadio 1 |
| **Contaminazione** | guadagno forte su una sola fonte di AggreFact | D2C solo con documenti non presenti in AggreFact (13-grammi); RAGTruth escluso | togliere la fonte e riportare con e senza |
| **Etichette generate sbagliate** | campione con errori; giudice che scarta > 40% | etichetta dall'operazione + giudice diverso | Brier/smoothing; rivedere i prompt |
| **ANLI difficile = ANLI rumoroso** | molti errori "sicuri" con etichette discutibili | errori sicuri ≤ 30% del sottoinsieme | smoothing su quelle righe; tetto più basso |
| **Chunking che spezza le catene** | "evidenza divisa" fallisce; LFQA/ExpertQA peggiori | chunk grandi per l'LLM (≥ 4k) | `min`×`max` sulle frasi del claim (K2) |
| **Positivi troncati in training** | evidenza tagliata → etichetta sbagliata | i positivi non si spezzano ([03](03_rendering_formato.md) §4) | budget più alto o scarto |
| **Prompt diverso da quello di produzione** | calo appena cambia una parola | domanda unica congelata in `question.json` | normalizzare l'input nel codice chiamante |
| **Lettere invertite** | la loss non scende o sale | test di rendering (A = falso in Winnow, A = sì in JPT) | correggere i target |
| **Margine ridotto ≠ forward completo** | probabilità del training diverse dall'inferenza | test di [05](05_training.md) §6 (softcapping incluso) | applicare la stessa trasformazione ai logit |
| **Conclusione sbagliata sull'architettura** | si sceglie D1 senza sapere se un base farebbe uguale | fase A e controllo C1 obbligatori | — |
| **Specializzazione non dichiarata** | qualcuno usa il modello per altre domande | nome `-fc`, model card | — |
| **Overfitting del dev** | ottimo su `dev_select`, peggio su `dev_confirm` | criteri di adozione ≥ 0,5 punti | conferma su `dev_confirm` |
| **Test guardato troppo presto** | — | test solo dopo `frozen.json` | dichiararlo |
| **Licenze** | — | [08](08_merge_export.md) §7 | ricetta con dati permissivi |

## 2. Checklist per step

### Step 0 — Split
- [ ] Split congelati (riuso di `data/finetune/splits/` se esiste già per il piano generale).
- [ ] `test_long` definito.
- [ ] Decontaminazione contro AggreFact e test_ood; nessun `doc_id` condiviso.

### Step 0a — Approccio
- [ ] Backbone principale e controlli (A2, A3, A5) decisi.
- [ ] Deciso se la distillazione è un requisito.

### Step 1 — Baseline
- [ ] B1 (una domanda) e B2 (tre domande) sul nostro dev; B3 per scegliere la domanda.
- [ ] B4 (modelli base), B5–B7 (FactCG, MiniCheck, Bespoke).
- [ ] Domanda di produzione scritta in `finetune/question.json`.

### Step 2a — Fonti
- [ ] `anli_hard` prodotto solo dal train, con quota di errori sicuri registrata.
- [ ] (Se CG2C-MHQA) builder HotpotQA/MuSiQue decontaminati.

### Step 2b — Generazione
- [ ] Generazione solo dal train; `doc_id` ereditato.
- [ ] Documenti D2C decontaminati contro AggreFact.
- [ ] Tasso di accettazione ≥ 200 esempi per famiglia; 30 letti a mano con < 3 errori.

### Step 2c — Replay
- [ ] Confermato che il modello non deve restare generale (altrimenti → piano generale).
- [ ] (Se P-KL) probabilità del modello di partenza precalcolate.

### Step 2d — Varianti
- [ ] Domanda unica fissata; varianti solo se P-VAR.

### Step 2e — Mix
- [ ] `fc_pilot`, `fc_stage1`, `fc_stage2` prodotti; positivi 45–55%; "non stabilito" ≥ 40% dei negativi; ≥ 20% di righe lunghe nello stadio 2.

### Step 3 — Rendering
- [ ] Prompt identici ai benchmark con la domanda scelta.
- [ ] Lettere e target verificati; positivi lunghi non spezzati.

### Step 4 — Loss
- [ ] Funzioni testate sui casi numerici di [04](04_loss.md).

### Step 5 — Training
- [ ] Test "margine ridotto = forward completo", "passo zero = baseline", "overfit di 64 esempi".
- [ ] Monitoraggio: BAcc, AUC, soglia ottimale, BAcc a 0,5, ECE, solo claim, righe lunghe.
- [ ] Ottimizzatore nuovo allo stadio 2.

### Step 6 — Valutazione
- [ ] Soglia da `dev_select`, confermata; soglia 0,5 riportata; secondaria dichiarata.
- [ ] Comportamentali, inclusi "claim parziale" ed "evidenza divisa".
- [ ] Riferimenti esterni con lo stesso protocollo.

### Step 7 — Esperimenti
- [ ] Fase A completata e interpretata.
- [ ] C1–C3 eseguiti.

### Step 8 — Export
- [ ] Fusione su bf16 verificata; GGUF verificato (Winnow).
- [ ] `decision.json` con domanda unica, soglia, chunking.
- [ ] (Opzionale) testa estratta verificata; studente distillato valutato.
- [ ] `frozen.json` prima del test; model card che dichiara la specializzazione.
