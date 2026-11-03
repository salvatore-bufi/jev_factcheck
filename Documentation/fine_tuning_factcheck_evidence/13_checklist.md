# Checklist

## 1. Per passo

### Passo 1 — Ambiente e modelli
- [ ] `flash-linear-attention` e `causal-conv1d` installati e importabili sulla macchina GPU (JPT).
- [ ] `evidence/models.py` carica d1-3B, JPT-9B, Winnow-12B con `attn_implementation="sdpa_capture"`.
- [ ] Self-test superato per i tre modelli: identità del verdetto, layer, riga di attenzione contro "eager", memoria a 8k e 16k token.

### Passo 2 — Righe dalle fonti
- [ ] `sentences.py` scritto, con hash registrato.
- [ ] Indice sqlite delle pagine FEVER; FEVER detokenizzato (train, paper_dev, paper_test).
- [ ] `hotpot_qa` (distractor) scaricato; HoVer con distrattori e gold allineate; controllo su 100 pagine.
- [ ] SciFact (validation) e WiCE (test) come righe di test.
- [ ] C2D e D2C segmentati.
- [ ] Gold mappate su una sola frase nostra ≥ 90% per fonte; 30 righe per fonte lette a mano.

### Passo 3 — Dati generati
- [ ] C2D con posizioni, varianti negative e sotto-claim; CG2C con indice di frase per tripla.
- [ ] Ablazione con d1: conferma del giudice ≥ 80%, accordo con JPT ≥ 0,7.
- [ ] Righe sparse (estremi, distrattori, oltre 1.024 e 4.096 token) e negativi sparsi; nessun riferimento rotto su 30 lette.

### Passo 4 — Split e mix
- [ ] Blocchi `dev_evid_*`, `test_evid_*` congelati con `manifest.json`; distribuzioni delle distanze salvate.
- [ ] `ev_head` con tutti i vincoli di [04](04_split_e_mix.md) §3 rispettati.

### Passo 5 — Formato
- [ ] B-SAND e B-FT-SAND misurati; preambolo scelto; verdetto per modello deciso; `question.json` aggiornato.

### Passo 6 — Rendering ed estrazione
- [ ] `escape_with_map` identico a `json.dumps` / `safe_data` su 1.000 documenti.
- [ ] Test di posizione (200 su 200), escaping, identità del prompt, verdetto dalle feature.
- [ ] Feature di d1 per `ev_head` e blocchi dev, con `meta.json`.

### Passo 7 — Baseline
- [ ] E-RAND, E-LEX, E-EMB, E-ATT (preambolo e nativo) su `dev_evid_select`; E-ABL su 300 righe per fonte e per modello.

### Passo 8 — Testa e loss
- [ ] Test numerici delle loss superati; test della testa (forme, maschere, righe `none`, una sola frase).

### Passo 9 — Training
- [ ] Overfit di 64 righe; H0 riproducibile.
- [ ] Regola di selezione (`τ_e` o δ) fissata sul dev e confermata.
- [ ] (Fase 2) passo zero e verdetto entro i limiti a ogni checkpoint.

### Passo 10 — Valutazione
- [ ] Metriche per fonte, fasce di distanza (frasi e token), sufficienza, negativi accesi.
- [ ] Test comportamentali generati e congelati.
- [ ] Latenza A / B / C.

### Passo 11 — Esperimenti
- [ ] Scala H0 → H2c, ablazioni H3–H9, condizionati decisi; registro compilato.
- [ ] J0 e W0; controlli C1–C4.

### Passo 12 — Export
- [ ] Verifiche del server superate; `frozen.json` prima dei test; test eseguiti una volta; model card.

## 2. Se succede…

| Sintomo | Prima cosa da controllare | Poi |
|---|---|---|
| La testa non impara, F1 vicino a E-RAND | decodifica di `sent_tok_spans` ([06](06_rendering_ed_estrazione.md) §6): mappa degli offset, escaping, preambolo | `meta.json` delle feature contro la configurazione |
| H0 ≈ E-LEX, fallisce sulle parafrasi e sulle frasi ponte | quota di righe HoVer, CG2C, C2D nel mix | più negativi lunghi con frasi simili al claim; H-MIL con μ_neg = 1 |
| F1 buono su FEVER, basso su HoVer e oltre 10 frasi | H3a: l'aggregatore sta lavorando? | più righe sparse; H-PAIR; H3c con 4 layer |
| Crollo oltre 1.024 token (Winnow) o 4.096 (d1, JPT) | quota di righe lontane nel mix | H2b (attenzione), layer del mix più profondi; fase 2 |
| Mostra molte più frasi delle gold | `s_0` con i negativi: H-MIL attivo? | δ sul dev; ω più alto |
| Si spegne sui positivi incerti | recall nella fascia soglia … soglia + 0,15 | μ_neg più basso |
| Evidenzia spesso la prima frase o il titolo | distribuzione delle posizioni gold nel mix | H9 (senza embedding di posizione) |
| Precisione bassa ma sufficienza alta | lettura a mano dei "falsi positivi" (spesso frasi valide non annotate) | ω FEVER più basso; metrica principale sulla sufficienza per quella fonte |
| Aggiungere le etichette per ablazione peggiora (H8) | conferma del giudice e accordo d1–JPT | solo righe concordi, o `mil_pos` al posto dell'ablazione |
| Grande differenza fra semi | feature di attenzione (pesi concentrati su una testa) | dropout delle feature più alto, o togliere l'attenzione |
| Verdetto cambiato in fase 1 | non deve succedere: rendering diverso fra estrazione e piano binario | test di identità del prompt |
| Verdetto peggiorato in fase 2 | λ, lr della LoRA | ancoraggio β·KL; `stop_grad` su `mil_neg`; tornare alla fase 1 |
| Latenza oltre il budget | tempo di testa e pooling misurati a parte | meno layer nel mix, togliere la media o l'attenzione, aggregatore a 1 layer |
| JPT molto lento in estrazione o nel server | kernel DeltaNet importati davvero (`fla`) | batch per lunghezza; estrazione su più job |
| Spazio su disco insufficiente per le feature | numero di layer del mix | solo i layer con peso su d1; attenzione in fp16; shard per blocco |
