# Checklist e problemi frequenti

## 1. Checklist per passo

**Passo 1 — Modelli**
- [ ] Pacchetti per JPT installati (`flash-linear-attention`, `causal-conv1d`).
- [ ] I tre modelli si caricano con la funzione di attenzione che registra i pesi.
- [ ] Test superati: verdetto invariato, layer giusti, attenzione uguale alla modalità "eager", memoria.

**Passo 2 — Dati dalle fonti**
- [ ] `sentences.py` scritto e fissato.
- [ ] FEVER: indice delle pagine, testo rimesso in forma normale, train/dev/test.
- [ ] HoVer: `hotpot_qa` scaricato, documenti con pagine di disturbo, evidenza allineata (100 pagine controllate).
- [ ] SciFact e WiCE pronti come test.
- [ ] Almeno il 90% delle frasi di evidenza originali corrisponde a una nostra frase; 30 righe per fonte lette.

**Passo 3 — Altri dati**
- [ ] C2D con posizioni (con negativi e claim parziali); CG2C con la frase di ogni fatto.
- [ ] Etichette ricavate con d1: giudice d'accordo ≥ 80%, accordo con JPT ≥ 0,7.
- [ ] Righe con evidenza spostata, senza riferimenti rotti.

**Passo 4 — Split e mix**
- [ ] Set di valutazione congelati, con le distribuzioni delle distanze.
- [ ] Mix `ev_head` che rispetta tutti i vincoli.

**Passo 5 — Preambolo**
- [ ] Testo scelto; per ogni modello deciso se il verdetto si riusa o si riaddestra; `question.json` aggiornato.

**Passo 6 — Token e vettori**
- [ ] Posizioni dei token giuste (200 su 200), anche con caratteri speciali.
- [ ] Vettori di d1 salvati per il mix e per i set dev.

**Passo 7 — Metodi semplici**
- [ ] E-RAND, E-LEX, E-EMB, E-ATT su `dev_evid_select`; E-ABL su 300 righe per fonte.

**Passo 8 — Testa e loss**
- [ ] Test automatici superati.

**Passo 9 — Training**
- [ ] La testa impara a memoria 64 righe; H0 riproducibile.
- [ ] Regola per le frasi mostrate scelta e confermata.

**Passo 10 — Valutazione**
- [ ] Tutte le metriche, le prove di comportamento e la latenza A/B/C.

**Passo 11 — Esperimenti**
- [ ] Esperimenti principali e controlli su d1; configurazione ripetuta su JPT e Winnow; registro compilato.

**Passo 12 — Rilascio**
- [ ] Controlli del server superati; `frozen.json` scritto prima dei test; test eseguiti una volta; model card.

## 2. Se succede…

| Problema | Prima cosa da controllare | Poi |
|---|---|---|
| La testa non impara (F1 vicino al caso) | le posizioni dei token di ogni frase (passo 6, test) | le impostazioni dei vettori salvati (`meta.json`) |
| La testa fa come E-LEX: sbaglia frasi riscritte e frasi ponte | quante righe HoVer, CG2C, C2D ci sono nel mix | più negativi con frasi simili al claim; H-MIL con peso 1 |
| Bene su FEVER, male su HoVer e con frasi lontane | H3a: il transformer fra le frasi serve? | più righe spostate; H-PAIR; 4 layer |
| Male quando l'evidenza è lontana dal preambolo | quante righe lontane ci sono nel mix | attenzione come feature (H2b); fase 2 |
| Mostra troppe frasi | la regola di selezione (δ) e i negativi nel training | peso più alto per le frasi non di evidenza |
| Non mostra quasi nulla sui positivi incerti | peso dei negativi | abbassarlo |
| Mostra spesso la prima frase o il titolo | dove stanno le frasi di evidenza nel mix | H9 (senza posizione) |
| Precisione bassa ma sufficienza alta | leggere a mano le frasi "sbagliate": spesso sono evidenza valida non annotata | peso più basso per le frasi non segnate di quella fonte |
| Le etichette ricavate col modello peggiorano (H8) | accordo del giudice e fra d1 e JPT | usare solo le righe concordi, o i positivi senza evidenza |
| Risultati molto diversi fra semi | se la testa dipende da poche teste di attenzione | più dropout sull'attenzione, o toglierla |
| Il verdetto cambia nelle fasi 1a/1b | non deve succedere: il prompt è diverso da quello del passo 5 | test di identità del prompt |
| Il verdetto peggiora nella fase 2 | learning rate della LoRA, peso dell'evidenza | termine che tiene il verdetto vicino a quello di partenza; tornare alla fase 1 |
| Troppo lento | tempo della testa misurato a parte | meno layer, togliere media o attenzione, 1 layer di transformer |
| JPT molto lento | i pacchetti per DeltaNet sono davvero usati? | batch per lunghezza, più job |
| Spazio su disco insufficiente | quanti layer si salvano | solo i layer utili su d1 |
