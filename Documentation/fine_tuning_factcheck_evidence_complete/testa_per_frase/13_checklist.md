# Checklist e problemi frequenti

## 1. Checklist per passo

**Passo 1 — Modelli e hook per frase**
- [ ] `flash-linear-attention` e `causal-conv1d` installati e importabili; `punkt_tab` disponibile e `~/nltk_data` privata.
- [ ] `SentenceTap` e `sdpa_capture` funzionanti per KnowLine-4B; test superati (verdetto, layer, attenzione contro "eager", maschera, riassunti delle frasi, memoria).

**Passo 2 — Dati**
- [ ] Dataset costruiti con `--hover-db-fallback`; allineamento ≥ 90% per fonte.
- [ ] 30 positivi per fonte letti con le frasi marcate.

**Passo 3 — Dati generati** (facoltativo per le prime prove)
- [ ] Articoli con posizioni note; ablazione con conferma del giudice ≥ 80% e accordo con elephant-1 ≥ 0,7; evidenza spostata senza riferimenti rotti.

**Passo 4 — Split e mix**
- [ ] Split per `doc_id`; WiCE con `aggrefact_overlap` solo in test; dev e test di HoVer solo `exact`.
- [ ] Mix `ev_head` con i vincoli rispettati.

**Passo 5 — Preambolo**
- [ ] Testo scelto; verdetto riusato o riaddestrato; `question.json` aggiornato.

**Passo 6 — Dalle frasi ai token, estrazione**
- [ ] `escape_with_map`, `token_char_spans`, `sentence_token_spans` testati; frasi decodificate = frasi (200 su 200).
- [ ] Vettori di KnowLine-4B salvati per il mix e i set dev, con `meta.json`.

**Passo 7 — Baseline**
- [ ] E-RAND, E-LEX, E-EMB, E-ATT su `dev_evid_select`; E-ABL su 300 righe per fonte.

**Passo 8 — Testa e loss**
- [ ] `tests/test_evidence_head.py` passa (forme, invarianza al riempimento, esempi numerici).

**Passo 9 — Training**
- [ ] Impara a memoria 64 righe; H0 riproducibile; regola di selezione scelta e confermata.

**Passo 10 — Valutazione**
- [ ] Tutte le metriche, le prove di comportamento e la latenza A/B/C.

**Passo 11 — Esperimenti**
- [ ] Esperimenti principali e controlli su KnowLine-4B; configurazione ripetuta su elephant-1 e sugli altri modelli; registro compilato.

**Passo 12 — Rilascio**
- [ ] Divisore e `punkt_tab` inclusi nel rilascio; controlli del server superati; `frozen.json` prima dei test; test una volta; model card.

## 2. Se succede…

| Problema | Prima cosa da controllare | Poi |
|---|---|---|
| La testa non impara (F1 vicino al caso) | le posizioni dei token di ogni frase (passo 6, test) | le impostazioni dei vettori salvati (`meta.json`) |
| La testa fa come E-LEX: sbaglia frasi riscritte e frasi ponte | quante righe HoVer e generate ci sono nel mix | più negativi con frasi simili al claim; H-MIL con peso 1 |
| Bene su SciFact, male su HoVer e con frasi lontane | H3a: il transformer fra le frasi serve? | più righe spostate; H-PAIR; 4 layer di transformer |
| Male quando l'evidenza è lontana dall'inizio del documento | quante righe lontane ci sono nel mix | attenzione come feature (H2b); fase 2 |
| Mostra troppe frasi | la regola di selezione (δ) e i negativi nel training | ω più alto |
| Non mostra quasi nulla sui positivi incerti | peso di `mil_neg` | abbassarlo |
| Mostra spesso la prima frase o il titolo | dove stanno le frasi gold nel mix | H9 (senza posizione) |
| Precisione bassa ma sufficienza alta | leggere a mano le frasi "sbagliate": spesso sono evidenza valida non annotata | ω più basso per quella fonte |
| Le etichette ricavate col modello peggiorano (H8) | accordo del giudice e fra KnowLine ed elephant-1 | solo le righe concordi, o `mil_pos` al posto dell'ablazione |
| Risultati molto diversi fra semi | se la testa dipende da poche teste di attenzione (pesi di `W_a`) | dropout sull'attenzione più alto, o toglierla |
| Il verdetto cambia nelle fasi 1a/1b | non deve succedere: il prompt è diverso da quello del passo 5 | test di identità del prompt |
| Frasi diverse fra training e server | versione di NLTK o file `punkt_tab` diversi | caricare `punkt_tab` dalla repo; hash in `frozen.json` |
| Troppo lento | tempo della testa e degli hook misurati a parte | meno layer, togliere media o attenzione, 1 layer di transformer |
| Spazio su disco insufficiente | quanti layer si salvano | solo i layer con peso, o i migliori secondo la sonda |
