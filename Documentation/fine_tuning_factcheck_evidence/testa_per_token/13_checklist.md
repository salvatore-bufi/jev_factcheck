# Checklist e problemi frequenti

## 1. Checklist per passo

**Passo 1 — Modelli e hook per token**
- [ ] `flash-linear-attention` e `causal-conv1d` installati e importabili.
- [ ] `TokenTap` e `sdpa_capture` funzionanti per KnowLine-4B; test superati (verdetto, layer, attenzione contro "eager", maschera, documento giusto).
- [ ] PCA dei layer 23 e 31 di KnowLine-4B, con la varianza spiegata registrata.

**Passo 2 — Dati**
- [ ] `gold_char_sets` e `contra_char` aggiunti a `evidence/data/schema.py`, validati e testati.
- [ ] Dataset ricostruiti con `--hover-db-fallback`; 30 positivi per fonte letti con gli intervalli evidenziati.

**Passo 3 — Dati generati** (facoltativo per le prime prove)
- [ ] Articoli con posizioni esatte; ablazione con uscita in intervalli; evidenza spostata.

**Passo 4 — Split e mix**
- [ ] Split per `doc_id`; WiCE con `aggrefact_overlap` solo in test; dev e test di HoVer solo `exact`; `test_nopunct` costruito.
- [ ] Mix `ev_tok_head` con i vincoli rispettati.

**Passo 5 — Preambolo**
- [ ] Testo scelto; verdetto riusato o riaddestrato; `question.json` aggiornato.

**Passo 6 — Dai caratteri ai token, estrazione**
- [ ] `evidence/escape.py` con `escape_with_map` e `token_char_spans`; testato contro `json.dumps`.
- [ ] `tok_char` ricostruisce il documento (200 su 200); etichette dei token coprono le frasi gold.
- [ ] Vettori di KnowLine-4B salvati per il mix e i set dev, con `meta.json`.

**Passo 7 — Baseline**
- [ ] E-RAND, E-LEX, E-EMB, E-ATT su `dev_evid_select`; E-ABL su 300 righe per fonte.

**Passo 8 — Testa, loss, decodifica**
- [ ] `tests/test_token_head.py` passa (forme, invarianza al riempimento, esempio numerico, decodifica).

**Passo 9 — Training**
- [ ] Impara a memoria 64 righe; H0 riproducibile; parametri della decodifica scelti e confermati.

**Passo 10 — Valutazione**
- [ ] Metriche sulle frasi gold e sui caratteri; confronto con la testa per frase; `test_nopunct`; latenza A/B/C.

**Passo 11 — Esperimenti**
- [ ] H0–H7, controlli H8–H13, T-vs-F; registro compilato.

**Passo 12 — Rilascio**
- [ ] Controlli del server superati; `frozen.json` prima dei test; test una volta; model card.

## 2. Se succede…

| Problema | Prima cosa da controllare | Poi |
|---|---|---|
| La testa non impara (risultati vicini al caso) | `tok_char` ed etichette dei token (passo 6, test) | `meta.json` dei vettori contro la configurazione |
| Evidenzia paragrafi interi | peso di `L_out`; buco massimo della decodifica | ω più alto; α più basso (H13) |
| Tratti frammentati, parole isolate | `L_tv`; contesto locale (H1) | lunghezza minima della decodifica |
| Colpisce una frase gold ma non la "ponte" | contesto globale (H3), attenzione (H2) | più evidenza spostata; blocchi più piccoli (H7) |
| Fa come E-LEX (accende le parole del claim) | negativi nel mix e loro peso | più negativi con parole in comune col claim |
| Accende tratti sui claim non supportati | peso di `mil_neg` | δ più alto nella decodifica |
| Precisione bassa ma sufficienza alta | leggere a mano i "falsi positivi": spesso è evidenza valida non annotata | ω più basso per quella fonte |
| Peggio della testa per frase sulle metriche per frase | rumore delle etichette per token | righe generate a intervalli esatti (H10); peso di `L_hit` più alto |
| Spazio su disco insufficiente | numero di layer e di componenti PCA | 1 layer; 128 componenti |
| Latenza oltre il limite | proiezione, convoluzioni, transformer misurati a parte | blocchi più grandi; meno convoluzioni |
| Tratti con i caratteri sbagliati | mappa dell'escaping | test con documenti con `"`, `\`, a capo |
| Training o server molto lenti | i kernel DeltaNet sono davvero usati? | righe raggruppate per lunghezza |
