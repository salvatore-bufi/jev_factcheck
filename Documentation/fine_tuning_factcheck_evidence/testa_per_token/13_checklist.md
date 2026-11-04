# Checklist e problemi frequenti

## 1. Checklist per passo

**Passo 1 — Modelli e hook per token**
- [ ] Hook sui token del documento con proiezione nell'hook; test del piano per frase superati (verdetto, layer, attenzione, maschera).
- [ ] PCA per KnowLine-4B (layer 23 e 31) con la varianza spiegata registrata.

**Passo 2 — Dati**
- [ ] `gold_char_sets` e `contra_char` aggiunti a `evidence/data/schema.py`, validati e testati; dataset ricostruiti.
- [ ] 30 positivi per fonte letti con gli intervalli evidenziati.

**Passo 3 — Dati generati** (facoltativo per le prime prove)
- [ ] C2D con intervalli esatti; ablazione con uscita in intervalli; righe sparse.

**Passo 4 — Split e mix**
- [ ] Split per `doc_id`; WiCE con `aggrefact_overlap` fuori dal training; dev e test di HoVer solo `exact`.
- [ ] Mix `ev_tok_head` con i vincoli rispettati.

**Passo 5 — Preambolo**
- [ ] Scelto come nel piano per frase; `question.json` aggiornato.

**Passo 6 — Rendering ed estrazione**
- [ ] `escape_with_map` spostata in `evidence/escape.py`; mappa inversa testata.
- [ ] `tok_char` ricostruisce il documento (200 su 200); etichette dei token coprono le frasi gold.
- [ ] Vettori di KnowLine-4B salvati per il mix e i set dev.

**Passo 7 — Baseline**
- [ ] E-RAND, E-LEX, E-EMB, E-ATT; E-ABL su 300 righe per fonte.

**Passo 8 — Testa, loss, decodifica**
- [ ] `tests/test_token_head.py` passa (forme, esempio numerico, decodifica).

**Passo 9 — Training**
- [ ] Impara a memoria 64 righe; H0 riproducibile; parametri della decodifica scelti e confermati.

**Passo 10 — Valutazione**
- [ ] Metriche sui caratteri e sulle frasi gold; confronto con la testa per frase; prova senza punteggiatura; latenza.

**Passo 11 — Esperimenti**
- [ ] H0–H7, controlli H8–H13, T-vs-F; registro compilato.

**Passo 12 — Rilascio**
- [ ] Controlli del server superati; `frozen.json` prima dei test; model card.

## 2. Se succede…

| Problema | Prima cosa da controllare | Poi |
|---|---|---|
| La testa non impara (F1 vicino al caso) | `tok_char` e etichette dei token (passo 6, test) | `meta.json` dei vettori contro la configurazione |
| Evidenzia paragrafi interi | peso di `L_out`; decodifica (buco massimo troppo grande) | ω più alto; lunghezza massima dei tratti |
| Tratti frammentati, parole isolate | `L_tv`; contesto locale (H1) | lunghezza minima della decodifica |
| Colpisce una frase gold ma non la "ponte" | contesto globale (H3), attenzione (H2) | più righe sparse; blocchi più piccoli (H7) |
| Fa come E-LEX (accende parole del claim) | negativi nel mix e loro peso | più negativi con parole in comune col claim |
| Accende tratti sui claim non supportati | peso di `mil_neg` | soglia δ più alta |
| Precisione bassa ma sufficienza alta | leggere a mano i "falsi positivi": spesso evidenza valida non annotata | ω più basso per quella fonte |
| Peggio della testa per frase sulle metriche per frase | rumore delle etichette per token | righe C2D a intervalli esatti (H10); `L_hit` più forte |
| Spazio su disco insufficiente | numero di layer, componenti PCA | 1 layer; 128 componenti |
| Latenza oltre il limite | tempo di proiezione, convoluzioni, transformer misurati a parte | blocchi più grandi; meno convoluzioni |
| Tratti con caratteri sbagliati nel testo originale | mappa inversa dell'escaping | test con documenti con `"`, `\`, a capo |
