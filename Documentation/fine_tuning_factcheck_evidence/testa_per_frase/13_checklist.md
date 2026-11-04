# Comandi in ordine, checklist e problemi frequenti

## 1. Comandi in ordine

```sh
cd <repo>
F=data/evidence/features/knowline-4b; S=dataset/evidence/splits

# 0. test (CPU, secondi)
for t in tests/test_evidence_*.py; do python $t; done

# 1. split e mix (CPU, ~10 s) — già fatti il 2026-11-04
python -m evidence.data.make_splits
python -m evidence.data.make_mix --recipe evidence/recipes/ev_head.json

# 2-3. controlli su rendering e modello
python -m evidence.render 400                  # CPU: frasi = token
python -m evidence.model --selftest            # GPU: verdetto, riassunti, attenzione

# 4. preambolo (GPU, ~1.200 righe x 3 forward, ~25 min) — PRIMA dell'estrazione
python -m evidence.check_preamble

# 5. estrazione (GPU, ~2,5 righe/s: mix ~2,3 h, dev ~8 min; ~45 GB)
python -m evidence.extract --rows dataset/evidence/mixes/ev_head.jsonl $S/dev_evid_select.jsonl \
    $S/dev_evid_confirm.jsonl $S/dev_evid_neg.jsonl

# 6. baseline
python -m evidence.baselines --features $F/dev_evid_select --rows $S/dev_evid_select.jsonl --method rand lex att
python -m evidence.baselines --features $F/dev_evid_select --rows $S/dev_evid_select.jsonl --method abl --per-source 100

# 8. training: controllo, poi esperimenti uno alla volta
python -m evidence.train_head --config evidence/configs/h0.json --train $F/ev_head --overfit 64
for c in h0 h_loss h_mil h2a h2b h2c; do
  python -m evidence.train_head --config evidence/configs/$c.json --train $F/ev_head \
      --dev $F/dev_evid_select --confirm $F/dev_evid_confirm --dev-neg $F/dev_evid_neg
done
cat data/evidence/runs/registry.md

# 9. valutazione della configurazione scelta (dev)
python -m evidence.eval_evidence --run data/evidence/runs/knowline-4b/<id> --features $F/dev_evid_select \
    --rows $S/dev_evid_select.jsonl --sufficiency 300
python -m evidence.latency --run data/evidence/runs/knowline-4b/<id> --n 1000

# 11. rilascio, poi test una volta sola
python -m evidence.release --run data/evidence/runs/knowline-4b/<id> --name knowline-4b-fc-ev-v1
python -m evidence.extract --rows $S/test_evid_id.jsonl $S/test_evid_ood.jsonl
for t in test_evid_id test_evid_ood; do
  python -m evidence.eval_evidence --run releases/knowline-4b-fc-ev-v1 --features $F/$t --rows $S/$t.jsonl --sufficiency 0
done
python -m evidence.serve --run releases/knowline-4b-fc-ev-v1 --port 8090
```

## 2. Checklist

**Passo 1–2, dati**

- [x] dataset costruiti con `--hover-db-fallback`; allineamento ≥ 90% per fonte
- [x] split per `doc_id`, nessun documento condiviso; WiCE con `aggrefact_overlap` mai in train
- [x] mix `ev_head` con i vincoli rispettati

**Passo 2–3, rendering e modello**

- [x] `python -m evidence.render 400`: tutte le righe corrette
- [x] `python -m evidence.model --selftest` superato
- [x] `tests/test_evidence_render.py` superato

**Passo 4, preambolo**

- [ ] `check_preamble` lanciato; `preamble_choice.json` scritto; `acceptable` verificato

**Passo 5, estrazione**

- [ ] mix e dev estratti con il preambolo scelto; `report` stampa la quota di evidenza oltre 1.024 token (≥ 0,15)
- [ ] `skipped.json` controllato (righe saltate e motivo)

**Passo 6–9, baseline, testa, training, valutazione**

- [ ] baseline su `dev_evid_select`
- [x] `tests/test_evidence_head.py` superato
- [ ] `--overfit 64` sulle feature vere: score ≈ 1
- [ ] H0 → H2c nel registro, con le decisioni compilate
- [ ] configurazione finale con 3 semi; sufficienza ≥ 85%; latenza nell'obiettivo

**Passo 11, rilascio**

- [ ] release creata e verificata; server = valutazione su 200 righe
- [ ] test misurati una volta sola; `report.md` aggiornato

## 3. Se succede…

| Problema | Prima cosa da controllare | Poi |
|------------------------------------------|--------------------------------------------------|---------------------------------------------------|
| La testa non impara (F1 vicino a E-RAND) | `python -m evidence.render` sulle righe del mix | `meta.json` delle feature (preambolo, layer) |
| `--overfit 64` non arriva a ≈ 1 | posizioni delle frasi, etichette `gold` in `index.jsonl` | il codice della loss (`tests/test_evidence_head.py`) |
| Fa come E-LEX (sbaglia frasi "ponte") | quante righe HoVer ci sono nel mix | H3c4; H-PAIR; `h_mil1` |
| Bene su SciFact, male su HoVer o con frasi lontane | H3a: il transformer serve? | H3c4, H-PAIR |
| Mostra troppe frasi | $\delta$ in `selection.json`, `count_error_mean` | $\omega$ più alto (H7) |
| Non mostra quasi nulla sui positivi incerti | `recall_uncertain` | $\mu_{neg}$ più basso |
| Mostra spesso la prima frase o il titolo | posizione delle gold nel mix | H9 |
| Precisione bassa ma sufficienza alta | leggere le frasi "sbagliate": spesso sono evidenza valida non annotata | $\omega$ più basso per quella fonte |
| Risultati molto diversi fra semi | dipendenza da poche teste di attenzione | `att_dropout` più alto, o `use_att=False` |
| "extracted with different settings" | `meta.json` delle due cartelle | ri-estrarre con lo stesso preambolo e lo stesso codice |
| Il verdetto in `eval_evidence` cambia | non deve succedere nella fase 1a | prompt identico (`tests/test_evidence_render.py`) |
| Il server rifiuta la release (hash) | `sentences.py`, `render.py`, `model.py`, `punkt_tab` modificati? | ripristinarli, o ri-estrarre e riaddestrare |
| Out of memory in estrazione | righe oltre 16.000 token (`skipped.json`) | `--max-tokens` più basso |
| Troppo lento | tempi separati in `latency.json` | meno layer, `use_mean=False`, 1 layer di transformer |
