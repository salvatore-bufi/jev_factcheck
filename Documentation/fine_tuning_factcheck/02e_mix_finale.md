# Step 2e — I training set: pilota, stadio 1, stadio 2

## Obiettivo

Mettere insieme fonti esistenti ([02a](02a_dataset_fonti_esistenti.md)) e famiglie generate ([02b](02b_dataset_generazione_llm.md)) in tre training set:

| Mix | Uso | Righe |
|---|---|---:|
| `fc_pilot` | prima corsa di verifica, a uno stadio, senza dati generati nuovi | ~25.000 |
| `fc_stage1` | stadio 1 del curriculum: logica del supporto | ~45.000–60.000 |
| `fc_stage2` | stadio 2: distribuzione d'uso (testi reali, lunghi) | ~25.000–40.000 |

Dev e test **non** fanno parte dei mix: sono gli split congelati dello Step 0.

## 1. Perché due stadi

Sia MiniCheck sia FactCG addestrano in due fasi:

| | Stadio 1 | Stadio 2 |
|---|---|---|
| MiniCheck | 21k ANLI difficili + 7k C2D; 2 epoche; lr 5e-5 | 7k D2C; 1 epoca; lr 1e-5 |
| FactCG | NLI + C2D + CG2C-MHQA; 1 epoca | D2C + CG2C-Doc; 1 epoca; ottimizzatore nuovo |

Motivo: i documenti sintetici (C2D, CG2C-MHQA) hanno uno stile riconoscibile. Se il training finisse su quelli, il modello si adatterebbe allo stile. Il secondo stadio, su testi reali e con learning rate più basso, riporta il modello verso la distribuzione d'uso. Nell'esperimento D-CUR ([07](07_esperimenti.md)) si verifica se da noi serve.

## 2. Le ricette

Formato di `make_mix.py`: `{fonte: {"cap": N, "balance": true|false}}`, `cap = 0` significa "tutto". Ogni fonte è un file `data/finetune/splits/train/<fonte>.jsonl` (fonti esistenti) o `data/finetune/raw/<famiglia>.jsonl` (famiglie generate, già solo train).

### 2.1 `fc_pilot.json`

```json
{"minicheck": {"cap": 0, "balance": false},
 "anli_hard": {"cap": 6000, "balance": true},
 "summedits": {"cap": 0, "balance": false},
 "hover": {"cap": 2000, "balance": true},
 "vitaminc": {"cap": 1500, "balance": true},
 "fever": {"cap": 1500, "balance": true}}
```

MiniCheck (80% del rilasciato, ~11.500) + ANLI difficile 6.000 + SummEdits (~5.000) + HoVer, VitaminC, FEVER ≈ 27.000 righe.

### 2.2 `fc_stage1.json`

```json
{"minicheck_c2d": {"cap": 0, "balance": false},
 "c2d": {"cap": 15000, "balance": false},
 "anli_hard": {"cap": 20000, "balance": true},
 "cg2c_mhqa": {"cap": 10000, "balance": true},
 "contradict_claim": {"cap": 4000, "balance": false},
 "hover": {"cap": 6000, "balance": true},
 "vitaminc": {"cap": 5000, "balance": true},
 "fever": {"cap": 4000, "balance": true}}
```

### 2.3 `fc_stage2.json`

```json
{"minicheck_d2c": {"cap": 0, "balance": false},
 "d2c": {"cap": 20000, "balance": false},
 "summedits": {"cap": 0, "balance": false},
 "cg2c_doc": {"cap": 6000, "balance": true},
 "remove_evidence": {"cap": 5000, "balance": false},
 "averitec": {"cap": 0, "balance": false}}
```

`minicheck_c2d` e `minicheck_d2c` sono le righe di `minicheck` con `meta.subset` uguale a `c2d` o `d2c` (campo presente in ogni riga). `contradict_claim` e `remove_evidence` sono famiglie di soli negativi: bilanciano il mix insieme ai positivi delle stesse fonti di partenza.

## 3. Le modifiche a `make_mix.py`

Il codice attuale ([data_prep/make_mix.py](../../data_prep/make_mix.py)) legge `data/finetune/raw/`, decontamina, crea il proprio val e applica tetti e bilanciamento. Per questo piano:

1. **Lettura**: `--raw data/finetune/splits/train` per le fonti esistenti; per le famiglie generate, un secondo percorso (`--gen data/finetune/raw`). Le fonti `minicheck_c2d` / `minicheck_d2c` sono filtri su `minicheck` per `meta.subset`.
2. **Niente val interno**: con gli split dello Step 0 il passo "validation pool" va disattivato (opzione `--no-val`); tutte le righe lette sono train.
3. **Decontaminazione**: già fatta allo Step 0 per le fonti esistenti; per le famiglie generate va rifatta (contro AggreFact e test_ood), con lo stesso codice.
4. **Campi conservati**: `meta` intero (serve `neg_type`, `p0`, `pair_id`, `family`).
5. **Bilanciamento**: invariato (`take()`): con `balance: true` la fonte è ridotta a 50/50; con `false` si prende così com'è.

Pseudocodice del passo principale modificato:

```python
for src, cfg in recipe.items():
    rows = load_source(src)          # splits/train/<src>.jsonl, raw/<src>.jsonl, o filtro su minicheck
    rows = dedup_and_length_filter(rows, max_doc_chars=24000)
    if src in GENERATED:
        rows = decontaminate(rows, agg_shingles, agg_claims, ood_shingles, ood_claims)
    rows = take(rows, cfg.get("cap", 0), cfg.get("balance", False), rng)
    train_all += rows
```

## 4. Principi per le proporzioni

1. **Dati costruiti per il compito al centro**: MiniCheck, C2D, D2C, CG2C ≥ 50% di ciascuno stadio.
2. **Fonti grandi con tetti bassi**: FEVER e VitaminC insegnano scorciatoie; ≤ 10% ciascuna.
3. **Negativi "non stabilito" ≥ 40% dei negativi**: sono il caso che i nostri modelli sbagliano (dicono "sì" quando manca un pezzo).
4. **Documenti lunghi nello stadio 2**: almeno il 20% delle righe con documento > 2.000 token.
5. **Fonti piccole e uniche prese intere**: SummEdits, AVeriTeC.

## 5. Controlli sul mix prodotto — `finetune/check_mix.py`

```python
import json, collections
rows = [json.loads(l) for l in open("data/finetune/mix_fc_stage1/train.jsonl")]
neg = [r for r in rows if r["label"] == 0]
ne = sum(r.get("meta", {}).get("neg_type") == "not_established" for r in neg)
print("righe", len(rows), "positivi", round(sum(r["label"] for r in rows) / len(rows), 3))
print("negativi non stabiliti / negativi", round(ne / max(1, len(neg)), 3))
print(collections.Counter(r["source"] for r in rows).most_common())
```

| Controllo | Valore atteso |
|---|---|
| Quota di positivi | 45–55% |
| Negativi `not_established` / negativi (dove noto) | ≥ 40% |
| doc_id in comune con dev/test dello Step 0 | 0 |
| Righe generate partite da righe non di train | 0 |
| Documenti oltre il budget di token | < 1% per fonte (vedi [03](03_rendering_formato.md) §7) |
| Fonti escluse | RAGTruth e WiCE assenti |

Statistiche da salvare in `stats.json`: righe e positivi per fonte, quota `neg_type`, lunghezze in token (p50, p95) con il tokenizer di JPT.

## 6. Comandi

```sh
conda run -n factcheck python data_prep/make_mix.py --recipe data_prep/recipes/fc_pilot.json  --name fc_pilot  --raw data/finetune/splits/train --no-val
conda run -n factcheck python data_prep/make_mix.py --recipe data_prep/recipes/fc_stage1.json --name fc_stage1 --raw data/finetune/splits/train --gen data/finetune/raw --no-val
conda run -n factcheck python data_prep/make_mix.py --recipe data_prep/recipes/fc_stage2.json --name fc_stage2 --raw data/finetune/splits/train --gen data/finetune/raw --no-val
conda run -n factcheck python finetune/check_mix.py
```

(`--gen` e `--no-val` sono le due opzioni da aggiungere.)

## 7. Variante a uno stadio (controllo)

`fc_stage1 ∪ fc_stage2` mescolati in un solo file (`fc_onestage`), una epoca. Se l'esperimento D-CUR mostra un risultato uguale entro 0,5 punti, si tiene la versione a uno stadio, più semplice.

## Criteri per passare allo step successivo

- `mix_fc_pilot` prodotto e controllato; `mix_fc_stage1` / `mix_fc_stage2` dopo lo step 2b.
- Statistiche di lunghezza in token registrate.
