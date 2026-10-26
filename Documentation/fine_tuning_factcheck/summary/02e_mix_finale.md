# Step 2e — Il mix: curriculum a due stadi, come MiniCheck e FactCG

## Obiettivo

Mettere insieme le fonti ([02a](02a_dataset_fonti_esistenti.md)) e i dati generati ([02b](02b_dataset_generazione_llm.md)) in **due training set ordinati**: prima i dati che insegnano la logica del supporto, poi quelli che riportano il modello alla distribuzione d'uso. È la struttura usata da entrambi i paper e, insieme ai dati, è la parte della loro ricetta che si trasferisce a qualsiasi backbone.

## 1. Perché due stadi

| Paper | Stadio 1 | Stadio 2 |
|---|---|---|
| MiniCheck | 21k ANLI difficili + 7k C2D, 2 epoche, lr 5e-5 (1e-5 per RoBERTa) | 7k D2C, 1 epoca, lr 1e-5 |
| FactCG | NLI + C2D + CG2C-MHQA, 1 epoca | D2C + CG2C-Doc, 1 epoca, ottimizzatore nuovo |

Motivo dichiarato da MiniCheck: i documenti sintetici (C2D) hanno uno stile riconoscibile; se il training finisce su quelli, il modello si adatta allo stile. Il secondo stadio su testi reali corregge lo spostamento.

## 2. Le ricette

Formato: lo stesso JSON di `make_mix.py` (`{fonte: {"cap": N, "balance": bool}}`), una ricetta per stadio. Il mix legge solo `data/finetune/splits/train/`.

### `fc_pilot` — pilota a uno stadio (~25k coppie)

Serve a verificare che il training funzioni, prima di generare dati nuovi.

```json
{"minicheck": {"cap": 0, "balance": false},
 "anli_hard": {"cap": 6000, "balance": true},
 "summedits": {"cap": 0, "balance": false},
 "hover": {"cap": 2000, "balance": true},
 "vitaminc": {"cap": 1500, "balance": true},
 "fever": {"cap": 1500, "balance": true}}
```

### `fc_stage1` — logica del supporto (~45–60k coppie)

```json
{"minicheck_c2d": {"cap": 0, "balance": false},
 "c2d_ext": {"cap": 15000, "balance": false},
 "anli_hard": {"cap": 20000, "balance": true},
 "cg2c_mhqa": {"cap": 10000, "balance": true},
 "hover": {"cap": 6000, "balance": true},
 "vitaminc": {"cap": 5000, "balance": true},
 "fever": {"cap": 4000, "balance": true}}
```

### `fc_stage2` — distribuzione d'uso (~25–40k coppie)

```json
{"minicheck_d2c": {"cap": 0, "balance": false},
 "d2c_ext": {"cap": 20000, "balance": false},
 "summedits": {"cap": 0, "balance": false},
 "cg2c_doc": {"cap": 6000, "balance": true},
 "remove_evidence_long": {"cap": 5000, "balance": false},
 "averitec": {"cap": 0, "balance": false}}
```

(`minicheck` è da dividere in `minicheck_c2d` e `minicheck_d2c` usando `meta.subset` (`c2d` / `d2c`), già presente in ogni riga; `anli_hard` viene da [02a](02a_dataset_fonti_esistenti.md) §3.)

### Variante a uno stadio (controllo)

`fc_stage1 ∪ fc_stage2` mescolati, 1 epoca: esperimento D-CUR in [07](07_esperimenti.md). Se il risultato è uguale entro 0,5 punti, si tiene la versione a uno stadio (più semplice).

## 3. Principi per le proporzioni

1. **Dati costruiti per il compito al centro.** MiniCheck, C2D/D2C estesi e CG2C sono ≥ 50% di ciascuno stadio.
2. **Fonti grandi con tetti bassi.** FEVER e VitaminC hanno centinaia di migliaia di righe ma insegnano scorciatoie: ≤ 10% ciascuna.
3. **Negativi "non stabilito" ≥ 40% dei negativi.** Sono il caso che i nostri modelli sbagliano di più (soglia 0,90–0,95 di `support_simple`: dicono "sì" quando manca un pezzo).
4. **Documenti lunghi nello stadio 2.** Almeno ~20% delle righe con documento > 2.000 token (D2C su notizie e trascrizioni, SummEdits, `remove_evidence` su documenti lunghi).

## 4. Bilanciamento

50/50 per fonte di default; le famiglie generate si bilanciano con le loro coppie positive (stesso `doc_id`). Controllo finale per stadio: positivi 45–55%.

## 5. Controlli sul mix prodotto

| Controllo | Valore atteso |
|---|---|
| Positivi per stadio | 45–55% |
| Negativi "non stabilito" / negativi | ≥ 40% |
| `doc_id` in comune con dev/test dello Step 0 | 0 |
| Righe generate da righe non di train | 0 |
| Documenti oltre il budget di token (troncati) | < 1% per fonte, oppure chunking ([03](03_rendering_formato.md), §4) |
| Fonti escluse | RAGTruth sempre; WiCE salvo esperimento dedicato |

## 6. Comandi

```sh
conda run -n factcheck python data_prep/make_mix.py --recipe fc_pilot.json  --name fc_pilot
conda run -n factcheck python data_prep/make_mix.py --recipe fc_stage1.json --name fc_stage1
conda run -n factcheck python data_prep/make_mix.py --recipe fc_stage2.json --name fc_stage2
```

## Criteri per passare allo step successivo

- `mix_fc_pilot` prodotto e verificato (§5); `fc_stage1`/`fc_stage2` prodotti dopo lo step 2b.
- Statistiche di lunghezza in token (p50, p95) per stadio registrate.
