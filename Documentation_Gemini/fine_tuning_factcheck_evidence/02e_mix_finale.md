# Step 2e — Il mix: verdetto per tutti, evidenza dove è nota

## Obiettivo

Costruire il training set della fase di evidenza: le stesse righe del verdetto (piano binario) arricchite con `sentences` ed `evidence` dove disponibili, più le righe costruite per l'evidenza sparsa.

## 1. Composizione

Due mix, secondo la strategia di training ([05](05_training.md)):

| Mix | Per cosa | Contenuto |
|---|---|---|
| `ev_head` | testa su backbone **congelato** | **solo righe supportate con evidenza** (gold, construct, ablation, sparse). Il verdetto non si addestra |
| `ev_joint` | training **congiunto** LoRA + testa | mix del piano binario (stadio 2 o unione) + righe con evidenza; la loss per frase è mascherata dove manca |

### `ev_head` — pilota (~20k righe)

| Famiglia | Righe | `evidence_quality` | Peso nella loss per frase |
|---|---:|---|---:|
| FEVER SUPPORTS | 6.000 | gold | 1,0 |
| HoVer SUPPORTED | 5.000 | gold | 1,0 |
| C2D con posizioni | 2.000 | construct | 1,0 |
| MiniCheck / D2C / SummEdits per ablazione | 3.000 | ablation | 0,5 |
| Evidenza sparsa costruita | 4.000 | come la riga di origine | come la riga di origine |

### `ev_joint` — corsa piena

Il mix finale del piano binario ([../fine_tuning_factcheck/02e_mix_finale.md](../fine_tuning_factcheck/02e_mix_finale.md)), in cui:
- le righe supportate di FEVER e HoVer portano l'evidenza gold;
- si aggiungono CG2C con posizioni, C2D con posizioni e le righe sparse (~20–30% del totale);
- le negative restano tutte (servono al verdetto).

## 2. Principi

1. **Molta evidenza multi-frase.** Almeno il 40% delle righe con evidenza deve averne ≥ 2 frasi; FEVER ha soprattutto evidenza a una frase, HoVer e CG2C a 2–4.
2. **Distanza variata.** Distribuzione della distanza fra la prima e l'ultima frase gold: almeno il 25% oltre 10 frasi (con le righe sparse costruite).
3. **Domini diversi da Wikipedia.** C2D, CG2C su notizie e righe per ablazione su D2C/SummEdits ≥ 30% delle righe con evidenza: altrimenti la testa impara lo stile delle pagine Wikipedia.
4. **Documenti lunghi.** Righe con > 50 frasi ≥ 15%: in produzione la testa deve scegliere 2–3 frasi su centinaia.

## 3. Controlli

| Controllo | Atteso |
|---|---|
| Frasi di evidenza per riga (media) | 1,8–2,5 |
| Quota di frasi evidenza sul totale delle frasi | ~2–10% (classe molto sbilanciata: [04](04_loss.md) §2) |
| Righe con `sentences` incoerenti con `document` | 0 |
| `doc_id` condivisi con dev/test | 0 |
| Distrattori delle righe sparse presi da documenti di dev/test | 0 |

## 4. Comandi

```sh
conda run -n factcheck python data_prep/make_mix.py --recipe ev_head.json  --name ev_head
conda run -n factcheck python data_prep/make_mix.py --recipe ev_joint.json --name ev_joint
```

(`make_mix.py` va esteso per conservare `sentences`, `evidence`, `evidence_quality`.)

## Criteri per passare allo step successivo

- `mix_ev_head` prodotto con le statistiche di §3.
- Distribuzione della distanza fra frasi gold registrata.
