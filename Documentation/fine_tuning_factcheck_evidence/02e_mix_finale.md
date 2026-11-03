# Step 2e — Il mix: verdetto per tutti, evidenza dove è nota, negativi per la soglia

## Obiettivo

Costruire il training set della fase di evidenza: le righe con evidenza nota, le righe costruite per l'evidenza sparsa, e — novità rispetto al piano precedente — una quota di **negativi lunghi** per il termine MIL ([04](04_loss.md) §3). Il mix è lo stesso per i tre modelli; cambia solo il rendering.

## 1. Composizione

Due mix, secondo la fase di training ([05](05_training.md)):

| Mix | Per cosa | Contenuto |
|---|---|---|
| `ev_head` | testa su backbone **congelato** (fasi 1a, 1b) | righe supportate con evidenza (gold, construct, ablation, sparse) + negativi lunghi. Il verdetto non si addestra |
| `ev_joint` | training **congiunto** LoRA + testa (fase 2) | mix del piano binario + righe di `ev_head` |

### `ev_head` — pilota (~26k righe)

| Famiglia | Righe | `evidence_quality` | Termine di evidenza | Peso |
|---|---:|---|---|---:|
| FEVER SUPPORTS | 6.000 | gold | supervisionato | 1,0 |
| HoVer SUPPORTED | 5.000 | gold | supervisionato | 1,0 |
| C2D con posizioni | 2.000 | construct | supervisionato | 1,0 |
| MiniCheck / D2C / SummEdits per ablazione | 3.000 | ablation | supervisionato | 0,5 |
| Evidenza sparsa costruita | 4.000 | come la riga di origine | supervisionato | come la riga di origine |
| **Negativi lunghi** (FEVER REFUTES/NEI con pagina, HoVer NOT_SUPPORTED, C2D negativi, negativi sparsi) | 6.000 | — | MIL negativo | 0,3 (H-MIL: 0 / 0,3 / 1) |
| Positivi senza evidenza (solo per H-MIL) | 0 (3.000 in H-MIL) | none | MIL positivo | 0,3 |

I negativi sono ~23% del mix: abbastanza per insegnare la soglia, non tanto da spingere la testa a spegnersi sui positivi.

### `ev_joint` — corsa piena

Il mix finale del piano binario ([../fine_tuning_factcheck/02e_mix_finale.md](../fine_tuning_factcheck/02e_mix_finale.md)), in cui:
- le righe supportate di FEVER e HoVer portano l'evidenza gold;
- si aggiungono CG2C con posizioni, C2D con posizioni e le righe sparse (~20–30% del totale);
- le negative restano tutte (servono al verdetto); quelle con documento lungo hanno anche il termine MIL.

Tutte le righe sono renderizzate nel formato scelto in [02d](02d_varianti_domanda.md) (sandwich).

## 2. Principi

1. **Molta evidenza multi-frase.** Almeno il 40% delle righe con evidenza deve averne ≥ 2 frasi; FEVER ha soprattutto evidenza a una frase, HoVer e CG2C a 2–4.
2. **Distanza variata.** Distribuzione della distanza fra la prima e l'ultima frase gold: almeno il 25% oltre 10 frasi (con le righe sparse costruite).
3. **Lontano dal preambolo.** Almeno il 15% delle righe con evidenza oltre 1.024 token dall'inizio del documento (misurato con il tokenizer di ciascun modello; i tre tokenizer danno lunghezze diverse).
4. **Domini diversi da Wikipedia.** C2D, CG2C su notizie e righe per ablazione su D2C/SummEdits ≥ 30% delle righe con evidenza: altrimenti la testa impara lo stile delle pagine Wikipedia.
5. **Documenti lunghi.** Righe con > 50 frasi ≥ 15%: in produzione la testa deve scegliere 2–3 frasi su centinaia.
6. **Negativi con frasi simili al claim.** Nei negativi lunghi, almeno metà ha una frase con sovrapposizione lessicale alta con il claim (BM25 sopra la mediana): sono quelli che insegnano qualcosa.

## 3. Controlli

| Controllo | Atteso |
|---|---|
| Frasi di evidenza per riga positiva (media) | 1,8–2,5 |
| Quota di frasi evidenza sul totale delle frasi dei positivi | ~2–10% (classe molto sbilanciata: [04](04_loss.md)) |
| Quota di negativi in `ev_head` | 20–25% |
| Righe con `sentences` incoerenti con `document` | 0 |
| `doc_id` condivisi con dev/test | 0 |
| Distrattori delle righe sparse presi da documenti di dev/test | 0 |

## 4. Comandi

```sh
conda run -n factcheck python data_prep/make_mix.py --recipe ev_head.json  --name ev_head
conda run -n factcheck python data_prep/make_mix.py --recipe ev_joint.json --name ev_joint
```

(`make_mix.py` va esteso per conservare `sentences`, `evidence`, `evidence_quality` e per le quote di negativi.)

## Criteri per passare allo step successivo

- `mix_ev_head` prodotto con le statistiche di §3.
- Distribuzione delle distanze (fra frasi gold, dal preambolo in token per modello) registrata.
