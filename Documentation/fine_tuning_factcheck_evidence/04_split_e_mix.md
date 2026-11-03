# Passo 4 — Split e mix di training

## In breve

Due cose:
1. **Split**: si decide una volta per tutte quali righe servono per scegliere (dev) e quali per la misura finale (test), e non si cambiano più. Nessun documento può stare sia in training sia in dev/test.
2. **Mix**: si costruisce l'insieme di training della testa, con le giuste proporzioni di casi facili, difficili e negativi.

Script: `evidence/make_splits.py`, `evidence/make_mix.py`. Gli split sono gli stessi per i tre modelli.

## 1. Regole degli split

- Si parte dagli split del piano binario ([../fine_tuning_factcheck/00b_split_e_protocollo.md](../fine_tuning_factcheck/00b_split_e_protocollo.md)): una riga che lì è in dev o test resta in dev o test.
- Si divide per documento (`doc_id`), non per riga.
- FEVER: train → training, `paper_dev` → dev, `paper_test` → test. HoVer: il dev ufficiale si divide a metà per documento (dev / test).
- I dati generati e spostati nascono solo da righe di training, con frasi di disturbo di training.

## 2. I set di valutazione

| Set | Che cosa contiene | Righe | A che cosa serve |
|---|---|---:|---|
| `dev_evid_select` | positivi con evidenza nota del dev (FEVER, HoVer, C2D con posizioni) + versioni spostate | 3.000 | tutte le scelte: configurazione, soglie, esperimenti |
| `dev_evid_confirm` | come sopra, altre righe | 2.000 | conferma delle scelte |
| `dev_evid_neg` | negativi del dev con documenti lunghi (FEVER REFUTES, HoVer NOT_SUPPORTED) | 1.500 | controllare che la testa non dia punteggi alti a frasi su claim non supportati |
| `test_evid_id` | positivi del test di FEVER e HoVer, 10% di C2D | 3.000 | misura finale, testi simili al training |
| `test_evid_ood` | SciFact, WiCE | tutte | misura finale, testi diversi |
| `test_evid_sparse` | da `test_evid_id`, con evidenza spostata lontano | 1.500 | misura finale, evidenza lontana |

Per ogni set si salvano anche:
- quante frasi di evidenza ha ogni riga;
- quanto sono lontane fra loro (vicine / 2–9 / 10–19 / 20 o più frasi);
- quanto è lontana l'ultima frase di evidenza dal claim del preambolo, in token (meno di 1.024 / 1.024–4.096 / oltre 4.096), per ciascun modello.

```sh
python -m evidence.make_splits --binary-splits <manifest del piano binario> --out data/evidence/splits/
```

## 3. Il mix `ev_head` (training della sola testa)

Ogni riga ha un tipo, che dice come la loss la usa ([08](08_testa_e_loss.md) §3):
- `supervised`: positivo con evidenza nota → le frasi di evidenza devono superare la soglia, le altre no;
- `mil_neg`: negativo → **nessuna** frase deve superare la soglia;
- `mil_pos`: positivo senza evidenza nota → almeno una frase deve superare la soglia (solo nell'esperimento H-MIL).

| Famiglia | Righe | Tipo | Peso della riga | Peso delle frasi non di evidenza |
|---|---:|---|---:|---:|
| FEVER supportati | 6.000 | `supervised` | 1,0 | 0,3 |
| HoVer supportati | 5.000 | `supervised` | 1,0 | 0,5 |
| C2D / CG2C con posizioni | 2.000 | `supervised` | 1,0 | 1,0 |
| Etichettati col modello | 3.000 | `supervised` | 0,5 (ripetuti altrove: 0,3) | 0,5 |
| Evidenza spostata | 4.000 | come l'origine | come l'origine | come l'origine |
| Negativi con documenti lunghi | 6.000 | `mil_neg` | 1,0 | — |
| Positivi senza evidenza (solo H-MIL) | 3.000 | `mil_pos` | 1,0 | — |

Perché il peso delle frasi non di evidenza è basso per FEVER (0,3): in FEVER gli annotatori hanno segnato un gruppo di frasi sufficiente, non tutte. Una frase non segnata può essere evidenza valida, quindi la si penalizza poco.

Lo script controlla questi vincoli e si ferma se non sono rispettati:

| Vincolo | Valore |
|---|---|
| Positivi con 2 o più frasi di evidenza | ≥ 40% |
| Positivi con evidenza distante 10 o più frasi | ≥ 25% |
| Positivi con evidenza oltre 1.024 token dal preambolo | ≥ 15% |
| Positivi da testi non Wikipedia | ≥ 30% |
| Documenti con più di 50 frasi | ≥ 15% |
| Negativi | 20–25% |
| Negativi con una frase molto simile al claim (per parole) | ≥ 50% dei negativi |
| Documenti in comune con dev o test | 0 |

## 4. Il mix `ev_joint` (training anche del modello, fase 2)

Il mix di training del piano binario ([../fine_tuning_factcheck/02e_mix_finale.md](../fine_tuning_factcheck/02e_mix_finale.md)), in cui le righe che hanno evidenza la portano con sé, più le righe di `ev_head` (fino al 20–30% del totale). Tutte le righe servono al verdetto; il termine di evidenza si applica solo dove c'è un tipo.

```sh
python -m evidence.make_mix --recipe recipes/ev_head.json  --out data/evidence/mixes/ev_head.jsonl
python -m evidence.make_mix --recipe recipes/ev_joint.json --out data/evidence/mixes/ev_joint.jsonl
```

La ricetta (`recipes/*.json`) contiene famiglie, numeri, pesi e seme: lo stesso mix si ricostruisce identico.
