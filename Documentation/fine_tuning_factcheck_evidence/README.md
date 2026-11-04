# fine_tuning_factcheck_evidence — documentazione dell'implementazione

Documentazione del sistema **implementato** in [`evidence/`](../../evidence/): fact checking con evidenza a livello di frase, in un solo forward di KnowLine-4B.

| Cartella / file | Contenuto |
|---|---|
| [`testa_per_frase/`](testa_per_frase/00_overview.md) | la documentazione, passo per passo (00–15), con formule, figure e riferimenti al codice |
| [`figures/`](figures/) | le figure (PNG per la lettura, PDF vettoriale); sorgenti TikZ in `figures/src/`, ricompilabili con `sh figures/src/build.sh` |
| [`pdf/`](pdf/) | tutta la documentazione in un PDF, generato con `sh pdf/build.sh` |

**Da dove partire:**

- [00 overview](testa_per_frase/00_overview.md);
- [08 testa e loss](testa_per_frase/08_testa_e_loss.md) per l'architettura;
- [15](testa_per_frase/15_recap_architettura_layer_loss.md) per tutte le formule in una pagina;
- [13](testa_per_frase/13_checklist.md) per i comandi.

**Riferimenti:**

- Piano originale, con tutte le varianti non implementate: [`../fine_tuning_factcheck_evidence_complete/`](../fine_tuning_factcheck_evidence_complete/testa_per_frase/00_overview.md).
- Guida pratica al codice: [`evidence/code_report.md`](../../evidence/code_report.md).
- Stato del lavoro: [`CLAUDE_README.md`](../../CLAUDE_README.md).

Nota sui riferimenti nel codice: le intestazioni di `config.py`, `data/make_splits.py`, `data/__init__.py` ed `evidence/README.md` puntano sia a questa documentazione sia al piano. Le altre intestazioni citano `.../testa_per_frase/NN_*.md` con numeri di sezione (§) **del piano**, che sta in `../fine_tuning_factcheck_evidence_complete/`; i file di questa cartella con lo stesso numero trattano lo stesso passo. `sentences.py`, `render.py` e `model.py` non sono stati toccati: il loro sha256 definisce dataset, feature e release.
