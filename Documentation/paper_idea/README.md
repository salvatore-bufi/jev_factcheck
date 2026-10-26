# Idea di paper: decision model tipizzati per il fact checking documento–claim

Cartella di lavoro per decidere **che paper scrivere**, **con quali esperimenti** e **dove sottometterlo**. Prima versione: 8 ottobre 2026. **Rivista il 26 ottobre 2026** dopo una nuova ricerca bibliografica: in diciotto giorni il campo è cambiato molto, e alcune affermazioni della prima versione non reggono più. Il dettaglio di cosa è cambiato, con le fonti, è in [09_aggiornamento_ottobre_2026.md](09_aggiornamento_ottobre_2026.md).

## La risposta breve

1. **"Primo studio dei TDM come fact checker" non è più vero.** Jev è già stato valutato su **tutto il test di LLM-AggreFact** (78,6 zero-shot, una domanda, soglia 0,5, ECE 0,128: [2609.37647](https://arxiv.org/abs/2609.37647)) e su 495 claim di AggreFact contro un LLM di frontiera, con cascate ([2610.20634](https://arxiv.org/abs/2610.20634)). Quattro TDM sono stati usati come verificatori di premesse estratte da revisioni scientifiche ([PremiseCheck, 2610.21046](https://arxiv.org/abs/2610.21046)). Otto TDM sono stati confrontati su RAGTruth, con un test sulla formulazione della domanda ([Arize](https://arize.com/blog/decision-model-benchmark/)). Sulla **leaderboard di LLM-AggreFact** i primi posti sono ora di decision model (nimble-v4-max 79,5, Clef 79,0, Decider v1.2 78,7, Jev 1.13 78,7, d1-ultra 78,5), inviati dai produttori con un protocollo non documentato.

2. **Anche il metodo della Parte B è in gran parte pubblicato in forma generale.** Fine-tuning LoRA sulle lettere delle opzioni con ancoraggio KL ([LLM-as-Jev, 2610.02076](https://arxiv.org/abs/2610.02076)), effetti del fine-tuning sulla calibrazione e nessuna perdita sulle conoscenze generali ([Releasing the Inner Jev, 2610.17205](https://arxiv.org/abs/2610.17205)), coppie contrastive con certificato di rimozione dell'evidenza ([PACT, 2609.35865](https://arxiv.org/abs/2609.35865)). E le letture **senza training** di un LLM qualsiasi (AnyJev, LLM-as-Jev) arrivano vicino ai TDM: l'audit del filone ([2609.32160](https://arxiv.org/abs/2609.32160)) conclude che la lettura tipizzata "non ha mostrato un vantaggio di accuratezza indipendente" rispetto alla lettura delle probabilità delle etichette.

3. **Il buco che resta è preciso, ed è quello dove questa repo è forte:**
   - nessuno ha valutato **i TDM aperti** (oggi ~15 nella repo, da 3B a 27B, cinque architetture di lettura) **con un unico protocollo controllato** sul grounding; le voci della leaderboard non dicono domanda, soglia, chunking;
   - nessuno ha sfruttato l'**esperimento naturale "stessa base, ricette diverse"**: almeno sei TDM su Qwen3.5-9B (elephant-1, Nimble-9B, JPT-9B, d3-flash, StartLux-9B, Kev-9B), due su Qwen3.8-27B (Clef, Nimble-v4-max), uno su Gemma-4-12B (Winnow), uno su Qwen3.5-4B (KnowLine). Confrontati con la loro base letta in modo ingenuo **e** con una lettura calibrata senza training, rispondono alla domanda aperta dell'audit **sul compito dove conta di più**;
   - nessuno ha misurato **quanto della classifica dipende dal protocollo** (soglia, formulazione e polarità della domanda, una o tre domande, composizione della richiesta, troncamento, contaminazione). Dai nostri dati, le scelte di protocollo spostano un modello di 1,5–3,6 punti, più della distanza fra i primi cinque della leaderboard (1 punto).

4. **Il paper consigliato cambia di conseguenza**: un **paper di analisi** (Parte A riorientata), con un solo esperimento di fine-tuning, quello che nessuno ha fatto e che costa poco: *a parità di dati di grounding, partire da un TDM è meglio che partire dalla sua base?* Il metodo completo (Parte B) e il set diagnostico completo diventano il secondo paper.

5. **Scadenza**: ACL 2027 (Kyoto, 17–22 agosto 2027), sottomissione ARR entro il **4 gennaio 2027** (data ufficiale, ora confermata): mancano circa 10 settimane. Il rischio di essere anticipati è alto: **preprint su arXiv entro fine novembre**.

## I file

| File | Contenuto |
|---|---|
| [01_stato_arte_e_novita.md](01_stato_arte_e_novita.md) | Cosa esiste oggi (TDM, lavori su TDM e grounding, checker specializzati, letture senza training), dove resta il buco |
| [02_opzioni_di_paper.md](02_opzioni_di_paper.md) | Le alternative riviste, con pro, contro, costi e la nuova raccomandazione |
| [03_parte_A_valutazione.md](03_parte_A_valutazione.md) | Domande di ricerca, modelli, baseline, dati, metriche, esperimenti del paper di analisi |
| [04_parte_B_finetuning.md](04_parte_B_finetuning.md) | L'esperimento di fine-tuning del primo paper (B-ctrl) e il metodo completo per il secondo |
| [05_set_diagnostico.md](05_set_diagnostico.md) | Il set diagnostico: versione ridotta per il primo paper, versione completa per il secondo |
| [06_venue_calendario_rischi.md](06_venue_calendario_rischi.md) | Conferenze, calendario a ritroso fino al 4 gennaio, checklist di rigore, rischi |
| [07_scheletro_paper.md](07_scheletro_paper.md) | Titoli, abstract di prova, sezioni, figure, obiezioni dei revisori |
| [08_benchmark_e_modelli.md](08_benchmark_e_modelli.md) | La suite di benchmark e i modelli, aggiornati ai rilasci di ottobre |
| [09_aggiornamento_ottobre_2026.md](09_aggiornamento_ottobre_2026.md) | Registro della ricerca del 26 ottobre: cosa è uscito, cosa cambia per noi, fonti |

## Cosa abbiamo già (e cosa manca)

Numeri dal test di LLM-AggreFact (29.320 righe), regola a tre domande e soglia 0,30 fisse (quelle di Jev), media per fonte; da `table.md` e dai run locali in `results/`.

| Elemento | Stato |
|---|---|
| Jev-1.13 su tutto il test (79,09, dev-tuned) | fatto, con due limiti da dichiarare: domande e soglia scelte sul dev di AggreFact, e un controllo intermedio sul test ([../../legacy/results/INTERIM_AMENDMENT.md](../../legacy/results/INTERIM_AMENDMENT.md)) |
| TDM aperti su tutto il test, con cache di probabilità (e logit dove esistono) | fatto: Winnow-12B BF16 76,95, NVFP4 77,49, elephant-1 76,70, Rune-26B 76,24, JPT-9B 76,04, KnowLine-4B 75,65, d1-3B 71,84, CLM-8B 49,98 (la regola a tre domande non funziona per CLM: vedi `models/clm-8b/README.md`) |
| TDM aperti con adapter pronto, run completo da fare | d3-flash, d3-turbo (serve una GPU da 56 GiB+), StartLux-9B-Instant, StartLux-27B-AutoThink, Nimble-9B-v3 |
| TDM da aggiungere (usciti dopo l'8 ottobre) | Clef 27B (già nella cache HF), Nimble-v4 9B / v4-max 27B, pplx-decider v1.2 / flash, Drex v1.5, Kev-9B/27B; Jev 1.14 |
| Checker specializzati | FactCG 75,64 (fatto); Bespoke-MiniCheck-7B (adapter fatto, solo 32 righe) |
| Grid search su regola, temperatura e soglia, senza inferenza | fatto (`grid_search.py`): base dell'analisi di sensibilità al protocollo |
| Split nostri train/dev/test e test fuori dominio | da fare ([../fine_tuning_factcheck/00b_split_e_protocollo.md](../fine_tuning_factcheck/00b_split_e_protocollo.md)); molti dataset sono già nella cache HF |
| Basi generiche lette allo stesso modo (ingenua e calibrata senza training) | da fare: è il cuore del paper (RQ2) |
| Calibrazione, robustezza, costi misurati in modo sistematico | latenza, throughput, memoria ed energia già misurati dal benchmark; calibrazione e robustezza da fare |
| Fine-tuning | da fare: solo B-ctrl per il primo paper ([04](04_parte_B_finetuning.md)) |
