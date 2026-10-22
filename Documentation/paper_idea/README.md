# Idea di paper: decision model tipizzati per il fact checking documento–claim

Cartella di lavoro per decidere **che paper scrivere**, **con quali esperimenti** e **dove sottometterlo**. Scritta l'8 ottobre 2026.

## La risposta breve

1. **La novità c'è, ma il campo corre.** Dal rilascio di Jev (15 settembre 2026) sono usciti decine di paper sui *typed decision model* (TDM). L'audit dei primi 28 ([arXiv 2609.32160](https://arxiv.org/abs/2609.32160)) dice che **nessuno** si concentra su fact checking, allucinazioni, NLI o verifica di claim, e nessuno fa fine-tuning dei modelli aperti. Il lavoro più vicino ([JEV-as-a-Judge, 2609.26550](https://arxiv.org/html/2609.26550v1)) usa 240 esempi di HaluEval, solo con Jev. Lo spazio "TDM come fact checker su documento" è libero **oggi**: conviene muoversi in fretta e rendere pubblico un preprint presto.

2. **Non partirei da un dataset nuovo.** Proporre un dataset è un contributo a sé: richiede annotazione umana, documentazione, licenze e baseline, cioè mesi di lavoro e costi. Per questa domanda di ricerca esistono già benchmark solidi (LLM-AggreFact e altri). Il contributo più forte e più rapido è **un'analisi rigorosa dei TDM come fact checker**, più **un metodo per specializzarli senza rompere ciò che li rende utili**.

3. **Il piano consigliato: un paper principale, con un preprint anticipato.**
   - **Parte A, valutazione** (in gran parte già fatta in questa repo): TDM aperti e chiusi contro checker specializzati (MiniCheck, FactCG, Bespoke) e contro LLM generici letti allo stesso modo, su LLM-AggreFact **zero-shot**, su domini fuori distribuzione, con calibrazione e test di robustezza.
   - **Parte B, metodo**: fine-tuning di Winnow-12B e JPT-9B con loss sulle lettere di risposta e replay per non dimenticare, misurando quanto si guadagna sul fact checking e quanto si perde sulle decisioni generali.
   - **Contributo secondario facoltativo**: un piccolo **set diagnostico** (coppie minime, rimozione dell'evidenza, distrattori) ricavato da test esistenti e validato a mano su un campione. È molto più economico di un dataset nuovo ed è utile a tutta la comunità.

4. **Dove sottometterlo**: ACL 2027 (Kyoto, agosto 2027) tramite ACL Rolling Review; secondo il sito di ACL 2027, l'ultimo ciclo ARR utile è **gennaio 2027**. In alternativa EMNLP 2027 (scadenza attesa verso maggio–giugno 2027), oppure il workshop FEVER se si vuole uscire prima con la sola Parte A. Le date esatte vanno verificate sui siti ufficiali.

## I file

| File | Contenuto |
|---|---|
| [01_stato_arte_e_novita.md](01_stato_arte_e_novita.md) | Cosa esiste già (TDM, fact checker specializzati, LLM-as-judge), dove sta il buco, cosa si rischia |
| [02_opzioni_di_paper.md](02_opzioni_di_paper.md) | Le alternative (solo valutazione, metodo, dataset nuovo, set diagnostico) con pro, contro, costi e raccomandazione |
| [03_parte_A_valutazione.md](03_parte_A_valutazione.md) | Domande di ricerca, modelli, baseline, dati, metriche, esperimenti, tabelle e figure della parte di valutazione |
| [04_parte_B_finetuning.md](04_parte_B_finetuning.md) | Il metodo di specializzazione e gli esperimenti che servono per un paper |
| [05_set_diagnostico.md](05_set_diagnostico.md) | Come costruire e validare il set diagnostico (contributo secondario) |
| [06_venue_calendario_rischi.md](06_venue_calendario_rischi.md) | Conferenze, calendario a ritroso, checklist di rigore, rischi e come gestirli |
| [07_scheletro_paper.md](07_scheletro_paper.md) | Titoli possibili, abstract di prova, sezioni, tabelle e figure |

## Cosa abbiamo già (e cosa manca)

| Elemento | Stato |
|---|---|
| Jev-1.13 su tutto il test di LLM-AggreFact (79,09, dev-tuned) | fatto, ma con due limiti da dichiarare: domande e soglia scelte sul dev di AggreFact, e un controllo intermedio sul test (vedi `results/INTERIM_AMENDMENT.md`) |
| Winnow-12B (NVFP4 e BF16), JPT-9B, Rune-26B, FactCG su tutto il test, con cache di logit e probabilità | fatto |
| Soglie fittate sul dev di AggreFact per i modelli aperti | fatto |
| Split nostri train/dev/test e test fuori dominio | da fare ([../fine_tuning/00b_split_e_protocollo.md](../fine_tuning/00b_split_e_protocollo.md)) |
| Baseline con LLM generici letti allo stesso modo (stesso prompt, logit delle lettere) | da fare: è la prima obiezione che farà un revisore |
| Calibrazione, robustezza, costi e latenza misurati in modo sistematico | in parte (latenze nei log), da completare |
| Fine-tuning | da fare ([../fine_tuning/](../fine_tuning/00_overview.md)) |
