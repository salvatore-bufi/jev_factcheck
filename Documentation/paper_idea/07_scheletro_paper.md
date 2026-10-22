# Scheletro del paper

Formato ACL: 8 pagine (long paper) + riferimenti + Limitations + appendici illimitate.

## 1. Titoli possibili

Se il contributo principale è il **metodo** (fine-tuning con guadagno netto):
- *Specializing Typed Decision Models for Grounded Fact Checking without Forgetting*
- *One Forward Pass to Check a Claim: Adapting Typed Decision Models to Document-Grounded Verification*

Se il contributo principale è l'**analisi**:
- *Are Typed Decision Models Good Fact Checkers? An Evaluation on Document-Grounded Claim Verification*
- *Calibrated, Cheap, and Grounded? Typed Decision Models as Fact Checkers*

## 2. Abstract di prova (versione "metodo", numeri da riempire)

> Typed decision models (TDMs) answer typed questions about a structured state with one forward pass and return a probability for every option, promising cheap and calibrated decisions. We present the first systematic study of TDMs as document-grounded fact checkers. On LLM-AggreFact, evaluated zero-shot with all choices made on held-out data from other sources, open TDMs reach [X] balanced accuracy, [above / below] specialized checkers of similar size, and [Y] points above their own base LLMs read in the same way, showing that decision training [does / does not] transfer to grounding. Their probabilities are [well / poorly] calibrated in domain but thresholds transfer [poorly] across domains. Decomposing the decision into typed sub-questions [helps / does not help]. We then specialize two open TDMs with LoRA, optimizing only the option-letter distribution read at inference and replaying self-distilled generic decisions. The adapted models gain [Z] points zero-shot on LLM-AggreFact and [W] on out-of-domain benchmarks while losing at most [V] points on general decision benchmarks. We release code, adapters, predictions, and a diagnostic set of [N] controlled perturbations.

## 3. Sezioni (versione completa, ~8 pagine)

| Sezione | Contenuto | Pagine |
|---|---|---:|
| 1. Introduction | i TDM: perché sono interessanti (una passata, probabilità, costo); il buco (nessuna valutazione sul fact checking documento–claim, nessun fine-tuning dei modelli aperti); domande di ricerca; contributi in 4 punti | 1 |
| 2. Background & Related Work | TDM (Jev, Winnow, JPT, Laya, ...); fact checker specializzati (MiniCheck, FactCG, AlignScore, ...); LLM-as-judge e calibrazione; adattamento senza oblio | 0,75 |
| 3. Task and Evaluation Protocol | definizione del compito; prompt e domande tipizzate; protocollo zero-shot (dev nostro, test una volta); suite (AggreFact, fuori dominio, diagnostico, oblio); metriche | 1 |
| 4. How Good Are TDMs as Fact Checkers? | RQ1–RQ3 e RQ6: tabella principale, TDM contro basi generiche, calibrazione e trasferimento della soglia, costo | 1,5 |
| 5. What Do TDMs Get Wrong? | RQ4–RQ5: scomposizione in domande, set diagnostico, analisi degli errori | 1 |
| 6. Specializing TDMs | metodo (loss sulle lettere, replay, dati, varianti); risultati; controlli (basi generiche, loss, replay, dati) | 1,75 |
| 7. Discussion | quando usare un TDM e quando un checker specializzato; cascate; implicazioni per i costruttori di TDM | 0,5 |
| 8. Conclusion | | 0,25 |
| Limitations | lingua, supporto contro verità, modelli chiusi, calibrazione fra domini, dati sintetici | (fuori conteggio) |
| Appendici | prompt completi; tabelle per fonte; iperparametri; costi e calcolo; datasheet del set diagnostico; analisi degli errori estesa; checklist dell'audit | |

## 4. Contributi (da scrivere nell'introduzione)

1. **Il primo studio sistematico dei typed decision model come fact checker documento–claim**: [K] TDM aperti e chiusi contro checker specializzati e contro le loro basi generiche, con un protocollo zero-shot su LLM-AggreFact e test fuori dominio.
2. **Un'analisi della calibrazione e della robustezza**: trasferimento delle soglie fra domini, scomposizione in domande tipizzate, set diagnostico di perturbazioni controllate.
3. **Un metodo di specializzazione** che ottimizza la sola distribuzione sulle opzioni e usa replay auto-distillato, con misura esplicita dell'oblio.
4. **Rilascio** di codice, adapter, previsioni e set diagnostico.

## 5. Figure chiave

| Figura | Messaggio in una frase |
|---|---|
| Fig. 1 (pagina 1) | schema: stato (documento, claim) + domande tipizzate → una passata → probabilità per opzione; confronto con checker specializzato e LLM generativo |
| Fig. 2 | qualità contro costo per 1.000 claim: i TDM sul fronte di Pareto? |
| Fig. 3 | diagrammi di affidabilità: in dominio contro fuori dominio |
| Fig. 4 | sensibilità e invarianza sul set diagnostico, per modello |
| Fig. 5 | specializzazione contro oblio: guadagno su AggreFact (asse x) contro calo su JevBench/Kev (asse y), per configurazione |

## 6. Le obiezioni prevedibili dei revisori, e la risposta pronta

| Obiezione | Risposta (con riferimento all'esperimento) |
|---|---|
| "Un LLM qualsiasi letto con le log-probabilità fa lo stesso" | RQ2 / B-ctrl: confronto diretto con le basi generiche, stesso prompt e stessa lettura |
| "Avete scelto soglia e prompt sul benchmark" | protocollo zero-shot: tutto scelto su dati nostri; la soglia dal dev di AggreFact è solo un confronto secondario dichiarato |
| "Il fine-tuning rovina il modello" | oblio misurato su tre benchmark esterni; curva specializzazione–oblio |
| "Contaminazione dei dati di training" | decontaminazione documentata, con numeri per fonte |
| "Jev è chiuso, non è riproducibile" | conclusioni principali sui modelli aperti; versione, date e risposte di Jev rilasciate |
| "Differenze piccole, entro il rumore" | intervalli di confidenza per documento, test appaiati, 3 seed |
| "Solo inglese" | dichiarato nelle limitazioni; eventuale estensione futura |

## 7. Prossimi passi concreti

1. Decidere fra paper unico (consigliato) e due paper ([02_opzioni_di_paper.md](02_opzioni_di_paper.md), Opzione 5).
2. Completare lo Step 0 e la baseline con il protocollo zero-shot: è la base di tutte le tabelle.
3. Aggiungere le baseline B1 (basi generiche) e B3 (Bespoke-MiniCheck-7B, MiniCheck-Flan-T5-L): danno subito la tabella principale.
4. Valutare se uscire con un preprint della Parte A a metà novembre.
