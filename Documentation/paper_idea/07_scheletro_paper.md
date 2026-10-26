# Scheletro del paper

Formato ACL: 8 pagine (long paper) + riferimenti + Limitations + appendici. Rivisto il 26 ottobre 2026 per il paper di analisi ([02](02_opzioni_di_paper.md), Opzione 2).

## 1. Titoli possibili

Se il risultato di RQ2 è "la ricetta conta poco, la base e la lettura contano molto":
- *Same Backbone, Same Verdicts? What Decision Training Adds to Grounded Fact Checking*
- *Decision Models Are Their Backbones: A Controlled Study of Small Typed Decision Models as Fact Checkers*

Se il risultato di RQ2 è "la ricetta conta":
- *Same Backbone, Different Verdicts: What Decision Training Adds to Grounded Fact Checking*

Se il messaggio principale è RQ3:
- *Is It the Model or the Protocol? Typed Decision Models on Grounded Fact Checking*

## 2. Abstract di prova (numeri da riempire)

> Typed decision models (TDMs) answer typed questions about a state in one forward pass and return a probability for every option. They now top the LLM-AggreFact leaderboard, but their entries come from vendors with undocumented protocols, and recent work finds that the typed readout has no independent accuracy advantage over reading label probabilities from an ordinary LLM. We focus on TDMs that run on a single commodity GPU: [K] open models up to 9B parameters, five readout architectures from 0.4B to 9B, evaluated under a single frozen protocol on LLM-AggreFact and [M] out-of-domain grounding benchmarks, against specialized checkers and against their own base models read both naively and with training-free calibration. Exploiting [seven] TDMs built on the same 9B backbone and [five] on the same 4B backbone, we find that decision training adds [X] points of balanced accuracy over the calibrated base, while recipes on the same backbone differ by [Y]; the best of them comes within [D] points of 27B and closed TDMs. Protocol choices that leaderboards leave unstated (threshold, question polarity, one versus three questions, request composition) move a single model by up to [Z] points, more than the [1.3]-point spread among the five best leaderboard entries up to 9B, and change the ranking (Kendall τ = [t]). TDMs admit [W]% of unsupported claims at matched balanced accuracy, rarely separate contradicted from unstated content, and their thresholds transfer [poorly] across domains. Finally, training a grounding checker from a TDM rather than from its base [helps / does not help] ([V] points). We release predictions with full option distributions for every model, the evaluation harness, and a diagnostic suite.

## 3. Sezioni (~8 pagine)

| Sezione | Contenuto | Pagine |
|---|---|---:|
| 1. Introduction | i TDM in cima alla leaderboard del grounding; due domande aperte (cosa aggiunge l'addestramento rispetto alla base; quanto è solido il numero); contributi | 1 |
| 2. Background & Related Work | TDM e ricette aperte; valutazioni di Jev (incluso AggreFact); TDM per claim verification (PremiseCheck, cascate RAG); checker specializzati; letture senza training e calibrazione; fragilità | 0,75 |
| 3. Setup | compito; perché fino a 9B; modelli raggruppati per base; tre letture della base; protocollo congelato; dati e contaminazione; metriche (inclusa l'ammissione dei non supportati) | 1 |
| 4. RQ1–RQ2, RQ8: What does decision training add, and does the decision leaderboard predict it? | tabella principale; base ingenua contro calibrata contro TDM, per gruppo; dispersione delle ricette | 1,5 |
| 5. RQ3: Model or protocol? | sensibilità a soglia, polarità, numero di domande, composizione, troncamento, contaminazione; classifiche sotto protocolli diversi | 1 |
| 6. RQ4–RQ6: What do TDMs get wrong? | errori asimmetrici, contraddetto contro non stabilito, calibrazione e trasferimento, set diagnostico ridotto, analisi degli errori | 1,25 |
| 7. RQ7 and B-ctrl: Using and training TDMs as checkers | costo, accordo fra modelli, cascate contro escalation casuale; TDM contro base come punto di partenza per il fine-tuning | 1 |
| 8. Discussion & Conclusion | quando usare un TDM, un checker, una base calibrata; raccomandazioni per chi pubblica risultati di TDM sul grounding (protocollo minimo da dichiarare) | 0,5 |
| Limitations | lingua, supporto contro verità, contaminazione, modelli chiusi, bersaglio mobile, dati sintetici | (fuori conteggio) |
| Appendici | prompt e runtime di ogni modello; tabelle per fonte; varianti di protocollo complete; iperparametri di B-ctrl; costi ed energia; schede del set diagnostico; checklist dell'audit punto per punto | |

## 4. Contributi (da scrivere nell'introduzione)

1. **Il primo confronto controllato di molti TDM aperti piccoli (fino a 9B) sul fact checking su documento**: [K] modelli, stesse domande, stessi dati, stesse regole, con le distribuzioni complete delle opzioni rilasciate.
2. **Una risposta, sul grounding, alla domanda aperta dell'audit**: grazie a più TDM sulla stessa base, separiamo il contributo della base, della lettura e dell'addestramento da decision model.
3. **Una misura della sensibilità al protocollo** che mostra quanto delle differenze riportate dipende da scelte non dichiarate, e un **protocollo minimo** da dichiarare.
4. **Un'analisi degli errori e della calibrazione** orientata al deployment (ammissione dei non supportati, contraddetto contro non stabilito, trasferimento delle soglie) e un set diagnostico ridotto.
5. **Un esperimento controllato di fine-tuning**: TDM contro base come punto di partenza per un checker.

## 5. Figure chiave

| Figura | Messaggio in una frase |
|---|---|
| Fig. 1 (pagina 1) | per ogni base, la base ingenua, la base calibrata e i TDM come punti sullo stesso asse di BAcc: quanto della distanza è lettura e quanto è addestramento |
| Fig. 2 | la classifica cambia con il protocollo (grafico a "bump"), con la banda dei primi cinque della leaderboard |
| Fig. 3 | ammissione dei non supportati contro BAcc: modelli con la stessa BAcc e guardiani molto diversi |
| Fig. 4 | diagrammi di affidabilità in dominio e fuori dominio |
| Fig. 5 | qualità contro costo (ed energia) per 1.000 claim; Jev e i 27B come riferimento |

## 6. Le obiezioni prevedibili dei revisori, e la risposta pronta

| Obiezione | Risposta |
|---|---|
| "Jev su AggreFact è già stato valutato ([2609.37647](https://arxiv.org/abs/2609.37647))" | lo riproduciamo come controllo; il contributo sono i TDM aperti, il confronto con le basi e il protocollo |
| "La leaderboard ha già questi numeri" | protocollo non dichiarato; mostriamo che il protocollo vale quanto le differenze; i nostri numeri sono confrontabili fra loro |
| "Gli LLM sono già TDM (AnyJev, LLM-as-Jev)" | è la nostra ipotesi nulla, testata sul grounding con la lettura calibrata senza training |
| "Avete scelto soglia e prompt sul benchmark" | tutto scelto sul nostro dev; la grid search sul test è esplorativa e dichiarata |
| "Contaminazione" | media senza RAGTruth; fuori dominio scelto fuori dalla Decision Index; colonna `contamination_identifier` |
| "I modelli cambiano ogni settimana" | data di congelamento; il risultato di RQ3 non dipende dai modelli; harness e previsioni rilasciati |
| "Differenze piccole, entro il rumore" | intervalli per documento, test appaiati; per B-ctrl seed multipli o bootstrap |
| "Il fine-tuning è un controllo, non un metodo" | sì, per scelta: risponde a una domanda precisa; il metodo è un lavoro separato |
| "È un confronto su Qwen" | il 68% dei TDM fino a 9B della Decision Index, e tutti i primi 25, sono su Qwen: è lo stato dell'arte, e il disegno "stessa base" lo richiede; la generalità è verificata su tre famiglie non Qwen (Gemma-4-E4B, Ministral-3-8B, LFM2.5) e tre encoder (E2b), con il limite dichiarato che i non Qwen sono più deboli |
| "Perché solo fino a 9B?" | costo, deployment su una GPU, fine-tuning accessibile, gruppi per base più ricchi; i più grandi come riferimento |
| "Solo inglese" | dichiarato nelle limitazioni |

## 7. Prossimi passi concreti

1. Confermare l'Opzione 2, il vincolo dei 9B e la data di congelamento (30 novembre).
2. Step 0 (split, dev, fuori dominio): è la base di tutte le tabelle.
3. Finire i run dei TDM fino a 9B con adapter pronto; scrivere gli adapter delle basi con le due letture.
4. Jev 1.13 e 1.14 con il protocollo zero-shot; riprodurre il 78,6 di [2609.37647](https://arxiv.org/abs/2609.37647).
5. Preparare il preprint con A1–A3 per il 30 novembre.
