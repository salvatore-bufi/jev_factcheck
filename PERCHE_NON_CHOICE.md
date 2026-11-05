In core/questions.py esiste già il tipo choice. Il front-end di KnowLine accetta fino a 255 opzioni (MAX_LABELS = 255 in knowline_server.py) e le etichetta A, B, C…
Quindi si può fare una seconda domanda con una frase per opzione, tipo "Quale frase del documento supporta il claim?", da affiancare a support_simple. Non serve addestrare niente.
Perché non è equivalente alla testa
Sceglie una frase sola. La risposta choice è un softmax su un solo token-etichetta, quindi le frasi competono tra loro. Molte evidenze però sono fatte di più frasi: HoVer è multi-hop, con 2–4 frasi. Con un top-k o una soglia sulle probabilità hai comunque punteggi relativi, non un "sì/no" per frase. La testa invece dà un punteggio indipendente a ogni frase.
Costa di più. Serve un secondo prefill, e le frasi entrano nel prompt due volte: nel documento e nelle opzioni, quindi circa il doppio dei token. La cache del prefisso fa risparmiare solo la parte del documento. La testa misurata aggiunge +0,5% di latenza, perché lavora sullo stesso forward del verdetto.
È un formato che il modello non ha mai visto. KnowLine è addestrato su Sì/No e su choice con poche opzioni. Con 30–80 opzioni ci si può aspettare bias di posizione (prime o ultime lettere) e probabilità mal calibrate. In più non si può riaddestrare: non abbiamo una LoRA.
Il legame col verdetto si perde. La testa legge gli stati nascosti del forward che ha prodotto il verdetto, quindi l'evidenza è "quello che il modello ha guardato per decidere". Una choice separata è un'altra domanda, e può indicare una frase anche quando il verdetto è "non supportato", o il contrario.
I documenti lunghi pesano. Più frasi significa più opzioni e un prompt più lungo. Il read_twice di Gen5.5 si disattiva sopra twice_max_tokens.
Cosa ti consiglio
Aggiungerlo come baseline E-CHOICE in evidence/baselines.py, accanto a E-LEX, E-ATT ed E-ABL. Stesse metriche, stessa divisione A/B dei dev:

se batte la testa, o ci arriva vicino, hai una soluzione senza training e la scelta è facile;
se resta sotto, hai un argomento concreto per la testa, cioè un "perché non chiedere semplicemente al modello?" con i numeri.
Il risultato di E-ABL fa pensare che la choice possa avere problemi. Su 5 righe KnowLine diceva "supportato" anche con una frase sola (F1 0,45), quindi non sembra molto selettivo sulle singole frasi. È solo un'indicazione, da verificare.

Se vuoi, scrivo io E-CHOICE: probabilità per frase e selezione con soglia o top-k, scelte sulla metà A, come per le altre baseline. Va lanciato su GPU con --per-source 100 come E-ABL.