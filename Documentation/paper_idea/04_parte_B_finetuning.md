# Parte B — Fine-tuning: un esperimento nel primo paper, il metodo nel secondo

Il piano tecnico completo è in [../fine_tuning_factcheck/](../fine_tuning_factcheck/00_overview.md). Qui c'è solo ciò che serve **per i paper**. Rivisto il 26 ottobre 2026.

## 1. Cosa è cambiato

La prima versione proponeva quattro contributi tecnici. Oggi tre sono pubblicati in forma generale (per TDM generali, non per il grounding):

| Contributo previsto | Già pubblicato | Cosa resta nostro |
|---|---|---|
| Loss solo sulle lettere delle opzioni | LLM-as-Jev ([2610.02076](https://arxiv.org/abs/2610.02076)), Releasing the Inner Jev ([2610.17205](https://arxiv.org/abs/2610.17205)), Our Own Jev ([2610.29052](https://arxiv.org/abs/2610.29052)) | niente come metodo; resta la scelta motivata |
| Replay / ancoraggio contro l'oblio | ancoraggio KL in LLM-as-Jev; Inner Jev: nessun calo su BoolQ, MMLU, ARC... dopo il training; calibrazione in calo su 3 modelli su 4 | la misura sul grounding, non il metodo |
| Negativi per cancellazione dell'evidenza e coppie minime | PACT ([2609.35865](https://arxiv.org/abs/2609.35865)): coppie contrastive con certificato di rimozione della frase decisiva, termine di "necessità dell'evidenza", consistenza alle permutazioni | l'applicazione ai dati di grounding (MiniCheck, FactCG) |
| Robustezza alle varianti di domanda | in parte in PACT (permutazioni) e negli studi di fragilità | il test su polarità e composizione della richiesta |

Inoltre la guida di fine-tuning è passata a **una sola domanda `noul`** con loss binaria sul margine fra le due lettere, d1-3B per iterare e replay **non** obbligatorio ([../fine_tuning_factcheck/summary/02c_replay_e_forgetting.md](../fine_tuning_factcheck/summary/02c_replay_e_forgetting.md)). Il paper va allineato a quella guida.

## 2. Nel primo paper: B-ctrl

### L'ipotesi

> A parità di dati di grounding, di loss e di budget, un checker ottenuto da un **TDM** è migliore (BAcc fuori dominio, calibrazione) di uno ottenuto dalla **sua base generica**.

Nessuno l'ha testata. Le ricette pubblicate addestrano TDM generali; i checker specializzati (MiniCheck, FactCG) partono da basi generiche. La risposta dice se l'addestramento da decision model è un buon punto di partenza per un checker, che è la domanda pratica di chi deve costruirne uno.

### Il disegno minimo

| Elemento | Scelta |
|---|---|
| Coppia principale | elephant-1 contro Qwen3.5-9B (oppure JPT-9B contro Qwen3.5-9B-Base); replica a metà taglia: KnowLine-4B (o pplx-decider-flash) contro Qwen3.5-4B; **replica non Qwen**: Winnow-E4B contro Gemma-4-E4B-it; pilota su d1-3B contro LFM2.5-VL-3B. Tutto entro il vincolo dei 9B |
| Dati | mix di stadio 1 della guida (fonti pubbliche + MiniCheck C2D/D2C + ANLI difficile), stesso per entrambe ([../fine_tuning_factcheck/02e_mix_finale.md](../fine_tuning_factcheck/02e_mix_finale.md)) |
| Domanda | una `noul` (quella scelta sul nostro dev secondo [../fine_tuning_factcheck/02d_varianti_domanda.md](../fine_tuning_factcheck/02d_varianti_domanda.md)) |
| Loss | CE binaria sul margine fra le due lettere ([../fine_tuning_factcheck/04_loss.md](../fine_tuning_factcheck/04_loss.md)) |
| Training | LoRA, stesso rank, passi e learning rate; 3 seed se il budget lo consente, altrimenti 1 seed + bootstrap |
| Valutazione | AggreFact zero-shot (con e senza RAGTruth), fuori dominio, ECE; facoltativo: un sottoinsieme della Decision Index pubblica per sapere quanto il TDM resta generale |
| Curva | BAcc contro numero di esempi (es. 0, 2k, 8k, 32k): il vantaggio del TDM, se c'è, dovrebbe essere massimo con pochi dati |

### Possibili esiti

| Esito | Lettura |
|---|---|
| TDM-FT ≫ base-FT, soprattutto con pochi dati | l'inizializzazione da decision model aiuta: consiglio pratico ai costruttori di checker |
| TDM-FT ≈ base-FT | il valore è nei dati: coerente con l'audit; RQ2 e B-ctrl si confermano a vicenda |
| TDM-FT < base-FT | il post-training da decision model irrigidisce il modello: risultato sorprendente, da controllare con un secondo seed e un secondo TDM |

Tutti e tre sono pubblicabili dentro il paper di analisi.

### Costo

| Voce | Stima |
|---|---|
| Codice di training (già progettato nella guida) | 1,5–2 settimane |
| Pilota d1-3B (2 corse) | 1–2 giorni su una GPU da 24 GB |
| Coppia 9B, 2 corse × (1–3 seed), più la curva sui dati | ~1 settimana su L40S (LoRA in BF16) o su una GPU da 24 GB con QLoRA |
| Coppia 4B (facoltativa) | 2–3 giorni |
| Valutazione | 2–3 giorni |

## 3. Nel secondo paper: il metodo completo

Se B-ctrl dà un vantaggio al TDM, il secondo paper diventa "come specializzare un TDM per il grounding", con la novità spostata dove la letteratura non arriva:

1. **Dati di grounding con certificato** (stile PACT, ma sulle ricette C2D/D2C/CG2C): coppie minime e rimozione dell'evidenza, con il termine di necessità dell'evidenza.
2. **Tre classi invece di due**: supportato / contraddetto / non stabilito, per affrontare il "terzo valore soppresso" ([Sys1Cal](https://arxiv.org/abs/2609.35342)) e la debolezza della domanda sulla contraddizione che vediamo nei nostri dati ([03](03_parte_A_valutazione.md) §2).
3. **Robustezza a polarità e composizione** come obiettivo di training, misurata sul set diagnostico completo.
4. **Confronto con lo stato dell'arte fino a 9B**: Nimble-v4 9B (77,8, solo API), Bespoke-MiniCheck-7B (77,4), Clef-flash (76,6) e i checker; i 27B (79,0–79,5) come tetto di riferimento. Obiettivo realistico: un 9B (o un 4B) che superi tutti i modelli fino a 9B e si avvicini ai 27B.
5. **Oblio e calibrazione**: oblio sulla Decision Index pubblica (attenzione: contiene RAGTruth e HoVer, da escludere dal conto); calibrazione fuori dominio.

| Esperimento | Perché |
|---|---|
| B-main | TDM-FT contro i migliori zero-shot e i checker |
| B-loss | CE contro Brier contro CE + termini PACT |
| B-anchor | con e senza ancoraggio KL ai target del modello di partenza (P-KL della guida) |
| B-data | leave-one-source-out o almeno tre mix |
| B-3way | due classi contro tre |
| B-seed | 3 seed per la configurazione finale |

Tempi: 2–3 mesi. Venue: EMNLP 2027 o il ciclo ARR utile successivo ([06](06_venue_calendario_rischi.md)).

## 4. Cosa rilasciare

- Adapter LoRA di B-ctrl (licenze: JPT e Nimble non commerciali; elephant-1 MIT; KnowLine, pplx-decider-flash e Qwen3.5 Apache 2.0) e codice di training.
- Ricette dei dati (script, non i testi di terze parti se le licenze non lo consentono).
- Cache delle previsioni su tutti i test, come già fatto per gli altri modelli.
