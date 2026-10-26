# Step 2c — Replay e oblio: perché non servono e cosa fare invece

## 1. Che cos'è il replay e perché qui non serve

Quando si fa il fine-tuning di un modello su un solo compito, il modello tende a peggiorare sugli altri (*catastrophic forgetting*). Per un decision model che deve restare generale, la difesa è il **replay**: mescolare al training domande di altro tipo, con target presi dal modello di partenza.

In questo piano il modello serve **solo** a verificare claim. Se peggiora su altre domande (routing di ticket, classificazioni di email, ...), non si perde nulla di ciò che si usa. Quindi:

- nessuna domanda di replay da generare;
- nessun target del modello di partenza da calcolare per domande generiche;
- nessun benchmark di oblio (JevBench, Kev-v9, typed-decisions) nella valutazione;
- il training contiene solo esempi di fact checking a due opzioni.

Se in futuro lo stesso modello dovesse servire anche per altre domande, questo piano non basta: va seguito quello in `Documentation/fine_tuning/`.

## 2. Che cosa si può comunque "dimenticare"

| Rischio | Come si vede | Perché succede |
|---|---|---|
| **Calibrazione** | probabilità quasi tutte vicine a 0 o 1; ECE in aumento; soglia ottimale lontana da 0,5 | la cross-entropy continua a spingere anche sugli esempi già giusti |
| **Generalizzazione** | ottimo su test_id, peggio della baseline su test_ood | il modello impara lo stile dei dati sintetici |
| **Robustezza del prompt** | crollo se cambia una parola del prompt | solo se in produzione il prompt non è identico ([02d](02d_varianti_domanda.md)) |

Difese già previste negli altri step: loss sulle sole lettere ([04](04_loss.md)), una epoca per stadio, secondo stadio su testi reali ([02e](02e_mix_finale.md)), soglia e temperatura rifittate sul dev.

## 3. Un ancoraggio opzionale: KL verso il modello di partenza

### 3.1 Cosa fa

Si aggiunge alla loss un termine che penalizza l'allontanamento dalle probabilità del modello di partenza **sugli stessi esempi di fact checking**:

```text
L = CE(lettere, etichetta) + β · KL(p0 ‖ p)
KL(p0 ‖ p) = p0 · ln(p0 / p) + (1 − p0) · ln((1 − p0) / (1 − p))
```

dove p0 è P(supportato) del modello di partenza e p quella del modello in training.

Esempio: p0 = 0,80, p = 0,60 → KL = 0,8 · ln(0,8/0,6) + 0,2 · ln(0,2/0,4) = 0,8 · 0,288 + 0,2 · (−0,693) = 0,230 − 0,139 = **0,092**.

Effetto: dove il modello di partenza era già giusto il training non lo sposta inutilmente; dove sbagliava, la CE prevale (con β piccolo).

### 3.2 Perché non è il default

Con la domanda singola il modello di partenza dice "sì" troppo facilmente (soglia ottimale 0,90–0,95). Un β alto **conserverebbe anche questo difetto**. Si usa solo nell'esperimento P-KL ([07](07_esperimenti.md)), con β ∈ {0,05; 0,1}, se il modello fine-tunato peggiora la calibrazione o test_ood.

### 3.3 Come preparare p0 (una volta, prima del training)

1. Scrivere il mix in un file temporaneo nel formato degli split (una riga per esempio).
2. Calcolare p con `finetune/score.py` e il modello di partenza, con la stessa domanda e lo stesso troncamento del training.
3. Salvare p0 nella riga (`meta.p0`), che il rendering copia nel campo `teacher_p` ([03](03_rendering_formato.md)).

Costo: un'inferenza sul mix (~1 h per JPT su una L40S per ~60k esempi brevi; documenti lunghi di più).

Per ANLI difficile p0 c'è già (`meta.p0` da [02a](02a_dataset_fonti_esistenti.md) §3).

## 4. Cosa misurare al posto dell'oblio

| Misura | Su cosa | Allarme |
|---|---|---|
| ECE e Brier | dev_select | sopra la baseline |
| Distanza fra BAcc a soglia 0,5 e a soglia fittata | dev_select | > 1 punto |
| BAcc su test_ood | alla fine | sotto la baseline |
| (facoltativo) 200 domande di `LocalLLaMA/typed-decisions` test | | solo per sapere quanto il modello si è specializzato; nessun vincolo |

## Criteri per passare allo step successivo

- Confermato che il modello deve solo verificare claim (altrimenti → piano in `Documentation/fine_tuning/`).
- Se si farà P-KL: p0 calcolata e salvata nelle righe del mix.
