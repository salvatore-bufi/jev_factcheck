# Step 2c — Replay e oblio: il rischio vero è peggiorare il verdetto

## Il punto

Come nel piano binario ([../fine_tuning_factcheck/02c_replay_e_forgetting.md](../fine_tuning_factcheck/02c_replay_e_forgetting.md)), le altre domande del decision model non servono: **nessun replay** di domande generiche.

Il rischio è un altro. Se la loss di evidenza modifica (tramite la LoRA) gli stessi stati da cui si legge il verdetto, il verdetto può peggiorare o cambiare calibrazione senza che nessuno lo stia guardando. Il formato sandwich aggiunge un secondo rischio: un preambolo mai visto in training può spostare il verdetto anche senza nessun training ([01](01_baseline.md), B-SAND).

## Come si protegge il verdetto

| Strumento | Come | Quando |
|---|---|---|
| **Backbone congelato** | fasi 1a e 1b: si addestra solo la testa; LoRA e lettere non cambiano. Vale anche per H-FAITH, che calcola gradienti attraverso il backbone ma non ne aggiorna i pesi | **primo tentativo**: il verdetto è identico per costruzione |
| Verdetto riaddestrato con il preambolo | se B-FT-SAND perde più di 0,3 punti ([01](01_baseline.md) §1) | prima della fase 1 |
| Loss del verdetto sempre attiva | nel training congiunto ogni batch contiene anche la CE sulle lettere, su tutte le righe | fase 2 |
| Ancoraggio al verdetto | `β · KL(p_verdetto_congelato ‖ p)` sulle lettere | fase 2, se la calibrazione si sposta |
| λ piccolo | peso della loss di evidenza 0,2–0,5 | default della fase 2 |
| Monitoraggio | BAcc, AUC, ECE e soglia ottimale del verdetto su `dev_select` a ogni checkpoint | sempre |

Regola di adozione: un checkpoint con migliore evidenza si accetta solo se il verdetto su `dev_select` resta entro 0,3 punti di BAcc e 0,01 di ECE dal modello del piano binario (nel formato scelto).

## Il termine MIL sui negativi nel training congiunto

Nella fase 2 il termine "nessuna frase sopra la soglia" sui negativi passa anche per la LoRA. Spinge gli stati delle frasi dei documenti non supportati verso il basso: è coerente con il verdetto, ma è un secondo segnale sul verdetto che non passa dalle lettere. Si controlla con H6 ([07](07_esperimenti.md)) e, se sposta la calibrazione, si applica con `stop_grad` sul backbone (agisce solo sulla testa).

## Dati senza evidenza nel training congiunto

Le righe negative e quelle supportate senza evidenza restano nel mix: servono al verdetto. Togliere le negative dal training congiunto sbilancerebbe il verdetto verso "supportato".

## Criteri per passare allo step successivo

- Deciso, per ciascun modello, se il verdetto del piano binario si riusa così com'è nel formato sandwich o si riaddestra.
- Scelto se partire da backbone congelato (consigliato) o da training congiunto.
- Probabilità del verdetto congelato precalcolate sul mix, se si userà l'ancoraggio.
