# Step 2c — Replay e oblio: il rischio vero è peggiorare il verdetto

## Il punto

Come nel piano binario ([../fine_tuning_factcheck/02c_replay_e_forgetting.md](../fine_tuning_factcheck/02c_replay_e_forgetting.md)), le altre domande del decision model non servono: **nessun replay** di domande generiche.

Qui c'è un rischio nuovo: la loss di evidenza modifica (tramite la LoRA) gli stessi stati nascosti da cui si legge il verdetto. Il verdetto può peggiorare, o cambiare calibrazione, senza che nessuno lo stia guardando.

## Come si protegge il verdetto

| Strumento | Come | Quando |
|---|---|---|
| **Backbone congelato** | si addestra solo l'aggregatore sugli stati del modello del verdetto; LoRA e lettere non cambiano | **primo tentativo** (H1 in [07](07_esperimenti.md)): il verdetto è identico per costruzione |
| Loss del verdetto sempre attiva | nel training congiunto ogni batch contiene anche la CE sulle lettere, su tutte le righe (positive e negative) | se si passa al training congiunto |
| Ancoraggio al verdetto | `β · KL(p_verdetto_congelato ‖ p)` sulle lettere | se la calibrazione del verdetto si sposta |
| λ piccolo | peso della loss di evidenza 0,2–0,5 | default del training congiunto |
| Monitoraggio | BAcc, AUC, ECE e soglia ottimale del verdetto su `dev_select` a ogni checkpoint | sempre |

Regola di adozione: un checkpoint con migliore evidenza si accetta solo se il verdetto su `dev_select` resta entro 0,3 punti di BAcc e 0,01 di ECE dal modello del piano binario.

## Dati senza evidenza nel training congiunto

Le righe negative e quelle supportate senza evidenza (`evidence_quality = none`) restano nel mix: servono al verdetto. La loss per frase le maschera. Togliere le negative dal training congiunto sbilancerebbe il verdetto verso "supportato".

## Criteri per passare allo step successivo

- Scelto se partire da backbone congelato (consigliato) o da training congiunto.
- Probabilità del verdetto congelato precalcolate sul mix, se si userà l'ancoraggio.
