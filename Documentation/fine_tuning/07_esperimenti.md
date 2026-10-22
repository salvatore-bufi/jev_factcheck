# Step 7 — Piano degli esperimenti

## Obiettivo

Arrivare alla configurazione finale con il minimo numero di corse, cambiando **una cosa alla volta** e decidendo con criteri fissati prima. Ogni esperimento si fa prima su **JPT-9B** (più veloce); le scelte vincenti si ripetono su **Winnow-12B**.

## Regole generali

1. Ogni esperimento ha un ID, una configurazione di riferimento e una sola variazione.
2. Si giudica su `dev_select`, sui test comportamentali e sui benchmark di oblio ([06_valutazione.md](06_valutazione.md)). I test (test_id, test_ood, AggreFact) solo alla fine; LLM-AggreFact non guida nessuna scelta.
3. Una variazione si adotta se migliora la BAcc macro su `dev_select` di almeno **0,5 punti** (sotto, è nel rumore fra seed) **senza** peggiorare oblio (> 2 punti) o calibrazione (ECE +0,02).
4. Ogni esperimento si registra nella tabella di §8.

## Fase 0 — Baseline (senza training)

| ID | Cosa | Domanda | Costo |
|---|---|---|---|
| B0 | Regola di Jev a 3 domande, soglia 0,30 | numeri già noti | 0 |
| B1 | Soglia fittata sul dev | già fatto: +0,7 JPT, +0,4 Winnow | 0 |
| B2 | Soglia + temperatura sul dev | quanto vale ricalibrare | 0 (dalle cache) |
| B3 | Solo `support_simple` | quanto vale la regola a 3 domande | 0 (dalle cache) |
| B4 | Modelli base (Gemma-4-12B, Qwen3.5-9B) | punto di partenza del controllo C1 | ~2 h GPU |
| B5 | Blocchi C e D sui modelli di partenza | baseline di robustezza e oblio | ~2 h GPU |

**Uscita**: la baseline vera = migliore fra B1–B3 su `dev_select`, confermata su `dev_confirm`. B1–B3 si fittano sul nostro dev, non più su quello di AggreFact ([01_baseline.md](01_baseline.md)).

## Fase 1 — Pilota (JPT, `pilot_text`, S1)

Configurazione di riferimento **P0**: LoRA r16, lr 2e-5, 1 epoca, CE sulle lettere, replay 10% (λ = 1), varianti di domanda 30%, 4.096 token.

| ID | Variazione | Domanda | Decisione |
|---|---|---|---|
| **P0** | — | il fine-tuning funziona? | se non batte la baseline su `dev_select`, **fermarsi** e controllare rendering e loss prima di tutto il resto |
| P1 | lr 1e-5 / 5e-5 | sensibilità al learning rate | tenere il migliore |
| P2 | Brier al posto della CE | calibrazione e BAcc | Brier se ECE migliora senza perdere BAcc |
| P3 | SFT classico sul token di risposta | quanto vale la loss sulle lettere | conferma della scelta (atteso: peggiore in ECE) |
| P4 | senza replay | quanto oblio evita il replay | se l'oblio senza replay è < 1 punto, ridurre il replay |
| P5 | senza varianti di domanda | quanto valgono per la stabilità | tenere se riducono la varianza fra formulazioni |

Costo: ~6 corse × 5–10 h.

## Fase 2 — Dati (JPT, configurazione migliore della Fase 1)

| ID | Variazione | Domanda |
|---|---|---|
| D1 | Leave-one-source-out (una corsa per fonte tolta) | quale fonte aiuta e quale danneggia |
| D2 | Solo MiniCheck + ANLI (≈ ricetta di Bespoke-MiniCheck) | confronto con il miglior modello aperto |
| D3 | + FactKG trasformato in prosa ([02b](02b_dataset_generazione_llm.md), §3) | i ragionamenti di FactKG aiutano se il formato è giusto? |
| D4 | + FactKG in triple (com'è oggi) | solo come confronto con D3 |
| D5 | Scala: 10k / ~29k / ~85k coppie | curva di apprendimento: conviene la corsa piena? |
| D6 | + WiCE decontaminato | guadagno sospetto su WiCE? riportare a parte |
| D7 | FEVER senza claim con negazioni esplicite | il modello impara "not → falso"? (test "solo claim") |
| D8 | bilanciamento 50/50 contro distribuzione naturale | effetto su soglia e BAcc |
| D9 | + coppie generate (`remove_evidence`, `contradict_claim`) | i negativi difficili migliorano coppie minime e rimozione dell'evidenza? |
| D10 | + dati CG2C | il multi-hop sintetico aiuta HoVer e i domini nuovi? |

D1 è costoso (7 corse): si può fare sul sottoinsieme da 10k per risparmiare.

**Come leggere D1.** Se togliendo AVeriTeC la BAcc macro su `dev_select` (calcolata sulle fonti rimaste, per confrontare a parità di fonti) sale di 0,6 e i test comportamentali non cambiano, AVeriTeC (formato domanda–risposta) probabilmente disturba: ridurne la quota.

## Fase 3 — Formato e capacità (Winnow)

Si porta su Winnow la configurazione migliore di JPT (ricetta e iperparametri), poi:

| ID | Variazione | Domanda |
|---|---|---|
| W0 | configurazione migliore di JPT, su Winnow | il risultato si trasferisce? |
| F1 | S3 (tre domande) invece di S1 | la regola a 3 domande migliora dopo il training? |
| F2 | 8.192 token invece di 4.096 | aiuta sui documenti lunghi (LFQA, MediaSum, MeetingBank)? |
| F3 | rank 8 / 32 / 64 | capacità dell'adapter |
| F4 | 2 epoche | sovra-adattamento e calibrazione |
| F5 | nomi dei campi variati | robustezza a integrazioni diverse |

## Fase 4 — Loss contrastive (opzionale)

Solo dopo aver fissato la configurazione della Fase 3 ([04_loss.md](04_loss.md), §8):

| ID | Variazione | Domanda |
|---|---|---|
| X1 | + ranking logistico sulle coppie VitaminC (μ = 0,3) | migliorano coppie minime e riassunti? |
| X2 | X1 + coppie generate | più tipi di coppia aiutano? |
| X3 | + coerenza fra varianti | la stabilità migliora senza perdere BAcc? |
| X4 | μ ∈ {0,1; 0,3; 0,5} | quanto peso dare alla loss di coppia |

## Controlli finali (obbligatori)

| ID | Cosa | Perché |
|---|---|---|
| C1 | stesso training finale su Gemma-4-12B e Qwen3.5-9B base | se il modello base arriva allo stesso punto, il vantaggio è nei dati, non nel decision model |
| C2 | 3 seed della configurazione finale | varianza |
| C3 | Bespoke-MiniCheck-7B e FactCG con la stessa suite | riferimenti esterni nelle stesse condizioni |

## Calendario indicativo

| Settimana | Lavoro |
|---|---|
| 1 | Step 0 (split congelati, [00b](00b_split_e_protocollo.md)); Fase 0 (baseline sul nostro dev); codice di rendering, loss e training ([03](03_rendering_formato.md)–[05](05_training.md)); generazione di replay e varianti dal solo train |
| 2 | Fase 1 su JPT |
| 3 | Fase 2 su JPT (D2, D5, D9 prima; D1 sul sottoinsieme) |
| 4 | Fase 3 su Winnow (L40S) |
| 5 | controlli C1–C3, congelamento, test, export ([08](08_merge_export.md)) |
| poi | Fase 4 e dati CG2C se restano margini |

## 8. Registro degli esperimenti (modello)

Un file `finetune/experiments.md` (o un foglio) con una riga per corsa:

| ID | Modello | Mix | Variazione | BAcc `dev_select` | AUC `dev_select` | ECE `dev_select` | KL replay | Oblio (Jev/Kev/typed) | Solo claim | Durata | Decisione |
|---|---|---|---|---:|---:|---:|---:|---|---:|---|---|
| B1 | JPT | — | soglia da `dev_select` | | | | — | — / — / — | | 0 | baseline |
| P0 | JPT | pilot_text | — | | | | | | | | |
