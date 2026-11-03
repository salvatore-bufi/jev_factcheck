# Fact checking binario con evidenza evidenziata — panoramica

Questa cartella estende il piano binario [../fine_tuning_factcheck/](../fine_tuning_factcheck/00_overview.md). Il compito resta binario (documento + claim → supportato / non supportato), con un'uscita in più:

> **se il claim è supportato**, il modello indica le frasi del documento che lo stabiliscono.

Non si evidenzia nulla nei casi negativi: né le frasi che contraddicono il claim, né le parti del claim che mancano nel documento. Se il verdetto è "non supportato", l'uscita è solo il verdetto.

Vincolo di progetto: **la velocità deve restare quella di un decision model**, cioè un solo forward sul prompt e nessun token generato.

Revisione del 28 ottobre 2026: il piano è riscritto per i **tre modelli del piano binario** (d1-3B, JPT-9B, Winnow-12B) e integra le proposte di [idea.md](idea.md), adattate alle architetture reali di questi modelli ([00c](00c_modelli_e_architetture.md)). Cosa si è preso da idea.md, e in che forma, è in [00a](00a_scelta_approccio.md) §8.

## 1. Il problema in una pagina

**Input**: documento e claim (il documento sempre fornito, nessun retrieval).

**Output**:

```json
{"p_supported": 0.93, "supported": true,
 "evidence": [{"sentence": 0,  "start": 0,    "end": 61,   "score": 0.91},
              {"sentence": 39, "start": 4210, "end": 4262, "score": 0.84}]}
```

Se `supported` è `false`, `evidence` è vuoto.

**Esempio con evidenza sparsa** (il caso difficile):

> Claim: "Nancy R. Howell teaches at a seminary in Kansas City."
>
> S1 (inizio del documento): "Saint Paul School of Theology is a seminary located in Kansas City, Missouri."
> … 38 frasi irrilevanti …
> S40 (fine): "Nancy R. Howell is Professor of Theology at Saint Paul School of Theology."
>
> Verdetto: supportato. Evidenza: S1 e S40.

## 2. I modelli

| Modello | Ruolo | Architettura (dal config) | d |
|---|---|---|---:|
| **d1-3B** (LFM2.5-VL-3B) | **si itera qui**: tutti gli esperimenti H prima su d1 | 30 layer: 22 convoluzioni corte + 8 di attenzione piena | 2.048 |
| **JPT-9B** (Qwen3.5-9B) | conferma a 9B | 32 layer: 24 Gated DeltaNet (ricorrenti) + 8 di attenzione piena | 4.096 |
| **Winnow-12B** (Gemma 4, BF16) | candidato di massima qualità | 48 layer: 40 a finestra scorrevole (1.024 token) + 8 globali | 3.840 |

Fuori perimetro: CLM-8B (non legge lettere: è una somiglianza fra embedding, non c'è un `Answer:` da cui partire), Rune-26B (51,6 GB, oltre il budget di una GPU per il fine-tuning), Winnow NVFP4 sul server llama.cpp (non espone gli stati nascosti). Dettagli e conseguenze in [00c](00c_modelli_e_architetture.md).

Il fatto che conta per tutto il piano: **in tutti e tre i modelli solo 8 layer mescolano informazione a lunga distanza con un'attenzione esplicita**. Da lì si leggono le feature di attenzione (§3) e lì si può applicare una maschera per frase.

## 3. L'architettura scelta (in breve)

Un solo forward del decision model del piano binario, con il **formato sandwich**: il claim compare una volta *prima* del documento (preambolo) e una volta *dopo*, nella posizione nativa. Dallo stesso forward si leggono tre cose:

```text
prompt: [preambolo: claim] [prompt nativo: documento S1 … Sn, claim, domanda] Answer:
                │                               │
          forward unico del decision model (LoRA del piano binario)
                │                               │
  per ogni frase k:                             q = stati in "Answer:"
   h_k  = stati di fine frase, mix di 4 layer   │
   m_k  = media dei token della frase           ├─ logit delle lettere → P(supportato)   (come oggi)
   a_k  = attenzione di "Answer:" sulla frase   │
          negli 8 layer globali (8 × teste)     │
                ▼                               ▼
  aggregatore: proiezione → 256 + 2 layer di transformer NON causale su [x_0 = q, x_1 … x_n]
                │
  logit_k = v·X_k ;  soglia dell'istanza s_0 = v·X_0
                │
  frase k mostrata se logit_k > s_0  (e solo se P(supportato) > soglia del verdetto)
```

Perché ogni pezzo (dettagli e alternative scartate in [00a](00a_scelta_approccio.md)):

| Scelta | Motivo |
|---|---|
| Un forward, nessuna generazione | le citazioni generate costerebbero da +50% a +100% di latenza |
| Formato sandwich | le frasi vedono il claim (preambolo); `Answer:` ha il claim accanto, nell'ordine nativo: il verdetto del piano binario resta utilizzabile |
| Stati di fine frase + media della frase, da più layer | nessun marcatore nel testo; il token finale di una frase è spesso punteggiatura, e l'ultimo layer è specializzato a predire la lettera |
| Attenzione di `Answer:` sulle frasi | il modello calcola già dove guarda per decidere: 8 × 16 (o 8 × 32) numeri per frase, gratis |
| Aggregatore bidirezionale | la prima frase non vede l'ultima; l'aggregatore le fa comunicare, e comunicare con `q` |
| Soglia dell'istanza `s_0` | nessuna soglia fissa da tarare; con catene multi-hop la soglia si abbassa per la frase ponte |
| Testa mostrata solo sui positivi | il compito è evidenziare solo quando l'evidenza c'è |

Costo aggiuntivo: qualche milione di operazioni sull'aggregatore e il ricalcolo di una riga di attenzione per layer globale. Trascurabile rispetto al prefill, da misurare (§6).

## 4. Il percorso

| Step | File | Contenuto |
|---|---|---|
| 0. Split | [00b_split_e_protocollo.md](00b_split_e_protocollo.md) | split del piano binario + sottoinsiemi con evidenza annotata |
| 0a. Approccio | [00a_scelta_approccio.md](00a_scelta_approccio.md) | perché una testa nello stesso forward; evidenza sparsa; valutazione delle idee di idea.md |
| 0c. Modelli | [00c_modelli_e_architetture.md](00c_modelli_e_architetture.md) | d1-3B, JPT-9B, Winnow-12B: layer, stati da leggere, cosa si può mascherare, come servirli |
| 1. Baseline | [01_baseline.md](01_baseline.md) | formato (nativo / claim prima / sandwich), evidenza senza training, riferimento offline |
| 2a. Fonti | [02a_dataset_fonti_esistenti.md](02a_dataset_fonti_esistenti.md) | FEVER, HoVer, SciFact, WiCE: evidenza per frase; negativi utili |
| 2b. Generazione | [02b_dataset_generazione_llm.md](02b_dataset_generazione_llm.md) | C2D/CG2C con posizioni note, ablazione offline, evidenza sparsa costruita |
| 2c. Replay | [02c_replay_e_forgetting.md](02c_replay_e_forgetting.md) | il rischio è peggiorare il verdetto |
| 2d. Formato | [02d_varianti_domanda.md](02d_varianti_domanda.md) | il formato sandwich, per modello |
| 2e. Mix | [02e_mix_finale.md](02e_mix_finale.md) | righe con evidenza, negativi per il MIL, augmentation |
| 3. Rendering | [03_rendering_formato.md](03_rendering_formato.md) | segmentazione, posizioni, estrazione di stati e feature di attenzione |
| 4. Loss | [04_loss.md](04_loss.md) | ZLPR con soglia appresa, MIL sui negativi, varianti |
| 5. Training | [05_training.md](05_training.md) | testa, fase 1a (stati salvati), 1b (backbone congelato in linea), 2 (congiunto) |
| 6. Valutazione | [06_valutazione.md](06_valutazione.md) | F1, insieme completo, distanza, sufficienza, latenza |
| 7. Esperimenti | [07_esperimenti.md](07_esperimenti.md) | piano H, in ordine di costo |
| 8. Export | [08_merge_export.md](08_merge_export.md) | server per ciascun modello, stati e logit dallo stesso forward |
| Rischi | [09_rischi_checklist.md](09_rischi_checklist.md) | |

## 5. Le decisioni chiave

1. **Si parte dal piano binario.** Domanda, dati, loss sulle lettere e protocollo sono quelli di [../fine_tuning_factcheck/](../fine_tuning_factcheck/00_overview.md); qui si aggiunge l'evidenza.
2. **Formato sandwich come default**, scelto contro "claim prima del documento" con la misura B-SAND / B-ORD ([01](01_baseline.md)). Il prompt nativo resta intatto dopo il preambolo: è ciò che permette di riusare il verdetto del piano binario.
3. **Prima il verdetto, poi l'evidenza.** La testa si addestra prima su backbone congelato (il verdetto non cambia di un bit), poi, solo se serve, insieme alla LoRA.
4. **Si itera su d1-3B**, si conferma su JPT-9B e Winnow-12B.
5. **La loss usa anche i negativi** (nessuna frase deve superare la soglia dell'istanza), non solo i positivi annotati.
6. **Evidenza sparsa trattata esplicitamente**: aggregatore bidirezionale, documento intero in un forward (niente chunking con `max`), dati con evidenza messa apposta lontana.
7. **Velocità misurata**: la latenza con testa deve essere entro +3% di quella del solo verdetto.

## 6. Obiettivi misurabili

| Obiettivo | Soglia |
|---|---|
| Verdetto | nessun calo rispetto al modello del piano binario (BAcc e AUC su `dev_select`, entro 0,3 punti) |
| F1 per frase sull'evidenza (dev con evidenza) | > baseline lessicale (E-LEX) di almeno 15 punti; obiettivo indicativo ≥ 0,75 su FEVER/HoVer |
| Recall dell'insieme completo (tutte le frasi gold evidenziate) | ≥ 0,65 su HoVer (multi-hop) |
| Calo con la distanza fra frasi di evidenza | < 5 punti di F1 fra "adiacenti" e "oltre 20 frasi" |
| Sufficienza (le frasi evidenziate da sole danno "supportato") | ≥ 85% dei positivi veri |
| Latenza | ≤ +3% rispetto al solo verdetto (per d1-3B, dove il prefill è breve: ≤ +3% o ≤ 2 ms, il maggiore dei due) |

## 7. Tempi (stime)

| Fase | Tempo |
|---|---|
| Builder con evidenza (FEVER, HoVer, SciFact, WiCE) | 1–2 giorni di codice |
| Rendering sandwich + estrazione di stati e attenzione per i tre modelli | 2–3 giorni di codice |
| Etichettatura per ablazione offline e generazione con posizioni | 1–2 giorni di GPU (con d1-3B come etichettatore, meno) |
| Esperimenti H su d1-3B (stati salvati) | minuti per prova; 2–3 giorni in tutto |
| Conferma su JPT-9B e Winnow-12B | estrazione degli stati ~1–3 h per modello, poi come su d1 |
| Training congiunto LoRA + testa (se serve) | come una corsa del piano binario (+10–20%) |
