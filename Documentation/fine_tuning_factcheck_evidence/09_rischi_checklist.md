# Rischi e checklist

## 1. Rischi principali

| Rischio | Come si riconosce | Prevenzione | Rimedio |
|---|---|---|---|
| **Posizioni dei token sbagliate** | la testa non impara; F1 vicino a E-RAND | test di [03](03_rendering_formato.md) §6 (decodifica degli intervalli delle frasi) | correggere la mappa degli offset (escaping JSON di d1 e Winnow, preambolo) |
| **Segmentazione diversa fra training e produzione** | buono in valutazione, frasi sfasate in produzione | un solo modulo `sentences.py`, con hash in `decision.json` | riaddestrare la testa se il segmentatore cambia |
| **Il preambolo sposta il verdetto** | B-SAND / B-FT-SAND sotto B-1D / B-FT | misura prima di tutto ([01](01_baseline.md) §1) | verdetto riaddestrato con il preambolo; o formato nativo se H4 dice che il preambolo vale poco |
| **Il claim non arriva alle frasi lontane** | F1 crolla oltre 1.024 token (Winnow) o oltre 4k (d1, JPT) | metrica per distanza in token; righe "lontano dal preambolo" nel mix | più dati lontani; feature di attenzione (H2b); fase 2 |
| **Evidenza sparsa non trovata** | F1 alto su FEVER, basso su HoVer e oltre 10 frasi | aggregatore bidirezionale, preambolo, righe sparse costruite | più dati sparsi; H3c; H-PAIR; fase 2 |
| **Verdetto peggiorato** | BAcc o ECE del verdetto cambiano | fasi 1 con backbone congelato; λ basso in fase 2 | tornare alla fase 1; ancoraggio β·KL |
| **La testa si spegne sui positivi incerti** | recall bassa sulle righe con P vicina alla soglia del verdetto | μ_neg basso; controllo per fascia di P ([04](04_loss.md) §3) | ridurre μ_neg; `min_evidence = 1` in produzione |
| **Annotazioni incomplete** | frasi giuste contate come errori; precisione bassa ma sufficienza alta | ω basso per FEVER; sufficienza come metrica | leggere a mano i "falsi positivi" |
| **Bias di posizione** | evidenzia spesso la prima frase o il titolo | righe sparse con evidenza in posizioni varie; H9 | togliere l'embedding di posizione |
| **Scorciatoia lessicale** | la testa ≈ E-LEX; fallisce su parafrasi e frasi ponte; negativi "accesi" | negativi lunghi con frasi simili (MIL); dati con parafrasi e frasi ponte | più negativi di questo tipo; μ_neg più alto |
| **Etichette per ablazione rumorose o legate a d1** | H8: aggiungerle peggiora; accordo d1–JPT basso | giudice LLM, peso 0,5, controllo di accordo | etichettare con il modello stesso, o MIL positivo al posto dell'ablazione |
| **Feature di attenzione fragili** | grande differenza fra seed; i pesi del mix su un solo layer o una testa | dropout delle feature; C1 con 3 seed | togliere le feature (H2b = no) |
| **Hook di attenzione sbagliati** | E-ATT a livello di E-RAND; differenza con "eager" | test contro "eager" ([00c](00c_modelli_e_architetture.md)) | correggere scaling, softcap, ripetizione delle teste KV |
| **H-FAITH degenere** | la testa accende tutte le frasi | sempre insieme al termine supervised e alla sparsità | peso più basso; abbandonare H-FAITH |
| **JPT lento in transformers** | estrazione e server molto più lenti del previsto | installare i kernel DeltaNet nella settimana 1 | estrazione a batch più piccoli; server dedicato |
| **Spazio su disco della fase 1a** | > 80 GB per JPT e Winnow | ridurre i layer del mix dopo H2a su d1 | estrazione a pezzi; fp16 per le attenzioni |
| **Documenti oltre il contesto** | troncamento silenzioso | `max_document_tokens` in `decision.json`; segnalazione | spezzare a monte, sapendo che l'evidenza fra pezzi si perde |
| **Latenza** | p95 sopra +3% (più probabile su d1, dove il prefill è breve) | pooling sulla GPU, solo n vettori trasferiti; aggregatore piccolo | togliere feature (attenzione, media), ridurre i layer del mix o dell'aggregatore |
| **Frasi lunghe** | evidenziato un paragrafo intero | spezzare frasi > 60 parole | granularità più fine (fuori perimetro) |
| **Fughe** | distrattori o etichette da righe di test | generazione e ablazione solo sul train | rifare gli split di evidenza |

## 2. Checklist per step

### Step 0 — Split
- [ ] Sottoinsiemi `dev_evid_*`, `dev_evid_neg`, `test_evid_id`, `test_evid_ood`, `test_evid_sparse` congelati.
- [ ] Distribuzione delle distanze (fra frasi gold; dal preambolo in token per modello) registrata.

### Step 0a / 0c — Approccio e modelli
- [ ] Architettura approvata (sandwich, stati da più layer, attenzione, aggregatore, soglia dell'istanza).
- [ ] Indici dei layer e hook di attenzione verificati per d1, JPT, Winnow.
- [ ] Budget di latenza e lunghezza massima del documento fissati.

### Step 1 — Baseline
- [ ] B-ORD, B-SAND (P-A, P-B), B-FT-SAND misurati; formato e preambolo scelti.
- [ ] E-LEX, E-EMB, E-ATT su `dev_evid_select`; E-ABL su campione.

### Step 2a — Fonti
- [ ] FEVER dalle `lines` (anche REFUTES/NEI); HoVer riallineato (100 pagine verificate), anche NOT_SUPPORTED; SciFact e WiCE come test.
- [ ] `sentences` coerenti con `document` per tutte le righe.

### Step 2b — Etichette generate
- [ ] C2D con frasi marcate; CG2C con indice di frase per tripla.
- [ ] Ablazione offline con d1: conferma del giudice ≥ 80%, accordo con JPT ≥ 0,7.
- [ ] Righe sparse (anche oltre 1.024 / 4.096 token) senza riferimenti rotti.

### Step 2c — Verdetto
- [ ] Per modello: verdetto riusato o riaddestrato con il preambolo.
- [ ] Scelta fase 1 / fase 2; probabilità del verdetto precalcolate se serve l'ancoraggio.

### Step 2d — Formato
- [ ] `question.json` con `preamble` e `verdict_reused`.

### Step 2e — Mix
- [ ] `ev_head` con ≥ 40% di righe a più frasi di evidenza, ≥ 25% oltre 10 frasi, ≥ 15% oltre 1.024 token, 20–25% di negativi lunghi.

### Step 3 — Rendering
- [ ] Test di posizione, escaping, preambolo, attenzione e coerenza training/produzione superati per i tre modelli.

### Step 4 — Loss
- [ ] Funzioni testate sugli esempi numerici; α, ω, μ_neg fissati.

### Step 5 — Training
- [ ] Feature estratte verificate; overfit di 64 righe; gradino zero = H0.
- [ ] Regola di selezione (`s_0` + δ, o `τ_e`) su `dev_evid_select`, confermata.
- [ ] (Fase 2) verdetto entro i limiti.

### Step 6 — Valutazione
- [ ] F1, insieme completo, fasce di distanza (frasi e token), sufficienza, negativi accesi.
- [ ] Verdetto identico (fasi 1) o entro i limiti (fase 2).
- [ ] Latenza ≤ +3% per modello.

### Step 7 — Esperimenti
- [ ] Scala H0 → H2c e ablazioni registrate su d1; condizionati decisi; J0, W0; controlli C1–C4.

### Step 8 — Export
- [ ] Server per modello che restituisce risposte, `q` e feature per frase da un forward.
- [ ] Offset sul testo originale verificati.
- [ ] `frozen.json` prima del test; model card con "evidenzia solo se supportato".
