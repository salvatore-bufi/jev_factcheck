# Dataset per il fine-tuning (claim–documento, senza retrieval)

Obiettivo: scegliere i dati con cui fare fine-tuning di **Winnow-12B** e **JPT-9B** sul task di LLM-AggreFact: *dato un documento e un claim, il documento supporta l'intero claim?* (etichetta 1 = supportato, 0 = contraddetto **oppure** non stabilito). Il documento è sempre fornito: nessuna fase di retrieval né in training né in inferenza.

## 0. Aggiornamento del 2026-10-08: obiettivo e tre chiarimenti

- **Obiettivo: un fact checker generale.** Il modello finale riceve un documento in prosa e un claim in prosa (una frase o un breve testo) e deve funzionare bene in generale, non solo su LLM-AggreFact. Quindi i dati di training devono essere **testo contro testo**, e la valutazione si fa su una suite più ampia (domini diversi, test comportamentali, oblio): vedi [finetuning_strategy.md](finetuning_strategy.md), §6. LLM-AggreFact resta il riferimento per confrontarsi con la leaderboard.
- **FactCG oggi non è usato nei dati.** Nella repo c'è solo il *modello* FactCG-DeBERTa (`FactCG/`), valutato come riferimento (75,64). Il suo metodo di generazione dati (CG2C) è nella lista dei candidati (§4) e ora è raccomandato come prossima fonte da integrare: genera claim in prosa da documenti in prosa, quindi è perfettamente in linea con l'obiettivo. Dettagli in §4.1.
- **FactKG non si usa così com'è.** Oggi `build_pairs.py` costruisce il "documento" di FactKG come lista di triple `soggetto | relazione | oggetto`: non è il formato che il modello vedrà in uso. FactKG entra nel training **solo dopo essere stato trasformato in prosa** con un LLM e controllato in automatico (§3, punto 5), e comunque come esperimento (D3 della strategia). Le ricette consigliate in §6 lo escludono. Il codice non è stato modificato: le ricette `pilot`/`core` di `make_mix.py` lo includono ancora, quindi per ora si usano le ricette JSON di §6.

Strumenti nella repo:
- [`data_prep/build_pairs.py`](../data_prep/build_pairs.py): estrae le coppie (documento, claim, etichetta) da ogni fonte, nello stesso formato JSONL.
- [`data_prep/make_mix.py`](../data_prep/make_mix.py): decontamina contro LLM-AggreFact, bilancia, divide train/val senza documenti in comune.

## 1. Criteri di valutazione

Ogni dataset è giudicato su sei assi, in ordine di peso:

1. **Aderenza al task**: evidenza in prosa fornita, claim da verificare *per intero*, etichetta binaria supportato / non supportato. I dataset in cui l'evidenza va cercata (FEVER, HoVer) sono utilizzabili solo se l'evidenza gold è annotata e può essere materializzata come documento.
2. **Qualità delle etichette**: annotazione umana o sintetica verificata; assenza di scorciatoie note (es. la negazione "not" che predice REFUTES in FEVER).
3. **Generalità e copertura**: varietà di domini e di tipi di errore (entità, numeri, date, attribuzioni, negazioni, claim composti, informazione mancante), compresi i punti deboli misurati sui nostri modelli (vedi §2).
4. **Contaminazione** con LLM-AggreFact (misurata, §5).
5. **Dimensione** utile (non serve molto: Bespoke-MiniCheck-7B arriva a 77,4 con 35k esempi).
6. **Licenza** (JPT-9B è già CC BY-NC 4.0, quindi una licenza NC non peggiora JPT; per Winnow, Apache 2.0, conta di più).

## 2. Da dove partiamo

Balanced accuracy macro sulle 11 fonti del test LLM-AggreFact, regola fissa a 3 domande di Jev ([results/runs/dev_threshold_summary.md](../results/runs/dev_threshold_summary.md)):

| Modello | @0.30 | @soglia dev | AUC macro |
|---|---:|---:|---:|
| jev-1.13.0 (riferimento, dev-tuned) | 79,09 | — | 86,89 |
| Winnow-12B-BF16 | 76,99 | 77,42 | 85,02 |
| Winnow-12B-NVFP4 | 77,49 | 77,41 | 85,46 |
| JPT-9B | 76,04 | 76,71 | 84,57 |
| Bespoke-MiniCheck-7B (leaderboard) | 77,4 | | |

Fonti deboli (alla soglia dev): JPT-9B su **CNN 67,9, MediaSum 72,9, ExpertQA 57,5**, XSum 72,6; Winnow-BF16 su **MediaSum 72,0, XSum 73,4, CNN 72,7**, ExpertQA 59,5. Il pattern comune: **riassunti** (notizie, trascrizioni di meeting/interviste) con errori sottili. ExpertQA è a ~59 per tutti i modelli della leaderboard, quindi è probabilmente un tetto dovuto al rumore delle etichette e non va inseguito.

Conseguenza per la scelta dei dati: servono esempi con **documenti lunghi in prosa, claim multi-fatto ed errori sottili** (entità, numeri, attribuzioni), più che altri claim enciclopedici a singolo salto.

## 3. Classifica dei dataset (qualità attesa dopo il fine-tuning)

Coppie effettivamente estratte da `build_pairs.py` (run completo del 2026-10-08). "Doc p50" = lunghezza mediana del documento in caratteri (≈ 4 caratteri per token).

| # | Dataset | Coppie train / dev | % supportati (train) | Doc p50 / p95 | Fascia | Licenza |
|---:|---|---:|---:|---:|:-:|---|
| 1 | **MiniCheck C2D + D2C** | 14.395 / — | 46% | 921 / 1.703 | A | MIT |
| 2 | **ANLI R1–R3** | 162.865 / 3.200 | 32% | 313 / 512 | A | CC BY-NC 4.0 |
| 3 | **SummEdits** | 6.348 / — | 38% | 2.482 / 11.636 | A | CC BY 4.0 |
| 4 | **HoVer** | 17.000 / 3.745 | 61% | 1.135 / 2.181 | A− | CC BY-SA 4.0 |
| 5 | **FactKG** (solo se trasformato in prosa) | 81.342 / 12.587 | 49% | 1.328 / 2.984 | B se in prosa; **non usare in triple** | CC BY-NC-SA 4.0 |
| 6 | **FEVER** | 144.411 / 9.926 | 55% | 1.556 / 4.347 | B | CC BY-SA 3.0 |
| 7 | **VitaminC** | 370.653 / 63.054 | 50% | 166 / 337 | B | CC BY-SA 3.0 |
| 8 | **AVeriTeC** | 2.873 / 462 | 30% | 605 / 2.122 | B− | CC BY-NC 4.0 |
| 9 | **SciFact** (consigliato come test fuori dominio, non in training) | 919 / 340 | 40% | 1.358 / 3.003 | C | CC BY-NC 2.0 |
| 10 | **WiCE (solo train, decontaminato)** | 3.470 / 949 | 60% | 5.582 / 22.605 | C (rischio) | CC BY-SA 4.0 |
| — | RAGTruth | 15.090 / — | — | — | **escluso** | MIT |
| — | LLM-AggreFact | — | — | — | **escluso** (licenza; solo valutazione) | — |
| — | ClaimDB | — | — | — | **escluso** (serve SQL) | — |
| — | TSVer | — | — | — | **escluso** | — |
| — | LIAR, RealFactBench | — | — | — | **escluso** (nessun documento) | — |
| — | LoReFact | — | — | — | **non disponibile** | — |
| — | Multimodale ("Piecing It All Together") | — | — | — | **escluso** (immagini/tabelle) | — |

### Perché questo ordine

**1. MiniCheck C2D + D2C (fascia A).** È l'unico dataset costruito *esattamente* per LLM-AggreFact: claim multi-fatto da verificare contro più frasi di un documento. C2D parte da claim umani e genera documenti sintetici; D2C parte da documenti umani (notizie) e genera claim per porzioni del documento, con negativi ottenuti rimuovendo le frasi che supportano un fatto. Bespoke-MiniCheck-7B (77,4, il miglior modello aperto della leaderboard) è stato addestrato su 21k ANLI + 14k C2D/D2C. Unico limite: 14k esempi, quindi da usare tutto.

**2. ANLI (fascia A).** NLI avversariale in tre round, premesse brevi (Wikipedia, notizie, narrativa). Gli esempi sono stati scritti apposta per ingannare i modelli, quindi insegnano a non fidarsi della sovrapposizione lessicale. Va bene per il task binario perché *entailment → 1, neutral e contradiction → 0* è proprio la definizione di "supportato" di AggreFact. Limite: documenti corti (p50 313 caratteri), quindi da solo non insegna a cercare l'evidenza in un testo lungo. Squilibrato (32% positivi): bilanciare.

**3. SummEdits (fascia A).** 6.348 coppie documento–riassunto in 10 domini (notizie, podcast, meeting, email, chiamate commerciali, documenti legislativi, articoli scientifici, ...). I negativi sono modifiche *minime* di un riassunto corretto, verificate da annotatori: è il tipo di errore sottile che i nostri modelli sbagliano su CNN, XSum, MediaSum e MeetingBank. È il dataset che colpisce più direttamente i punti deboli di §2. Documenti lunghi (p95 11.636 caratteri): utile anche per allenare contesti lunghi.

**4. HoVer (fascia A−).** Claim umani multi-hop (2–4 pagine Wikipedia). Il documento è il testo introduttivo di tutte le pagine gold, mescolate, quindi il modello deve combinare più paragrafi. NOT_SUPPORTED in HoVer fonde già refuted e NEI, cioè coincide con la nostra etichetta 0. Limite: dominio enciclopedico, nessun riassunto.

**5. FactKG (solo se trasformato in prosa).** Ha due pregi: è grande (81k) e bilanciato, e ogni claim è etichettato con il tipo di ragionamento (one-hop, congiunzione, esistenza, multi-hop, negazione), anche in stile colloquiale. È utile per insegnare negazione, esistenza e claim composti.

**Così com'è, però, non va usato.** Il "documento" prodotto oggi da `build_pairs.py` è una lista di triple `soggetto | relazione | oggetto` prese dal grafo DBpedia, mescolate con triple distrattrici. Il modello finale riceverà testi in prosa: addestrarlo su liste di triple gli insegna a leggere un formato che in uso non vedrà mai, e può peggiorare proprio la lettura della prosa. A questo si aggiungono due fonti di rumore: le entità letterali (date, numeri) portano con sé archi inversi poco pertinenti, e per i claim falsi di tipo *existence* il documento semplicemente non contiene la relazione (etichetta corretta, ma esempio troppo facile).

**Come renderlo utilizzabile: trasformare le triple in prosa** con un LLM (modelli suggeriti nella strategia, §3.4):
1. Per ogni claim, passare all'LLM **solo le triple** (evidenza + distrattori), **mai il claim**, così il testo non può copiarne la formulazione.
2. Chiedere un paragrafo in stile enciclopedico che dica tutti e soli i fatti delle triple (prompt sotto).
3. **Controllo automatico di fedeltà**: per ogni tripla, un secondo passaggio (stesso LLM o un altro) risponde "il paragrafo afferma questo fatto? sì/no"; e una domanda finale "il paragrafo contiene fatti non presenti nell'elenco?". Si scartano i paragrafi che perdono anche una sola tripla di evidenza o aggiungono fatti.
4. Il resto della pipeline non cambia: la coppia (paragrafo, claim) eredita l'etichetta di FactKG.

```text
Write one encyclopedic paragraph in English that states all of the following facts and nothing else.
Use natural sentences, vary the order, and do not add any fact, date, number or opinion that is not listed.
Facts (subject | relation | object):
{TRIPLES}
Return only the paragraph.
```

Anche in prosa, FactKG resta un dataset di frasi enciclopediche brevi: al massimo il 15–20% del mix, e il suo contributo va misurato (esperimenti D3 contro D4 della strategia: in prosa contro triple).

**6. FEVER (fascia B).** Molto grande ma con limiti noti: claim scritti per mutazione da annotatori, e scorciatoie solo-claim (parole come "not", "only", "never" predicono REFUTES senza guardare il documento). Per i SUPPORTS/REFUTES il documento è l'intera pagina gold (fino a 40 frasi), non solo la frase di evidenza, così il modello deve localizzarla. Per NOT ENOUGH INFO non esiste evidenza: lo script abbina la pagina il cui titolo è il più lungo n-gramma del claim (costruzione FEVER-NLI). Questi negativi sono più facili dei REFUTES. Utile in quota limitata, filtrando i claim con negazioni esplicite se un'ablazione mostra che il modello impara la scorciatoia.

**7. VitaminC (fascia B).** Coppie contrastive da revisioni di Wikipedia: due evidenze quasi identiche (spesso cambia solo un numero o una data) con etichette opposte. Insegna sensibilità a numeri, date e quantità, cioè gli errori tipici dei riassunti. Limite: evidenza di 1–2 frasi (p50 166 caratteri). Enorme: ne basta un campione (3k–15k).

**8. AVeriTeC (fascia B−).** Claim *reali* di 50 organizzazioni di fact-checking, l'unico dataset con la distribuzione della disinformazione vera. Il documento è l'evidenza ricostruita dagli annotatori come coppie domanda–risposta (con fonte), senza la giustificazione che rivelerebbe il verdetto. Limiti: piccolo (2.873), formato Q/A non naturale, 70% negativi; Conflicting/Cherrypicking escluso perché non binarizzabile. Buono come dato di diversità, non come base.

**9. SciFact (fascia C).** Claim biomedici contro abstract. Di alta qualità ma minuscolo (919) e di dominio. Per un fact checker generale vale di più come **test fuori dominio** (mai visto in training) che come 0,6% del mix: lo escludono le ricette consigliate in §6.

**10. WiCE (fascia C, rischio).** Claim di Wikipedia contro la pagina web citata: sarebbe ottimo (documenti reali e lunghi, etichetta "partially supported"), ma **il test di WiCE è una delle 11 fonti di LLM-AggreFact**. Il 13% dei documenti del train è risultato sovrapposto ai documenti di AggreFact (§5). Dopo la decontaminazione resta comunque la stessa distribuzione della fonte di test, quindi un guadagno su WiCE non sarebbe credibile. Lo includo solo nella ricetta `extended`, per un'ablazione esplicita.

### Esclusi

- **RAGTruth**: anche il suo *train* condivide i documenti con il test di AggreFact: **594 documenti su 600 campionati** superano la soglia di sovrapposizione (§5). Addestrarci significherebbe vedere i documenti di test. Escluso da ogni ricetta (resta disponibile in `build_pairs.py` solo per analisi).
- **LLM-AggreFact**: il README del dataset permette la valutazione ed esclude esplicitamente pre-training e fine-tuning ([aggrefact_protocol_notes.md](aggrefact_protocol_notes.md)). Il dev serve per scegliere soglie e checkpoint, il test si guarda una volta sola.
- **ClaimDB**: l'evidenza sono database SQLite con milioni di righe; non entra nel contesto e richiede tool calling / SQL. È un task diverso.
- **TSVer**: 304 claim reali, ma le serie temporali non sono incluse nei file scaricati (vanno ricostruite da Our World in Data) e il dev ha 24 esempi. Troppo piccolo e costoso da materializzare.
- **LIAR, RealFactBench**: claim senza documento di evidenza (open-world): richiederebbero retrieval.
- **LoReFact**: nella cartella c'è solo il PDF ("link unavailable"). Interessante per la componente logica (fatti veri collegati da un'implicazione falsa); da riconsiderare se i dati vengono pubblicati.

## 4. Altri dataset trovati con la ricerca (non ancora integrati)

Candidati per una seconda fase, in ordine di interesse. Prima di usarli: aggiungere un builder in `build_pairs.py` e passarli per la decontaminazione.

| Dataset | Cosa contiene | Perché è interessante | Attenzione |
|---|---|---|---|
| **FactCG CG2C-MHQA** ([paper](https://arxiv.org/pdf/2501.17144), [codice](https://github.com/derenlei/FactCG)) — **priorità alta** | Claim multi-hop sintetici in prosa generati da grafi di contesto estratti da documenti (HotpotQA, MuSiQue), etichette dalla struttura del grafo | FactCG-DeBERTa-L arriva a 75,6 su AggreFact con 0,4B parametri grazie a questi dati; testo contro testo | I dati non sono rilasciati: vanno rigenerati con il loro codice (MIT). Vedi §4.1 |
| **LLM-Oasis** ([paper](https://arxiv.org/abs/2411.19655v3), Babelscape) | Testi fattuali/non fattuali generati da claim di Wikipedia falsificati; test gold annotato a mano | Descritto come la più grande risorsa per valutatori di fattualità end-to-end | Task end-to-end (senza documento esplicito nella versione principale): serve la variante con evidenza |
| **AmbiFC** ([paper](https://arxiv.org/abs/2104.00640v3)) | 10k claim, 50k passaggi Wikipedia annotati per passaggio, etichette soft | Annotazione per passaggio = coppie claim–documento naturali; le etichette soft aiutano la calibrazione | Ambiguità alta: usare solo i casi con accordo |
| **DocNLI** (`tasksource/doc-nli`) | ~1,4M coppie NLI a livello di documento | Premesse lunghe | Negativi generati con regole, rumorosi; include dati di riassunto (CNN/DailyMail) da decontaminare |
| **FEVEROUS** (`fever/feverous`) | Claim con evidenza testo + tabelle Wikipedia | Ragionamento su tabelle linearizzate | Serve linearizzare le tabelle; distribuzione simile a FEVER |
| **HealthVer, COVID-Fact** (`dwadden/healthver_entailment`, `dwadden/covidfact_entailment`) | Claim sanitari contro abstract | Diversità di dominio | Piccoli, dominio stretto; repository con script di caricamento |
| **Climate-FEVER** (`tdiggelm/climate_fever`) | Claim sul clima con 5 frasi di evidenza annotate | Claim reali | Evidenza a livello di frase, molti disaccordi |
| **FEVER-NLI** (`pietrolesci/nli_fever`) | FEVER già convertito in coppie NLI con contesto recuperato | Alternativa pronta al nostro builder FEVER | Stesse scorciatoie di FEVER |
| **HaluBench** (`PatronusAI/HaluBench`, [paper](https://arxiv.org/html/2407.08488v2)) | 15k triple contesto–domanda–risposta (DROP, CovidQA, PubMedQA, FinanceBench, HaluEval, RAGTruth) | Domini finanziario/medico | **Contiene RAGTruth**: solo valutazione, o escludere quella parte |
| **HalluMix** (`quotientai/HalluMix`, [paper](https://arxiv.org/html/2505.00506v1)) | Benchmark multi-dominio di rilevamento allucinazioni, multi-documento | Contesti lunghi e multi-documento | Mescola dataset esistenti: controllare sovrapposizioni con AggreFact |
| **HalluLens** ([paper](https://aclanthology.org/2025.acl-long.1176.pdf)) | Claim annotati contro passaggi di riferimento (supported / refuted / unverifiable) | Annotazione umana recente | Pensato come benchmark |

### 4.1 FactCG / CG2C: cosa è e come usarlo

**Cosa è.** FactCG ([paper](https://arxiv.org/pdf/2501.17144), NAACL 2025) propone CG2C (*Context Graph to Claim*), un metodo per generare dati di training sintetici a partire da **soli documenti in prosa**. Secondo il paper:
1. dal documento si estrae un grafo di contesto (entità come nodi, relazioni come archi, ciascun arco legato alla frase che lo esprime);
2. si campiona un sottografo multi-hop (2–4 archi collegati);
3. un LLM scrive un claim in prosa che esprime quel sottografo → esempio **supportato**;
4. per i negativi si altera il sottografo (si sostituisce un nodo, cioè un'entità o un valore, oppure si toglie dal documento la frase di un arco) → esempio **non supportato**.

L'etichetta viene dalla **struttura del grafo**, non dal giudizio di un LLM: è il punto di forza rispetto ai dati etichettati da LLM. Con questi dati (da HotpotQA e MuSiQue, "CG2C-MHQA") un DeBERTa da 0,4B arriva a 75,6 su LLM-AggreFact.

**Perché è adatto al nostro obiettivo.** Documento e claim sono entrambi in prosa; il grafo è solo uno strumento interno di generazione e il modello non lo vede mai. È la differenza fondamentale con FactKG, dove il grafo *è* il documento.

**Disponibilità.** Il [repository](https://github.com/derenlei/FactCG) (licenza MIT) rilascia il codice e il modello, **non i dati generati**: vanno rigenerati.

**Come usarlo da noi (lavoro futuro):**
- Documenti di partenza in prosa e non presenti in LLM-AggreFact: i paragrafi di HoVer (già in `wiki_wo_links.db`), le pagine FEVER (già in `Datasets/Fever/`), notizie recenti, trascrizioni di meeting (decontaminate).
- LLM per estrazione del grafo e scrittura dei claim: uno di quelli suggeriti nella strategia (§3.4), es. Qwen3-30B-A3B in locale per il volume.
- Controllo automatico: scartare i claim positivi che un secondo LLM giudica non supportati dal documento, e i negativi che giudica supportati.
- Volume iniziale: 5k–10k coppie, esperimento D9 della strategia.

## 5. Decontaminazione (misurata)

`make_mix.py` scarta una coppia se il claim normalizzato coincide con un claim di LLM-AggreFact (dev o test), oppure se almeno il **20%** dei 13-grammi di parole del documento (campionati 1 su 8 via hash) compare nei documenti di AggreFact.

Risultati sul run completo (tutte le coppie train + dev di ogni fonte, 2026-10-08):

| Fonte | Coppie scartate per sovrapposizione | Commento |
|---|---:|---|
| RAGTruth | **14.769 documenti + 30 claim su 15.090 (98%)** | Train e test condividono i documenti sorgente → escluso |
| WiCE (train + dev) | **1.328 documenti + 4 claim su 4.419 (30%)**; sul solo train 80 su 594 campionati (13%) | Stessa distribuzione di una fonte di test → solo ablazione |
| VitaminC | 980 documenti + 11 claim su 433.707 (0,2%) | Frasi Wikipedia in comune con fonti di AggreFact; trascurabile |
| FEVER | 171 documenti + 5 claim su 154.337 (0,1%) | Pagine Wikipedia comuni; trascurabile |
| ANLI | 38 documenti + 2 claim | trascurabile |
| MiniCheck C2D/D2C | 18 documenti | probabilmente notizie in comune con AggreFact-CNN: tolte |
| HoVer | 16 documenti | trascurabile |
| AVeriTeC | 1 claim | |
| FactKG, SummEdits, SciFact | 0 | |

Altre pulizie di `make_mix.py` (numeri in `data/finetune/mix_<nome>/decontamination.json`):
- **Duplicati** (stesso documento e stesso claim): 8.230 in FEVER (claim ripetuti), 3.760 in VitaminC.
- **Righe di train con un documento presente nel val**: 37.702 in ANLI (le premesse si ripetono fra train e dev), 4.826 in FEVER, 356 in FactKG, 292 in SciFact. Vengono tolte dal train, così il val misura la generalizzazione a documenti nuovi.

## 6. Ricette pronte e come usarle

Mix già generati in `data/finetune/` (git-ignorati, si rigenerano con i comandi sotto):

| Ricetta | Train | Val | Composizione del train |
|---|---:|---:|---|
| `pilot` | 32.872 | 4.204 | MiniCheck 6k, ANLI 6k, SummEdits 4k, HoVer 4k, FactKG 4k, FEVER 3k, VitaminC 3k, AVeriTeC 2,9k (tutto); bilanciato 50/50 tranne AVeriTeC |
| `core` | 106.082 | 4.480 | MiniCheck 13,6k (tutto), FactKG 20k, FEVER 20k, ANLI 15k, VitaminC 15k, HoVer 13,2k, SummEdits 5,8k, AVeriTeC 2,9k, SciFact 0,6k |
| `extended` | 108.999 | 4.590 | `core` + WiCE 2,9k (decontaminato): solo per l'ablazione D6 della strategia |

**Ricette consigliate (dal 2026-10-08): senza FactKG e senza SciFact.** FactKG resta fuori finché non è trasformato in prosa (§3, punto 5); SciFact resta fuori per usarlo come test fuori dominio. Il codice non è stato modificato: basta salvare questi JSON e passarli a `--recipe`.

`pilot_text.json` (~28.900 train / ~3.600 val):
```json
{"minicheck": {"cap": 6000, "balance": true}, "anli": {"cap": 6000, "balance": true},
 "summedits": {"cap": 4000, "balance": true}, "hover": {"cap": 4000, "balance": true},
 "fever": {"cap": 3000, "balance": true}, "vitaminc": {"cap": 3000, "balance": true},
 "averitec": {"cap": 0, "balance": false}}
```

`core_text.json` (~85.500 train / ~3.600 val):
```json
{"minicheck": {"cap": 0, "balance": false}, "hover": {"cap": 15000, "balance": true},
 "fever": {"cap": 20000, "balance": true}, "anli": {"cap": 15000, "balance": true},
 "vitaminc": {"cap": 15000, "balance": true}, "summedits": {"cap": 0, "balance": false},
 "averitec": {"cap": 0, "balance": false}}
```

```sh
python data_prep/make_mix.py --recipe pilot_text.json     # -> data/finetune/mix_pilot_text/
python data_prep/make_mix.py --recipe core_text.json      # -> data/finetune/mix_core_text/
```

Quando si toccherà il codice: aggiornare `RECIPES` in `make_mix.py` di conseguenza, e aggiungere a `build_pairs.py` FactKG in prosa e CG2C.

Il val contiene fino a 600 righe bilanciate per fonte, prese dal dev originale della fonte (o da una fetta del 5% per hash del documento se la fonte non ha un dev), senza documenti in comune con il train. In `core_text` le fonti di fascia A (MiniCheck, ANLI, SummEdits, HoVer) pesano circa il 56% del train (47.586 su 85.458).

```sh
conda activate factcheck

# 1. Coppie per fonte -> data/finetune/raw/<fonte>.jsonl (FEVER e FactKG: alcuni minuti)
python data_prep/build_pairs.py all
python data_prep/build_pairs.py factkg hover --limit 200      # prova veloce

# 2. Mix decontaminato -> data/finetune/mix_<nome>/{train,val}.jsonl, stats.json, decontamination.json
python data_prep/make_mix.py --recipe pilot
python data_prep/make_mix.py --recipe core
python data_prep/make_mix.py --recipe miaricetta.json --name prova   # {"fonte": {"cap": N, "balance": true}}
```

Note pratiche:
- Le fonti remote si scaricano con `datasets` da Hugging Face; a volte il proxy risponde `500`: basta rilanciare.
- FEVER usa i parquet di `fever/fever` già nella cache HF (`~/.cache/huggingface/hub/datasets--fever--fever`) e le pagine in `Datasets/Fever/data/wiki-pages/`.
- FactKG carica `dbpedia_2015_undirected_light.pickle` (1,5 GB); la versione completa da 5,8 GB non serve.
- Ogni riga ha `doc_id` (chiave di gruppo per gli split), `orig_label` (serve a derivare i target delle tre domande, vedi la strategia §3.1) e `meta` (es. tipi di ragionamento FactKG, `truncated`).
- I documenti oltre 24.000 caratteri sono troncati in `build_pairs.py` e scartati in `make_mix.py` (`--max-doc-chars`).

## 7. Fonti

- MiniCheck: https://arxiv.org/pdf/2404.10774 · dati: https://huggingface.co/datasets/lytang/C2D-and-D2C-MiniCheck
- Bespoke-MiniCheck-7B (ricetta 21k ANLI + 14k C2D/D2C): https://huggingface.co/bespokelabs/Bespoke-MiniCheck-7B
- FactCG / CG2C: https://arxiv.org/pdf/2501.17144 · https://github.com/derenlei/FactCG
- VitaminC: https://arxiv.org/pdf/2103.08541
- LLM-Oasis: https://arxiv.org/abs/2411.19655
- AmbiFC: https://arxiv.org/abs/2104.00640
- HaluBench / Lynx: https://arxiv.org/html/2407.08488v2
- HalluMix: https://arxiv.org/html/2505.00506v1
- HalluLens: https://aclanthology.org/2025.acl-long.1176.pdf
- Schede dei dataset locali: `Datasets/*/` e [Datasets/DATASETS_OVERVIEW.md](../Datasets/DATASETS_OVERVIEW.md)
