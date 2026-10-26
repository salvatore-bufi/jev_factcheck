# Stato dell'arte e novità

Aggiornato al 26 ottobre 2026. Il registro completo della ricerca, con tutte le fonti, è in [09_aggiornamento_ottobre_2026.md](09_aggiornamento_ottobre_2026.md).

## 1. Quattro filoni che si incontrano

### 1.1 Typed decision model (TDM)

Modelli che non generano testo: leggono uno stato e domande tipizzate (*choice*, *noul* sì/no, *score*) e restituiscono una probabilità per opzione in una sola passata.

Questo studio si limita ai modelli **fino a 9B parametri** (taglia nominale); i più grandi compaiono solo come riferimento.

Il filone è esploso: su arXiv ci sono ~200 paper su Jev e i TDM dal 15 settembre, e la **Decision Index** (la classifica dei TDM, edizione 0.4 dal 20 ottobre) conta più di 200 sistemi. Ci sono TDM chiusi di quasi tutti i grandi fornitori e decine di TDM aperti, quasi tutti ottenuti post-addestrando poche basi (Qwen3.5-9B, Qwen3.8-27B, Gemma-4). Le famiglie che contano per noi:

| Modello | Tipo | Base | Note |
|---|---|---|---|
| Jev 1.13 / **1.14** (TypeSafe) | chiuso, API | — | 1.14 dal 14/10 (immagini, tipo `box`, contesto lungo "danger zone"); `jev-latest` passa a 1.14; 1.13 resta, con 60 giorni di preavviso prima del ritiro |
| OpenAI Decisions, Microsoft Decision-1, d1-ultra (Liquid) | chiusi, API | — | OpenAI ora in beta pubblica |
| **Clef-flash** (Cloudflare, 1/10) | aperto, Apache 2.0 | **Qwen3.5-9B** | testa congiunta sopra il backbone; 76,6 su LLM-AggreFact (leaderboard). Clef (27B, 79,0) solo come riferimento |
| Nimble-v4 9B (Bespoke Labs) | **solo API** (non su Hugging Face) | — | 77,8 su LLM-AggreFact, primo fino a 9B; Nimble-v4-max (27B, 79,5, primo assoluto) solo come riferimento |
| **pplx-decider-flash v1.2** (Perplexity, 15/10) | aperto, Apache 2.0 | **Qwen3.5-4B** | la versione 27B fa 78,7 su LLM-AggreFact (riferimento) |
| **Tacit-9B / 8B / 4B / 2B** (AnyJev, Nokia) | aperti | Qwen3.5-9B, Qwen3-8B, Qwen3-4B, Qwen3.5-2B | autodistillazione del ragionamento in una passata |
| Strands Decider 2B (AWS), Intern-Decision 0,8B / 2B / 4B, SII-Decision-4B | aperti | Qwen3.5-2B e altre (da verificare) | la parte bassa della curva di taglia |
| Drex v1.5 (nace-ai) | aperto, licenza RAIL | MiMo-V2.6-Distill-Qwen-9B | 76,5 su LLM-AggreFact |
| Kev-9B / Kev-4B (J. Palmer) | aperti, Apache 2.0 | Qwen3.5-9B-**Base** / Qwen3.5-4B-**Base** | partono dalla variante Base, non dall'Instruct |
| elephant-1 (Baseten) | aperto (MIT) | Qwen3.5-9B | nella repo: 76,70 |
| JPT-9B, Nimble-9B-v3, d3-flash, StartLux-9B | aperti | Qwen3.5-9B (StartLux: architettura Qwen3.5, base non dichiarata) | nella repo |
| Winnow-12B, Rune-26B-A4B | aperti | Gemma-4 | nella repo (76,95 e 76,24), **sopra il vincolo**: solo riferimento |
| KnowLine-4B, d1-3B, Laya (~0,4B, encoder), CLM-8B (contrastivo) | aperti | Qwen3.5-4B, LFM2.5, ModernBERT, Qwen3-8B | nella repo (Laya da aggiungere) |

Ricette pubblicate per costruire TDM aperti: AnyJev ([2610.00831](https://arxiv.org/abs/2610.00831)), LLM-as-Jev ([2610.02076](https://arxiv.org/abs/2610.02076)), Releasing the Inner Jev ([2610.17205](https://arxiv.org/abs/2610.17205)), SII-Decision ([2610.03935](https://arxiv.org/abs/2610.03935)), Our Own Jev ([2610.29052](https://arxiv.org/abs/2610.29052)), Intern-Decision ([2610.22131](https://arxiv.org/abs/2610.22131)).

**Valutazioni generali dei TDM** utili come cornice:
- **Audit dei primi 28 paper** ([2609.32160](https://arxiv.org/abs/2609.32160)): la lettura tipizzata "non ha mostrato un vantaggio di accuratezza indipendente" rispetto alla lettura delle probabilità delle etichette; i vantaggi chiari sono latenza e costo; checklist di 14 punti (§4).
- **Valutazione di Jev su 37 dataset** ([2609.37647](https://arxiv.org/abs/2609.37647)): buona calibrazione sulle *choice*; sulle *noul* buon ordinamento ma soglia 0,5 inadatta; soglie fittate su 1.000 esempi non sempre trasferiscono.
- **JEVal** ([2610.03935](https://arxiv.org/abs/2610.03935)): i TDM vanno bene quando la decisione si ricava dall'evidenza disponibile, ma sovrastimano la probabilità dell'opzione scelta.
- **Limiti della singola passata** ([2610.24929](https://arxiv.org/abs/2610.24929)): Jev e le repliche aperte falliscono quando la decisione richiede passi intermedi (più aggiornamenti dello stesso fatto) e seguono un riassunto vecchio dello stato.
- **Fragilità**: un contesto naturale aggiunto ribalta il 61–73% delle decisioni corrette ([JevOut, 2609.30243](https://arxiv.org/abs/2609.30243)); aggiungere altre domande alla stessa richiesta cambia le risposte dei TDM aperti ([2610.14631](https://arxiv.org/abs/2610.14631)); la formulazione negata della stessa domanda inverte o annulla l'AUC di cinque TDM su otto ([Arize](https://arize.com/blog/decision-model-benchmark/)).
- **Calibrazione**: nelle risposte binarie Jev sembra sopprimere un terzo valore, "non so" ([Sys1Cal, 2609.35342](https://arxiv.org/abs/2609.35342)). Nel fact checking è esattamente la differenza fra "contraddetto" e "non stabilito".

### 1.2 TDM applicati al grounding e al fact checking (il nostro filone diretto)

| Lavoro | Cosa fa | Cosa **non** fa |
|---|---|---|
| **Evaluating and Benchmarking Jev** ([2609.37647](https://arxiv.org/abs/2609.37647), 29/9) | Jev zero-shot su **tutto il test di LLM-AggreFact**: **78,6** [77,6–79,6], una domanda *noul*, soglia 0,5, ECE 0,128; Qwen3.8-27B (76,9) e Gemma-4-E4B (75,6) letti allo stesso modo | nessun TDM aperto, nessun checker specializzato, nessuna analisi per fonte (solo l'intervallo 59,8–90,6), nessun test fuori dominio, una sola formulazione |
| **Jev-as-a-Judge for RAG Claim Verification** ([2610.20634](https://arxiv.org/abs/2610.20634), 30/9) | Jev contro un LLM di frontiera su **495 claim di AggreFact**: pari in BAcc a 1/187 del costo, ma Jev ammette il 23,2% dei claim non supportati contro il 14,1%; la cascata non batte un controllo con escalation casuale | campione piccolo, solo Jev |
| **PremiseCheck** ([2610.21046](https://arxiv.org/abs/2610.21046), 15/10) | Jev, Kev-27B, Microsoft Decision-1, Perplexity Decider come verificatori di premesse estratte da revisioni scientifiche; da giudici olistici (AUROC 50–57) a quasi pari a GPT-5.5 con la scomposizione | dominio specifico (revisioni), niente calibrazione, niente LLM-AggreFact, niente fine-tuning |
| **Arize, decision model benchmark** (ottobre, blog) | 8 TDM su RAGTruth (2.675 righe): Jev 85,5, Clef 85,2, d1 84,8, OpenAI 81,5, Kev-9B 76,8...; **sensibilità alla polarità** della domanda | blog, non paper; soglia fittata su metà dei dati; un solo benchmark |
| JEV-as-a-Judge ([2609.26550](https://arxiv.org/abs/2609.26550), v5) | Jev come giudice con escalation; 240 casi di HaluEval | |
| Radiologia ([2609.27607](https://arxiv.org/abs/2609.27607)), citazioni finanziarie ([2610.08675](https://arxiv.org/abs/2610.08675)) | Jev come verificatore di supporto in un dominio, con coppie minime controllate | domini singoli, solo Jev |
| **Leaderboard di LLM-AggreFact** | nimble-v4-max 79,5, Clef 79,0, Decider v1.2 78,7, Jev 1.13 78,7, d1-ultra 78,5, nimble-v4 77,8, Clef Flash 76,6, Drex v1.5 76,5 | protocollo non documentato (domanda, soglia, chunking); nessuna calibrazione |

Da notare: la **Decision Index** contiene RAGTruth (2.700 richieste) e HoVer fra i benchmark pubblici, e ANLI, ContractNLI, NLI4CT fra quelli di NLI. I produttori di TDM ottimizzano esplicitamente per la Decision Index, quindi **la fonte RAGTruth di LLM-AggreFact va considerata a rischio di contaminazione** per tutti i TDM recenti (§5).

### 1.3 Fact checker specializzati su documento

- **MiniCheck** ([2404.10774](https://arxiv.org/abs/2404.10774), EMNLP 2024): introduce LLM-AggreFact e i dati sintetici C2D/D2C; Bespoke-MiniCheck-7B (77,4) **non è più** il miglior modello aperto della leaderboard: lo superano Nimble-v4-max, Clef, Decider e Nimble-v4, tutti TDM.
- **FactCG** ([2501.17144](https://arxiv.org/abs/2501.17144), NAACL 2025): DeBERTa da 0,4B a 75,6.
- AlignScore, SummaC, HHEM, Granite Guardian 3.3 (76,5), Lynx/HaluBench, HalluGuard.
- Lavori recenti: aggregazione di giudici economici ([RAIM, 2609.39229](https://arxiv.org/abs/2609.39229), entro 1,9 punti dal primo della leaderboard), self-play per rilevatori ([2607.07993](https://arxiv.org/abs/2607.07993)), le metriche di attribuzione non trasferiscono fra dataset ([2606.23915](https://arxiv.org/abs/2606.23915)), le etichette dei benchmark di allucinazione dipendono dalla formulazione del criterio ([2610.08026](https://arxiv.org/abs/2610.08026)).

### 1.4 LLM generici letti come TDM, senza training

- **AnyJev** ([2610.00831](https://arxiv.org/abs/2610.00831), Nokia): livello "L0" senza training: rotazione ciclica delle opzioni contro il bias di posizione e divisione per il prior delle etichette stimato senza etichette. Su Gemma-4-31B, 60,22 sulla Decision Index 0.3 (5° su 149) senza alcun training.
- **LLM-as-Jev** ([2610.02076](https://arxiv.org/abs/2610.02076)): gli LLM moderni sono già TDM; il fine-tuning serve soprattutto ai modelli piccoli.
- **Releasing the Inner Jev** ([2610.17205](https://arxiv.org/abs/2610.17205)): Gemma-4-31B zero-shot passa da 72,9 a 79,6 di Capability con la sola temperatura per tipo di domanda.

Conseguenza per noi: la baseline B1 non può essere solo la base "letta in modo ingenuo" (softmax sulle lettere). Serve anche **la base letta in modo calibrato senza training** (rotazione delle opzioni + temperatura), altrimenti un revisore dirà che il vantaggio del TDM è solo calibrazione che si ottiene gratis.

## 2. Il buco, rivisto

| Domanda | Coperta da qualcuno? |
|---|---|
| Quanto vale Jev su LLM-AggreFact intero? | **sì** (78,6 a soglia 0,5: [2609.37647](https://arxiv.org/abs/2609.37647)). Per noi diventa un numero da riprodurre, non un contributo |
| Quanto valgono **i TDM aperti piccoli (fino a 9B)** con **un unico protocollo controllato** su AggreFact e fuori dominio? Quanto distano dai TDM grandi e chiusi? | **no**: solo voci di leaderboard inviate dai produttori, protocollo ignoto |
| Cosa aggiunge l'addestramento da decision model **rispetto alla base**, sul grounding? Varia molto fra ricette sulla stessa base? | **no** sul grounding (solo Qwen/Gemma accanto a Jev, che non ha una base nota); l'audit lo indica come domanda aperta in generale |
| Quanto della classifica dipende dal protocollo (soglia, formulazione, polarità, una o tre domande, composizione della richiesta, troncamento)? | **no** (studi di sensibilità esistono su altri compiti, non sul grounding e non su molti TDM) |
| I TDM distinguono "contraddetto" da "non stabilito"? Gli errori sono simmetrici? | in parte: asimmetria degli errori per Jev su 495 claim; ipotesi del "terzo valore soppresso" su dati sintetici |
| Le probabilità sono calibrate sul grounding? La soglia trasferisce fra domini? | in parte: solo l'ECE di Jev su AggreFact (0,128); niente trasferimento fra domini, niente TDM aperti |
| Gli errori dei TDM sono correlati fra loro e con i checker specializzati? (cascate, ensemble) | in parte: errori condivisi fra Jev e LLM su rubriche ([2609.29769](https://arxiv.org/abs/2609.29769)); non sul grounding, non fra TDM sulla stessa base |
| Partire da un TDM è meglio che partire dalla base, per addestrare un checker di grounding? | **no** (le ricette pubblicate addestrano TDM generali, non checker; il controllo con la base non c'è) |

## 3. Cosa rende il contributo difendibile oggi

1. **Ampiezza controllata sui TDM piccoli**: ~15–20 TDM aperti fino a 9B, cinque architetture di lettura (lettere su LLM causale, attenzione bidirezionale sul prompt, testa congiunta, encoder, contrastivo), da 0,4B a 9B, tutti eseguibili su una sola GPU da 24–48 GB, **stesse domande, stessi dati, stesse regole**, con cache di probabilità e logit rilasciate. Nessuna voce della leaderboard è confrontabile così.
2. **L'esperimento naturale "stessa base, ricette diverse"**: sette TDM su Qwen3.5-9B e tre su Qwen3.5-4B (contando Kev sulle varianti Base), contro la base letta in modo ingenuo e calibrato. Separa tre effetti: base, lettura, addestramento da decision model.
3. **Sensibilità al protocollo**, misurata senza nuova inferenza grazie alla grid search già pronta: se le scelte di protocollo valgono più della distanza fra i modelli, la leaderboard va letta diversamente. È un risultato utile a tutto il campo.
4. **Errori, calibrazione, trasferimento**: asimmetria degli errori (claim non supportati ammessi), contraddetto contro non stabilito, ECE per dominio, trasferimento della soglia, accordo fra modelli.
5. **Un esperimento di fine-tuning mirato** (B-ctrl): stesso training di grounding su un TDM e sulla sua base.
6. **Rispetto della checklist dell'audit** (§4), dichiarato punto per punto.

## 4. La checklist dell'audit (da rispettare)

| ID | Requisito | Come lo copriamo |
|---|---|---|
| B1 | baseline con log-probabilità delle etichette | basi dei TDM (Qwen3.5-9B, Qwen3.5-4B e le varianti Base, LFM2.5-VL-3B, Qwen3-8B) con lo stesso prompt; lettura ingenua **e** calibrata senza training (AnyJev L0 o temperatura) |
| B2 | cascate generative economiche | LLM piccolo che genera "Yes/No" (con e senza ragionamento breve); cascata TDM → LLM con il controllo a escalation casuale di [2610.20634](https://arxiv.org/abs/2610.20634) |
| B3 | modelli specifici del compito | Bespoke-MiniCheck-7B, MiniCheck-Flan-T5-L, FactCG, AlignScore, Granite Guardian 3.3 |
| T1 | dispersione fra modelli | molti TDM e più baseline; intervalli di confidenza |
| T2 | invarianza delle decisioni | polarità e parafrasi della domanda, distrattori, posizione dell'evidenza, composizione della richiesta |
| T3 | scambio dei nomi delle opzioni | inversione e rinomina delle opzioni e dei campi |
| T4 | provenienza dei dati | tabelle di provenienza; contaminazione (RAGTruth nella Decision Index; colonna `contamination_identifier` di AggreFact) |
| R1 | versioni e date esatte | revisioni dei pesi (già in `run.json`), versione di Jev (1.13 e 1.14), date delle chiamate |
| R2 | intervalli di confidenza | bootstrap per documento |
| R3 | scelte fatte dopo la valutazione | registro degli esperimenti; dichiarazione del controllo intermedio di Jev; la grid search sul test è solo esplorativa |
| C1 | calibrazione su dati tenuti da parte | ECE e Brier su test fuori dominio |
| C2 | accuratezza ad alta confidenza | copertura selettiva |
| C3 | soglie congelate prima del test | protocollo dello Step 0 |
| C4 | latenza con e senza cache | già misurata dal benchmark (sonda a cache fredda, throughput per batch) |

## 5. Rischi per la novità

- **Scoop**: altissimo. In sei settimane sono usciti tre lavori su TDM e AggreFact o claim verification, e i produttori inviano TDM alla leaderboard ogni settimana. Contromisura: preprint della parte di analisi entro fine novembre; ricontrollare arXiv ogni settimana con le chiavi del §6.
- **"È solo benchmarking"**: il rischio resta. La risposta è un'idea scientifica netta (RQ2: base contro ricetta; RQ3: protocollo contro modello), non la quantità di modelli.
- **Contaminazione**: RAGTruth è nella Decision Index, i dati di training dei TDM non sono pubblici. Riportare sempre la media di AggreFact anche **senza RAGTruth**, e verificare la colonna `contamination_identifier`.
- **Bersaglio mobile**: Jev 1.14, nuove versioni ogni settimana. Congelare la lista dei modelli a una data (es. 30 novembre) e dichiararla.
- **Modelli chiusi**: conclusioni principali sui modelli aperti; per i chiusi, versione e date.

## 6. Chiavi per la ricerca settimanale

arXiv (API: `abs:Jev`, `abs:"decision model" AND abs:typed`, `abs:AggreFact`, `abs:MiniCheck`), la leaderboard di LLM-AggreFact, la Decision Index, Semantic Scholar con "typed decision model fact", "System One grounding", "decision model hallucination". Lo script usato il 26 ottobre è descritto in [09](09_aggiornamento_ottobre_2026.md).
