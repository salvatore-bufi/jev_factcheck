# Selezione dei modelli (fino a 9B) e delle loro controparti LLM

Scritto il 26 ottobre 2026. Fonti: i dati della [Jev Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index) (file `data/index.json` dello Space, generato il 26/10 alle 15:30 UTC, edizione 0.4, 229 sistemi, Jev di riferimento 1.14), i metadati e le model card su Hugging Face, la leaderboard di LLM-AggreFact e i run di questa repo. I numeri della Decision Index sono **inviati o rieseguiti dai manutentori della classifica**: vanno citati come tali, non come misure nostre.

## 1. Cosa ci dicono i benchmark dei decision model (e perché giustificano la scelta)

### 1.1 Sotto i 9B l'addestramento da decision model conta molto; sopra quasi niente

La Decision Index contiene anche **LLM generici letti senza alcun training** (voci "inference technique"). Il confronto con i TDM addestrati della stessa taglia è netto:

| Taglia | Migliore lettura senza training | Migliore TDM addestrato | Distanza (Full 0.4) |
|---|---|---|---:|
| ~4B (Qwen3.5-4B / Qwen3-4B) | Ryotide 31,99; Jobe 30,35; openvons 28,78; SemIf 26,71 | KnowLine-4B 55,99; StartLux-4B 54,80; pplx-decider-flash 54,29 | **~+24** |
| ~31B (Gemma-4-31B) | FlexDM 67,56; AnyJev L0 63,65; Decider chat 61,63 | Jeffan-31b 69,75 (e GLiDE 27,8B 71,52) | **~+2–4** |

Questo sposta la domanda aperta dell'audit ([2609.32160](https://arxiv.org/abs/2609.32160): "la lettura tipizzata non ha un vantaggio indipendente") **verso i modelli piccoli**. Sopra i 30B un LLM letto bene è già quasi un TDM; sotto i 9B no. Il vincolo dei 9B è quindi la zona dove RQ2 ("base contro ricetta") ha più da dire, e lo testiamo sul grounding.

### 1.2 Sul grounding i migliori sotto i 9B valgono i migliori 27–31B (secondo la Decision Index)

Media dei cinque benchmark di grounding e NLI della Decision Index (RAGTruth, HoVer, ANLI, ContractNLI, NLI4CT; punteggi grezzi), che chiamiamo **DI-G5**:

| Modello | Taglia | Full 0.4 | DI-G5 |
|---|---:|---:|---:|
| GLiDE 1.2 / Jeffan-31b (i primi assoluti) | 27,8B / 31B | 71,5 / 69,8 | 85,4 / 85,4 |
| **elephant-1** | 9,4B | 60,3 | **85,4** |
| StartLux-9B | 9,7B | 58,1 | 84,5 |
| Jeffan-4B / StartLux-4B / KnowLine-4B | ~4,5B | 52–56 | 82–83 |
| Jev 1.13 (chiuso) | — | 59,2 | 76,0 |

Sul sottoinsieme di grounding il miglior 9B **raggiunge** i 27–31B e **supera Jev di 9 punti**. Ma sul test di LLM-AggreFact i nostri 9B stanno 2–3 punti **sotto** Jev (elephant-1 76,70, JPT-9B 76,04 contro 78,6 di Jev a soglia 0,5). Due letture possibili, entrambe interessanti:
- i benchmark di grounding della Decision Index sono **ottimizzati** dai produttori (Amber e Crivo dichiarano di essere addestrati sugli split di training del kit della Decision Index, RAGTruth compreso);
- oppure misurano una capacità diversa da quella di AggreFact.

Ne nasce una domanda nuova, economica e utile a tutto il campo: **RQ8, la classifica generale dei decision model predice la qualità sul grounding fuori dalla classifica?** ([11](11_design_esperimenti.md), E9).

### 1.3 La classifica generale non ordina come AggreFact

| Modello | Full 0.4 (rank su 229) | AggreFact (nostro run, regola fissa) |
|---|---:|---:|
| elephant-1 | 60,26 (41°) | 76,70 |
| KnowLine-4B | 55,99 (64°) | 75,65 |
| JPT-9B | 44,08 (109°) | 76,04 |
| d1-3B | 31,12 (154°) | 71,84 |
| CLM-8B | 6,49 (222°) | 49,98 (58,78 con una domanda) |

JPT-9B è 68 posizioni sotto elephant-1 nella Decision Index ma quasi alla pari su AggreFact. Bastano cinque modelli per vedere che la classifica generale non è un buon criterio di selezione per il grounding: per questo **non selezioniamo solo per punteggio** (§2).

### 1.4 Altri segnali utili alla selezione

- **Polarità**: Clef-flash ha 84,5 su ContractNLI ma **35,6 su RAGTruth**, dove la domanda della Decision Index è negativa ("la risposta contiene qualcosa che la fonte non supporta?"). Lo stesso schema (RAGTruth vicino o sotto il caso, il resto normale) compare in d3-nano, d3-lite, d3-edge, lev, Cam1. È coerente con l'inversione di polarità trovata da [Arize](https://arize.com/blog/decision-model-benchmark/). Questi modelli sono il banco di prova naturale per RQ3.
- **Calibrazione**: ECE della Decision Index fra 0,015 (StartLux-0.8B) e 0,123 (Nimble-9B-v3) fra i candidati: abbastanza dispersione per un'analisi di calibrazione.
- **Famiglie a più taglie con lo stesso runtime**: StartLux (0,8 / 2 / 4 / 9B), d3 (0,6 / 0,8 / 2 / 4 / 8,4B), OneJev 1.6 (0,8 / 4 / 9B), JPT e Kev (0,8 / 4 / 9B). Danno una **curva di taglia a ricetta fissa** quasi senza costo di ingegneria.
- **Voci di AggreFact dichiarate dai produttori**: Clef Flash 76,6 e Drex v1.5 76,5 sulla leaderboard. Riprodurle col nostro protocollo è un test diretto di RQ3.

### 1.5 Perché il confronto è centrato su Qwen (e come lo bilanciamo)

Un revisore lo chiederà quasi certamente. La risposta ha tre parti.

**È un fatto dell'ecosistema, non una nostra scelta.** Nella Decision Index, fra i 141 sistemi fino a ~10B: 96 (68%) partono da una base Qwen; tolti gli encoder, la quota sale a circa il 90%. **I primi 25 sistemi fino a 9B per punteggio sono tutti Qwen**, e il miglior non Qwen (Standard One 8B, su Ministral-3-8B) è a 42,9 contro 60,3 di elephant-1. Studiare i TDM piccoli di oggi significa, per forza, studiare soprattutto TDM su Qwen3.5.

**Per RQ2 la concentrazione è un vantaggio.** Il disegno "stessa base, ricette diverse" richiede molti TDM sulla stessa base: solo Qwen3.5-9B e Qwen3.5-4B ne hanno abbastanza (sette e cinque) per stimare la dispersione fra ricette. La base fissa elimina anche un confondente pratico: stesso tokenizer, stesse lettere delle opzioni, stessa lunghezza dei prompt.

**Ma la generalità va dimostrata, non assunta.** Aggiungiamo quindi un **blocco non Qwen** con gruppi controllati più piccoli (§3.1bis): Gemma-4-E4B (tre TDM), Ministral-3-8B (due), LFM2.5 (due), più singoli modelli su altre basi (K2-Horizon, MiniCPM5) e tre famiglie di encoder (DeBERTa, ModernBERT, Ettin). Su questi ripetiamo E2 (base contro ricetta), E4 (protocollo) ed E10 (B-ctrl) e verifichiamo se le conclusioni tengono per famiglia ([11](11_design_esperimenti.md), E2b). Il limite che resta va scritto: i TDM non Qwen sono anche più deboli, quindi "famiglia" e "qualità" sono in parte confusi; lo dichiariamo e lo controlliamo confrontando modelli di qualità simile (es. d1-3B con i Qwen da 2B).

## 2. Criteri di selezione

| # | Criterio | Perché |
|---|---|---|
| C1 | **Fino a 9B parametri** (taglia nominale; i "9B" di Qwen3.5 hanno 9,4–9,7B con la torre visiva) | vincolo di progetto: una GPU da 24–48 GB, anche per il fine-tuning; zona dove l'addestramento da decision model conta (§1.1) |
| C2 | **Pesi aperti e runtime degli autori disponibile** | protocollo "runtime nativo, nessuna modifica" (lo stesso della Decision Index: "native inference helpers, prompts and readout rules are preserved"); riproducibilità |
| C3 | **Copre un contrasto controllato** | stessa base con ricette diverse (RQ2); stessa ricetta a taglie diverse; Instruct contro Base come punto di partenza; architettura di lettura |
| C4 | **Rappresentativo dello stato dell'arte nella sua taglia** | un paper che valuta solo modelli deboli non dice nulla; scegliamo i migliori della Decision Index per taglia, **ma non solo loro** (§1.3) |
| C5 | **Segnale specifico sul grounding** | DI-G5 alto o anomalo (polarità), voce su AggreFact da riprodurre |
| C6 | **Costo di ingegneria** | adapter già nella repo, o stesso runtime di un modello già fatto |
| C7 | **Licenza dichiarata** | rilascio delle previsioni; i modelli non commerciali sono ammessi ma segnalati |
| C8 | **Esclusioni** | sopra i 9B; senza runtime documentato; copertura parziale sulla Decision Index senza motivo noto; versioni superate della stessa famiglia (Nimble v2, OneJev 1.5); varianti di lingua (Saracura PT-BR); modelli hobby con Full < 30 a ≥ 4B, salvo che servano a un contrasto |
| C9 | **Varietà di famiglia** | almeno tre famiglie di base non Qwen con un gruppo controllato (§1.5), anche se più deboli |

## 3. La lista

Colonne della Decision Index: Full 0.4, ECE, DI-G5. "Costo" = stato nella repo: **fatto** (run completo su AggreFact), **pronto** (adapter esistente), **stesso runtime** (adapter quasi gratis), **nuovo**.

### 3.1 Livello A: i modelli del paper (tutta la suite, tutte le varianti di protocollo)

**Gruppo Qwen3.5-9B (Instruct)**, il contrasto principale di RQ2:

| # | Modello | Ricetta e lettura | Licenza | DI: Full / ECE / G5 | AggreFact | Costo | Perché è nella lista |
|--|---------|------------|-------|---------|-------|------|----------------------|
| A1 | **elephant-1** (Baseten) | fine-tuning completo; attenzione bidirezionale a blocchi | MIT | 60,3 / 0,058 / 85,4 | 76,70 | fatto | il migliore fino a 9B (C4); DI-G5 pari ai 27B |
| A2 | **StartLux-9B-Instant** | fine-tuning completo; prompt bidirezionale | CC BY-NC | 58,1 / 0,046 / 84,5 | — | pronto | secondo per taglia; capofila di una famiglia a 4 taglie |
| A3 | **d3-flash** (vLLM-SR) | fine-tuning completo; attenzione non causale; 255 codici | Apache 2.0 | 54,3 / 0,056 / 77,6 | — | pronto | lettura a codici; famiglia a 5 taglie con polarità anomala nelle taglie piccole |
| A4 | **OneJev 1.6-9B** | fine-tuning completo; runtime OneJev (lo stesso di elephant-1) | Apache 2.0 | 53,1 / **0,022** / 82,3 | — | stesso runtime | il meglio calibrato del gruppo; famiglia a 3 taglie |
| A5 | **Nimble-9B-v3** (Bespoke) | LoRA r64; 255 codici | CC BY-NC | 51,1 / **0,123** / 77,6 | — | pronto | il peggio calibrato: contrasto per RQ5; LoRA contro fine-tuning completo |
| A6 | **Clef-flash** (Cloudflare) | fine-tuning completo; testa congiunta | Apache 2.0 | 44,4 / 0,025 / 63,8 | 76,6 (produttore) | nuovo | voce di AggreFact da riprodurre (RQ3); anomalia di polarità su RAGTruth |
| A7 | **Tacit-9B** (AnyJev) | autodistillazione del ragionamento, senza etichette umane | Apache 2.0 | — / — / — | — | nuovo (runtime AnyJev, che serve comunque per le basi) | l'unica ricetta senza etichette: separa "dati etichettati" da "formato" |

**Gruppo Qwen3.5-9B-Base** (partenza dal modello pre-addestrato):

| # | Modello | Ricetta | Licenza | DI: Full / ECE / G5 | AggreFact | Costo | Perché |
|--|---------|------------|-------|---------|-------|------|----------------------|
| A8 | **JPT-9B** | LoRA; lettere | CC BY-NC | 44,1 / 0,077 / 74,1 | 76,04 | fatto | basso nella Decision Index ma alla pari su AggreFact (§1.3) |
| A9 | **Kev-9B** | LoRA + testa a puntatore | Apache 2.0 | 40,9 / 0,045 / 66,0 | — | nuovo | molto usato in letteratura (Arize; Kev-27B in PremiseCheck); testa a puntatore |

**Gruppo Qwen3.5-4B**, la replica a metà taglia:

| # | Modello | Ricetta | Licenza | DI: Full / ECE / G5 | AggreFact | Costo | Perché |
|--|---------|------------|-------|---------|-------|------|----------------------|
| A10 | **KnowLine-4B-Gen5.5** | due SFT LoRA mediate; lettura doppia dei prompt brevi | Apache 2.0 | 56,0 / 0,047 / 82,0 | 75,65 | fatto | il migliore 4B |
| A11 | **StartLux-4B-Instant** | come A2 | CC BY-NC | 54,8 / 0,052 / 82,9 | — | stesso runtime | ricetta di A2 a metà taglia |
| A12 | **pplx-decider-flash v1.2** | fine-tuning completo; non causale | Apache 2.0 | 54,3 / 0,052 / 80,7 | — | nuovo | grande produttore; la versione 27B è 78,7 su AggreFact |
| A13 | **d3-mini** | come A3 | Apache 2.0 | 50,0 / 0,056 / 75,6 | — | stesso runtime | ricetta di A3 a metà taglia |
| A14 | **OneJev 1.6-4B** | come A4 | Apache 2.0 | 48,9 / 0,025 / 80,6 | — | stesso runtime | ricetta di A4 a metà taglia |

**Altre basi e architetture**:

| # | Modello | Base | Licenza | DI: Full / G5 | AggreFact | Costo | Perché |
|--|---------|------------|-------|---------|-------|------|----------------------|
| A15 | **d1-3B** (Liquid) | LFM2.5-VL-3B (architettura ibrida, non Qwen) | LFM Open | 31,1 / 73,7 | 71,84 | fatto | unica base non Qwen; parole yes/no invece di lettere |
| A16 | **Drex v1.5** (nace-ai) | MiMo-V2.6-Distill-Qwen-9B (base distillata dal ragionamento, a sua volta da Qwen3.5-9B) | RAIL | 45,8 / 80,6 | 76,5 (produttore) | nuovo | voce di AggreFact da riprodurre; base "con ragionamento" |
| A17 | **GLiClass v4 edge** (knowledgator) | DeBERTa-v3-large (0,44B), encoder | Apache 2.0 | 31,2 / 75,3 | — | nuovo | **stessa base di FactCG**: TDM-encoder contro checker specializzato a parità di backbone |

### 3.1bis Livello A, blocco non Qwen (la verifica di generalità)

| # | Modello | Base | Licenza | DI: Full / G5 | Costo | Perché |
|--|---------|------------|-------|---------|------|----------------------|
| N1 | **Standard One 8B SH** (StandardThinking) | Ministral-3-8B-Instruct | Apache 2.0 | 42,9 / 77,8 | nuovo | il miglior TDM non Qwen fino a 9B |
| N2 | **Jevstral 8B** | Ministral-3-8B-**Base** | Apache 2.0 | 32,2 / 50,3 | nuovo | stessa famiglia di N1, partenza dalla Base: replica non Qwen del contrasto Instruct contro Base |
| N3 | **Winnow-E4B** (EldanRing) | Gemma-4-E4B-it (8,0B serviti) | Apache 2.0 | 32,7 / 65,8 | stesso runtime di Winnow-12B | stesso autore e ricetta di Winnow-12B, entro il vincolo |
| N4 | **Strands Decider E4B** (AWS) | Gemma-4-E4B-it | Apache 2.0 | — | nuovo | grande produttore; stessa base di N3 |
| N5 | **Gevva E4B** | Gemma-4-E4B-it, convertito in encoder | Apache 2.0 | 29,2 / 50,1 | nuovo | stessa base di N3–N4 con lettura da encoder: architettura a base fissa |
| A15 | d1-3B (già nel livello A) | LFM2.5-VL-3B (da LFM2.5-2.6B-Base) | LFM Open | 31,1 / 73,7 | fatto | gruppo LFM |
| N6 | LFM2.5-2.6B-RLCD | LFM2.5-2.6B-Base | da verificare | 8,2 / 40,7 | nuovo | stessa base di d1-3B, ricetta RLCD; debole, utile solo come secondo punto del gruppo |

Encoder (tre famiglie): GLiClass v4 edge (DeBERTa-v3-large, già nel livello A), **Amber 0,4B** (ModernBERT-large, DI 26,0 / 70,4; CC BY-NC) e **Crivo-1B** (Ettin, DI 29,3 / 73,8; CC BY-NC). Attenzione: Amber e Crivo dichiarano di essere addestrati sugli split di training del kit della Decision Index.

Singoli su altre basi (livello B): Lev1 3.7B (K2-Horizon, 35,4 / 69,6), Aurelis-2B (MiniCPM5-2B, 32,6 / 70,7).

### 3.2 Livello B: le curve di taglia (AggreFact + due test fuori dominio, solo le domande principali)

| Famiglia | Taglie da aggiungere | Full 0.4 per taglia | Note |
|---|---|---|---|
| StartLux v1.1 | 2B, 0,8B | 40,1 / 28,9 | ECE 0,035 / 0,015; DI-G5 77,9 / 74,2: tiene sul grounding anche a 0,8B |
| d3 | nano 2B, lite 0,8B | 34,3 / 26,2 | RAGTruth 29,5 / 29,7: polarità che crolla con la taglia |
| OneJev 1.6 | 0,8B | 29,6 | |
| JPT | 4B, 0,8B | 39,6 / 15,5 | base dichiarata diversa fra card e Decision Index (Instruct contro Base): da verificare |
| Kev | 4B (0,8B facoltativo) | 34,9 / 16,3 | sulle varianti Base |

### 3.3 Livello C: controlli e casi limite (già fatti o economici)

| Modello | Perché |
|---|---|
| CLM-8B (fatto) | lettura contrastiva: controllo negativo che mostra quanto la formulazione delle opzioni conti (49,98 con tre domande, 58,78 con una) |
| Laya (0,42B, encoder) | il TDM aperto più citato (Arize, Red Hat, PremiseCheck); Full 9,2: pavimento della scala |
| Intern-Decision-4B | ECE 0,028 ma Full 38,7: buon esempio di "calibrato ma poco accurato"; facoltativo |

### 3.4 Le controparti LLM (B1)

Ogni gruppo di TDM ha la sua base, letta con lo stesso prompt del TDM e con un prompt neutro comune, in tre modi:

| Lettura | Come | Costo |
|---|---|---|
| **R0, ingenua** | softmax sulle lettere (o codici) delle opzioni all'ultima posizione, attenzione causale | 1 passata per domanda |
| **R1, calibrata senza training** | AnyJev L0: rotazione delle opzioni + divisione per il prior delle etichette stimato senza etichette ([2610.00831](https://arxiv.org/abs/2610.00831)) | 2 passate per domanda binaria |
| **R2, R1 + temperatura** | temperatura fittata sul nostro dev | nessuna inferenza in più |

| Base | Licenza | TDM che ne derivano (livello A/B) | Uso |
|---|---|---|---|
| **Qwen3.5-9B** (Instruct, nella cache) | Apache 2.0 | elephant-1, StartLux-9B, d3-flash, OneJev-9B, Nimble-v3, Clef-flash, Tacit-9B | RQ2 principale; anche generativo (B2) con e senza ragionamento |
| **Qwen3.5-9B-Base** | Apache 2.0 | JPT-9B, Kev-9B | prompt di completamento, non chat: il prompt a chat non funziona sulle basi pre-addestrate ([2610.17205](https://arxiv.org/abs/2610.17205)) |
| **Qwen3.5-4B** | Apache 2.0 | KnowLine, StartLux-4B, pplx-flash, d3-mini, OneJev-4B | replica di RQ2; generativo (B2) |
| Qwen3.5-4B-Base | Apache 2.0 | Kev-4B (JPT-4B da verificare) | livello B |
| Qwen3.5-2B, Qwen3.5-0,8B | Apache 2.0 | StartLux, d3, OneJev piccoli | curva di taglia della distanza TDM − base |
| **LFM2.5-VL-3B** / LFM2.5-2.6B-Base | LFM Open | d1-3B, LFM2.5-RLCD | gruppo non Qwen |
| **Gemma-4-E4B-it** | Apache 2.0 | Winnow-E4B, Strands Decider E4B, Gevva E4B | gruppo non Qwen principale |
| **Ministral-3-8B-Instruct** / **-Base** | Apache 2.0 | Standard One 8B / Jevstral 8B | gruppo non Qwen; replica Instruct contro Base |
| **MiMo-V2.6-Distill-Qwen-9B** | MIT | Drex v1.5 | "base che ragiona" contro Qwen3.5-9B |
| DeBERTa-v3-large | MIT | GLiClass v4 edge | non si legge senza testa: la controparte è FactCG (checker addestrato sullo stesso backbone) |
| InternLM2.5-7B-chat | — | Bespoke-MiniCheck-7B (checker, non TDM) | numero della leaderboard (63,6 zero-shot) come controparte già pubblicata |

### 3.5 Checker specializzati (B3) e riferimenti

| Gruppo | Modelli |
|---|---|
| Checker fino a 9B | Bespoke-MiniCheck-7B (adapter pronto), FactCG-DeBERTa-L (fatto), MiniCheck-Flan-T5-L, Granite Guardian 3.3 8B; facoltativi HHEM-2.1, AlignScore |
| Riferimenti fuori vincolo (una riga, nessuna analisi) | Jev 1.13 e 1.14; Winnow-12B e Rune-26B (già misurati); i 27B della leaderboard (Nimble-v4-max 79,5, Clef 79,0, Decider 78,7) |

## 4. Quanti sono, quanto costano

| Livello | Modelli | Di cui già fatti o pronti | Adapter nuovi davvero diversi |
|---|---:|---:|---|
| A, blocco non Qwen | 6 (+2 encoder) | 0 (Winnow-E4B usa il runtime di Winnow-12B) | Standard One, Jevstral, Strands E4B, Gevva, LFM-RLCD, Amber, Crivo |
| A (TDM) | 17 | 8 (A1, A2, A3, A5, A8, A10, A15 + OneJev con il runtime di A1) | Clef-flash, Tacit (con AnyJev), Kev, pplx-flash, Drex, GLiClass; StartLux-4B, d3-mini, OneJev riusano runtime esistenti |
| B (taglie) | ~8 | 0 | nessuno: stessi runtime del livello A |
| C (controlli) | 3 | 1 | Laya |
| Basi (B1) | 10 (+2 piccole), di cui 4 non Qwen | Qwen3.5-9B in cache | un solo adapter generico (R0/R1/R2) per tutte le basi Qwen |
| Checker (B3) | 4 | 2 | MiniCheck-FT5, Granite Guardian |

Con il vincolo dei 9B tutto entra in GPU da 24–48 GB, e i run possono andare in parallelo su più GPU. Il costo di calcolo è in [11](11_design_esperimenti.md) §8.

## 5. Da verificare prima del congelamento (30 novembre)

- Base effettiva di JPT-4B e Intern-Decision-4B (la card dice Instruct, la Decision Index dice Base).
- Licenza e runtime di Tacit-9B (Apache 2.0 su HF; runtime AnyJev) e di GLiClass v4 edge (Apache 2.0).
- Che i runtime di StartLux-4B, d3-mini e OneJev 1.6 siano davvero gli stessi dei fratelli maggiori (le card dicono di sì).
- Eventuali nuovi TDM fino a 9B entrati nella Decision Index dopo il 26 ottobre: ammessi solo se cambiano un contrasto, non per inseguire la classifica.
