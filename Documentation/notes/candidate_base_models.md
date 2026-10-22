Models to transform in decision models and try.

### ≤ 4B
1. **Qwen3-4B-Instruct-2507** (and the Thinking-2507 variant): the strongest small model; handles long context and fine-tunes well
2. Phi-4-mini-instruct (3.8B): strong reasoning for its size
3. Gemma 3 4B: 128k context and good multilingual support
4. SmolLM3-3B: fully open training recipe, and it has a reasoning mode
5. Llama 3.2 3B: well supported by every tool, so it's a safe baseline

### 4–10B
1. **Qwen3-8B**: the default choice for fine-tuning at this size
2. NVIDIA Nemotron-Nano-9B-v2: a hybrid Mamba design that handles long documents quickly
3. Llama 3.1 8B: the most-used base for fact-checking/NLI fine-tunes, so you can compare against prior work
4. IBM Granite 3.3 8B: Granite Guardian, IBM's groundedness-checking model, is built on this family, so it fits your task well
5. OLMo 3 7B: training data is fully open, so you can check for contamination with AggreFact sources

### 10–15B
1. **Qwen3-14B**
2. Phi-4 (14B) or Phi-4-reasoning: very strong at reasoning-style checking
3. Gemma 3 12B
4. DeepSeek-R1-Distill-Qwen-14B: useful if you want chain-of-thought verdicts
5. Mistral Nemo 12B: Apache license, 128k context

### 15–20B (few good options here)
1. **Apriel-1.5-15B-Thinker** (ServiceNow)
2. Moonlight-16B-A3B (Moonshot): mixture-of-experts (MoE), cheap to run
3. Ling-lite-1.5 (~16.8B, MoE)
4. DeepSeek-V2-Lite (15.7B): old now, only worth it as a reference point
5. Honestly, it's probably better to skip this bracket. Either go up to gpt-oss-20b (21B total) or stay at 14B.

### 20–25B
1. **Mistral Small 3.2 24B**: dense model, Apache license, a very good base for fine-tuning
2. Magistral Small (24B): the reasoning variant of Mistral Small
3. gpt-oss-20b (21B total / 3.6B active, MoE): fast, with adjustable reasoning effort
4. ERNIE-4.5-21B-A3B: MoE
5. Reka Flash 3 (21B)

### 25–30B
1. **Qwen3-30B-A3B-Instruct-2507**: about 30.5B in total, but runs at the speed of a ~3B model
2. Qwen3-30B-A3B-Thinking-2507
3. Gemma 3 27B: dense, so it's simpler to fine-tune than an MoE
4. NVIDIA Nemotron-3-Nano-30B-A3B: hybrid MoE, good with long context
5. (A 26B-A4B-style MoE like the Rune model you're testing also belongs here.)

### 30–35B
1. **Qwen3-32B**: the strongest dense model in this bracket
2. OLMo 3 32B (Think/Instruct): fully open data
3. GLM-4-32B-0414
4. IBM Granite 4.0-H-Small (32B total / 9B active, hybrid design)
5. EXAONE 4.0 32B (non-commercial license) or QwQ-32B

### Things to check for fact-checking specifically
- **Baselines to beat:** MiniCheck (Flan-T5-L and Bespoke-MiniCheck-7B) was built for LLM-AggreFact. If a fine-tuned 8B model doesn't beat Bespoke-MiniCheck-7B, it isn't worth the cost.
- **Context length:** AggreFact's grounding documents can be long. Fine-tune with at least 8–32k context, or split documents into chunks and combine the chunk scores the way MiniCheck does.
- **How the model outputs a verdict:** a classifier head or a supported/unsupported token-probability score is cheaper to evaluate and easier to calibrate. A reasoning ("thinking") model helps on multi-hop claims but makes inference slower.
- **Contamination:** several AggreFact sources are public. A model with open data (OLMo 3, SmolLM3) lets you confirm it hasn't seen them in training.
- **Fine-tuning MoE vs dense:** LoRA works more simply on dense models (Qwen3-14B/32B, Mistral Small, Gemma 3). With MoE models, check that your framework handles expert layers properly.

If you want only three to start with, I'd pick **Qwen3-4B-2507 → Qwen3-14B → Qwen3-30B-A3B or Mistral Small 3.2**. They share one fine-tuning setup (except Mistral) and show you clearly how results change with model size.





**FINE TUNE DECISION MODELS**
Attenzioni specifiche
Calibrazione: il valore di questi modelli sono le probabilità calibrate. Un SFT normale tende a renderle overconfident. Usa Brier o cross-entropy solo sulle lettere delle opzioni, learning rate basso, e un piccolo replay di domande generiche, perché i loro dati di training originali non sono pubblici.
Prima la soglia, poi il fine-tuning: la soglia 0.30 è stata fittata sulle probabilità di Jev, non su questi modelli. Parte del distacco da Jev (79.09) potrebbe sparire solo ricalibrando la soglia sui tuoi dati di validazione, senza allenare nulla. Misuralo prima, così sai quanto guadagno arriva davvero dal fine-tuning.
Dati: LLM-AggreFact non si può usare per il training (vedi benchmark_notes.md). Ti servono altri dataset, oppure dati sintetici stile MiniCheck/FactCG.
Controllo: fai fine-tuning anche di un modello base generico della stessa taglia con gli stessi dati (Gemma-4-12B per Winnow, Qwen3.5-9B per JPT). Solo così puoi dimostrare che partire da un decision model aiuta davvero.
In sintesi: Winnow-12B come candidato principale, JPT-9B per iterare in fretta sul portatile, e la lista di modelli generici di prima solo come controlli.






**Ricalcolare dai risultati**
Ogni benchmark salva in cache le risposte grezze (answers.jsonl con le probabilità; per Winnow e Rune i logit delle opzioni). Soglia e regola vengono applicate solo quando si genera il report, quindi:


python jpt-9b/benchmark.py --offline --threshold 0.45
python winnow-12b/benchmark.py --offline --threshold 0.45 --temperature 1.5   # Winnow/Rune: anche la temperatura
Il report si ricostruisce in pochi secondi, senza GPU.

Il problema: da dove prendi la nuova soglia
Se provi più soglie sul test e tieni la migliore, il risultato non vale più. È esattamente ciò che vieta il tuo PROTOCOL.md: soglia fittata solo sul dev e test valutato una volta sola. Lo dicono anche gli help di --threshold: "do not choose a threshold from test results".

La procedura corretta:

Esegui ogni modello sul dev split. Ce l'hai già in locale (lytang___llm-aggre_fact/.../llm-aggre_fact-dev.arrow). Basta il campione del protocollo, fino a 110 esempi per fonte ed etichetta, cioè qualche migliaio di righe e non 29k. Questa è l'unica inferenza nuova.
Fitta la soglia (ed eventualmente la temperatura) sul dev, con la stessa griglia 0.01–0.99 che hai usato per Jev.
Applica quella soglia al test con --offline. Il test non va rieseguito, usi la cache che hai già.
Al momento i benchmark.py puntano al test split pinnato. Per il passo 1 serve un piccolo flag --split dev (con output in una cartella separata) più uno script che fitta la soglia sulla cache del dev. Se vuoi lo aggiungo io.

Nel frattempo
Nei report.json c'è già la ROC AUC per fonte (metrics.py:27), che non dipende dalla soglia. Confrontando le AUC di JPT, Winnow, Rune e Jev vedi subito quali modelli ordinano meglio gli esempi, indipendentemente dalla soglia 0.30. Se l'AUC di un modello è vicina a quella di Jev ma la balanced accuracy è più bassa, il distacco viene quasi tutto dalla soglia.