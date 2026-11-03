# Step 0a — La scelta dell'approccio, spiegata

Questo file spiega **perché** si fa quello che descrivono gli altri file. Contiene tutto ciò che serve sapere di MiniCheck e FactCG per lavorare: non serve leggere i paper.

## 1. Le tre famiglie di fact checker documento–claim

| Famiglia | Esempi | Come funziona | Pro | Contro |
|---|---|---|---|---|
| Classificatore encoder | FactCG-DeBERTa (0,4B), MiniCheck-RoBERTa/DeBERTa | un encoder bidirezionale legge "documento + claim" e una piccola testa produce 2 logit (supportato / no) | velocissimo, anche su CPU | contesto corto (512–2.048 token), tetto di qualità ~75–76 su LLM-AggreFact |
| Seq2seq piccolo | MiniCheck-Flan-T5-Large (0,8B) | il modello genera la risposta; si leggono le probabilità di uno o due token ("1"/"0" o "Yes"/"No") | veloce | stesso tetto |
| LLM con risposta sì/no | Bespoke-MiniCheck-7B, i nostri decision model | un LLM legge il prompt; si leggono i logit di due token di risposta | qualità più alta, contesto lungo | GPU necessaria |

## 2. MiniCheck, in breve

**Obiettivo**: un verificatore piccolo ed economico che giudichi se un documento supporta un claim generato da un LLM.

**Contributo**: due ricette di dati sintetici (le procedure complete sono in [02b](02b_dataset_generazione_llm.md)):

- **C2D (claim → documento)**: si parte da un claim, lo si scompone in fatti atomici, per ogni fatto si fanno scrivere **due frasi che solo insieme lo implicano**, si fa scrivere un documento che contiene tutte le frasi (positivo), poi documenti a cui manca una frase (negativi). Insegna a combinare più frasi e a dire "no" quando manca un pezzo.
- **D2C (documento → claim)**: si parte da un articolo vero, lo si divide in pezzi, si fa riassumere ogni pezzo, si tolgono frasi dal pezzo e si controlla quali fatti del riassunto restano supportati. Insegna lo stesso su testo reale.

**Training**: 14k esempi sintetici (7k C2D + 7k D2C) + 21k esempi di ANLI su cui modelli precedenti sbagliavano. Due stadi: prima ANLI + C2D (2 epoche), poi solo D2C (1 epoca, learning rate più basso). Fine-tuning completo, cross-entropy.

**Inferenza**: il documento si divide in chunk di ~400 token (o 500 parole) sui confini di frase; si valuta ogni chunk; il punteggio è il **massimo** sui chunk; soglia fissa 0,5.

**Risultati su LLM-AggreFact** (BAcc macro): MiniCheck-Flan-T5-L 75,0; MiniCheck-RoBERTa-L 73,5; MiniCheck-DeBERTa-L 73,1. La stessa ricetta su un LLM da 7B (InternLM2.5-7B-chat, che parte da 63,6) dà **Bespoke-MiniCheck-7B: 77,4**.

**Altri risultati utili**: scomporre il claim in fatti atomici e verificarli uno per uno migliora di soli +0,3 punti a un costo 2–4 volte maggiore; risolvere i pronomi del claim non serve.

## 3. FactCG, in breve

**Obiettivo**: insegnare il ragionamento **multi-hop** (combinare fatti che parlano di entità collegate).

**Contributo**: la ricetta **CG2C (context graph → claim)** (procedura completa in [02b](02b_dataset_generazione_llm.md)):

1. con un LLM si estrae dal documento un **grafo**: triple (entità 1, entità 2, relazione), ciascuna legata alla frase da cui viene;
2. si sceglie un **cammino** di 2–4 triple collegate (meglio lineare, senza rami);
3. un LLM scrive un claim che attraversa tutto il cammino → positivo;
4. un LLM **toglie dal documento una relazione** del cammino, con modifiche minime → negativo (il claim non è più verificabile);
5. si scartano i negativi che un modello NLI già indovina (si tengono solo quelli difficili).

Due varianti: **CG2C-MHQA**, che parte da dataset di domande multi-hop (HotpotQA, MuSiQue), e **CG2C-Doc**, che parte da documenti qualsiasi.

**Training**: stadio 1 = NLI + C2D di MiniCheck + CG2C-MHQA; stadio 2 = D2C di MiniCheck + CG2C-Doc; 1 epoca ciascuno, ottimizzatore nuovo al secondo stadio. Gli stessi backbone di MiniCheck. Per DeBERTa: lr 1e-5, warmup 6%, batch effettivo ~64, input fino a 2.048 token, chunk da 550 parole, `max` sui chunk.

**Risultati**: FactCG-DeBERTa-v3-L **75,6** su LLM-AggreFact (rimisurato in questa repo: 75,64). Migliore di MiniCheck-DeBERTa (73,1) con lo stesso backbone: la differenza è tutta nei dati.

**Cosa ne ricaviamo**: MiniCheck e FactCG **non introducono architetture**. Il loro valore è: (a) le ricette di dati, (b) il curriculum in due stadi, (c) il chunking con `max`, (d) l'obiettivo di una soglia vicina a 0,5. Tutte e quattro si applicano a qualunque backbone.

## 4. Il nostro punto di partenza: i decision model dello studio

I modelli sono i 10 decision model dello studio focalizzato, tutti costruiti su Qwen3.5 (6 sul 9B, 4 sul 4B), più i due backbone ([00](00_overview.md) §2). Ognuno riceve uno **stato** (qui documento e claim) e una **domanda** con opzioni, e restituisce una probabilità per opzione in un solo forward.

Cosa li distingue, e perché conta per il fine-tuning:

| Aspetto | Varianti nei 10 modelli | Conseguenza per il training |
|---|---|---|
| Prompt | ognuno il suo (stato come JSON, come righe, come testo indentato; domanda e opzioni in formati diversi) | si renderizza con il codice degli autori, mai con un prompt nostro ([03](03_rendering_formato.md)) |
| Attenzione | causale (KnowLine, OneJev, Nimble), bidirezionale sul prompt nei layer pieni (elephant-1 a blocchi, StartLux, d3-flash, pplx, Jeffan), non dichiarata (Clef) | la stessa maschera va usata in training, altrimenti si addestra un altro modello |
| Lettura | due righe della testa del vocabolario (lettere o codici: 8 modelli) o una testa propria (pplx a 255 vie, Clef congiunta) | la loss si calcola sui due punteggi che il modello usa davvero ([04](04_loss.md)) |
| Training originale | full fine-tune, LoRA (Nimble), media di due LoRA (KnowLine), distillazione (pplx) | si aggiunge sempre una **nuova** LoRA sopra i pesi pubblicati |

### 4.1 Cosa si sa già

Dai run della repo sul test di AggreFact, con la regola a tre domande di Jev e soglia 0,30: elephant-1 76,70, KnowLine-4B 75,65. I numeri dello studio focalizzato (domanda Q1, soglia fittata su AF-dev, tutti e 10 i modelli e i due backbone) arrivano dai run RUN-01…RUN-05 ([../paper_idea/15_focused_10_tdm_experiment_guide.md](../paper_idea/15_focused_10_tdm_experiment_guide.md) §7): quando sono pronti si copiano in [01](01_baseline.md) §1 e non si rifanno.

Riferimenti esterni: Bespoke-MiniCheck-7B 77,4 e FactCG 75,6 su LLM-AggreFact. Il primo obiettivo del fine-tuning è superare 77,4 con un modello da 4B o 9B.

### 4.2 I due modelli su cui si lavora per primi

| | KnowLine-4B (A11) | elephant-1 (A1) |
|---|---|---|
| Repository | `PelaAI/KnowLine-4B-Gen5.5` | `baselabs/elephant-1` |
| Base e training originale | Qwen3.5-4B; due LoRA SFT mediate e fuse | Qwen3.5-9B; full fine-tune |
| Codice degli autori | front end nel repository (`knowline_server.py`, `knowline_layout.py`), backend transformers | motore `qev` di OneJev (copiato in `models/elephant-1/qev`) con `serve.patch` |
| Prompt | stato come testo indentato in un turno utente; turno con istruzione, `Question:` e opzioni `A. Yes: …` / `B. No: …`; risposta aperta con `Answer:`. Richieste fino a 3.000 token **scritte due volte** | sistema + `<state>` (JSON indentato) come prefisso comune; ogni domanda è un suffisso `Question: … Options: A. yes: … B. no: … Answer with one letter: A, B.` |
| Attenzione | causale | bidirezionale a blocchi nei layer pieni (il prefisso vede sé stesso, ogni suffisso vede prefisso e sé stesso); i layer DeltaNet restano causali |
| Lettura | softmax sui token delle etichette; temperatura 1,50 già incorporata nella normalizzazione finale | softmax sui logit delle lettere A/B (testa del vocabolario in float32), temperatura 1 |
| Memoria per il training LoRA in bf16 | ~9 GB di pesi: entra in 24 GB | ~19 GB di pesi: GPU da 45 GB |

Perché si itera su KnowLine-4B: è causale e legge due token della testa del vocabolario, quindi il training è quello standard di un LLM; costa meno della metà di un 9B. Le scelte di ricetta (dati, loss, iperparametri) si fanno qui e si confermano su elephant-1.

## 5. Perché non cambiare la testa del modello

All'ultima posizione del prompt il modello produce uno stato nascosto h. Nei modelli con lettura dalla testa del vocabolario, il logit di un token è z_k = w_k · h, e si leggono solo le due opzioni:

```text
P(vero) = softmax([z_falso, z_vero] / T)[vero] = σ( (w_vero − w_falso) · h / T )
```

È **una regressione logistica su h**, con pesi già inizializzati da quello che il modello sa. Togliere la testa e metterne una nuova `Linear(d → 2)` non aggiunge capacità, parte da pesi casuali (al passo 0 il modello è al 50% invece che al suo livello) e cambia il modo di servirlo.

Esempio numerico: z_falso = 2,1, z_vero = 0,3 → P(vero) = σ(0,3 − 2,1) = σ(−1,8) = 0,14. Una testa nuova darebbe ~0,5 per qualunque coppia.

Per i due modelli con una **testa propria** (pplx-decider-flash, Clef-Flash) vale lo stesso principio: si tiene la loro testa, con i suoi pesi, e si addestra insieme alla LoRA (è piccola).

Le due cose utili di una "testa binaria" si ottengono comunque:
- **memoria**: in training si calcolano solo i due punteggi delle opzioni invece dei ~248.000 logit del vocabolario ([05](05_training.md));
- **servizio come classificatore**: si può estrarre il vettore (w_vero − w_falso) alla fine ([08](08_merge_export.md)).

## 6. Decision model o backbone? (B-ctrl)

È la domanda dell'esperimento B-ctrl del paper: **a parità di dati e di ricetta di fine-tuning, un checker ottenuto da un decision model è migliore di uno ottenuto dal suo backbone?** Nessuno l'ha misurato: i checker specializzati (MiniCheck, FactCG) partono da LLM generici, le ricette di fine-tuning pubblicate addestrano decision model generali.

| A favore del decision model | Contro |
|---|---|
| le probabilità sono già concentrate sulle due opzioni; la lettura è già calibrata dagli autori | con dati buoni la differenza di partenza può sparire (InternLM2.5-7B passa da 63,6 a 77,4 con i dati di MiniCheck) |
| meno dati e passi per arrivare allo stesso punto (da verificare) | ricette con attenzione o testa proprie complicano il training |

Si risponde con le coppie elephant-1 / Qwen3.5-9B e KnowLine-4B / Qwen3.5-4B, poi con gli altri decision model ([07](07_esperimenti.md), fase B). Il backbone si addestra con il prompt neutro dello studio e le opzioni A/B in ordine casuale ([03](03_rendering_formato.md) §2.4).

## 7. Decision model o encoder stile FactCG?

| | Decision model 9B | Decision model 4B | Encoder 0,4B |
|---|---|---|---|
| AggreFact atteso dopo il training | ≥ 77,4 | da misurare | ~75–76 |
| Contesto | 32.768 token (limite dei runtime degli autori) | idem | 2.048 (DeBERTa) / 8.192 (ModernBERT) |
| Multi-hop fra parti lontane | possibile in un chunk lungo | possibile in un chunk lungo | solo dentro un chunk |
| Costo | un prefill di 9B | un prefill di 4B | un forward di 0,4B per chunk (anche CPU) |

L'encoder vince solo sul costo. Se serve ancora di meno (CPU, milioni di claim), si **distilla** un encoder dal decision model fine-tunato ([08](08_merge_export.md) §8).

## 8. La decisione

1. **Backbone**: decision model con **una sola domanda sì/no**.
   - **KnowLine-4B per iterare**: LoRA in bf16 su 24 GB, corse brevi;
   - **elephant-1 come modello principale a 9B**;
   - gli altri 8 decision model con la ricetta finale (fase B di [07](07_esperimenti.md)).
2. **Domanda**: si sceglie sul nostro dev fra Q1, Q2 e Q4 dello studio ([02d](02d_varianti_domanda.md)).
3. **Uscita**: i punteggi delle due opzioni che il modello usa già; nessuna testa nuova.
4. **Dati e curriculum**: ricette MiniCheck/FactCG + fonti pubbliche, due stadi.
5. **Controlli**: stesso training sui due backbone (B-ctrl) e su un encoder (A5).

## 9. Cosa si perde

- Il modello fine-tunato **non è più un decision model generale**: altre domande possono peggiorare. Si dichiara nel nome (`knowline-4b-fc`, `elephant-1-fc`, …) e nella model card.
- Non distingue "contraddetto" da "non stabilito": l'uscita è binaria.
- Le licenze restano quelle dei pesi di partenza: StartLux e Nimble sono non commerciali (CC BY-NC).

## Criteri per passare allo step successivo

- Decisione registrata: KnowLine-4B per gli esperimenti, elephant-1 per la conferma, gli altri 8 e i due backbone per B-ctrl.
