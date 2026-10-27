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

## 4. Il nostro punto di partenza: i decision model

d1-3B (LFM2.5-VL-3B), JPT-9B (Qwen3.5-9B) e Winnow-12B (Gemma-4-12B) sono LLM addestrati a rispondere a **domande tipizzate** su uno stato: il prompt contiene lo stato (qui: documento e claim) e la domanda; il modello mette quasi tutta la probabilità su due risposte (le lettere A/B per JPT e Winnow, le parole yes/no per d1-3B). La probabilità di "sì" si legge dai logit di quelle due risposte, in un solo forward.

Misure su LLM-AggreFact test (run di ottobre 2026, `results/lytang--LLM-AggreFact/test/`; analisi in [../risultati_aggrefact_2026-10.md](../risultati_aggrefact_2026-10.md)). Q1 = `support_simple`; Q4 = `unsupported_detail` letta come 1 − P(sì). "Migliore" = soglia scelta sul test stesso, quindi ottimistica:

| Modello | Taglia | Regola | AUC macro | BAcc @0,5 | BAcc migliore (soglia) | Latenza p50 | Memoria GPU |
|---|---:|---|---:|---:|---:|---:|---:|
| d1-3B | 3B | Q1 | 83,51 | 72,48 | 74,97 (0,73) | **38 ms** | **11,6 GB** |
| d1-3B | | Q4 | 84,41 | 74,30 | 75,59 (0,67) | | |
| JPT-9B | 9B | Q1 | 84,15 | 74,32 | 76,43 (0,86) | 172 ms | ~43 GB* |
| JPT-9B | | Q4 | 84,30 | 76,74 | 76,94 (0,63) | | |
| JPT-9B | | 3 domande | 84,48 | 77,11 | 77,21 (0,61) | | |
| Winnow-12B | 12B | Q1 | 84,03 | 74,91 | 76,85 (0,93) | 169 ms | ~43 GB* |
| Winnow-12B | | Q4 | **85,04** | 76,77 | 77,37 (0,26) | | |
| Winnow-12B | | 3 domande | 84,99 | 76,38 | 77,71 (0,14) | | |
| Rune-26B-A4B | 26B | Q1 | 84,91 | 76,65 | 77,89 (0,75) | 1.450 ms | ~43 GB* + 17 GB CPU |
| CLM-v0.1-8B | 8B | Q1 | 65,11 | 58,78 | 58,78 (0,50) | 101 ms | |
| Bespoke-MiniCheck-7B | 7B | una domanda | | | 77,4 (leaderboard) | | |
| FactCG-DeBERTa-v3-L | 0,4B | classificatore | | 75,64 (run precedente) | | | |

\* memoria riservata da vLLM (92% della scheda), non usata davvero. Latenze su NVIDIA L40S, una riga alla volta, tre domande.

Numeri storici con soglia fittata sul dev di Jev (run precedente): una domanda Q1 76,69 (Winnow, soglia 0,95) e 76,20 (JPT, soglia 0,90).

Quattro fatti:

1. **Già zero-shot, con una sola domanda, i decision model grandi sono al livello di FactCG e MiniCheck dopo il loro training**: con Q4 e soglia fissa 0,5, JPT e Winnow fanno 76,7–76,8, contro 75,6 di FactCG e 75,0 di MiniCheck-FT5.
2. **Q1 dice "sì" troppo facilmente** (soglie migliori 0,73–0,93). È il difetto principale che il fine-tuning deve correggere.
3. **Q4 (`unsupported_detail`, con le descrizioni delle opzioni) è una domanda singola migliore di Q1** per d1, JPT e Winnow: AUC più alta e ~2 punti in più di BAcc a soglia 0,5 (Rune fa eccezione). Va confermato sul nostro dev ([02d](02d_varianti_domanda.md)).
4. **d1-3B è a meno di 1 punto di AUC dai modelli da 9–12B**, con latenza 4–5 volte più bassa e un quarto della memoria. La sua BAcc a soglia fissa è più bassa (72–74), ma la differenza è soprattutto di soglia e calibrazione, cioè ciò che il fine-tuning corregge più facilmente.

### 4.1 d1-3B in dettaglio

| Aspetto | Valore |
|---|---|
| Repository | `LiquidAI/d1-3B`, revisione `051bcc464b01b9f92942b364d9586b0ef5912432`, codice proprio (`trust_remote_code=True`) |
| Base | LFM2.5-VL-3B: 3,12B parametri, di cui ~400M nel codificatore visivo (SigLIP2), non usato qui |
| Backbone testuale | `lfm2`: 30 layer, 22 a convoluzione corta e 8 ad attenzione piena (layer 2, 5, 9, 13, 17, 21, 24, 27), hidden 2.048, vocabolario 128.000, embedding legate, contesto 32.768 token |
| Prompt | stato come JSON indentato, poi `QUESTION:` e la domanda; per una `noul` con criteri due righe `Yes: <descrizione di true>` / `No: <descrizione di false>`; poi `Reply with yes or no only.` e l'intestazione dell'assistente ([03](03_rendering_formato.md) §3b) |
| Lettura | P(sì) = softmax(max(z_yes, z_Yes), max(z_no, z_No, z_NO)); nessuna temperatura |
| Interfaccia | `system_one(state, questions)` e `system_one_batch(...)`: più domande sullo stesso stato in una passata (lo stato è letto una volta, le domande sono rami) |
| Licenza | LFM Open License v1.0: uso commerciale consentito solo a entità con fatturato annuo sotto 10 milioni di dollari (vale anche per i modelli derivati) |
| Dichiarati dagli autori | 8 ms per decisione su RTX 4090 con CUDA graph; primo decision model sotto 10B sul Decision Index 0.2.1 |

Perché conta per il fine-tuning:
- **training molto più economico**: LoRA in bf16 con 8k token sta comodamente nei 24 GB del portatile (pesi ~7 GB); una corsa del pilota dura una frazione di quella di JPT;
- **servizio economico**: ~4–5 volte meno latenza, ~4 volte meno memoria, energia per 1.000 righe 12,8 kJ contro ~35 kJ di JPT e Winnow;
- **rischio**: parte un po' più in basso; la qualità sui documenti lunghi è inferiore (BAcc aggregata 66 sopra 16.000 caratteri, contro 71 di JPT e Winnow).

## 5. Perché non cambiare la testa del modello

Nel decision model, all'ultima posizione del prompt il modello produce uno stato nascosto h (un vettore di dimensione d: 2.048 per d1-3B, 4.096 per JPT, 3.840 per Winnow). La testa del vocabolario calcola un logit per ogni token: z_k = w_k · h. Si leggono solo le due risposte:

```text
P(vero) = softmax([z_falso, z_vero] / T)[vero] = σ( (w_vero − w_falso) · h / T )
```

È **una regressione logistica su h**, con pesi u = (w_vero − w_falso)/T: esattamente ciò che fa la testa binaria di un classificatore, ma **già inizializzata** con quello che il modello sa. (Per Winnow i logit passano prima da un *softcapping* 30·tanh(z/30): la funzione resta determinata dalle due righe w_vero e w_falso, solo non è più perfettamente lineare. Per d1-3B ogni risposta è il massimo su più forme dello stesso token — `yes`/`Yes` e `no`/`No`/`NO` — quindi il margine è max(z_yes, z_Yes) − max(z_no, z_No, z_NO): ancora una funzione di poche righe della testa.)

Togliere la testa del vocabolario e mettere un `Linear(d → 2)` nuovo:

- non aggiunge capacità (stessa forma di funzione);
- parte da pesi casuali: al passo 0 il modello è al 50% invece che al 76–77%;
- richiede di addestrare anche la testa e di servire il modello con una classe diversa.

Esempio numerico (Winnow, coppia supportata): z_falso = 2,1, z_vero = 0,3 → P(vero) = σ(0,3 − 2,1) = σ(−1,8) = 0,14. Con u = w_vero − w_falso si ottiene u · h = −1,8 e lo stesso 0,14. Una testa nuova darebbe ~0,5 per qualunque coppia.

Le due cose utili di una "testa binaria" si ottengono senza cambiarla:

- **memoria**: in training si calcolano solo le due righe w_vero, w_falso (prodotto [B, d] × [d, 2]) invece dei ~250.000 logit del vocabolario ([05](05_training.md));
- **servizio come classificatore**: si può estrarre u alla fine ([08](08_merge_export.md)).

## 6. Decision model o LLM base?

| A favore del decision model | Contro |
|---|---|
| AUC 83,5–85 con una domanda già zero-shot (BAcc 72–77); LLM generici di taglia simile fanno ~70–73 di BAcc (Llama-3.1-8B 70,3; Qwen2.5-7B 72,8) | con dati buoni la differenza di partenza può sparire: InternLM2.5-7B passa da 63,6 a 77,4 |
| le probabilità sono già concentrate sulle due lettere | JPT è CC BY-NC 4.0 |
| meno dati e passi per arrivare allo stesso punto (da verificare) | formati particolari (A = falso in Winnow) |

Si decide con un controllo: **la stessa ricetta su LFM2.5-VL-3B, Qwen3.5-9B base e Gemma-4-12B-it** (esperimenti A2 e C1 in [07](07_esperimenti.md)). Se il base arriva entro 0,5 punti dal decision model, il backbone si sceglie per licenza e costo.

## 7. Decision model o encoder stile FactCG?

| | Decision model 9–12B | Decision model 3B (d1) | Encoder 0,4B |
|---|---|---|---|
| AggreFact atteso dopo il training | ≥ 77,4 | da misurare (parte da AUC 84,4) | ~75–76 |
| Contesto | decine di migliaia di token | 32.768 token | 2.048 (DeBERTa) / 8.192 (ModernBERT) |
| Multi-hop fra parti lontane | possibile in un chunk lungo | possibile in un chunk lungo | solo dentro un chunk |
| Costo | ~170 ms per riga (L40S, 3 domande) | ~40 ms per riga | un forward di 0,4B per chunk (anche CPU) |

L'encoder vince solo sul costo, e d1-3B riduce molto il divario di costo restando un decision model a contesto lungo. Se serve ancora di meno (CPU, milioni di claim), si **distilla** un encoder dal decision model fine-tunato ([08](08_merge_export.md) §8).

## 8. La decisione

1. **Backbone**: decision model con **una sola domanda sì/no**.
   - **d1-3B per iterare**: LoRA in bf16 sul portatile, corse brevi; è anche il candidato per la produzione se contano latenza e costo (verificare la licenza: commerciale solo sotto 10 M$ di fatturato);
   - **Winnow-12B come modello per la massima accuratezza** (AUC più alta, Apache 2.0);
   - **JPT-9B** come terzo punto di confronto (licenza non commerciale).
   Le scelte di ricetta (dati, loss, iperparametri) si fanno su d1-3B e si confermano su Winnow; se dopo il training d1-3B arriva entro ~1 punto da Winnow su dev e test_ood, è il modello da rilasciare.
2. **Domanda**: Q4 (`unsupported_detail`, letta come 1 − P(sì)) è la candidata principale, Q1 e Q3 le alternative; si sceglie sul nostro dev ([02d](02d_varianti_domanda.md)).
3. **Uscita**: logit delle due risposte, nessuna testa nuova.
4. **Dati e curriculum**: ricette MiniCheck/FactCG + fonti pubbliche, due stadi.
5. **Controlli**: stesso training su LLM base (A2) e su encoder (A5) per confermare la scelta con numeri nostri.
6. **Esclusi**: CLM-8B (AUC 65 con la domanda singola, regola a tre domande inutilizzabile).

## 9. Cosa si perde

- Il modello fine-tunato **non è più un decision model generale**: altre domande possono peggiorare. Si dichiara nel nome (`d1-3b-fc`, `jpt-9b-fc`, `winnow-12b-fc`) e nella model card.
- Non distingue "contraddetto" da "non stabilito": l'uscita è binaria.

## Criteri per passare allo step successivo

- Decisione registrata: d1-3B per il pilota e gli esperimenti, Winnow per la conferma e la versione di massima accuratezza, JPT come confronto, controlli A2 e A5 in programma.
