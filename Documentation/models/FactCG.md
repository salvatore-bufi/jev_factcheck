# Analisi FactCG

**Paper:** *FactCG: Enhancing Fact Checkers with Graph-Based Multi-Hop Data*

**Link:** https://aclanthology.org/2025.naacl-long.258/

---

## 1. Il Problema: La Carenza di Ragionamento Multi-Hop

I fact-checker attuali non sanno unire i concetti ai diversi hop perché non sono mai stati addestrati a farlo.

---

## 2. Architettura dei Dati Sintetici: CG2C (Context Graph to Claim)

La soluzione proposta è **CG2C**, un generatore di dati che usa i grafi per forzare il modello a imparare percorsi logici complessi.


### Estrazione del Sotto-Grafo di contesto Multi-Hop

Poiché i claim generati dai LLM sono condizionati dai documenti di base, spesso non sono completamente decontestualizzati e nascondono catene di ragionamento implicite. Per estrarre la catena logica completa di ciascun claim, gli autori utilizzano i LLM attraverso tre  step : 
-  **Costruzione del grafo di contesto $\mathcal{G}$**: A partire dal documento $doc$, il sistema segue l'approccio basato su prompt di GraphRAG per estrarre triple nel formato $\langle \text{entità, entità, relazione} \rangle$. 
La relazione è una breve frase che descrive come le due entità sono connesse in base al contesto del documento (ed è non direzionale). Tutte le triple estratte vengono poi raggruppate in modo che ogni tripla all'interno di un cluster condivida almeno un'entità.  
- **Mappatura del sotto-grafo $\mathcal{G}_c$**: Per ogni claim $c$, l'LLM individua le triple ad esso correlate per isolare uno specifico sotto-grafo di contesto $\mathcal{G}_c$. Questo sotto-grafo contiene esattamente le informazioni presenti in $c$ che necessitano di verifica rispetto al documento $doc$. 
-  **Conteggio degli Hop**: Il numero di triple connesse in $\mathcal{G}_c$ definisce il numero di "hop" (i passaggi logici) richiesti per il claim $c$. Se il sotto-grafo $\mathcal{G}_c$ contiene elementi disgiunti, il conteggio degli hop viene calcolato sul sotto-grafo più esteso.  

### Generazione CG2C-MHQA 
Variante generata a partire da dataset pubblici di Multi-Hop QA (come HotpotQA e Musique).
Ogni campione di dati di partenza contiene un documento sorgente $doc$, una domanda $q$, una risposta $ans$ e le frasi di supporto $doc_{support}$ (dove $doc_{support} \subset doc$ e fornisce l'evidenza per la risposta).  
- **Generazione del Claim (Campione Positivo)**: Un LLM converte la coppia $\langle ans, q \rangle$ in una singola frase dichiarativa, definita come claim $c$. Il campione positivo è quindi definito dalla coppia $\langle doc, c \rangle$.  
- **Corruzione (Campione Negativo)**: Per creare campioni negativi, il sistema esegue i seguenti passaggi:  Estrae un grafo di base $\mathcal{G}_s$ a partire dalle frasi di supporto $doc_{support}$.
  Sfrutta la coppia $\langle \mathcal{G}_s, c \rangle$ per estrarre il sotto-grafo di contesto $\mathcal{G}_c$ (usando la stessa strategia di GraphRAG definita nella Sezione 2.1).  Dal sotto-grafo $\mathcal{G}_c$, seleziona casualmente una tripla e usa un LLM per rimuovere quella specifica relazione dal documento originale, creando il documento corrotto $doc_{neg}$. Il campione negativo diventa $\langle doc_{neg}, c \rangle$. 
-  **Filtro Logico NLI**: Le coppie generate vengono valutate da un modello RoBERTa-MNLI per assicurarsi che i dati negativi siano effettivamente difficili. L'output del modello NLI ($y$) viene mappato nelle etichette finali tramite la funzione $f(y)$:
  $$f(y) = \begin{cases} \text{Grounded} & \text{if } y = \text{Entailment} \\ \text{Ungrounded} & \text{if } y \in \{\text{Contradiction}, \text{Neutral}\} \end{cases}$$
I campioni che il modello NLI riesce a predire correttamente vengono eliminati.



### Generazione CG2C-DOC 
- **Estrazione context graph**: da un documento $doc$ , esattamente come in Estrazione del Sotto-Grafo di contesto Multi-Hop.
- Estrazione sottografi con specifici numero di hops e dimensioni.
- **Generazione claim**: uso LLM per generare claim $c$ che includa tutt i nodi del grafo
- **Positive data sample**: creato dalla coppia $\langle doc, c \rangle$
- **Negative data sample**: estraiamo relazione da sottografo (random) e chiediamo all LLM di rimuoverla dal doc, ottenendo $\langle doc_{neg}, c \rangle$



---





### La Classificazione delle Topologie dei Grafi (Con Esempi)

Il sistema estrae programmaticamente sotto-grafi di diverse forme. Gli autori distinguono tra grafi lineari (*Without Branch*) e ramificati (*With Branch*).
**Meglio i grafi lienari:** Addestrare sui grafi **lineari (Without Branches)** produce risultati migliori (BAcc 77.2% vs 77.1% su DeBERTa). I claim ramificati possono infatti essere facilmente scomposti mentalmente in sotto-claim a 1-hop separati, riducendo la complessità del ragionamento richiesto al modello.

---

## 3. Esempio di Generazione (CG2C-Doc)

Il processo di estrazione da documenti generici (CG2C-Doc) è illustrato nella Figura 2 del paper e segue questi step esatti:

**1. Il Documento Originale:**

> *Paragrafo A:* "L'India [...] è il settimo Paese più grande per area e il secondo più popoloso, con oltre 1.2 miliardi di persone."
> *Paragrafo B:* "Indogrammodes è un genere di falene della famiglia Crambidae. Contiene una sola specie, Indogrammodes pectinicornalis, che si trova in India."
> 
> 

**2. Costruzione del Grafo e Sotto-Grafo:**
L'LLM mappa le entità in una catena (Multi-Hop):
`Falene` $\rightarrow$ `Indogrammodes` $\rightarrow$ `Indogrammodes pectinicornalis` $\rightarrow$ `India` $\rightarrow$ `Area`.

**3. Generazione del Claim (Campione Positivo):**
L'LLM scrive una frase che attraversa tutti i nodi del sotto-grafo:

> *"Il genere Indogrammodes in India, il settimo Paese più grande del mondo, contiene solo una specie di falena, che è Indogrammodes pectinicornalis."* $\rightarrow$ Etichetta: **Vero (1)**.
> 
> 

**4. Corruzione del Documento (Campione Negativo):**
Il sistema rimuove intenzionalmente *una singola relazione* dal testo originale. In questo caso, il Paragrafo A viene riscritto rimuovendo il dettaglio sull'area:

> *Paragrafo A Corrotto:* "L'India [...] è un Paese situato in Asia meridionale. È il secondo Paese più popoloso, con oltre 1.2 miliardi di persone."
> 
> 

Ora il claim generato allo step 3 non è più verificabile perché manca il ponte logico sull'area dell'India. $\rightarrow$ Etichetta: **Falso/Non Supportato (0)**.

---
## Varianti del modello
- FactCG-RBT (basato su RoBERTa): Utilizza un approccio standard di classificazione su coppie di sequenze (sequence pair classification). Prende in input la coppia documento-claim e restituisce direttamente una probabilità.  

- FactCG-FT5 (basato su Flan-T5): Essendo un modello generativo, utilizza un template di input basato su istruzioni (nello specifico, il template FLAN usato per il dataset ANLI). Durante la fase di inferenza, il sistema non gli fa generare testo libero, ma legge i logit (i punteggi grezzi di probabilità) associati esclusivamente alle parole "Yes" (Sì) e "No". Questi logit vengono usati come punteggi di classificazione finale.  

- FactCG-DBT (basato su DeBERTa): Usa lo stesso template di istruzioni FLAN impiegato per FT5, ma viene addestrato per eseguire una classificazione di frasi diretta, una scelta che si è rivelata empiricamente molto efficace. I logit grezzi generati dal modello passano attraverso una funzione matematica (softmax) per essere convertiti in probabilità da 0 a 1. Infine, si applica la soglia $\theta$ per ottenere l'output binario $y$ (1 = Supportato, 0 = Non Supportato)


---

## 4. Architettura dettagliata (dal codice)

> **Fonti:** paper (App. A.1–A.6) e codice ufficiale [derenlei/FactCG](https://github.com/derenlei/FactCG): `factcg/grounding_model.py` (modello), `factcg/dataloader.py` (formato input), `train.py`/`train.sh` (training), `factcg/inference.py` (inferenza). A differenza di MiniCheck, qui **anche il codice di training è pubblico**, quindi ogni dettaglio si può verificare.

### 4.1 Idea generale
Come MiniCheck, FactCG **non inventa un'architettura**: usa gli **stessi tre backbone** (*"using the same backbone models"*) e cambia solo i dati di training (CG2C). Tutti e tre i modelli sono implementati in un'unica classe PyTorch Lightning, `GroundingModel`, che è una versione semplificata del `BERTAlignModel` di AlignScore: tiene solo la parte binaria. Il fine-tuning è **completo**: tutti i pesi vengono aggiornati, nessun freezing, nessun LoRA.

Struttura comune del codice:
```python
self.base_model = <backbone>                                   # dipende dalla variante
self.bin_layer  = nn.Linear(hidden_size, 2)                    # testa binaria: 0 = ungrounded, 1 = grounded
self.dropout    = nn.Dropout(p=0.1)
```

| Variante | Backbone (init) | Formato input | Testa | Teste rimosse |
| --- | --- | --- | --- | --- |
| **FactCG-RBT** | RoBERTa-large **dal checkpoint AlignScore** | coppia di sequenze `(doc, claim)` | `pooler` di RoBERTa + `bin_layer` **ereditati da AlignScore** | `tri_layer` (3-way) e `reg_layer` (regressione) di AlignScore |
| **FactCG-DBT** | `microsoft/deberta-v3-large` (vanilla) | template FLAN, **sequenza singola** | `ContextPooler` + `bin_layer` **nuovi, init random** | nessuna (il backbone non ha teste) |
| **FactCG-FT5** | `google/flan-t5-large` | template FLAN, sequenza singola | **LM head** del decoder ristretta ai token "No"/"Yes" | nessuna (il `bin_layer` viene creato ma non è mai usato) |

### 4.2 FactCG-RBT: testa ereditata da AlignScore, teste in più eliminate
Questa è la variante in cui le teste vengono davvero **eliminate**:
1. Si costruisce un `RobertaModel` vuoto (`RobertaModel(AutoConfig...)`, pesi casuali) più il `bin_layer`.
2. Si carica il checkpoint Lightning di **AlignScore-large**. AlignScore ha un encoder condiviso e 3 teste sul `pooler_output`: `bin_layer` (2 classi), `tri_layer` (3 classi NLI) e `reg_layer` (score continuo).
3. L'hook `on_load_checkpoint` **cancella dallo state dict** `tri_layer.*`, `reg_layer.*` e `bin_finetuned_layer.*`. Restano encoder, pooler e **`bin_layer` di AlignScore**, che riparte dai pesi già addestrati. Il paper lo conferma: *"We load both the base model and the binary classification layer."*

```
tokenizer(doc, claim)  →  <s> doc </s></s> claim </s>        (sequence-pair classification classica)
        │
RoBERTa-large (pesi AlignScore)
        │  h_[<s>]
pooler: tanh(W_p · h_[<s>])            ← da AlignScore
        │
dropout(0.1) → bin_layer: Linear(1024→2)  ← da AlignScore (le altre 2 teste eliminate)
        │
softmax → P(grounded)
```
In sostanza la testa binaria di AlignScore viene "specializzata" sul grounding di documenti multi-hop.

### 4.3 FactCG-DBT: testa nuova da zero
```python
self.base_model = AutoModel.from_pretrained("microsoft/deberta-v3-large")      # solo encoder, pre-addestrato
self.pooler = DebertaV2ForSequenceClassification(config).pooler                # istanziato da config → pesi RANDOM
self.bin_layer = nn.Linear(1024, 2)                                            # RANDOM
```
- `AutoModel` carica **solo l'encoder** pre-addestrato (MLM/RTD), senza teste.
- Il **pooler** viene "preso in prestito" da una `DebertaV2ForSequenceClassification` costruita *dalla config* e non da pesi pre-addestrati, quindi è **inizializzato a caso**. È il `ContextPooler` di DeBERTa: prende $h_{[CLS]}$, applica dropout → `dense(1024→1024)` → **GELU**.
- `bin_layer` è un `Linear(1024→2)` nuovo, anche lui casuale.

```
"{doc}\n\nChoose your answer: based on the paragraph above can we conclude that \"{claim}\"?
 \n\nOPTIONS:\n- Yes\n- No\nI think the answer is "
        │   (una sola sequenza: [CLS] prompt [SEP], niente coppia)
DeBERTa-v3-large (24 layer, hidden 1024, posizioni relative → input fino a 2048 token)
        │  h_[CLS]
ContextPooler: dense(1024→1024) + GELU      ← NUOVO
        │
dropout(0.1) → bin_layer: Linear(1024→2)    ← NUOVO
        │
softmax → P(grounded)
```
**Particolarità:** anche se DeBERTa è un encoder, gli si passa **il prompt istruzione di FLAN** (lo stesso di FT5) invece della coppia `[CLS] doc [SEP] claim [SEP]`. Il paper la chiama *"sentence classification"*: il modello classifica l'intero prompt come un'unica sequenza, e gli autori riportano che empiricamente funziona meglio.
Il checkpoint su HF (`yaxili96/FactCG-DeBERTa-v3-Large`) è salvato come `DebertaV2ForSequenceClassification` standard: `pooler` → `pooler`, `bin_layer` → `classifier`.

### 4.4 FactCG-FT5: classificazione con la LM head ("Yes"/"No")
Non viene aggiunta nessuna testa. Il `bin_layer` esiste nel codice ma per T5 non è mai chiamato.
```python
decoder_input_ids = zeros((B, 1))                       # solo il token di start del decoder (pad = 0)
logits = T5(input_ids, attention_mask, decoder_input_ids).logits.squeeze(1)   # (B, 32128)
bi_label_score = logits[:, [465, 2163]]                 # 465 = "▁No"  → classe 0,  2163 = "▁Yes" → classe 1
```
- Si fa **un solo passo di decoding** e, dei 32K logit del vocabolario, si tengono solo quelli di **"No"** e **"Yes"**. Questi due logit si comportano come l'output di una testa a 2 classi.
- **Anche in training** la loss è una cross-entropy **su questi 2 logit soltanto**, non la loss seq2seq su tutto il vocabolario. È una differenza rispetto a MiniCheck-FT5, che usa i target testuali "1"/"0" (vedi minicheck.md, 4.4).
- Gli ID 465/2163 sono stati verificati con il tokenizer di `google/flan-t5-large`.

### 4.5 Loss
La stessa per tutte le varianti (`grounding_model.py`):
$$\mathcal{L} = \frac{1}{N}\sum_{i=1}^{N} \frac{\text{CE}(z_i, y_i)}{\ln 2}, \qquad z_i \in \mathbb{R}^2$$
È una cross-entropy binaria a 2 logit (`reduction='sum'`, poi divisa per il numero di esempi), espressa in **bit** dividendo per $\ln 2$, una convenzione ereditata da AlignScore. È solo un fattore di scala. Etichetta 1 = Grounded, 0 = Ungrounded.

### 4.6 Pipeline di inferenza
Da `inference.py`:
1. Il documento viene spezzato in frasi (NLTK) e le frasi vengono raggruppate in **chunk sequenziali** fino a **550 token** (contati con `nltk.word_tokenize`). Nel paper: 400 per RBT, 550 per DBT e FT5.
2. Ogni chunk viene inserito nel template insieme al claim, poi forward e softmax: $p_k = P(\text{grounded} \mid \text{chunk}_k, c)$.
3. **Max sui chunk**: $\text{score} = \max_k p_k$ (in stile SummaC/AlignScore; il codice restituisce anche l'indice del chunk migliore).
4. Soglia $\theta$: **0.5 fissa**, oppure scelta **per dataset** sul validation set di LLM-AggreFact con due decimali. Gli autori preferiscono la seconda.

---

## 5. Dettagli di Addestramento

Loss: cross-entropy binaria (vedi 4.5).

FactCG viene addestrato con un rigoroso processo a due stadi per garantire l'assorbimento di capacità logiche:

| Stadio | Dati Utilizzati | Scopo |
| --- | --- | --- |
| **Stadio 1** (1 Epoca) | Dati NLI base + MiniCheck C2D + **CG2C-MHQA** | Insegnare i fondamentali dell'inferenza logica e il multi-hop basato su domande (sfruttando dataset come HotpotQA e Musique).
| **Stadio 2** (1 Epoca) | MiniCheck D2C + **CG2C-Doc** | Adattare il modello a documenti realistici (news, articoli) e rafforzare il ragionamento multi-hop lineare generato autonomamente.

Lo stadio 2 riparte dal checkpoint finale dello stadio 1 (`--ckpt-path`), con optimizer e scheduler nuovi. **FactCG-RBT non usa ANLI** nello stadio 1 (solo C2D + CG2C-MHQA), come MiniCheck-RBTA, perché AlignScore l'aveva già visto.

**Iperparametri (paper App. A.1 + `train.py`/`train.sh`):**

| Parametro | Valore |
| --- | --- |
| Batch | 2 per GPU × gradient accumulation 8 (× 3 GPU in `train.sh`, 4 nel paper) → batch effettivo 48–64 |
| Learning rate | Paper: **1e-5** (RBT, DBT), **5e-5** (FT5). Attenzione: il default di `train.py` è 5e-5 e `train.sh` non lo sovrascrive, quindi lo script pubblico allena DBT a 5e-5 |
| Optimizer | AdamW, $\epsilon$ = 1e-6, weight decay 0 (bias e LayerNorm esclusi) |
| Scheduler | warmup lineare sul 6% degli step, poi decadimento lineare a 0 |
| Precisione | fp16 (RBT, DBT), fp32 (FT5) |
| Lunghezza max input | 2048 token in training e inferenza (DeBERTa e T5 hanno posizioni relative) |
| Split | 95% train / 5% validation, shuffle |
| Parallelismo | DDP (PyTorch Lightning), `seed=2024` |

**Infrastruttura e Costi:**
Sorprendentemente, l'addestramento è super-efficiente. Su 4 GPU Nvidia Quadro RTX 8000 il training richiede ~**7 min** per RBT, ~**18 min** per DBT (DeBERTa-v3-large, ~0.4B parametri) e ~**32 min** per FT5 (~0.8B). L'innovazione non è nella computazione bruta, ma nell'ingegneria del dato sintetico.

### Differenze architetturali rispetto a MiniCheck

| | MiniCheck | FactCG |
| --- | --- | --- |
| Backbone | RoBERTa (AlignScore), DeBERTa-v3, Flan-T5 | **identici** |
| Input DeBERTa | coppia `[CLS] chunk [SEP] claim [SEP]` | **template istruzione FLAN** come sequenza singola |
| Testa DeBERTa | `ContextPooler` + Linear(1024→2) nuovi | identica (nuova, random) |
| RoBERTa | encoder + testa binaria AlignScore | identica, esplicitamente: carica `bin_layer`, scarta `tri_layer`/`reg_layer` |
| Flan-T5 | target "1"/"0", loss seq2seq, score = P("▁1") contro P("▁") | template FLAN, loss CE **sui soli logit "No"/"Yes"** |
| Chunk inferenza | 400 token (encoder) / 500 parole (T5) | 400 (RBT) / 550 (DBT, FT5) |
| Aggregazione chunk | max | max |
| Soglia | 0.5 fissa | 0.5 fissa oppure tarata per dataset |

---

## 6. Metriche Avanzate: Il Connected Reasoning (CoRe)

Per dimostrare che FactCG non "tira a indovinare" sfruttando bias del testo, gli autori usano il framework **CoRe (Connected Reasoning)** sul dataset WiCE.

### La Matematica del CoRe

Sia $\mathcal{E}_i$ l'insieme delle frasi di evidenza minime necessarie per verificare un claim. Rimuovendo una di queste frasi essenziali ($e_i^-$) dal documento, il claim diventa inverificabile (creando uno scenario di "ragionamento disconnesso").

Le due metriche misurano:

1. **$Accuracy_{CoRe}$**: La proporzione di casi in cui il modello indovina *entrambi* gli scenari (giudica Vero il documento intero e Falso il documento bucato).


2. **$Precision_{CoRe}$**: Quante delle predizioni "Vere" fatte dal modello poggiano effettivamente su un ragionamento connesso (e non sul tirare a indovinare).



### Risultati CoRe

* **MiniCheck-DBT:** Precision CoRe 37.81%.


* **FactCG-DBT:** Precision CoRe **44.78%**.


* **MiniCheck-RBT:** Precision CoRe 34.32%.


* **FactCG-RBT:** Precision CoRe **54.04%**.



FactCG dimostra una capacità nettamente superiore di ancorare le proprie decisioni all'intera catena di prove, anziché farsi ingannare da artefatti testuali locali.

---

## 7. L'Ingegneria dei Prompt (Dalle Appendici)

Il paper svela i prompt esatti usati per GPT-4o, mostrando come si aggirano le limitazioni strutturali dei LLM.

**Estrazione dei Grafi (Table 13):**
Per evitare che l'LLM perda il filo, viene forzato un formato rigido con delimitatori speciali (`{tuple_delimiter}` e `{group_delimiter}`).

* *Esempio di output imposto:* `Hunt {tuple_delimiter} government {tuple_delimiter} Hunt said something about the government`.



**La Corruzione Mirata (Table 16):**
Per creare i campioni negativi, all'LLM non viene chiesto semplicemente di "rendere il claim falso". Viene fornita la coppia di entità (es. `India, Area`) e il testo, con l'istruzione rigorosa: *"Remove the relation between two provided entities in below provided sentences with minimal changes."*. Questo assicura che il documento rimanga linguisticamente fluido e coerente, rendendo l'individuazione dell'allucinazione estremamente sfidante per il classificatore finale.