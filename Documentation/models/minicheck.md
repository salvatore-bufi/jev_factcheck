# Riassunto  : MiniCheck
**Paper:** *MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents* (Tang, Laban, Durrett)

**Link:** https://aclanthology.org/2024.emnlp-main.499.pdf 

---

## 1. Obiettivo 
**MiniCheck**, una serie di modelli di fact-checking altamente efficienti. A differenza della maggior parte dei lavori precedenti che affrontano il controllo delle allucinazioni in modo empirico, questo paper formalizza il processo di verifica dei fatti come una funzione di discriminazione matematica.

### Definizione Formale del Task
Assumiamo che l'output di un LLM sia una sequenza di frasi (claim) $c = [c_1, ..., c_{|c|}]$. 
Per ogni frase $c_i$, esiste un set di documenti di supporto (grounding documents) $\mathcal{D}_i = \{D_{i,1}, ..., D_{i,|\mathcal{D}_i|}\}$.
L'obiettivo è apprendere un modello discriminatore $M$ tale che:
$$M(D_{i,j}, c_i) \rightarrow \{0,1\}$$
dove $1$ indica che il claim è supportato (supported) e $0$ indica che non è supportato (unsupported) dal documento.
Per giudicare l'intera frase $c_i$ rispetto a tutti i documenti recuperati, il sistema calcola il massimo:
$$\max_j M(D_{i,j}, c_i)$$
Il claim è considerato vero se e solo se esiste *almeno un* documento che lo supporta.

>> **Goal: Creazione di 1 dataset**
 $$ \{(D_i, c_i, y_i)\}_{1}^{N} $$

di $N$ documenti $D_i$ accoppiati a claim $c_i$ e label $y_i$

---

## 2. Generazione Sintetica dei Dati: Il metodo C2D (Claim to Doc)
La sfida principale è che i modelli esistenti non sanno ragionare su **fatti multipli** che richiedono **frasi multiple** per essere validati. Per insegnare questo, gli autori costruiscono il dataset C2D definendo un set di trasformazioni matematiche sui claim.

### Step 1: Decomposizione
Un claim originale $c$ viene scomposto in un set di $l$ fatti atomici tramite la funzione $Decomp$:
$$Decomp(c) = \{a_1, ..., a_l\}$$
> **Esempio (Tijuana Stadium):**
> *Claim ($c$)*: "Oltre 5.000 membri della carovana soggiornavano allo Stadio di Tijuana, una struttura con una capacità di 3.000 persone."
> *Fatto $a_1$*: "Oltre 5.000 membri soggiornavano allo Stadio".
> *Fatto $a_2$*: "Lo Stadio ha una capacità di 3.000 persone."

 

### Step 2: Espansione dei Fatti Atomici
Per ogni fatto atomico $a_i$, un LLM (GPT-4) genera una *coppia* di frasi che devono essere lette congiuntamente per implicare il fatto:
$$SentPair(a_i) = (s_{i,1}, s_{i,2}) \quad \forall i \in \{1, ..., l\}$$
> **Esempio su $a_2$:**
> *$s_{2,1}$*: "Lo stadio di Tijuana ha recentemente subito lavori di ristrutturazione che hanno raddoppiato la sua capacità."
> *$s_{2,2}$*: "Prima di queste modifiche, lo stadio poteva ospitare 1.500 spettatori."
> *(Solo combinando $1.500 \times 2$ si ottiene la capacità di 3.000 indicata in $a_2$)*.


> **Recap:**
> * ogni  claim diviso in fatti atomici $a_i$.
> * ogni fatto $a_i$ diviso in due frasi $(s_{i,1}, s_{i,2})$


### Step 3 e 4: Generazione Documento Positivo e Ablazione (Hard-Negatives)
Tutte le frasi generate vengono unite in un pool $s = \{s_{1,1}, s_{1,2}, ..., s_{l,1}, s_{l,2}\}$. 
> Quindi $s$ rappresenta il claim !

Viene creato il documento, chiede a LLM (GPT4) di creare un documento che menziona tutte le frasi in $s$:
$$D = PassageGen(s)$$
Tramite questi passaggi si genera quindi la tupla del dataset di train: **$(D, c, y=1)$**.

> Per costruzione, un fatto atomico $a_i$ nel claim $c$ è supportato dalla coppia $s_{i,1}, s_{i,2}$.
Rimuovendo quindi una frase $s_{j,k}$ -> $a_j$ non è più supportato -> $c$ non più supportato.

Per creare un **documento negativo sfidante**, si rimuove selettivamente una singola frase $s_{i,j}$ dal pool $s$:
$$D'_{a_{i\backslash j}} = PassageGen(s \backslash s_{i,j})$$
> **Esempio di Ablazione:**
> Rimuovendo $s_{2,2}$ (la frase sui 1.500 spettatori originali), il nuovo documento $D'$ menziona i lavori di raddoppio ma *non dice più* quale fosse la base di partenza.
> Pertanto, $a_2$ (capacità 3.000) non è più matematicamente deducibile.
> Tupla generata: **$(D'_{a_{2\backslash 2}}, c, 0)$**.

### Step 5: Augmentation (Power Set)
L'obiettivo di questo step è non sprecare questo documento bucato utilizzandolo solo per un banale "Falso" sull'intero claim originale, ma creare nuove tuple di addestramento sfruttando tutte le combinazioni logiche possibili dei fatti.

Vengono calcolati tutti i sottoinsiemi possibili dei fatti atomici: $Power(a)$.

l'insieme di tutti i suoi possibili sottoinsiemi. Nel nostro caso, partiamo dai fatti atomici in cui è stato diviso il claim originale: $a = \{a_1, a_2, ..., a_l\}$.

 Per ogni combinazione $a'$, un LLM genera un nuovo "sotto-claim" $c'$ unendo assieme in modo fluido solo i fatti di quella specifica combinazione ($c' = Merge(a')$)


 Nello step precedente (Ablazione), avevamo creato dei documenti negativi ($D'$). Un documento $D'$ è stato "bucato" intenzionalmente rimuovendo una frase chiave, in modo tale che non supporti più un fatto specifico (chiamiamolo $a_2$), pur continuando a supportare perfettamente tutti gli altri fatti ($a_1$, $a_3$, ecc.).

 L'augmentation consiste nel prendere questo documento bucato $D'$ e testarlo contro tutti i sotto-claim appena generati tramite il Power Set.La regola per le etichette ($y \in \{0,1\}$) diventa:
 -  Etichetta 0 (Unsupported): Se il sotto-claim $c'$ contiene il fatto $a_1$ (quello per cui il documento è stato bucato), l'intero sotto-claim crolla. Il documento non può supportarlo.
 - Etichetta 1 (Supported): Se il sotto-claim $c'$ non contiene il fatto $a_1$, allora il documento può supportarlo! Questo perché il documento contiene ancora le prove per tutti gli altri fatti.

La regola per le etichette diventa rigorosa:
* Se $a_i \notin a'$: La tupla è $(D'_{a_{i\backslash j}}, c', 1)$ (il fatto rimosso non serve a questo specifico sub-claim).
* Se $a_i \in a'$: La tupla è $(D'_{a_{i\backslash j}}, c', 0)$.


>>> Esempio
- Sottoinsieme 1 ($\{a_1\}$):$c'_1$ = "Oltre 5.000 membri soggiornavano allo Stadio."
- Sottoinsieme 2 ($\{a_2\}$):$c'_2$ = "Lo Stadio ha una capacità di 3.000 persone.
- "Sottoinsieme 3 ($\{a_1, a_2\}$):$c'_3$ = "Oltre 5.000 membri soggiornavano allo Stadio, una struttura con una capacità di 3.000 persone." (Questo coincide con il claim originale)

Incrociamo il nostro documento bucato ($D'_{a_{2\backslash 2}}$) con ciascuno di questi tre nuovi claim ($c'$).

Guardiamo il fatto (in questo caso $a_2$) e contorlliamo se è presente nel nuovo sub-claim

| Tupla di Addestramento Generata | Condizione Matematica | Spiegazione Logica | Etichetta Finale ($y$) |
| --- | --- | --- | --- |
| $(D'_{a_{2\backslash 2}}, c'_1, y)$ | $a_2 \notin a'$ | Il claim $c'_1$ parla solo delle 5.000 persone. Il documento bucato ha le frasi intatte per dimostrarlo. Il fatto sabotato ($a_2$) non serve.| $1$ (Supported) 
| $(D'_{a_{2\backslash 2}}, c'_2, y)$ | $a_2 \in a'$ | Il claim $c'_2$ parla solo della capacità di 3.000 persone. Il documento è stato bucato su quell'informazione (manca il 1.500).| $0$ (Unsupported)
| $(D'_{a_{2\backslash 2}}, c'_3, y)$ | $a_2 \in a'$ | Il claim $c'_3$ richiede entrambi i fatti. Poiché $a_2$ non è dimostrabile, l'intera frase crolla.| $0$ (Unsupported)

---

## 3. Generazione Sintetica dei Dati: Il metodo D2C (Doc to Claim)
Per evitare che il modello impari solo su testi "sintetici" (causando distribution shift), il metodo D2C parte da un documento reale $D_{real}$.

### Chunking e Sommario
Il documento viene diviso in 3 chunk: $\{D_1, D_2, D_3\}$. 
Viene generato un sommario per ogni chunk: $c = \{c_1, c_2, c_3\}$. Assumiamo $(D_i, c_i, 1)$.

Esattamente come avveniva nel C2D, il riassunto $c_i$ viene trattato come un claim da analizzare:  
- Decomposizione: $c_i$ viene scomposto nei suoi fatti atomici $a_i = \{a_{i,1}, ..., a_{i,l}\}$.  
- Power Set: Si calcolano tutti i sottoinsiemi possibili di questi fatti, chiamati $a'_i$ (es. solo il primo fatto, solo il secondo, entrambi, ecc.).  
- Merge: Si generano nuovi sub-claim  combinando i fatti di ogni sottoinsieme: $Aug(c_i) = \{Merge(a'_i)\}$. 


### Ablazione del Documento
Qui si mettono alla prova i sub-claim generati, alterando il testo sorgente.

Un chunk originale $D_i$ non è altro che una concatenazione di $n$ frasi reali: $s = \{s_{i,1}, ..., s_{i,n}\}$.  

Data una frase $s_{i,j}$ dentro il chunk $D_i$, creiamo un chunk ablato (bucato),  rimuovendo iterativamente una frase $s_{i,j}$:
$$D'_{i \backslash j} = Concat(s \backslash \{s_{i,j}\})$$
Si valuta se ogni fatto atomico $a_{i,k}$ di $c_i$ è ancora supportato:
$$L^{-j}(a_{i,k}) = Ent(D'_{i \backslash j}, a_{i,k}) \in \{0,1\}$$

Se TUTTI i fatti atomici contenuti nel sub-claim $a'_i$ risultano ancora supportati dal documento bucato (cioè $L = 1$ per tutti), si crea la tupla positiva: $(D'_{i \backslash j}, Merge(a'_i), 1)$. 
> **Esempio Pratico (Box Office e Scioperi):**
> *Chunk $D_2$ originale*: "...Kraven the Hunter e Ghostbusters costeranno quest'anno 400 milioni... le speranze per il 2024 sono distrutte".
> *Claim $c_2$*: "I rinvii costeranno potenzialmente 500 milioni."
> Se rimuoviamo la frase che cita l'impatto economico specifico, la funzione $L^{-j}$ sul fatto atomico "costeranno 500 milioni" restituisce $0$.

### Cross-Document Augmentation
Si verifica se i fatti del sommario $c_i$ possono essere supportati da *un altro chunk* $D_j$ (con $j \neq i$):
$$L^{D_j}(a_{i,k}) = Ent(D_j, a_{i,k})$$
Questo crea associazioni non banali tra parti diverse dello stesso articolo originale.

---

## 4. Architettura, Formato dell'Output e Training

> **Fonti:** paper (Sez. 5 e App. G), codice di inferenza ufficiale ([`minicheck/inference.py`](https://github.com/Liyan06/MiniCheck/blob/main/minicheck/inference.py)), config dei checkpoint su HuggingFace (`lytang/MiniCheck-*`). Il **codice di training di MiniCheck non è stato rilasciato** (il repo contiene solo inferenza e generazione dati): quello che segue sul training viene dal paper, quello sull'architettura dal codice di inferenza e dai checkpoint. Il codice è dichiaratamente *"adapted from AlignScore"*.

### 4.1 Idea generale: nessuna architettura nuova
MiniCheck **non introduce un'architettura nuova**. Prende tre backbone esistenti e li fa diventare **classificatori binari** $(D, c) \rightarrow P(\text{supported})$ con un fine-tuning completo: tutti i pesi vengono aggiornati, nessun layer è congelato, nessun adapter o LoRA. Il contributo è tutto nei dati (C2D + D2C). Per ottenere il classificatore si usano due strategie diverse:

| Modello | Backbone | Parametri | Tipo | Come produce lo score | Testa di classificazione |
| --- | --- | --- | --- | --- | --- |
| **MiniCheck-RBTA** | RoBERTa-large **già addestrato da AlignScore** | ~355M | Encoder-only | `RobertaForSequenceClassification`, softmax su 2 logit | Testa binaria (2 classi) sopra il token `<s>` |
| **MiniCheck-DBTA** | `microsoft/deberta-v3-large` (vanilla) | ~435M (304M backbone + 131M embedding, vocab 128K) | Encoder-only | `DebertaV2ForSequenceClassification`, softmax su 2 logit | **Testa nuova** inizializzata a caso |
| **MiniCheck-FT5** | `google/flan-t5-large` | ~780M | Encoder-decoder | Riusa la **LM head** del decoder: probabilità del token "1" contro "0" | **Nessuna testa nuova** |

### 4.2 MiniCheck-RBTA (RoBERTa da AlignScore)
**Punto di partenza: AlignScore.** AlignScore (Zha et al., 2023) è un RoBERTa-large già addestrato su ~4.7M esempi di "allineamento" (NLI, QA, parafrasi, fact verification, summarization). Nel codice di AlignScore (`BERTAlignModel`) il modello ha **un encoder condiviso e tre teste lineari** sopra il `pooler_output`:
```
RobertaModel ──► pooler_output = tanh(W_p · h_[<s>])  (1024-d)
                    ├── bin_layer : Linear(1024 → 2)   # aligned / not-aligned
                    ├── tri_layer : Linear(1024 → 3)   # entail / neutral / contradict
                    └── reg_layer : Linear(1024 → 1)   # score continuo (regressione)
```
**Cosa fa MiniCheck:** *"We fine-tune the tuned roberta-large model from AlignScore with a binary classification head"*. Tiene l'encoder (con il pooler) già specializzato sull'allineamento e lo usa con **una sola testa binaria**. Le teste 3-way e di regressione **non vengono usate**. Il checkpoint rilasciato è un `RobertaForSequenceClassification` standard, con `id2label = {0: "0", 1: "1"}`:
```
<s> chunk </s> claim </s>
        │
RoBERTa-large (24 layer, hidden 1024, 16 head)  ← pesi AlignScore, poi fine-tuning
        │  h_[<s>]
dense(1024→1024) + tanh  →  dropout  →  out_proj(1024→2)  →  softmax  →  P(y=1)
```
> **Testa riusata o nuova?** Il paper di MiniCheck dice solo *"with a binary classification head"*. FactCG, che segue la stessa ricetta, lo dice esplicitamente nel codice: carica encoder **e** `bin_layer` di AlignScore e butta via `tri_layer` e `reg_layer` (vedi FactCG.md). È quasi certamente quello che fa anche MiniCheck, ma il suo codice di training non è pubblico, quindi non si può verificare.

**Nota sul training:** RBTA viene addestrato **solo sui 14K dati sintetici, senza ANLI**. AlignScore aveva già visto ANLI nel suo pre-training.

### 4.3 MiniCheck-DBTA (DeBERTa-v3-large)
Qui il backbone è il DeBERTa-v3-large **vanilla**, che non ha nessuna testa di classificazione pre-addestrata. Si usa la classe standard di HuggingFace `DebertaV2ForSequenceClassification` con `num_labels=2`, quindi la testa è **nuova e inizializzata a caso**:
```
[CLS] chunk [SEP] claim [SEP]
        │
DeBERTa-v3-large (24 layer, hidden 1024, attention disentangled con posizioni relative)
        │  h_[CLS]
ContextPooler: dropout → dense(1024→1024) → GELU   ← nuovo, init random
        │
classifier: Linear(1024→2)                          ← nuovo, init random
        │
softmax → P(y=1)
```
Encoder e testa vengono addestrati insieme, end-to-end, con cross-entropy. Grazie alle posizioni relative di DeBERTa, la lunghezza massima in inferenza è **2048 token**, contro i 512 di RoBERTa che ha posizioni assolute (`max_position_embeddings=514`).

### 4.4 MiniCheck-FT5 (Flan-T5-large): classificazione via LM head
Flan-T5 è un modello **generativo encoder-decoder** e qui **non gli si aggiunge nessuna testa**. Il task diventa *text-to-text*:
- **Target di training:** la stringa `"1"` (supportato) oppure `"0"` (non supportato). È lo stesso schema del baseline T5-NLI-Mixed descritto nel paper. La loss è la cross-entropy seq2seq standard sui token target.
- **Input:** `"predict: " + chunk + "</s>" + claim`, troncato a 2048 token. T5 usa bias posizionali relativi, quindi va oltre i 512.
- **Inferenza (un solo passo di decoding):** al decoder si dà solo il token di start (`decoder_input_ids = [0]`) e si leggono i logit del **primo** token generato, limitati a due ID del vocabolario:
  ```python
  label_logits = logits[:, [3, 209]]   # 3 = "▁" (no support), 209 = "▁1" (support)
  probs = softmax(label_logits)        # P(supported) = probs[:, 1]
  ```
  Perché proprio `3`? Il tokenizer T5 spezza `"0"` in `["▁", "0"]` (ID `[3, 632]`), mentre `"1"` è un unico token `"▁1"` (ID `209`). Il primo token della risposta "0" è quindi lo spazio `"▁"` (ID 3). Si è verificato direttamente con il tokenizer di `google/flan-t5-large`.

In pratica la LM head (vocab 32K) **funziona già da classificatore**: dei suoi 32K logit se ne tengono solo 2 e si fa una softmax su quei due.

### 4.5 Pipeline di inferenza (comune ai tre modelli)
Dal codice ufficiale:
1. **Sentence split** del documento (NLTK `sent_tokenize`, mantenendo i `\n`).
2. **Chunking** sui confini di frase: si aggiungono frasi finché il chunk non supera circa **400 token** del tokenizer (RBTA/DBTA) o **500 parole** separate da spazi (FT5).
3. Ogni chunk $D^{(k)}$ viene accoppiato allo **stesso claim** $c$ e passato al modello: $p_k = P(y=1 \mid D^{(k)}, c)$.
4. **Aggregazione max** sui chunk: $\text{score}(D,c) = \max_k p_k$. Basta che un chunk supporti il claim.
5. **Soglia fissa:** $\hat{y} = \mathbb{1}[\text{score} > 0.5]$.

Il claim **non** viene spezzato in frasi dal modello. Gli autori raccomandano di farlo prima, perché MiniCheck è un checker *a livello di frase*, e lasciano all'utente la scelta di come aggregare le frasi.

### 4.6 Iperparametri di training (App. G)
| | Init | Fase 1 (dati) | Fase 1 (hp) | Fase 2 (dati) | Fase 2 (hp) |
| --- | --- | --- | --- | --- | --- |
| **RBTA** | AlignScore-large | 7K C2D | 2 epoche, batch 4 × accum 2, lr **1e-5** | 7K D2C | 1 epoca, batch 4, lr 1e-5 |
| **DBTA** | deberta-v3-large | 21K ANLI + 7K C2D | 2 epoche, batch 4 × accum 2, lr **5e-5** | 7K D2C | 1 epoca, batch 4, lr 1e-5 |
| **FT5** | flan-t5-large | 21K ANLI + 7K C2D | 2 epoche, batch 4 × accum 2, lr **5e-5** | 7K D2C | 1 epoca, batch 4, lr 1e-5 |

- Loss: cross-entropy standard per tutti. Gli altri iperparametri sono quelli di default dei backbone.
- Le etichette ANLI *contradiction* e *neutral* vengono mappate entrambe su **unsupported (0)**.
- Il subset ANLI di 21K contiene gli esempi su cui i loro modelli di entailment già addestrati **sbagliavano** (hard examples).
- Hardware: una singola NVIDIA RTX A6000.

### 4.7 Estensione successiva: Bespoke-MiniCheck-7B (non nel paper)
Dopo il paper è stato rilasciato `bespokelabs/Bespoke-MiniCheck-7B`: un LLM da 7B (fine-tuning di InternLM2.5-7B-chat) addestrato sugli stessi dati e servito con vLLM. Anche qui non c'è una testa di classificazione:
- un prompt di sistema chiede di rispondere `"Yes"` o `"No"`; si genera **1 solo token** e $P(\text{supported})$ è la somma delle probabilità dei token che decodificano in "yes" (top-5 logprob);
- i chunk sono molto più grandi (~32K token);
- il claim viene spezzato in frasi e $\text{score} = \min_{\text{frase}} \max_{\text{chunk}} p$, cioè ogni frase deve essere supportata da almeno un chunk.

### 4.8 Soglia di decisione
Il paper utilizza una soglia $t$ per convertire in binario:
$$Prediction = \begin{cases} 1 & \text{se } M(D_i, c_i) > t \\ 0 & \text{altrimenti} \end{cases}$$
Normalmente $t = 0.5$. Il paper dimostra che i modelli addestrati sul dataset sintetico mantengono performance altissime **senza bisogno di ricalibrare la soglia $t$ per ogni diverso dataset** (nessun threshold tuning).

### Curriculum Learning (2-Stage Fine-Tuning)

Il dataset di addestramento finale è composto da circa 35K esempi: 14K esempi sintetici (C2D e D2C) uniti a un sottoinsieme di 21K esempi "difficili" prelevati dal dataset ANLI .

**Modelli Base (Backbone):**
Sono stati addestrati tre modelli:
*   **MiniCheck-RBTA:** Basato su RoBERTa-large pre-addestrato da AlignScore (355M parametri). Addestrato **solo** sui 14K sintetici, senza ANLI (vedi 4.2).
*   **MiniCheck-DBTA:** Basato su DeBERTa-v3-large (~435M parametri, di cui ~131M sono embedding).
*   **MiniCheck-FT5:** Basato su Flan-T5-large (770M parametri).

**Pipeline di Fine-Tuning (Prevenzione dell'Overfitting):**
Per evitare che i modelli si adattassero troppo allo stile di scrittura artificiale dei documenti C2D, gli autori hanno strutturato il training in due fasi :
1.  **Fase 1:** Fine-tuning per 2 epoche su 28K esempi (i 21K di ANLI + i 7K di C2D) per insegnare la logica formale di supporto/confutazione. RBTA usa solo i 7K di C2D.
2.  **Fase 2:** Ulteriore fine-tuning per 1 sola epoca usando esclusivamente i 7K esempi del D2C (basati su testi giornalistici reali). Questo ha permesso al modello di riadattarsi a una distribuzione linguistica realistica .




---

## 5. Risultati Quantitativi e Benchmark LLM-AGGREFACT

Il benchmark **LLM-AGGREFACT** aggrega 10 dataset. La valutazione ufficiale si basa sulla **Balanced Accuracy (BAcc)** per mitigare il fatto che i $TP$ e $TN$ non sono equamente distribuiti:
$$BAcc = \frac{1}{2}\left(\frac{TP}{TP+FN} + \frac{TN}{TN+FP}\right)$$

### Prestazioni e Costi (Senza Threshold Tuning)


MiniCheck-FT5 raggiunge un'accuratezza statisticamente indistinguibile da Claude-3 Opus e vicinissima a GPT-4, ma il costo scende di $107 / 0.24 \approx 445$ volte.

---

## 6. Sfatare i Miti: Decomposizione e Decontestualizzazione

### Claim Decomposition
La letteratura precedente (es. *Factcheck-GPT*) assumeva che $M(D_i, c_i)$ fosse debole, e suggeriva di calcolare sempre $\min_k M(D_i, a_k)$ sui fatti atomici. 
Il paper dimostra matematicamente che:
$$BAcc\left( M_{GPT-4}(D_i, c_i) \right) = 75.3\%$$
$$BAcc\left( \min_k M_{GPT-4}(D_i, a_k) \right) = 75.6\%$$
Il delta è solo del +0.3%, ma il numero di chiamate al modello aumenta di $l$ (dove $l$ è il numero medio di fatti per claim, solitamente da 2 a 4). Il costo sale a \$212. Pertanto, i modelli moderni ben addestrati non necessitano di decomposizione.

### Claim Decontextualization
Similmente, l'espansione dei pronomi o dei riferimenti per rendere ogni frase $c_i$ dipendente solo da sé stessa (Decontextualization) ha variato le performance di MiniCheck-FT5 solo dal 74.7% al 74.1%. Il meccanismo di attention dei modelli Transformer moderni (specie in un setup retrieval in cui $D_i$ e $c_i$ condividono implicitamente lo stesso contesto) gestisce efficacemente i riferimenti anaforici e cataforici senza preprocessing esplicito.
"""

