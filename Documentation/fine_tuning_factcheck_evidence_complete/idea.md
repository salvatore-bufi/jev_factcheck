L'architettura proposta nei tuoi documenti (`00_overview.md` e `00a_scelta_approccio.md`) è già ingegnerizzata con estrema lucidità: invertire l'ordine in **Claim $\rightarrow$ Documento** per aggirare il limite dell'attenzione causale, estrarre gli stati nascosti $h_k$ all'ultimo token di ogni frase senza alterare il testo con marcatori artificiali, e risolvere l'evidenza sparsa (multi-hop) con un **aggregatore bidirezionale leggero** alimentato anche dallo stato finale $q$ (`Answer:`), mantenendo l'overhead entro il +3% di un singolo prefill.

Tuttavia, nella letteratura recente su * mechanistic interpretability*, *dense retrieval* (late-interaction), *rationale extraction* e architetture *decoder-only*, esistono dinamiche interne ai Transformer che possono mettere in crisi questo setup o che, se sfruttate, possono alzare drasticamente l'F1 sull'evidenza (specialmente su dataset multi-hop come HoVer o SciFact) a costo computazionale quasi nullo.

Di seguito trovi **6 direttrici architetturali e di training** che estendono il tuo piano, complete di motivazione teorica, formulazione e diagrammi.

---

## 1. Sfruttare la gerarchia dei layer: Multi-Layer Feature Fusion e Q-K Attention Routing

### Il problema nel piano attuale

Nel tuo schema attuale, estrai $h_1 \dots h_n$ e $q$ dall'**ultimo layer** del LLM (subito prima della testa del linguaggio / verdetto).
Nei modelli decoder-only (Llama, Mistral, Qwen, Winnow), gli **ultimi 2-3 layer sono fortemente specializzati per la predizione del token successivo**: comprimono tutta l'informazione contestuale nel token finale (`Answer:`) per emettere il logit della lettera del verdetto, mentre le rappresentazioni dei token intermedi ($h_k$) tendono a "collassare" o a perdere la specificità semantica locale della frase a favore di feature globali o cosiddetti *attention sinks*.

Inoltre, durante il forward pass del backbone, il modello **ha già calcolato** le interazioni tra `Answer:` e il resto del documento tramite le matrici di attenzione dei vari layer: il tuo aggregatore bidirezionale da 256-d deve re-imparare da zero queste relazioni partendo solo dai vettori compressi finali.

### La soluzione da esplorare

1. **Estrazione da Layer Intermedi (Scalar Mix di tipo ELMo/Probing):** Invece di leggere $h_k$ dall'ultimo layer $L$, estrai gli stati da un sottoinsieme di layer medio-alti (es. $L-8, L-4, L-2, L$) dove la semantica della frase e il *bridging* multi-hop avvengono realmente, combinandoli con pesi scalari apprendibili $\alpha_l$.
2. **Iniezione delle feature di Q-K Attention native del Backbone:** Durante il singolo forward pass, il token finale `Answer:` (posizione $T$) interroga già tutti i token precedenti tramite le teste di attenzione ($H$ teste per $L$ layer). Per ogni frase $S_k$, puoi fare il *mean/max-pooling* dei pesi di attenzione che `Answer:` dedica ai token di $S_k$ negli ultimi $M$ layer. Otterrai un vettore gratuito di dimensione $M \times H$ per ogni frase, che rappresenta **dove il modello sta effettivamente guardando per decidere il verdetto**.

```
Prompt: [Claim] [S1 ... Sk ... Sn] [Answer:]
                   │                │
       ┌───────────┴────────────────┴───────────┐
       │      Backbone LLM (Singolo Forward)    │
       ├────────────────────────────────────────┤
       │ Layer L-6:  h_k^(L-6)                  │
       │ Layer L-3:  h_k^(L-3)                  │
       │ Layer L  :  h_k^(L)   +  Attn(q -> Sk) │
       └─────────────────┬──────────────┬───────┘
                         │              │ (Feature di routing native M*H)
        Somma pesata α_l │              ▼
                         ▼        [Proj_attn]
                   h_k_fused ────────► (+) ◄─── [Positional Emb]
                                        │
                                        ▼
                       [Aggregatore Bidirezionale 2-Layer]
                                        │
                                        ▼
                                 P(evidenza_k)

```

### Perché funziona e perché vale la pena

* **Allineamento intrinseco (Fedeltà):** Le *Induction Heads* e le *Retrieval Heads* nei layer medio-alti fanno già il lavoro sporco di collegare "Nancy R. Howell" (S40) a "Saint Paul School of Theology" (S1). Passare quelle attivazioni d'attenzione direttamente nell'input dell'aggregatore fornisce una scorciatoia diretta, accelerando la convergenza e impedendo che la testa di evidenza impari euristiche spurie slegate dal verdetto.


* **Costo:** Zero FLOP extra nel backbone; richiede solo di salvare un tensore ridotto durante il forward o usare un *forward hook* sui layer selezionati.

---

## 2. Dall'ultimo token allo "Span Pooling" intra-frase (Late-Interaction su Singolo Token vs Multi-Token)

### Il problema nel piano attuale

Rappresentare un'intera frase $S_k$ (che può arrivare fino a 60 parole prima di essere spezzata) esclusivamente con lo stato nascosto dell'**ultimo token** $h_k$ espone a due rischi critici:

1. **Collo di bottiglia del punto fermo:** L'ultimo token di una frase è quasi sempre un segno di punteggiatura (`.` o `;`). Nei LLM causali, i segni di punteggiatura agiscono spesso come *attention sinks* o punti di reset sintattico; lo stato nascosto sul punto fermo non è necessariamente un riassunto fedele di un'entità nominata all'inizio della frase.
2. **Diluizione nelle frasi lunghe:** Se una frase di 45 parole contiene l'evidenza cruciale nelle prime 5 parole e poi divaga in una subordinata irrilevante, lo stato sull'ultimo token sarà dominato dalla subordinata finale.

### La soluzione da esplorare: Attentive Span Pooling guidato da $q$ e dal Claim

Invece di selezionare solo l'indice dell'ultimo token di $S_k$, usa gli offset del tokenizer per prendere tutti i token $\{t_{k,1}, \dots, t_{k,m}\}$ appartenenti alla frase $k$, proiettali subito a dimensione ridotta ($d \rightarrow 256$) e applica un **Cross-Attention Pooling ultra-leggero** guidato dallo stato del claim $c$ e dallo stato finale $q$:

```
Token della frase Sk:   [t_k,1]   [t_k,2]  ...  [t_k,m] (es. ".")
                           │         │             │
Proiezione rapida (d->256) ▼         ▼             ▼
                        [v_k,1]   [v_k,2]  ...  [v_k,m]
                           │         │             │
                           └─────────┼─────────────┘
                                     ▼
Query di pooling:  u = W_q * [c ; q] ──► Softmax( u^T * v_k,i )
                                     │
                                     ▼
             Rappresentazione frase: x_k = Σ α_i * v_k,i  ⊕  v_k,m

```

### Perché funziona e perché vale la pena

* **Ispirato a ColBERT / Late-Interaction:** Permette di "pescare" esattamente il nome proprio, la data o il predicato dentro la frase $S_k$ che risponde a $q$ o al claim $c$, prima ancora che l'aggregatore bidirezionale fra le frasi entri in gioco.
* Concatenando il pooling pesato $\sum \alpha_i v_{k,i}$ con l'ultimo token $v_{k,m}$ ottieni sia il riassunto sequenziale di fine frase sia il picco di attivazione locale sull'entità chiave.
* **Costo:** Poiché la proiezione $d \rightarrow 256$ è una singola moltiplicazione matriciale eseguita una volta sola su tutti i token del documento ($N_{tok} \times d \times 256$), per 2.000 token e $d=4096$ parliamo di $\sim 4 \text{ GFLOP}$ totali (meno di $0.1 \text{ ms}$ su L40S).

---

## 3. Modellare la "Complanarità Multi-Hop": Matrice di Affinità a Coppie (Biaffine / Low-Rank Pairwise Head)

### Il problema nel piano attuale

Nel caso dell'evidenza sparsa:

* $S_1$: *"Saint Paul School of Theology is a seminary in Kansas City."*


* $S_{40}$: *"Nancy R. Howell is Professor at Saint Paul School of Theology."*



Con una testa finale puramente puntuale $P(\text{evidenza}_k) = \sigma(v \cdot X_k + b)$ e una Binary Cross-Entropy (BCE) indipendente per frase, il modello calcola la probabilità marginale di $S_1$ e di $S_{40}$ separatamente. Anche se l'aggregatore bidirezionale permette a $X_1$ e $X_{40}$ di comunicare, l'obiettivo di classificazione finale non premia esplicitamente il fatto che **$S_1$ è evidenza SOLO SE accompagnata da $S_{40}$** (e se nel documento ci fosse anche una frase $S_{15}$ quasi identica a $S_1$ ma riferita a un'altra università, una BCE indipendente rischierebbe di dare 0.55 a entrambe o sbagliare la coppia).

### La soluzione da esplorare: Scoring Congiunto + Edge Supervision

Aggiungi all'uscita dell'aggregatore bidirezionale un termine di **interazione bilineare a basso rango (Biaffine)** tra coppie di frasi, supervisionato sui dataset multi-hop (es. HoVer):

$$s_k^{\text{unary}} = v^T X_k + b$$

$$S_{k, j}^{\text{pair}} = (U_1 X_k)^T (U_2 X_j) \quad \text{con } U_1, U_2 \in \mathbb{R}^{64 \times 256}$$

Lo score finale della frase $k$ diventa la combinazione del suo contributo diretto e del supporto massimo (o *log-sum-exp*) che riceve da una frase "ponte" $j$ nel documento:

$$\text{logit}_k = s_k^{\text{unary}} + \tau \cdot \max_{j \neq k} \left( S_{k, j}^{\text{pair}} + \text{stop\_grad}(s_j^{\text{unary}}) \right)$$

```
 Uscita Aggregatore:    X_1        X_2   ...   X_40
                         │          │           │
           ┌─────────────┼──────────┼───────────┼─────────────┐
           ▼             │          │           │             ▼
    [Score Unario s_k]   │          ▼           │     [Matrice Coppie n x n]
     (Evidenza Singola)  └───► (U1*X_k)^T (U2*X_j) ◄──┘ (Evidenza Multi-Hop)
           │                                                  │
           └──────────────────────► (+) ◄─────────────────────┘
                                     │
                                     ▼
                           Logit Finale Frase k

```

### Perché funziona e perché vale la pena

* **Risoluzione esplicita dei ponti (Bridging):** Su HoVer (2-hop, 3-hop), se due frasi sono entrambe nel gold set dell'evidenza, durante il training puoi applicare una loss ausiliaria direttamente su $S_{k,j}^{\text{pair}} = 1$ per tutte le coppie $(k, j) \in \text{Gold} \times \text{Gold}$.


* Questo costringe l'aggregatore non solo a riconoscere le frasi rilevanti, ma a costruire uno **spazio metrico di co-occorrenza logica**, abbattendo il calo di F1 quando la distanza tra $S_1$ e $S_{40}$ supera le 20 frasi (uno dei tuoi KPI chiave).



---

## 4. Allineamento Causale: Contrastive Masking & Self-Consistency in Fase di Training

### Il problema nel piano attuale

Una delle criticità storiche dei modelli Multi-Task (Verdetto + Estrazione di Razionali/Evidenze) è il **disaccoppiamento tra spiegazione e decisione (Unfaithfulness)**:

* La testa del verdetto su `q` impara a dire `Supported = True` usando un proprio circuito interno.
* La testa di evidenza su $h_1 \dots h_n$ impara a imitare le annotazioni umane di FEVER/SciFact.


* Tuttavia, nulla garantisce che le frasi selezionate dalla testa di evidenza siano **effettivamente sufficienti e necessarie** al decision model per emettere quel verdetto (il tuo KPI di *Sufficienza $\ge 85\%$*).



### La soluzione da esplorare: Distillazione di Fedeltà / Attention-Masking Consistente (Solo in Training)

Anche se in **produzione** farai rigorosamente 1 solo forward pass, durante il **training** (o su un 20% dei batch di training) puoi sfruttare un secondo forward "leggero" (o una manipolazione della maschera di attenzione) per legare matematicamente le due teste:

1. **Forward 1 (Standard):** Passi il documento intero, calcoli i logit del verdetto $z_{\text{full}}$ e le probabilità di evidenza $p_1 \dots p_n$ per ogni frase.
2. **Costruzione della Soft-Mask (o Gumbel-Softmax Mask):** Crei un bias di attenzione per il token finale `Answer:` (o per gli ultimi 2 layer del LLM) tale per cui i token appartenenti alla frase $k$ vengono attenuati in base a $\log(p_k + \epsilon)$ (se la testa di evidenza dice che $p_k \approx 0$, il token finale `Answer:` viene "accecato" su quella frase).
3. **Vincolo di Sufficienza e Completezza (KL Divergence):**
* **Sufficienza:** Se `Answer:` guarda *solo* le frasi con $p_k$ alto, deve comunque produrre lo stesso logit $z_{\text{full}}$ (Supported).
* **Completezza (Comprehensive Drop):** Se `Answer:` viene accecato sulle frasi con $p_k$ alto (mascherando $1 - p_k$), la probabilità di `Supported` deve crollare verso `Not Supported` (o verso l'uniformità).



```
[Forward di Training]
       │
       ├──► Testa Evidenza ──► Probabilità frasi: p_1, p_2, ..., p_n
       │                                               │
       │    ┌──────────────────────────────────────────┘
       │    ▼ (Ricalcolo economico solo dell'ultimo layer su "Answer:")
       │  Maschera Attenzione A_new = A_orig + log(p_k)   ──► Logit_Sufficiente
       │  Maschera Inversa    A_inv = A_orig + log(1-p_k) ──► Logit_Svuotato
       │
       └──► Loss_Fedeltà = KL(Logit_Full || Logit_Sufficiente) + max(0, P(Supp|A_inv) - ম)

```

### Perché funziona e perché vale la pena

* **Costo di training quasi nullo se fatto solo sull'ultimo layer:** Non serve rifare il forward di tutto il modello da 12B! Basta ricalcolare **solo l'ultimo layer Transformer** (1/40 del costo) per il singolo token `Answer:` applicando la maschera di attenzione generata da $p_k$ sulle Key/Value già in cache.
* **Costo in inferenza:** **ZERO.** In produzione non esegui questo ramo, ma la testa di evidenza avrà imparato a selezionare esattamente le frasi che mantengono in piedi il logit del verdetto nel tuo specifico modello, portando la metrica di *Sufficienza* vicino al 95%+.



---

## 5. Architettura del Prompt: "Sandwich Prompting" (Claim-Doc-Claim)

### Il problema nel piano attuale

Hai giustamente deciso di mettere il **Claim prima del documento** (`[Claim] [Doc] [Answer:]`) perché altrimenti gli stati $h_k$ delle frasi non vedrebbero il claim a causa della maschera causale.
Tuttavia, quando il documento è lungo (es. 2.000 - 8.000 token):

1. Per le frasi finali ($S_{80} \dots S_{100}$) e per il token `Answer:`, il Claim si trova a **migliaia di token di distanza**. A causa del decadimento posizionale (RoPE decay) e della diluizione dell'attenzione sui contesti lunghi, la rappresentazione esatta dei dettagli del claim (es. negazioni sottili, date, numeri) si indebolisce.
2. Inoltre, come noti nel file `00a`, i modelli come Winnow e JPT sono nativamente addestrati con il **Claim dopo il documento**: invertire brutalmente l'ordine può degradare la qualità zero-shot/iniziale del verdetto.



### La soluzione da esplorare: Il formato "Sandwich" (`[Claim] [Documento] [Claim] [Answer:]`)

Invece di scegliere tra `Claim -> Doc` e `Doc -> Claim`, ripeti il claim sia prima che dopo il documento:

```text
[Istruzioni]
Claim to verify: {CLAIM}
Document:
S1 ... Sn
Verify the following claim based on the document above: {CLAIM}
Answer:

```

### Perché funziona e perché vale la pena

* **Il meglio dei due mondi in un singolo forward:**
* Ogni frase $S_k$ del documento vede il primo `{CLAIM}` che la precede: quindi gli stati nascosti $h_k$ sono condizionati dal claim fin dal token 1.


* Il token finale `Answer:` (da cui estrai $q$ e il verdetto) ha il secondo `{CLAIM}` immediatamente adiacente (nessun decadimento RoPE su documenti lunghi) e rispetta la struttura nativa `Documento -> Claim -> Answer` con cui il backbone è stato pre-addestrato.




* **Overhead:** Un claim è lungo in media 15–25 token. Ripeterlo su un prompt da 2.000 token aggiunge l'**1% di token** al prefill (~1.5 ms), ma stabilizza enormemente sia il verdetto sia il vettore $q$.



---

## 6. Funzione di Loss e Soglie: Set-Prediction e Calibrazione della Cardinalità (Multiple-Instance Learning)

### Il problema nel piano attuale

Usando una combinazione di BCE per frase + Ranking loss con una soglia fissa in inferenza (es. `score > 0.5`):

* Nei documenti lunghi ($n = 150$ frasi) in cui solo 1 o 2 frasi sono evidenza, lo sbilanciamento di classe è estremo (98% negativi).
* Se un claim richiede 2 frasi complementari ($S_1$ e $S_{40}$), spesso una delle due è molto esplicita (`score = 0.92`) e la frase "ponte" è più implicita (`score = 0.42`). Con una soglia statica perdi la frase ponte, facendo crollare la *Recall dell'insieme completo*.


* Inoltre, hai molti esempi nel dataset di training per il verdetto in cui sai che `supported = True`, ma **non hai le annotazioni gold delle frasi** (che attualmente mascheri completamente).



### Le 3 soluzioni di Loss da combinare

1. **Loss MIL (Multiple-Instance Learning) sui dati positivi SENZA annotazione di frase:**
Non sprecare i dati positivi privi di etichette a livello di frase mascherandoli del tutto!


* Se un esempio è **Non Supportato** ($Y = 0$), sappiamo con certezza matematica che **nessuna** frase (o combinazione) supporta il claim: puoi imporre $\max_k p_k \rightarrow 0$ su *tutti* i negativi (anche senza annotazioni di frase, aiutando l'aggregatore a sopprimere i falsi positivi lessicali dove una frase somiglia al claim ma lo contraddice!).
* Se un esempio è **Supportato** ($Y = 1$) ma privo di evidenza annotata, almeno una frase deve essere attiva: $\max_k p_k \rightarrow 1$ più una regolarizzazione di sparsità $L_1$ ($\frac{1}{n}\sum_k p_k$) per evitare che accenda tutto il documento.


2. **InfoNCE / ListNet Loss con temperatura appresa (invece di Margin Ranking a coppie):**
Tratta la distribuzione degli score sulle $n$ frasi e poni la distribuzione target uniforme sulle $K$ frasi gold ($1/K$ sulle frasi gold, $0$ sulle altre) usando la *Kullback-Leibler divergence* su $\text{Softmax}(s / \tau)$. Questo calibra naturalmente i punteggi relativi indipendentemente dalla lunghezza del documento.
3. **Predizione dinamica della soglia tramite un "Null/Stop Token" nell'aggregatore:**
Nell'input dell'aggregatore `[x_0, x_1, ..., x_n]`, usa il logit prodotto su `x_0` (lo stato globale $q$) come **soglia dinamica dell'istanza** $T_{\text{dyn}}$.
Una frase $k$ viene selezionata come evidenza se e solo se:



$$\text{logit}(X_k) > \text{logit}(X_0)$$



In fase di training, insegni al modello che tutte le frasi gold devono avere logit superiore a $X_0$, e tutte le frasi non-gold devono avere logit inferiore a $X_0$ (approccio *Threshold-Class* usato nel multi-label moderno, es. *ZLPR Loss - Zero-bounded Log-sum-exp & Pairwise Rank-based Loss* di Su et al.).

```
Logit Frasi dall'Aggregatore:
   S40 (0.89)  ───────────────────────────┐
   S1  (0.62)  ───────────────────────────┼──► Sopra x_0 ──► EVIDENZA ESTRATTA
 - - - - - - - - - [ Logit(x_0) = 0.51 ] -┼- - (Soglia dinamica dell'esempio)
   S12 (0.48)  ───────────────────────────┼──► Sotto x_0 ──► Scartate
   S8  (0.10)  ───────────────────────────┘

```

### Perché la ZLPR Loss + Soglia Dinamica cambia le regole del gioco

$$\mathcal{L}_{\text{ZLPR}} = \log \left( 1 + \sum_{i \in \Omega_{\text{neg}}} e^{s_i - s_0} \right) + \log \left( 1 + \sum_{j \in \Omega_{\text{pos}}} e^{s_0 - s_j} \right)$$

* Elimina l'iperparametro della soglia fissa in validazione.
* Quando il documento è complesso e multi-hop, lo stato globale $x_0$ (che sa che servono 2 o 3 frasi per chiudere la catena logica) abbassa automaticamente la propria asticella $s_0$ per far passare anche la frase ponte!



---

## Sintesi e Tabella di Priorità per il tuo Piano Esperimenti (`07_esperimenti.md`)

Ecco come integrare queste idee in modo incrementale rispetto alla tua architettura base, ordinandole per rapporto **Beneficio Atteso / Complessità di Implementazione**:

| ID Idea | Componente | Impatto su Latenza | Impatto Atteso (F1 / Multi-Hop) | Perché provarla subito |
| --- | --- | --- | --- | --- |
| **Idea 5** | **Sandwich Prompting** (`Claim -> Doc -> Claim`) | +1% token (~1.5 ms) | **Alto** (sia Verdetto che Evidenza) | Azzera il RoPE decay su doc lunghi e mantiene il formato nativo vicino ad `Answer:`. Zero modifiche al codice del modello. |
| **Idea 6** | **ZLPR Loss + Soglia Dinamica su $x_0$ + MIL sui negativi** | **0 ms** (solo training) | **Molto Alto** (+5-8 pt su Recall insieme completo) | Risolve il problema delle frasi "ponte" con score più basso e sfrutta i casi *Not Supported* per sopprimere i distrattori lessicali. |
| **Idea 1** | **Multi-Layer Fusion + Q-K Attention Routing** | < 0.2 ms | **Molto Alto** (F1 e Sufficienza) | Riutilizza i circuiti di attenzione che il 12B calcola già durante il forward pass invece di costringere l'aggregatore a indovinarli. |
| **Idea 2** | **Attentive Span Pooling intra-frase** | ~ 0.1 ms | **Medio-Alto** (su frasi lunghe >25 parole) | Evita il collo di bottiglia del punto fermo (`.`), catturando le entità all'interno della frase. |
| **Idea 3** | **Testa Biaffine Low-Rank per Coppie** | < 0.1 ms | **Alto su HoVer / Multi-Hop** | Modella esplicitamente la sinergia tra $S_1$ e $S_{40}$ quando sono distanti. |
| **Idea 4** | **Consistency Masking sull'ultimo layer (Training)** | **0 ms in prod** (+5% in train) | **Alto su metrica di Sufficienza** | Garantisce matematicamente che l'evidenza estratta sia la vera causa del verdetto del modello. |

Quale di questi blocchi vuoi approfondire a livello di codice PyTorch o integrare direttamente nelle specifiche dei file `03_rendering_formato.md`, `04_loss.md` e `05_training.md`?