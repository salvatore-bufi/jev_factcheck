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

## 4. Dettagli di Addestramento e Architettura di FactCG

Usano semplicemente la cross entropy. 

FactCG viene addestrato con un rigoroso processo a due stadi per garantire l'assorbimento di capacità logiche:

| Stadio | Dati Utilizzati | Scopo |
| --- | --- | --- |
| **Stadio 1** (1 Epoca) | Dati NLI base + MiniCheck C2D + **CG2C-MHQA** | Insegnare i fondamentali dell'inferenza logica e il multi-hop basato su domande (sfruttando dataset come HotpotQA e Musique).
| **Stadio 2** (1 Epoca) | MiniCheck D2C + **CG2C-Doc** | Adattare il modello a documenti realistici (news, articoli) e rafforzare il ragionamento multi-hop lineare generato autonomamente.

 |

**Infrastruttura e Costi:**
Sorprendentemente, l'addestramento è super-efficiente. Addestrare il modello FactCG-DBT (DeBERTa-v3-large da ~400 milioni di parametri) su 4 GPU Nvidia Quadro RTX 8000 richiede **solo 18 minuti**. L'innovazione non è nella computazione bruta, ma nell'ingegneria del dato sintetico.

---

## 5. Metriche Avanzate: Il Connected Reasoning (CoRe)

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

## 6. L'Ingegneria dei Prompt (Dalle Appendici)

Il paper svela i prompt esatti usati per GPT-4o, mostrando come si aggirano le limitazioni strutturali dei LLM.

**Estrazione dei Grafi (Table 13):**
Per evitare che l'LLM perda il filo, viene forzato un formato rigido con delimitatori speciali (`{tuple_delimiter}` e `{group_delimiter}`).

* *Esempio di output imposto:* `Hunt {tuple_delimiter} government {tuple_delimiter} Hunt said something about the government`.



**La Corruzione Mirata (Table 16):**
Per creare i campioni negativi, all'LLM non viene chiesto semplicemente di "rendere il claim falso". Viene fornita la coppia di entità (es. `India, Area`) e il testo, con l'istruzione rigorosa: *"Remove the relation between two provided entities in below provided sentences with minimal changes."*. Questo assicura che il documento rimanga linguisticamente fluido e coerente, rendendo l'individuazione dell'allucinazione estremamente sfidante per il classificatore finale.