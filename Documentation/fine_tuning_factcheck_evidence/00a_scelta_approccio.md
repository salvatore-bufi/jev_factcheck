# Step 0a — Come ottenere l'evidenza alla velocità di un decision model

## Obiettivo

Scegliere come produrre, insieme al verdetto, le frasi del documento che supportano il claim, con il vincolo di **un solo forward e nessun token generato**. Questo file confronta le alternative, affronta il caso dell'evidenza sparsa e valuta, una per una, le proposte di [idea.md](idea.md) sui tre modelli del piano (§8).

## 1. Il budget di tempo

Un decision model costa **un prefill**: legge il prompt e produce i logit di una posizione. Ordini di grandezza (da misurare in [06](06_valutazione.md) §4):

| Operazione | d1-3B | JPT-9B / Winnow-12B (bf16, L40S) |
|---|---|---|
| Prefill di ~2.000 token | decine di ms (~40 ms per riga nei benchmark) | ~150–300 ms (2 · 9–12 miliardi · 2.000 ≈ 36–48 TFLOP) |
| Un token generato | ~5–10 ms | ≥ ~20–25 ms (rilettura di 18–22 GB di pesi) |
| Una riga di attenzione (1 query × T chiavi) in 8 layer | microsecondi | microsecondi |
| Aggregatore su n = 60 frasi da 256 | < 1 ms | < 1 ms |

Ogni soluzione che **genera** o che fa **più forward** è fuori budget. Su d1-3B il prefill è breve: un overhead fisso di 1 ms pesa già il 2–3%. Per questo il vincolo di latenza ha anche una forma assoluta ([00_overview](00_overview.md) §6).

## 2. Le alternative

| Opzione | Forward | Token generati | Evidenza sparsa | Fedeltà | Esito |
|---|---|---|---|---|---|
| Ablazione: togliere una frase alla volta | n + 1 | 0 | sì (con ricerca greedy) | alta per costruzione | **scartata in produzione**; usata **offline** per etichettare i dati ([02b](02b_dataset_generazione_llm.md)) e come riferimento ([01](01_baseline.md)) |
| Citazioni generate (`Answer: B [S1][S40]`) | 1 | 3–10 | sì | buona se vincolata agli ID | **scartata**: +50–100% di latenza |
| Gradiente × input sul margine | 1 + backward | 0 | debole | bassa | scartata: backward ≈ 2× forward, mappe rumorose |
| Attenzione da `Answer:` verso le frasi | 1 | 0 | sì | discutibile | **baseline** senza training (E-ATT) e **feature** della testa (§8, idea 1) |
| Testa lineare per frase | 1 | 0 | **no** (§3) | buona | base della soluzione |
| Pointer `qᵀ W h_k` | 1 | 0 | parziale | buona | variante più semplice ([07](07_esperimenti.md)) |
| **Testa + aggregatore bidirezionale** | 1 | 0 | **sì** | buona | **scelta** |

## 3. Il problema dell'evidenza sparsa

In un modello decoder lo stato alla fine della frase k vede solo i token **precedenti**.

| Stato | Vede | Conseguenza |
|---|---|---|
| `h_1` (prima frase) | preambolo con il claim, S1 | non sa se più avanti c'è una frase che la completa |
| `h_40` (ultima) | claim, S1 … S39 | può sapere che S1 c'era e che insieme stabiliscono il claim |
| `q` (posizione `Answer:`) | tutto | sa quale catena regge il verdetto |

Esempio dove la testa per frase sbaglia: con una frase ponte che non nomina nulla del claim (per esempio "Saint Paul School of Theology moved to Leawood in 2013", quando il claim parla di Leawood solo attraverso S40), `h_1` non ha nessun indizio.

**Soluzione**: un aggregatore non causale sopra le rappresentazioni delle frasi.

```text
x_0 = P_q · q,   x_k = P_h · [h_k ; m_k] + P_a · a_k + E_pos[k]      (proiezioni → 256)
X   = TransformerEncoder(2 layer, 4 teste, 256) ( [x_0, x_1, …, x_n] )   # attenzione piena
logit_k = v · X_k + b ;   s_0 = v · X_0 + b                             # soglia dell'istanza
```

- Le frasi si vedono a vicenda: S1 "sa" di S40.
- `x_0` porta lo stato finale, che conosce il verdetto e la catena; il suo logit fa da soglia (§8, idea 6).
- Parametri: ~2–3M, contro 3–12 miliardi del backbone.

### Un limite in più, specifico dei nostri modelli

Che `h_k` "conosca il claim" dipende da come l'informazione viaggia dal preambolo alla frase k:

| Modello | Da dove passa il claim verso una frase lontana |
|---|---|
| d1-3B | solo dagli 8 layer di attenzione (le convoluzioni vedono 3 token) |
| JPT-9B | dagli 8 layer di attenzione e dallo stato ricorrente dei 24 layer DeltaNet (che decade) |
| Winnow-12B | dagli 8 layer globali; i 40 a finestra vedono 1.024 token (a catena un po' di più) |

Quindi il condizionamento di `h_k` sul claim si indebolisce con la **distanza in token dal preambolo**, oltre che con la distanza fra le frasi gold. Si misura a parte ([06](06_valutazione.md) §2). L'aggregatore compensa in parte, perché `x_0` porta il claim visto da vicino.

## 4. Il formato: perché il sandwich

Tre ordini possibili:

| Formato | Le frasi vedono il claim | `Answer:` ha il claim vicino | Prompt nativo dopo il preambolo | Verdetto del piano binario riusabile |
|---|---|---|---|---|
| Nativo (documento → claim) | no | sì | sì | sì |
| Claim → documento | sì | no (lontano quanto il documento) | no (ordine dei campi cambiato) | no: va riaddestrato |
| **Sandwich** (claim, poi prompt nativo) | sì | sì | **sì** | **probabilmente**: va misurato (B-SAND) |

Con il formato nativo nessuno stato `h_k` conosce il claim: tutto passa da `q`, e l'aggregatore deve fare da solo il confronto frase–claim. Con "claim prima" si risolve questo, ma si cambia il prompt che il verdetto del piano binario ha visto in training ([../fine_tuning_factcheck/02d_varianti_domanda.md](../fine_tuning_factcheck/02d_varianti_domanda.md) §4 fissa l'ordine nativo), e il claim finisce lontano dalla posizione di lettura.

Il sandwich tiene il meglio dei due: un **preambolo** con il claim, prima dello stato, e poi il prompt nativo **identico**. Costo: la lunghezza del claim, in media 15–40 token in più. Dettagli per modello in [02d](02d_varianti_domanda.md).

Scelta: sandwich come default; B-ORD e B-SAND ([01](01_baseline.md)) decidono, e H4 ([07](07_esperimenti.md)) misura quanto vale contro il formato nativo per l'evidenza.

## 5. Perché niente chunking

Con il chunking e la regola `max` (MiniCheck, FactCG), se S1 e S40 stanno in chunk diversi:
- nessun chunk da solo supporta il claim → verdetto "non supportato";
- e l'evidenza non esiste.

Il documento passa **intero** in un forward, entro il contesto: 32.768 token per d1-3B e per i benchmark di JPT e Winnow. Documenti oltre il contesto: fuori dal perimetro (vanno spezzati a monte, sapendo che l'evidenza fra pezzi diversi si perde).

## 6. Granularità: frasi, non token

| Unità | Pro | Contro |
|---|---|---|
| **Frase** | annotazioni disponibili (FEVER, HoVer, SciFact); leggibile per l'utente | frasi lunghe evidenziano anche parti irrilevanti |
| Token / span | preciso | quasi nessuna annotazione a questa granularità per il supporto; più rumoroso |

Si parte dalle frasi. Le frasi molto lunghe (> 60 parole) si spezzano su `;` e `:` ([03](03_rendering_formato.md)). I token servono solo **dentro** la rappresentazione della frase (media dei token, §8 idea 2), non come unità mostrata.

## 7. Raccomandazione

1. Verdetto dal piano binario, nel formato sandwich (riusato se B-SAND lo consente, altrimenti riaddestrato in quel formato).
2. Testa di evidenza = aggregatore bidirezionale a 2 layer su `[q, frasi]`; ogni frase rappresentata da stati di fine frase e media dei token (mix di layer) più le feature di attenzione di `Answer:`.
3. Loss con soglia dell'istanza (ZLPR) e MIL sui negativi ([04](04_loss.md)).
4. Training in tre tempi: stati salvati (fase 1a), backbone congelato in linea (1b) per ciò che non si può salvare, congiunto (2) solo se serve.
5. Si itera su d1-3B; JPT-9B e Winnow-12B confermano.
6. L'ablazione con n forward si usa solo offline: per etichettare dati e come riferimento di qualità.

## 8. Le proposte di idea.md, valutate sui nostri modelli

[idea.md](idea.md) propone sei estensioni. Qui ciascuna è confrontata con le architetture reali ([00c](00c_modelli_e_architetture.md)); i guadagni stimati in idea.md non sono misure e non si riportano. Ogni proposta diventa un esperimento con una regola di adozione ([07](07_esperimenti.md)).

### Idea 5 — Formato sandwich: adottata come default

È la più utile, e per un motivo in più rispetto a idea.md: risolve il conflitto con il piano binario, che fissa l'ordine nativo. La forma adottata non duplica campi nello stato (in JSON non si può): il claim va in un **preambolo** prima dello stato, e lo stato nativo resta com'è ([02d](02d_varianti_domanda.md)). In idea.md il secondo claim stava nella domanda; qui sta dove il modello lo ha sempre visto.

Correzione all'argomento di idea.md: più del decadimento di RoPE contano la finestra di 1.024 token di Winnow e le convoluzioni di d1, che limitano chi vede il claim (§3). Il sandwich aiuta `Answer:`; non risolve il condizionamento delle frasi lontane dal preambolo, che si misura.

### Idea 6 — Soglia dell'istanza (ZLPR) + MIL: adottata nella loss

- **ZLPR con `s_0` dallo stato `x_0`**: adottata come loss candidata principale contro BCE + ranking (H-LOSS). Toglie la soglia fissa `τ_e` e adatta il numero di frasi mostrate al caso.
- **MIL sui negativi** (nessuna frase sopra `s_0` se il claim non è supportato): adottata. Nella forma ZLPR è lo stesso termine con l'insieme dei positivi vuoto, quindi non serve una loss a parte. Porta nel training le frasi che somigliano al claim ma lo contraddicono, che sono proprio i falsi positivi di E-LEX.
- **MIL sui positivi senza annotazione** (almeno una frase sopra `s_0`, più sparsità): esperimento, in alternativa o in aggiunta alle etichette per ablazione (H-MIL).
- **ListNet / KL verso l'uniforme sulle gold**: non adottata come default. Con annotazioni incomplete (FEVER) la softmax toglie massa alle frasi valide non annotate; resta come variante di H-LOSS.

Attenzione: il termine sui negativi rende la testa dipendente dal verdetto (che legge da `x_0`). È coerente con la definizione (l'evidenza esiste solo se il claim è supportato), ma nei positivi con verdetto incerto la testa può spegnersi. Peso basso e controllo sulla recall delle righe vicine alla soglia.

### Idea 1 — Mix di layer e feature di attenzione: adottata, adattata

- **Mix di layer**: si leggono gli stati dopo gli ultimi layer globali e dall'ultimo layer, con pesi scalari appresi (softmax su 4–5 layer). Sostituisce l'esperimento "ultimo layer contro 2/3" del piano precedente.
- **Feature di attenzione**: l'idea dice "ultimi M layer"; nei nostri modelli l'attenzione esiste solo negli **8 layer globali** (d1: 2, 5, 9, 13, 17, 21, 24, 27; JPT: 3, 7, …, 31; Winnow: 5, 11, …, 47). Si usano tutti e 8: per ogni frase, la massa di attenzione di `Answer:` sui token della frase, per testa. Sono 8 × 32 = 256 numeri per d1, 8 × 16 = 128 per JPT e Winnow.
- Come si calcola: le implementazioni veloci (FlashAttention, SDPA) non restituiscono i pesi. Con un hook si salvano la query dell'ultima posizione e le chiavi, e si ricalcola **una riga** di softmax per testa: costo trascurabile, anche in produzione.
- Con la domanda Q4 del piano binario ("il claim contiene un dettaglio non stabilito?") `Answer:` può guardare soprattutto i punti problematici, non il supporto. È un motivo per apprendere una proiezione sulle feature, invece di usare l'attenzione grezza come in E-ATT.

### Idea 2 — Pooling dentro la frase: adottata in due gradini

Il rischio indicato (stato di fine frase dominato dalla punteggiatura o dalla coda della frase) è plausibile. Ma la versione attentiva di idea.md ha bisogno di **tutti** gli stati dei token, che non si possono salvare per la fase 1a (per JPT, 2.000 token × 4.096 × 2 byte = 16 MB per riga: 320 GB per 20k righe).

1. **Gradino fisso** (fase 1a): media dei token della frase, calcolata durante l'estrazione e salvata accanto allo stato di fine frase. Raddoppia lo spazio, nient'altro.
2. **Gradino attentivo** (fase 1b, backbone congelato in linea): il pooling guidato da `[claim; q]` di idea.md. Si prova solo se il gradino fisso guadagna sulle frasi lunghe.

### Idea 3 — Termine bilineare a coppie: esperimento condizionato

L'aggregatore ha già un'attenzione piena fra frasi, che è un'interazione bilineare a coppie: il guadagno di un termine esplicito non è scontato. Si prova (H-PAIR) solo se, dopo H0–H6, la recall dell'insieme completo su HoVer resta sotto 0,65 o il calo oltre 20 frasi supera 5 punti. La parte più promettente è la **supervisione sulle coppie gold** (perdita ausiliaria su (k, j) ∈ gold × gold), che sui dati a una frase (gran parte di FEVER) è vuota. Variante gratuita: come prior delle coppie, l'attenzione fra i token di fine frase negli 8 layer globali (causale: solo j < k, poi simmetrizzata).

### Idea 4 — Fedeltà tramite maschera in training: esperimento di ricerca, per ultimo

L'obiettivo (l'evidenza scelta è quella che regge il verdetto) è giusto, ma la versione di idea.md (maschera solo nell'ultimo layer, solo su `Answer:`) **non funziona**: nei layer precedenti l'informazione delle frasi è già passata nel flusso residuo di `Answer:` e delle frasi successive. Mascherare l'ultimo layer cambia poco il verdetto qualunque sia la maschera, e il segnale per la testa è debole.

Versione corretta: un bias `log p_k` (sufficienza) o `log(1 − p_k)` (completezza) sulle chiavi della frase k, per **tutte** le query fuori dalla frase, in **tutti** i layer di attenzione. Dove è fattibile:

| Modello | Maschera per frase | Perdita di informazione fuori dalla maschera |
|---|---|---|
| Winnow-12B | tutti i 48 layer sono di attenzione | nessuna: equivale a togliere la frase (a meno delle posizioni) |
| d1-3B | 8 layer di attenzione | solo le convoluzioni (3 token) ai confini fra frasi |
| JPT-9B | solo 8 layer su 32 | i 24 layer DeltaNet portano tutto: maschera solo approssimata |

Costo: due forward in più **con backward** attraverso il backbone (pesi congelati, ma serve il gradiente rispetto alla maschera): ~3–4× il costo di un passo della fase 1b. Zero in produzione. Rischio: la testa impara a seguire il modello, anche le sue scorciatoie; l'F1 rispetto alle gold può scendere mentre la sufficienza sale. Si adotta solo se la sufficienza sale **e** l'F1 non scende (H-FAITH, su d1-3B; poi Winnow).

Esiste già una versione offline e non differenziabile della stessa idea: le etichette per ablazione (E-ABL, [02b](02b_dataset_generazione_llm.md) §3). H-FAITH si prova solo se quelle non bastano.

### Riepilogo

| Idea | Esito | Dove | Esperimento |
|---|---|---|---|
| 5 Sandwich | default | [02d](02d_varianti_domanda.md), [03](03_rendering_formato.md) | B-SAND, H4 |
| 6 ZLPR + MIL | loss principale candidata | [04](04_loss.md) | H-LOSS, H-MIL |
| 1 Mix di layer + attenzione | primi gradini sopra H0 (estratti già nella fase 1a) | [03](03_rendering_formato.md), [05](05_training.md) | H2a, H2b |
| 2 Pooling nella frase | fisso nella fase 1a; attentivo condizionato | [05](05_training.md) | H2c, H2d |
| 3 Coppie | condizionato al gap multi-hop | [05](05_training.md) | H-PAIR |
| 4 Fedeltà con maschera | ricerca, per ultimo | [05](05_training.md) | H-FAITH |

## Criteri per passare allo step successivo

- Architettura approvata; formato sandwich come default, in attesa di B-SAND.
- Stabilito il budget di latenza e la lunghezza massima del documento in produzione.
