# Step 2b — Generare i dati: C2D e D2C (MiniCheck), CG2C (FactCG), negativi minimi

## Obiettivo

Produrre, con un LLM, coppie (documento, claim, etichetta) che i dataset pubblici non hanno o hanno in quantità insufficiente:

| Famiglia | Cosa insegna | Etichette prodotte | Stadio |
|---|---|---|---|
| **C2D** (claim → documento) | combinare più frasi; dire "no" quando manca un pezzo | 1 e 0 per costruzione | 1 |
| **D2C** (documento → claim) | lo stesso su testi reali (notizie, trascrizioni) | 1 e 0 verificati | 2 |
| **CG2C-MHQA** (grafo da domande multi-hop) | catene di 2–4 fatti | 1 e 0 per costruzione | 1 |
| **CG2C-Doc** (grafo da documenti) | catene su documenti qualsiasi | 1 e 0 per costruzione | 2 |
| **remove_evidence** | "il documento non lo dice" vale "no" | 0 | 2 |
| **contradict_claim** | contraddizioni minime (un numero, un nome) | 0 | 1 |

Questo file contiene **tutto** quello che serve: procedure, prompt esatti, schemi JSON, controlli, regole di etichettatura, esempi, costi.

## 1. Regole valide per tutte le famiglie

1. **Si parte solo dal train** congelato allo Step 0. Ogni riga generata eredita il `doc_id` della riga o del documento di partenza, così resta nel train.
2. **L'etichetta deriva dall'operazione**, non dal giudizio dell'LLM: se si cancella l'evidenza, l'etichetta è 0 perché l'evidenza non c'è più.
3. **Un secondo LLM giudice**, diverso dal generatore, verifica ogni esempio; si tengono solo quelli su cui operazione e giudice concordano (§8).
4. **Controlli automatici economici** prima del giudice (lunghezze, parole presenti, formato).
5. **Tutto in JSON**: ogni prompt chiede un solo oggetto JSON; il parser è `extract_json` di [llm_gen/tasks.py](../../llm_gen/tasks.py) (tollera ```json e testo attorno).
6. **Lingua**: inglese (come tutti i dati e i benchmark).
7. **Messaggio di sistema** (uguale per tutti i prompt, come in `llm_gen/tasks.py`):

```text
You help build training data for a fact-checking model. Follow the instructions exactly. Answer with a single JSON object and nothing else.
```

## 2. L'infrastruttura

### 2.1 Modelli

| Ruolo | Modello consigliato | Dove | Comando |
|---|---|---|---|
| Generatore | `Qwen/Qwen3-30B-A3B-Instruct-2507` (MoE, 3B parametri attivi: veloce) | 1 L40S | `vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 --max-model-len 32768 --gpu-memory-utilization 0.92 --port 8000` (se la memoria non basta in bf16, usare la variante FP8 dello stesso modello, se disponibile, o `--max-model-len 16384`) |
| Giudice | `openai/gpt-oss-120b` | 2 L40S | `vllm serve openai/gpt-oss-120b --tensor-parallel-size 2 --max-model-len 32768 --port 8001` |
| Alternativa leggera al giudice | `openai/gpt-oss-20b` | 1 L40S o il portatile | `vllm serve openai/gpt-oss-20b --port 8001` |
| Prove su pochi esempi | DeepSeek-V4.1-Flash via API NVIDIA (prototipo esistente) | API | `llm_gen/run.py` |

Tutti con licenza Apache 2.0: gli output si possono usare per addestrare senza vincoli. Con API commerciali bisogna invece controllare i termini (alcune vietano di addestrare modelli concorrenti sugli output).

Generatore e giudice non possono stare insieme sulle due L40S: si genera prima (una famiglia alla volta, su una scheda), poi si giudica tutto in blocco (due schede).

### 2.2 Il client verso un server locale

`llm_gen/client.py` oggi parla solo con l'API NVIDIA. Si aggiunge un client per un server compatibile OpenAI (vLLM):

```python
class LocalClient:
    """OpenAI-compatible client for a local vLLM server."""
    def __init__(self, base_url="http://127.0.0.1:8000/v1", model=None, temperature=0.7, top_p=0.95,
                 max_tokens=4096, timeout=600):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key="EMPTY", timeout=timeout)
        self.model = model or self.client.models.list().data[0].id
        self.temperature, self.top_p, self.max_tokens = temperature, top_p, max_tokens

    def chat(self, messages):
        r = self.client.chat.completions.create(model=self.model, messages=messages,
                                                temperature=self.temperature, top_p=self.top_p,
                                                max_tokens=self.max_tokens)
        return {"content": r.choices[0].message.content or "", "usage": r.usage.model_dump()}
```

Temperature consigliate: 0,7 per generare (varietà), 0 per giudicare e per i compiti di estrazione (grafo, scomposizione).

### 2.3 Esecuzione parallela

vLLM serve molte richieste insieme: inviarle in parallelo moltiplica la velocità. Schema (thread pool, ripresa dopo interruzioni):

```python
import concurrent.futures as cf, json
def run_parallel(items, fn, out_path, workers=32):
    done = set()
    if out_path.exists():
        done = {json.loads(l)["key"] for l in open(out_path)}
    todo = [it for it in items if it["key"] not in done]
    with open(out_path, "a") as f, cf.ThreadPoolExecutor(workers) as pool:
        for rec in pool.map(lambda it: safe(fn, it), todo):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()

def safe(fn, it):
    try:
        return {"key": it["key"], **fn(it)}
    except Exception as e:                       # si salva comunque, per ispezione
        return {"key": it["key"], "error": f"{type(e).__name__}: {e}"}
```

Ogni famiglia è una pipeline di più passi; ogni passo legge il file del passo precedente e scrive il suo in `data/finetune/gen/<famiglia>/<passo>.jsonl`. Se un passo si interrompe, lo si rilancia e riparte da dove era.

### 2.4 Il formato di una riga generata

Identico a `build_pairs.py`, con meta in più:

```json
{"id": "c2d:fever:train:75397#d0", "source": "c2d", "split": "train",
 "document": "...", "claim": "...", "label": 1, "orig_label": "supported (synthetic)",
 "doc_id": "fever:Nikolaj_Coster-Waldau", "meta": {
   "family": "c2d", "parent_id": "fever:train:75397", "generator": "Qwen3-30B-A3B-Instruct-2507",
   "judge": "gpt-oss-120b", "judge_verdict": "supported", "pair_id": "c2d:fever:train:75397",
   "neg_type": null, "facts": ["...", "..."], "removed": null}}
```

`pair_id` collega le righe nate dallo stesso documento di partenza (servono per la loss contrastiva opzionale, [04](04_loss.md) §5).

## 3. C2D — dal claim al documento (MiniCheck)

### 3.1 L'idea

Si vuole un documento in cui **ogni fatto del claim è stabilito solo combinando due frasi**. Allora togliendo una sola frase si ottiene un documento in cui quel fatto non è più stabilito, e si sa esattamente perché: l'etichetta è certa per costruzione.

### 3.2 Input

Claim reali **supportati** dal train di FEVER, HoVer e AVeriTeC (e, se si vuole, i riassunti di SummEdits con label 1), con:

- 8–40 parole;
- almeno due fatti distinti (si verifica al passo 1: se la scomposizione dà un solo fatto, si scarta).

Il documento di partenza **non** serve: C2D costruisce un documento nuovo. Il `doc_id` della riga generata è `c2d:<id del claim>`.

Quantità: per 10.000 esempi finali servono ~1.500 claim (ogni claim produce 6–10 righe, §3.8).

### 3.3 Passo 1 — Scomposizione in fatti atomici (`c2d_decompose`)

```text
Claim:
<<<
{claim}
>>>

Split the claim into atomic facts. An atomic fact states exactly one piece of information
(one entity, one property, one relation, one number or one date).
Rules:
- Every atomic fact must be a complete, self-contained sentence (resolve pronouns).
- Together the facts must say everything the claim says, and nothing more.
- Use between 2 and 4 facts. If the claim contains only one piece of information, return one fact.
Return JSON: {"facts": ["<fact 1>", "<fact 2>", ...]}
```

Controlli: 2–4 fatti (altrimenti si scarta il claim); nessun fatto più lungo del claim.

Esempio:

> Claim: "Over 5,000 caravan members stayed at the Tijuana stadium, a facility with a capacity of 3,000."
> a1: "Over 5,000 caravan members stayed at the Tijuana stadium."
> a2: "The Tijuana stadium has a capacity of 3,000 people."

### 3.4 Passo 2 — Due frasi per ogni fatto (`c2d_sentpair`)

Una chiamata per fatto:

```text
Fact:
<<<
{fact}
>>>

Write two sentences, S1 and S2, such that:
- S1 and S2 TOGETHER imply the fact, so that a careful reader can derive it.
- NEITHER sentence alone is enough to derive the fact.
- Each sentence is a natural sentence that could appear in a news article or an encyclopedia.
- Do not copy the fact's wording; use a different formulation, an intermediate entity,
  a calculation, a definition or a temporal step to connect S1 and S2.
Return JSON: {"s1": "<sentence>", "s2": "<sentence>", "how": "<one line: why both are needed>"}
```

Esempio per a2:

> S1: "The Tijuana stadium recently underwent renovations that doubled its capacity."
> S2: "Before the renovations, the stadium could hold 1,500 spectators."

**Verifica di necessità** (con il giudice, §8): per ogni fatto tre domande di verifica: (S1+S2, fatto) deve essere *supported*; (S1, fatto) e (S2, fatto) devono essere *not_enough_info*. Se una frase da sola basta, si rigenera la coppia (al massimo 2 tentativi) o si scarta il fatto.

### 3.5 Passo 3 — Il documento completo (`c2d_passage`)

```text
Sentences:
{numbered list of all S1/S2 sentences, shuffled}

Write a coherent passage of 120 to 250 words (one or two paragraphs) on the topic of these sentences.
Rules:
- Include EVERY sentence above. You may make minimal grammatical changes so the passage flows,
  but each sentence must keep its full meaning.
- Add a few sentences of neutral context that do not add any information about the facts these
  sentences establish.
- Do not state any of the following facts directly; they must only follow from the sentences above:
{list of atomic facts}
Return JSON: {"passage": "<the passage>"}
```

Il terzo punto è importante: senza di esso l'LLM tende a scrivere il fatto in chiaro ("so its capacity is now 3,000"), e il documento ablato del passo successivo lo conterrebbe ancora.

Controlli automatici: lunghezza 100–300 parole; per ogni frase S, almeno l'80% delle sue parole di contenuto compare nel passaggio.

Riga prodotta: `(D, claim, 1)`, `orig_label = "supported (synthetic)"`.

### 3.6 Passo 4 — Documenti ablati (`c2d_passage` di nuovo)

Per ogni fatto a_i e per ciascuna delle sue due frasi s_{i,j}: si richiama **lo stesso prompt** del passo 3 con l'elenco delle frasi **senza** s_{i,j} (e con lo stesso elenco di fatti da non scrivere in chiaro). Si ottiene D'_{i\j}.

Con 2 fatti si ottengono 4 documenti ablati; con 3 fatti 6; con 4 fatti 8. Per contenere i costi, si può prenderne uno solo per fatto (j scelto a caso).

Esempio: togliendo S2 ("Before the renovations, the stadium could hold 1,500 spectators") il documento dice che la capacità è raddoppiata ma non da quanto: a2 non è più stabilito.

### 3.7 Passo 5 — Sotto-claim del power set (`c2d_merge`)

Per ogni sottoinsieme non vuoto dei fatti (con 2 fatti: {a1}, {a2}, {a1,a2}; con 3: 7 sottoinsiemi; con 4: 15, se ne tengono al massimo 8 scelti a caso, sempre incluso l'insieme completo):

```text
Facts:
{list of the facts in the subset}

Merge these facts into ONE fluent claim that states all of them and nothing else.
Do not add or drop any information. If there is only one fact, rewrite it as a natural claim.
Return JSON: {"claim": "<merged claim>"}
```

Per l'insieme completo si può usare direttamente il claim originale.

### 3.8 Passo 6 — Le etichette

Sia c' il sotto-claim del sottoinsieme a' e D'_{i\j} il documento senza la frase j del fatto i.

| Documento | Sotto-claim | Etichetta | Motivo |
|---|---|---|---|
| D (completo) | qualsiasi c' | 1 | tutti i fatti sono stabiliti |
| D'_{i\j} | c' con a_i ∉ a' | 1 | il fatto tolto non serve a c' |
| D'_{i\j} | c' con a_i ∈ a' | 0 (`not_established`) | manca metà dell'evidenza di a_i |

Esempio (2 fatti, documento senza S2 di a2):

| Coppia | Etichetta |
|---|:-:|
| (D'_{2\2}, "Over 5,000 caravan members stayed at the Tijuana stadium.") | 1 |
| (D'_{2\2}, "The Tijuana stadium has a capacity of 3,000 people.") | 0 |
| (D'_{2\2}, claim completo) | 0 |

Il fatto che lo stesso documento bucato dia sia positivi sia negativi è ciò che impedisce al modello di imparare "documento sintetico corto = falso".

Righe per claim (2 fatti, un'ablazione per fatto): D × 3 sotto-claim + 2 documenti ablati × 3 sotto-claim = 9 righe, ~4 negative.

### 3.9 Verifica e costi

- Giudice su **tutti** i negativi e su un campione del 20% dei positivi (§8). Se il giudice dice *supported* per un negativo, il passaggio ha probabilmente scritto il fatto in chiaro: si scarta il documento ablato intero.
- Chiamate per claim (2 fatti): 1 scomposizione + 2 coppie di frasi + 6 verifiche di necessità + 1 passaggio + 2 passaggi ablati + 3 fusioni ≈ 15 chiamate generatore/giudice, ~1.000 token ciascuna → ~15k token per ~9 righe.
- 10.000 righe ≈ 1.100 claim ≈ 17M token: poche ore su una L40S con il generatore MoE.

## 4. D2C — dal documento al claim (MiniCheck)

### 4.1 L'idea

Partire da testi **reali** per non insegnare solo lo stile dei documenti sintetici. Si riassume un pezzo di documento (il riassunto è un claim supportato), poi si tolgono frasi e si controlla quali fatti del riassunto restano supportati.

### 4.2 Input: documenti reali non presenti nei test

| Corpus | Caricamento | Note |
|---|---|---|
| Notizie CNN/DailyMail | `abisee/cnn_dailymail`, config `3.0.0`, split **train**, campo `article` | AggreFact-CNN usa articoli di test di CNN/DM: si usano solo articoli di train **e** si scartano quelli con ≥ 20% di 13-grammi in comune con AggreFact |
| Notizie XSum | `EdinburghNLP/xsum`, split train, campo `document` | stesso controllo (AggreFact-XSum) |
| Trascrizioni di riunioni | `pszemraj/qmsum-cleaned` (Apache 2.0) | simili a MeetingBank e MediaSum; controllo a 13-grammi |
| Documenti di SummEdits | i `document` del train di SummEdits | già nel nostro train |

Si usano documenti di 400–3.000 parole. Il `doc_id` è `d2c:<corpus>:<id>`.

### 4.3 Passo 1 — Divisione in chunk

Il documento si divide in frasi (splitter di [03](03_rendering_formato.md) §6) e le frasi si raggruppano in **3 chunk consecutivi** di lunghezza simile. Ogni chunk D_k ha tipicamente 6–20 frasi.

### 4.4 Passo 2 — Riassunto di ogni chunk (`d2c_summarize`)

```text
Text:
<<<
{chunk}
>>>

Write a summary of the text in 2 or 3 sentences, as a careful journalist would.
Rules:
- Every statement must be fully supported by the text; do not add outside knowledge.
- Include specific details (names, numbers, dates) when the text gives them.
- Combine information from different parts of the text when natural.
Return JSON: {"summary": "<summary>"}
```

Riga prodotta: `(D_k, summary, 1)` dopo verifica del giudice (§8: deve dire *supported*; i riassunti a volte contengono errori, e quelli si scartano).

### 4.5 Passo 3 — Fatti atomici del riassunto

Stesso prompt `c2d_decompose` (§3.3), sul riassunto, con al massimo 6 fatti.

### 4.6 Passo 4 — Ablazione frase per frase, con un giudizio in blocco (`d2c_entail`)

Per ogni frase s_j del chunk (o, per contenere i costi, per le 4–6 frasi che contengono più parole dei fatti), si costruisce D_k \ s_j (il chunk senza quella frase). Poi **una** chiamata al giudice per documento ablato, con tutti i fatti:

```text
Text:
<<<
{chunk without sentence j}
>>>

For each fact below, decide whether the text establishes it, using ONLY the text
(faithful paraphrases and direct implications count).
Facts:
1. {fact 1}
2. {fact 2}
...
Return JSON: {"verdicts": [{"fact": 1, "verdict": "supported" | "contradicted" | "not_enough_info"}, ...]}
```

Si ottiene una tabella L(j, fatto) ∈ {supportato, non supportato}.

### 4.7 Passo 5 — Sotto-claim ed etichette

Sottoinsiemi dei fatti come in §3.7 (al massimo 6 per chunk), fusi con `c2d_merge`. Per ogni documento ablato D_k \ s_j e ogni sotto-claim c':

| Condizione | Etichetta |
|---|:-:|
| tutti i fatti di c' sono supportati in D_k \ s_j | 1 |
| almeno un fatto di c' non è supportato in D_k \ s_j, e nel chunk completo erano tutti supportati | 0 (`not_established`) |

Si tengono le righe in cui l'etichetta **cambia** rispetto al chunk completo (i casi informativi) e un campione delle altre.

### 4.8 Passo 6 — Chunk incrociati (cross-document)

Gli stessi fatti del riassunto di D_k si giudicano contro gli **altri** chunk dello stesso documento (D_h, h ≠ k) con il prompt `d2c_entail`. I sotto-claim tutti supportati da D_h danno positivi `(D_h, c', 1)`; quelli non supportati danno negativi. Sono coppie difficili: stesso argomento, dettagli diversi.

### 4.9 Esempio

> Chunk D_2 (notizia): "... Kraven the Hunter and Ghostbusters: Frozen Empire will cost the studio an estimated $400 million this year ... hopes for 2024 are dashed ..."
> Riassunto (claim): "The delays will cost the studio about $400 million and dash its hopes for 2024." → 1
> Togliendo la frase sui 400 milioni: il fatto "will cost about $400 million" diventa *not_enough_info* → (D_2 \ s, claim, 0), mentre (D_2 \ s, "The delays dash the studio's hopes for 2024.", 1).

### 4.10 Costi

Per chunk: 1 riassunto + 1 verifica + 1 scomposizione + ~5 giudizi in blocco + ~6 fusioni + 2 giudizi incrociati ≈ 16 chiamate, ~2.000 token l'una (i chunk sono più lunghi) → ~32k token per ~15–25 righe. 20.000 righe ≈ 30–40M token: 1 giorno su una L40S.

## 5. CG2C-MHQA — claim multi-hop da domande (FactCG)

### 5.1 L'idea

HotpotQA e MuSiQue hanno domande che richiedono 2–4 passaggi e indicano le frasi o i paragrafi di supporto. Si trasforma (domanda, risposta) in un claim dichiarativo (positivo); per il negativo si toglie dal documento **una relazione** della catena.

### 5.2 Input

Righe `hotpotqa_seed.jsonl` e `musique_seed.jsonl` ([02a](02a_dataset_fonti_esistenti.md) §4). Documento = paragrafi gold + 2–4 paragrafi distrattori, in ordine casuale, formattati come `Title: <titolo>\n<testo>`.

### 5.3 Passo 1 — Domanda e risposta → claim (`cg2c_claimify`)

```text
Question: {question}
Answer: {answer}

Rewrite the question and its answer as ONE declarative sentence (a claim) that states the answer.
Keep all the entities and conditions of the question. Do not add information.
Return JSON: {"claim": "<claim>"}
```

Esempio: "In which country is the moth genus whose only species is I. pectinicornalis found?" / "India" → "The moth genus whose only species is Indogrammodes pectinicornalis is found in India."

Riga positiva: `(documento, claim, 1)`.

### 5.4 Passo 2 — Grafo delle frasi di supporto (`cg2c_graph`)

Le frasi di supporto (HotpotQA: `supporting_facts`; MuSiQue: le frasi dei paragrafi `is_supporting`) si numerano e si passano all'LLM:

```text
Sentences:
[1] {sentence 1}
[2] {sentence 2}
...

Extract the knowledge graph stated by these sentences as triples.
A triple is (entity 1, entity 2, relation): the relation is a short phrase that describes how the two
entities are connected according to the sentence. Use the entity names as written in the sentences
(resolve pronouns to the entity they refer to). Every triple must come from exactly one sentence.
Return JSON: {"triples": [{"e1": "...", "e2": "...", "relation": "...", "sentence": <number>}, ...]}
```

### 5.5 Passo 3 — Le triple che servono al claim (`cg2c_subgraph`)

```text
Claim: {claim}
Triples:
[1] (e1, e2, relation)  -- from sentence 3
[2] ...

Select the triples that are needed to verify the claim: all of them together must be enough,
and removing any one must make the claim unverifiable.
Return JSON: {"needed": [<triple numbers>]}
```

Controllo: le triple scelte devono essere connesse (formano un cammino o un albero) e coprire almeno 2 frasi diverse; altrimenti la riga si usa solo come positivo.

Numero di hop = numero di triple scelte (se il sottografo ha componenti separate, quello della componente più grande). Si salva in `meta.hops`.

### 5.6 Passo 4 — Togliere una relazione (`cg2c_remove_relation`)

Si sceglie a caso una tripla fra quelle necessarie, (e1, e2, relazione), e la frase da cui viene:

```text
Sentence:
<<<
{sentence}
>>>

Remove the relation between "{e1}" and "{e2}" ({relation}) from the sentence, with minimal changes.
Rules:
- After the edit, the sentence must no longer state or imply that relation.
- It must not state the opposite either (no contradiction): the relation is simply absent.
- Keep everything else in the sentence; keep it fluent and grammatical.
- If the relation is the only content of the sentence, return an empty string.
Return JSON: {"edited_sentence": "<sentence or empty string>"}
```

Esempio: (Indogrammodes pectinicornalis, India, "is found in") nella frase "It contains only one species, Indogrammodes pectinicornalis, which is found in India." → "It contains only one species, Indogrammodes pectinicornalis."

Il documento negativo `doc_neg` è il documento con la frase sostituita. Riga: `(doc_neg, claim, 0)`, `neg_type = not_established`.

Controllo automatico: la frase modificata non contiene più e2 (o e1) se la relazione era l'unico punto in cui compariva; il resto del documento è identico carattere per carattere.

### 5.7 Passo 5 — Filtro di difficoltà

FactCG tiene solo le coppie che un modello NLI **sbaglia**. Da noi si usa il modello di partenza (JPT, domanda di produzione):

- negativi: si tengono quelli con P(supportato) ≥ 0,3 (il modello è ingannato o incerto) più un 20% scelto a caso fra gli altri;
- positivi: si tengono quelli con P(supportato) ≤ 0,7 più un 20% a caso fra gli altri.

Poi il giudice (§8): positivi *supported*, negativi *not_enough_info*.

### 5.8 Costi

Per domanda: 1 + 1 + 1 + 1 chiamate brevi (~600 token) + 2 giudizi ≈ 4k token per 2 righe. 20.000 righe ≈ 40M token.

## 6. CG2C-Doc — claim multi-hop da documenti qualsiasi (FactCG)

### 6.1 L'idea

Come §5, ma senza domande di partenza: il cammino si sceglie nel grafo del documento.

### 6.2 Input

Documenti in prosa del train, 150–1.500 parole: pagine di HoVer e FEVER (il loro `document`), documenti di D2C (notizie, trascrizioni). `doc_id` ereditato.

### 6.3 Passo 1 — Grafo del documento

Il documento si divide in frasi numerate e si usa il prompt `cg2c_graph` (§5.4) su tutte le frasi. Per documenti oltre ~60 frasi, si lavora su finestre di 30 frasi consecutive.

### 6.4 Passo 2 — Scelta dei cammini (codice, nessun LLM)

```python
import networkx as nx, random
def linear_paths(triples, min_hops=2, max_hops=4, max_paths=3, rng=random.Random(13)):
    """Cammini semplici di 2-4 triple che toccano almeno 2 frasi diverse."""
    g = nx.MultiGraph()
    for t in triples:
        g.add_edge(t["e1"].lower(), t["e2"].lower(), triple=t)
    found = []
    nodes = list(g.nodes)
    rng.shuffle(nodes)
    for src in nodes[:50]:
        for dst in nodes:
            if src >= dst:
                continue
            for path in nx.all_simple_edge_paths(g, src, dst, cutoff=max_hops):
                if len(path) < min_hops:
                    continue
                ts = [g.edges[e]["triple"] for e in path]
                if len({t["sentence"] for t in ts}) >= 2:
                    found.append(ts)
            if len(found) > 200:
                break
    rng.shuffle(found)
    return found[:max_paths]
```

Un cammino semplice è per definizione **lineare** (ogni nodo interno ha due archi nel cammino): FactCG ha trovato che i sottografi lineari funzionano un po' meglio di quelli ramificati, perché un claim ramificato si può scomporre mentalmente in claim indipendenti a un passo.

### 6.5 Passo 3 — Claim dal cammino (`cg2c_claim_from_path`)

```text
Triples (each with the sentence it comes from):
- ({e1}, {e2}, {relation})   [sentence: "{sentence text}"]
- ...

Write ONE fluent claim that states the information of ALL these triples together, as a chain
(mention every entity of the chain; you may refer to an intermediate entity by description,
e.g. "the seminary where she teaches", when it reads more naturally).
Do not add any information that is not in the triples.
Return JSON: {"claim": "<claim>"}
```

Esempio:

> Triple: (Nancy R. Howell, Saint Paul School of Theology, professor at) da S3; (Saint Paul School of Theology, United Methodist Church, affiliated with) da S9.
> Claim: "Nancy R. Howell is a professor at a seminary affiliated with the United Methodist Church." → 1

### 6.6 Passo 4 — Negativo

Una tripla del cammino scelta a caso → prompt `cg2c_remove_relation` (§5.6) sulla sua frase → `(doc_neg, claim, 0)`.

### 6.7 Filtro e verifica

Come §5.7. In `meta`: `hops`, `path` (le triple), `removed_triple`.

## 7. Negativi per modifica minima (prototipo esistente)

Già implementati in [llm_gen/tasks.py](../../llm_gen/tasks.py). Prompt esatti:

### 7.1 `remove_evidence` — da supportato a non stabilito

```text
Document:
<<<
{document}
>>>

Claim:
<<<
{claim}
>>>

The document currently supports the claim. Edit the document so that it NO LONGER supports the claim,
by DELETING the minimum amount of text: remove the sentences, clauses or lines that establish at least
one part of the claim.
Rules:
- Only delete text. Do not add, rewrite, paraphrase or reorder anything that remains.
- After the edit the document must not contradict the claim either: that part of the claim must simply
  be not established (missing evidence).
- Keep everything unrelated to the claim exactly as it is.
- If the claim has several parts, removing the evidence for one part is enough.
- If other passages still imply the removed fact, delete them too.
Return JSON:
{"edited_document": "<the full edited document>",
  "removed_text": ["<each deleted passage>"],
  "unsupported_part": "<the part of the claim that is no longer established>"}
```

Controlli automatici: documento più corto; ≥ 98% delle parole rimaste già presenti nell'originale (solo cancellazioni); `removed_text` non vuoto. Giudice: deve dire *not_enough_info* (se dice *contradicted*, la cancellazione ha creato una contraddizione: scartare).

Uso consigliato: righe supportate con documento **lungo** (SummEdits, D2C, FEVER) — insegna a dire "no" quando manca un dettaglio in un testo di migliaia di parole.

Limite: per documenti molto lunghi l'LLM deve riscrivere tutto il documento nella risposta. Oltre ~3.000 parole conviene chiedere solo `removed_text` e fare la cancellazione in codice (ricerca esatta del passaggio nel documento; se non si trova parola per parola, scartare).

### 7.2 `contradict_claim` — da supportato a contraddetto

```text
{stesso blocco Document / Claim}

The document supports the claim. Rewrite the claim with ONE minimal change so that the document clearly
CONTRADICTS it: replace one entity, number, date, quantity, place, relation or attribution with a value
that the document explicitly rules out.
Rules:
- Keep the rest of the claim word for word, and keep it fluent.
- The contradiction must be checkable from the document alone, without world knowledge.
- Do not simply insert a negation ("not", "never") unless no other change is possible.
Return JSON:
{"contradicted_claim": "<new claim>", "changed_from": "<original words>", "changed_to": "<new words>",
  "evidence": "<the sentence of the document that contradicts the new claim, copied exactly>"}
```

Controlli: claim cambiato; differenza ≤ 8 parole; la frase di evidenza compare nel documento; segnalazione se l'unica differenza è una negazione aggiunta (si tengono al massimo il 10% di questi). Giudice: *contradicted*.

Comandi del prototipo (API NVIDIA, per prove):

```sh
python llm_gen/run.py --task remove_evidence --source data/finetune/splits/train/summedits.jsonl --n 20
python llm_gen/run.py --task contradict_claim --source data/finetune/splits/train/hover.jsonl --n 20
```

## 8. Il giudice (`verify_label`)

Prompt (identico a quello del prototipo):

```text
{stesso blocco Document / Claim}

Judge the claim using ONLY the document (no outside knowledge).
- "supported": every part of the claim is established by the document (faithful paraphrases and direct
  implications count).
- "contradicted": the document states something incompatible with at least one part of the claim.
- "not_enough_info": not contradicted, but at least one part is not established.
Return JSON: {"verdict": "supported" | "contradicted" | "not_enough_info",
  "explanation": "<one sentence>"}
```

Regole di accettazione:

| Famiglia / tipo | Si tiene se il giudice dice |
|---|---|
| positivi (C2D, D2C, CG2C) | `supported` |
| negativi per frase o relazione tolta (C2D, D2C, CG2C, `remove_evidence`) | `not_enough_info` (con `contradicted` si scarta) |
| `contradict_claim` | `contradicted` |

Giudice a temperatura 0. Per i casi importanti (un campione di 500 per famiglia) si può chiedere il giudizio due volte con l'ordine Document/Claim invertito e tenere solo i casi concordi.

Tasso di scarto atteso: 10–30%. **Oltre il 40%** il prompt di generazione va rivisto (di solito: il generatore scrive i fatti in chiaro, o cancella troppo poco).

## 9. Raccolta finale

Script `llm_gen/collect.py` (da scrivere): legge gli output dei passi di ogni famiglia, applica le regole di §8, deduplica (stesso documento e stesso claim), e scrive `data/finetune/raw/<famiglia>.jsonl` più `<famiglia>.stats.json` con: righe per etichetta, quota scartata per motivo, distribuzione degli hop (CG2C), parole del documento p50/p95.

Le famiglie generate stanno nel train: `make_mix.py` le legge come fonti ([02e](02e_mix_finale.md)).

## 10. Controllo di qualità a mano

Per ogni famiglia, 30 esempi accettati scelti a caso, letti con questa scheda:

| Domanda | Sì / no |
|---|---|
| L'etichetta è corretta? | |
| Per i negativi: è chiaro quale parte del claim manca o è contraddetta? | |
| Il documento è naturale (non una lista di frasi incollate)? | |
| Il claim è naturale e non copia una frase del documento parola per parola? | |

Soglia: meno di 3 etichette sbagliate su 30. Altrimenti si rivede il prompt e si rigenera.

## 11. Volumi e calendario

| Famiglia | Pilota | Corsa piena | Tempo indicativo (L40S) |
|---|---:|---:|---|
| MiniCheck rilasciato (C2D + D2C) | 14k | 14k | già pronto |
| C2D esteso | 0 | 10–15k | ~0,5 giorni |
| D2C esteso | 3k | 15–20k | ~1 giorno |
| CG2C-MHQA | 0 | 10k | ~0,5 giorni |
| CG2C-Doc | 0 | 6k | ~0,5 giorni |
| `remove_evidence` su documenti lunghi | 2k | 5k | ore |
| `contradict_claim` | 2k | 4k | ore |
| Giudice su tutto | — | — | ~0,5–1 giorno su 2 L40S |

Ordine consigliato: D2C (riguarda il dominio più debole), poi C2D, poi CG2C.

## Criteri per passare allo step successivo

- Per ogni famiglia: tasso di accettazione misurato su ≥ 200 esempi; 30 letti a mano con < 3 errori.
- Documenti D2C decontaminati contro AggreFact (quota scartata registrata).
- File `data/finetune/raw/<famiglia>.jsonl` con `doc_id` ereditato e `meta.family`.
