# Step 3 — Rendering: dal mix ai token, esattamente come in produzione

## Obiettivo

Trasformare ogni riga del mix in un esempio pronto per il training, per ciascun modello:

- `input_ids`: il prompt **identico, token per token**, a quello che il modello vede nei benchmark e in produzione;
- la **maschera di attenzione** del modello (causale, bidirezionale sul prompt, a blocchi);
- `answer_pos`: la posizione in cui si legge la risposta;
- come si leggono le due opzioni: righe della testa del vocabolario (`yes_ids`, `no_ids`) o la testa propria del modello;
- `label` e `take`: quale risposta è giusta (con una domanda negativa un claim supportato vuole "no").

E definire come si gestiscono i documenti più lunghi del budget, in training e in inferenza.

## 1. Perché il prompt deve essere identico

Il modello ha imparato a rispondere su un formato preciso, con una maschera precisa. Se in training vedesse un prompt o una maschera diversi da quelli di produzione, si addestrerebbe un modello diverso da quello che poi si usa. Per questo **non si riscrive niente**: si importano le funzioni degli autori, che gli adapter della repo (`models/<modello>/adapter.py`) usano già e hanno verificato.

Attenzione in particolare alla maschera: per i modelli bidirezionali, caricare i pesi con transformers "normale" dà un modello **causale**, che risponde in modo diverso **senza dare errori** (lo dicono le model card di StartLux, elephant-1 e Jeffan). Il test di identità di §8 lo intercetta.

## 2. I modelli, raggruppati per come leggono la risposta

### 2.1 Lettura dalla testa del vocabolario, attenzione causale

| Modello | Codice degli autori (già usato dall'adapter) | Prompt | Opzioni lette |
|---|---|---|---|
| KnowLine-4B | `knowline_server.py`, `knowline_layout.py` (nel repository del modello): `render_request`, `score` | stato come testo indentato; turno con istruzione, `Question:`, `A. Yes: …` / `B. No: …`; risposta aperta con `Answer:`. Richieste fino a 3.000 token **scritte due volte** ("Read the same input again before answering:") | token delle etichette A/B; temperatura già nei pesi (T = 1) |
| OneJev1.6-9B | renderer di Flide (famiglia `onejev-labels-v2`) | prefisso comune (sistema + stato) e un suffisso per domanda: `Question: …`, `A. yes: …`, `B. no: …`, `Answer with one letter: A, B.` | lettere A/B (`slot_lm`), T = 1 |
| Nimble-9B-v3 | `models/nimble-9b/nimble` (copia del motore degli autori) | sistema; un turno utente JSON con contesto e schema di tutte le domande; `Requested field: "<id>"` | codici `A` = false, `B` = true fra 255 candidati; T = 1 |

### 2.2 Lettura dalla testa del vocabolario, attenzione bidirezionale sul prompt

Nei layer ad attenzione piena (3, 7, …, 31) ogni token del prompt vede anche quelli successivi; i 24 layer DeltaNet restano causali.

| Modello | Codice degli autori | Prompt | Maschera | Opzioni lette |
|---|---|---|---|---|
| elephant-1 | `models/elephant-1/qev` (motore OneJev `qev` con `serve.patch`) | sistema + `<state>` JSON indentato `</state>` come prefisso; un suffisso per domanda (`A. yes: …`, `B. no: …`, `Answer with one letter: A, B.`) | a blocchi: il prefisso vede sé stesso, ogni suffisso vede prefisso e sé stesso. Dichiarata in `modeling_overrides` del `config.json`: il motore la legge solo da una cartella locale | lettere A/B, testa in float32, T = 1 |
| StartLux-9B / 4B | pacchetto `startlux-decide` (gira come server vLLM con un plugin) | sistema, `Evidence:` con lo stato JSON compatto, `Question:`, `A) yes: …`, `B) no: …` | bidirezionale su tutto il prompt | lettere A/B; T per tipo (sì/no: 1,3396 il 9B, 1,2662 il 4B) |
| d3-flash | codice del modello (`trust_remote_code`): `prepare`, `run` | sistema, `State:` JSON, `Question:`, `A: No / false`, `B: Yes / true`, "Reply with only the code of the best option." | `noncausal_full_attention` | codici A/B fra 255; T = 1 |
| Jeffan-4B | `inference/jeffan_engine` (nel repository del modello) | sistema, `State:` JSON, `Question:`, `A: No / false`, `B: Yes / true` | bidirezionale nei layer pieni, una maschera per tipo di layer (`dm_config.json`) | codici A/B dalla testa del vocabolario; T = 1 |

StartLux è l'unico modello che gli autori eseguono solo con vLLM: per il training la maschera bidirezionale va riprodotta in transformers (§4) e verificata contro le risposte del server (§8).

### 2.3 Lettura da una testa propria

| Modello | Codice degli autori | Prompt | Maschera | Lettura |
|---|---|---|---|---|
| pplx-decider-flash | pacchetto `pplx_decider` 0.3.0 (nel repository), percorso di riferimento (`fast=False`) | tutte le domande in una sequenza, ognuna letta nella sua posizione; `A: No / false`, `B: Yes / true` | non causale nei layer pieni (installata con un hook) | testa a 255 vie (`readout.safetensors`, [255, 2.560]) sullo stato nella posizione di lettura, ristretta alle opzioni valide; T = 1,554 |
| Clef-Flash | `joint_schema_model.py` (nel repository) | sistema, `STATE:` JSON, `SCHEMA FIELDS:` con le domande e le opzioni, `JOINT SCHEMA DECISIONS:` | come caricata dal codice (non dichiarata) | testa congiunta (4 layer, larghezza 1.024) sugli stati finali: un logit per opzione; T = 1 |

### 2.4 I due backbone (controlli di B-ctrl)

Qwen3.5-9B e Qwen3.5-4B con il prompt neutro dell'adapter `llm-backbone` (documento, claim, domanda, `A. yes` / `B. no`, "Answer with the letter of the correct option only.", template di chat con thinking spento). In training l'ordine delle opzioni si sceglie **a caso per ogni esempio** (metà `A. yes`, metà `A. no`), così il modello non impara una preferenza di posizione; in valutazione si usano entrambi gli ordini e si fa la media (lettura R1 dello studio, senza la correzione del prior).

## 3. Una sola domanda per richiesta

Il training usa la domanda di `finetune/question.json`, da sola. Per i modelli con prefisso comune e un suffisso per domanda (elephant-1, OneJev, Nimble) la richiesta diventa una sola sequenza: prefisso + un suffisso. Per i modelli che leggono tutte le domande insieme (pplx, Clef) la richiesta contiene solo quella domanda: è la composizione C1 dello studio, che l'esperimento E4d confronta con le tre domande insieme.

## 4. Il codice: `finetune/render.py`

Un renderer per modello, con la stessa interfaccia. Ognuno **chiama le funzioni degli autori** (le stesse importate dall'adapter) e restituisce anche ciò che serve al forward di training:

```python
from dataclasses import dataclass

@dataclass
class Rendered:
    input_ids: list[int]
    answer_pos: int                  # posizione in cui si legge la risposta
    yes_ids: list[int] | None        # righe della testa del vocabolario (None per pplx e Clef)
    no_ids: list[int] | None
    mask: dict                       # {"kind": "causal"} | {"kind": "bidirectional_prompt", "prompt_len": n}
                                     # | {"kind": "blocks", "blocks": [[0, p], [p, q]]}  (applicata solo ai layer pieni)
    head_inputs: dict | None         # pplx / Clef: ciò che serve alla loro testa (posizione, opzioni valide)
    temperature: float

class Renderer:
    model_name: str
    def render(self, state: dict, question: dict) -> Rendered: ...
    def forward_model(self, model, batch):  # forward di training con la maschera giusta -> logit delle due opzioni [B, 2]
        ...
```

Per ogni modello `forward_model` fa ciò che fa il codice degli autori in inferenza, ma con i gradienti attivi:

| Modello | Come si applica la maschera in training |
|---|---|
| KnowLine-4B, OneJev, Nimble, backbone | maschera causale standard |
| elephant-1 | si carica il modello con il motore `qev` (che legge `modeling_overrides`) e si verifica che il suo forward supporti i gradienti; altrimenti si costruisce la maschera a blocchi e la si passa ai soli layer pieni |
| StartLux-9B / 4B | maschera bidirezionale sul prompt costruita da noi e passata ai soli layer pieni (via `AttentionInterface` di transformers, come per la cattura dell'attenzione nel piano dell'evidenza) |
| d3-flash, Jeffan-4B, pplx | il codice degli autori costruisce la maschera (per tipo di layer, o con un hook): si riusa il loro forward |
| Clef-Flash | il forward di `joint_schema_model.py` |

## 5. Formato di un esempio renderizzato

```json
{"id": "anli:train:...", "source": "anli_hard", "model": "knowline-4b",
 "input_ids": [248045, 846, ...], "answer_pos": 211,
 "yes_ids": [32], "no_ids": [33], "mask": {"kind": "causal"}, "temperature": 1.0,
 "take": "no", "label": 0, "neg_type": "contradicted", "teacher_p": 0.83, "pair_id": null}
```

(Id dei token illustrativi: si ricavano dal codice degli autori e si registrano.)

- `take`: `yes` se P(supportato) = P(sì) (Q1, Q4), `no` se P(supportato) = P(no) (Q2).
- `teacher_p`: P(supportato) del modello di partenza, solo se si usa l'ancoraggio ([02c](02c_replay_e_forgetting.md)).
- `pair_id`: per la loss contrastiva opzionale ([04](04_loss.md) §5).

## 6. Lo splitter di frasi: `finetune/sentences.py`

Usato da chunking, D2C, CG2C. Una sola funzione per tutto il progetto:

```python
import re
from nltk.tokenize import sent_tokenize      # serve il modello punkt_tab: nltk.download("punkt_tab")

def split_sentences(text):
    """-> lista di (inizio, fine) in caratteri. Ogni riga è un confine; le righe 'Title: ...'
    sono una frase a sé; dentro una riga le frasi sono quelle di NLTK."""
    spans = []
    for m in re.finditer(r"[^\n]+", text):
        line, base = m.group(0), m.start()
        if not line.strip():
            continue
        if line.startswith("Title: "):
            spans.append((base, m.end()))
            continue
        cursor = 0
        for sent in sent_tokenize(line):
            start = line.find(sent, cursor)
            if start < 0:
                continue
            spans.append((base + start, base + start + len(sent)))
            cursor = start + len(sent)
    return spans
```

Verificata su 1.500 documenti reali (300 ciascuno di HoVer, FEVER, SummEdits, MiniCheck, AVeriTeC): nessun carattere non spazio perso, nessuna sovrapposizione. Limite noto di NLTK: abbreviazioni come "b. ca." possono spezzare una frase; non è un problema per il chunking.

## 7. Documenti oltre il budget di token

### 7.1 In training

| Fase | Budget del prompt |
|---|---|
| Pilota (KnowLine-4B sul portatile) | 8.192 token |
| Corsa piena (L40S) | 8.192 token (16.384 per i modelli 4B se la memoria basta) |

Regola `fit_budget` per una riga il cui prompt supera il budget:

| Etichetta | Cosa fare | Perché |
|---|---|---|
| 0 | si divide il documento in chunk di frasi che stanno nel budget (§7.2) e si tiene **un** chunk scelto a caso, con etichetta 0 | se il documento intero non supporta il claim, nessuna sua parte lo supporta |
| 1 | **si scarta** la riga | un chunk potrebbe non contenere l'evidenza: l'etichetta 1 non sarebbe più garantita |

Si contano le righe scartate e quelle ridotte per fonte. Se una fonte perde più dell'1% dei positivi, si alza il budget o si accetta e si dichiara.

### 7.2 Chunking per frasi (training e inferenza)

```python
def chunk_by_tokens(text, claim, renderer, budget):
    """Raggruppa frasi consecutive finché il prompt (con questo claim) sta nel budget."""
    spans = split_sentences(text)
    chunks, cur = [], []
    def fits(parts):
        doc = " ".join(text[a:b] for a, b in parts)
        return len(renderer.ids({"document": doc, "claim": claim})) <= budget
    for sp in spans:
        if cur and not fits(cur + [sp]):
            chunks.append(cur); cur = [sp]
        else:
            cur.append(sp)
    if cur:
        chunks.append(cur)
    return [" ".join(text[a:b] for a, b in c) for c in chunks]
```

(Per efficienza, invece di rirenderizzare a ogni frase si può contare i token di ogni frase una volta e sommare, tenendo ~250 token di margine per il resto del prompt.)

### 7.3 In inferenza

```text
se il prompt intero sta in C token:   P = P(supportato | documento intero)
altrimenti:                           chunk di frasi ≤ C token;  P = max sui chunk
```

È la regola di MiniCheck e FactCG (che usano chunk di ~400–550 parole perché i loro encoder hanno contesto corto). Per un LLM i chunk possono essere molto più grandi: C si sceglie su dev_select fra {8.192, 16.384, 32.768} (esperimento K1); 32.768 è il limite dei runtime degli autori (oltre, rifiutano la richiesta). Più grande è C, meno catene di evidenza vengono spezzate fra due chunk; il costo cresce con la lunghezza.

I benchmark attuali **troncano** a 80.000 caratteri (75% inizio + 25% fine con `[Middle of long document omitted]`) invece di dividere: per la valutazione finale si usa la regola scelta in K1 e si dichiara.

## 8. Test obbligatori di `render.py`

| Test | Come | Atteso |
|---|---|---|
| **Identità con il benchmark** (il più importante) | 50 righe di `dev_select`: P(supportato) dal nostro `forward_model` (in eval, senza LoRA) contro le probabilità dell'adapter del benchmark sulla stessa riga e la stessa domanda | differenza media < 0,005, massima < 0,02 |
| Maschera | per i modelli bidirezionali, lo stesso calcolo con maschera causale deve dare probabilità **diverse** (se coincidono, la maschera non viene applicata) | differenza media > 0,01 |
| Opzioni | `yes_ids` / `no_ids` (o le opzioni della testa) uguali a quelli usati dal codice degli autori | identici |
| Target | con `take = yes` label 1 vuole "sì"; con `take = no` label 1 vuole "no" | |
| Frasi | ricopertura del documento senza sovrapposizioni | |
| Budget | un documento lungo produce chunk entro il budget; i positivi oltre il budget vengono scartati | |
| KnowLine "letto due volte" | una riga sotto i 3.000 token ha il documento due volte; una sopra, una volta | |

```python
def test_sentence_spans():
    for row in sample_rows(200):
        t = row["document"]; s = split_sentences(t)
        covered = "".join(t[a:b] for a, b in s)
        assert re.sub(r"\s", "", covered) == re.sub(r"\s", "", t)
        assert all(s[i][1] <= s[i + 1][0] for i in range(len(s) - 1))
```

Per StartLux il riferimento del test di identità è il server degli autori (vLLM con il plugin): è l'unico modo di sapere che la maschera ricostruita è quella giusta.

## 9. Statistiche da registrare

Per ogni mix e modello: esempi per fonte, lunghezza in token p50/p95/max, righe scartate o ridotte per budget, quota di target positivi, per KnowLine quota di righe lette due volte.

## Criteri per passare allo step successivo

- Tutti i test di §8 superati per il modello da addestrare (prima KnowLine-4B, poi elephant-1, poi gli altri).
- `rendered/<modello>/fc_pilot/` prodotto con le statistiche.
