# Passo 1 — I modelli e gli hook per token

## In breve

Serve un modulo, `evidence/models.py`, che per ciascun modello faccia **un forward** con il codice dei suoi autori e restituisca, dallo stesso forward:

1. i punteggi delle due opzioni nella posizione di risposta → P(supportato), come nel fine-tuning binario;
2. i vettori di **tutti i token del documento** in 2 layer, ridotti subito a 256 numeri per layer (§5);
3. per gli 8 layer ad attenzione piena, quanto la posizione di risposta "guarda" **ogni token** (una riga di pesi di attenzione per testa);
4. il vettore della posizione di risposta (`q`).

I punti 2–4 si ottengono con degli *hook*: piccole funzioni agganciate ai layer, che copiano i dati mentre il forward passa, senza cambiarlo.

## 1. Pacchetti

Ambiente `factcheck` della repo (transformers 5.14, torch, accelerate, peft), più:

```sh
pip install flash-linear-attention causal-conv1d safetensors rank_bm25 sentence-transformers
```

`flash-linear-attention` e `causal-conv1d` servono ai 24 layer Gated DeltaNet di Qwen3.5: senza, transformers usa un'implementazione lenta in torch. Controllo: `python -c "import fla, causal_conv1d"`.

## 2. L'architettura comune: Qwen3.5

Tutti i modelli sono Qwen3.5 con la stessa struttura:

| | Gruppo 4B | Gruppo 9B |
|---|---|---|
| Layer | 32 | 32 |
| Layer ad attenzione piena | 3, 7, 11, 15, 19, 23, 27, 31 | idem |
| Altri layer | 24 Gated DeltaNet: portano avanti un riassunto del testo letto, sempre causali | idem |
| Teste di query / di chiavi (layer pieni) | 16 / 4 | 16 / 4 |
| Lunghezza del vettore (d) | 2.560 | 4.096 |
| Tokenizer | Qwen3.5 (comune a tutti) | idem |

Layer da cui si leggono i vettori dei token: **23 e 31** (penultimo e ultimo layer pieno; il 31 è anche l'ultimo layer). L'esperimento H6 ([11](11_esperimenti.md)) prova 19, 23, 27, 31.

## 3. Come si esegue ciascun modello

Ogni modello si esegue con il codice dei suoi autori: prompt, maschera di attenzione e lettura della risposta sono i loro. Caricare i pesi con transformers "normale" darebbe, per i modelli bidirezionali, un modello causale che risponde diversamente **senza dare errori**.

| Modello | Codice degli autori | Prompt (schema) | Attenzione | Lettura della risposta | Posizione di risposta |
|---|---|---|---|---|---|
| **KnowLine-4B** | front end nel repository del modello (`knowline_server.py`, `knowline_layout.py`), backend transformers | stato come testo indentato in un turno utente; turno con istruzione, `Question:`, `A. Yes: …` / `B. No: …`; risposta aperta con `Answer:`. Richieste fino a 3.000 token **scritte due volte** | causale | token A/B; temperatura 1,50 già nei pesi | ultimo token |
| **elephant-1** | motore `qev` di OneJev (copia in `models/elephant-1/qev`) | sistema + `<state>` JSON indentato `</state>` come prefisso; la domanda come suffisso con `A. yes` / `B. no` | bidirezionale a blocchi nei layer pieni (prefisso con sé stesso, suffisso con prefisso e sé stesso), dichiarata in `modeling_overrides` del config | lettere A/B | ultimo token del suffisso |
| Jeffan-4B | `inference/jeffan_engine` (nel repository) | `State:` JSON, `Question:`, `A: No / false`, `B: Yes / true` | bidirezionale nei layer pieni | codici A/B | ultimo token |
| d3-flash | codice del modello (`trust_remote_code`) | `State:` JSON, `Question:`, `A: No / false`, `B: Yes / true` | non causale nei layer pieni | codici fra 255 | ultimo token |
| OneJev1.6-9B | renderer di Flide; in transformers come sequenza unica | prefisso (sistema + stato) e suffisso della domanda | causale | lettere A/B | ultimo token |
| Nimble-9B-v3 | motore degli autori (`models/nimble-9b/nimble`), LoRA su Qwen3.5-9B | turno JSON con contesto e schema, `Requested field: "<id>"` | causale | codici fra 255 | ultimo token |
| StartLux 9B/4B | pacchetto `startlux-decide` (gli autori usano vLLM con un plugin; in transformers la maschera va ricostruita) | `Evidence:` JSON compatto, `Question:`, `A) yes` / `B) no` | bidirezionale sul prompt | lettere A/B, temperatura per tipo | ultimo token |
| pplx-decider-flash | pacchetto `pplx_decider` (percorso di riferimento) | domande in una sequenza, ognuna letta nella sua posizione | non causale nei layer pieni | testa propria a 255 vie | posizione di lettura della domanda |
| Clef-Flash | `joint_schema_model.py` | `STATE:` JSON, `SCHEMA FIELDS:` | come caricato | testa congiunta propria | ultimo token prima della testa |

## 4. Trovare i layer

```python
import torch

def text_layers(model, n_layers=32):
    """La ModuleList dei layer testuali (i modelli caricati come multimodali hanno anche una torre visiva)."""
    found = [m for name, m in model.named_modules()
             if isinstance(m, torch.nn.ModuleList) and len(m) == n_layers and "visual" not in name and "vision" not in name]
    assert len(found) == 1
    return found[0]

def full_attention_layers(config):
    tc = getattr(config, "text_config", config)
    return [i for i, t in enumerate(tc.layer_types) if t == "full_attention"]      # [3, 7, 11, 15, 19, 23, 27, 31]
```

## 5. La riduzione dei vettori (PCA fissa)

I vettori dei token sono tanti (~1.500 per un documento HoVer, fino a ~18.000 per WiCE) e lunghi (2.560 numeri). Salvarli così occuperebbe terabyte. Si riducono a 256 numeri per layer con una **PCA calcolata una volta** per modello e layer:

```python
def fit_pca(samples, k=256):
    """samples: [M, D] vettori di token presi a caso. -> (media [D], proiezione [D, k], quota di varianza spiegata)."""
    mean = samples.mean(0)
    x = samples - mean
    U, S, V = torch.pca_lowrank(x, q=k, center=False)
    explained = (S ** 2).sum() / (x ** 2).sum()
    return mean, V[:, :k], float(explained)
```

1. 2.000 righe a caso del mix di training; per ognuna, 200 token a caso del documento; 400.000 vettori per layer.
2. `fit_pca` per i layer 23 e 31.
3. Salvataggio in `data/evidence_tok/pca/<modello>/layer<i>.safetensors` (media e proiezione) con la varianza spiegata. Se nel layer 31 è sotto il 70%, si passa a 384 componenti.

La PCA serve solo a salvare i vettori per la fase 1a. Nella fase 1b, con il modello in memoria, la testa impara la propria proiezione partendo dalla PCA ([09](09_training.md) §4).

## 6. Hook sui vettori dei token

```python
class TokenTap:
    """Per ogni layer scelto: i vettori dei token del documento, proiettati a 256, e il vettore della posizione di risposta."""
    def __init__(self, layers, layer_idx, pca):
        self.out, self.doc, self.answer_pos = {}, None, None         # doc = (primo, ultimo) token del documento nel prompt
        self.pca = pca                                                # {layer: (media [D], proiezione [D, 256])} sulla GPU
        self.handles = [layers[i].register_forward_hook(self._hook(i)) for i in layer_idx]

    def _hook(self, i):
        def hook(module, args, output):
            h = (output[0] if isinstance(output, tuple) else output)[0]          # [T, D]
            a, b = self.doc
            mean, proj = self.pca[i]
            self.out[i] = {"tok": ((h[a:b + 1].float() - mean) @ proj).half(),  # [n, 256]
                           "q": h[self.answer_pos].float()}                      # [D]
        return hook
```

La proiezione si fa sulla GPU dentro l'hook: dalla GPU escono 256 numeri per token e per layer. Non si conservano mai i vettori completi di tutti i token.

## 7. Hook sull'attenzione

Le implementazioni veloci dell'attenzione non restituiscono i pesi, e chiederli in modalità "eager" costa troppa memoria con documenti lunghi. Si registra una funzione di attenzione che calcola l'attenzione normale e, negli 8 layer pieni, ricalcola **una riga** di pesi: quella della posizione di risposta verso tutti i token. Costa pochissimo.

```python
from transformers import AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward

CAPTURE = {"rows": {}, "full": {3, 7, 11, 15, 19, 23, 27, 31}, "query_pos": -1, "inner": sdpa_attention_forward}

def sdpa_capture(module, query, key, value, attention_mask, scaling=None, **kw):
    out = CAPTURE["inner"](module, query, key, value, attention_mask, scaling=scaling, **kw)
    if module.layer_idx in CAPTURE["full"]:
        qp = CAPTURE["query_pos"]
        sl = slice(qp, qp + 1 if qp != -1 else None)
        k = key.repeat_interleave(query.size(1) // key.size(1), dim=1)           # 16 teste di query, 4 di chiavi
        s = (query[:, :, sl] @ k.transpose(-1, -2)).float() * scaling            # [B, 16, 1, T]
        if attention_mask is not None:                                           # la maschera del modello, anche bidirezionale
            s = s + attention_mask[:, :, sl, : s.size(-1)].float()
        CAPTURE["rows"][module.layer_idx] = s.softmax(-1)[:, :, 0]               # [B, 16, T]
    return out

AttentionInterface.register("sdpa_capture", sdpa_capture)
# caricamento: from_pretrained(..., attn_implementation="sdpa_capture")
```

- In transformers 5.14 la classe di attenzione di Qwen3.5 passa per `AttentionInterface` e ha `layer_idx`; `query` e `key` arrivano già pronte (dopo le rotazioni di posizione), `scaling` arriva dal modello.
- Se il codice degli autori installa una propria funzione di attenzione per la maschera bidirezionale (pplx, Jeffan, d3-flash, elephant-1), la si mette in `CAPTURE["inner"]` al posto di `sdpa_attention_forward`: la maschera resta la loro.

Dalla riga di pesi alla feature per token: `att[t] = [riga del layer l, testa j, posizione t]` per gli 8 layer e le 16 teste, cioè 128 numeri per token del documento.

## 8. Che cosa restituisce `evidence/models.py`

```python
def forward_tokens(model, rendered):
    """rendered: input_ids, answer_pos, doc_tok (primo, ultimo token del documento) -> dict:
       answer_logits [2]  punteggi delle opzioni sì/no (per P(supportato))
       h   [n, 2, 256]    vettori dei token del documento, layer 23 e 31, proiettati
       att [n, 128]       attenzione della posizione di risposta sui token del documento
       q   [2, D]         vettore della posizione di risposta, layer 23 e 31"""
```

## 9. Test

`python -m evidence.models --model knowline-4b --selftest`, poi per gli altri modelli:

| Test | Come | Deve dare |
|---|---|---|
| Verdetto invariato | 50 righe di dev, prompt senza preambolo: P(supportato) dal forward con gli hook contro l'adapter del benchmark | differenza media < 0,005, massima < 0,02 |
| Layer giusti | vettore dell'hook al layer i (prima della PCA) = `hidden_states[i + 1]` di transformers | identico |
| Attenzione giusta | 5 prompt da ~1.000 token, contro la modalità "eager" con la stessa maschera | differenza < 0,001 |
| Maschera conservata | per i modelli bidirezionali, il verdetto con e senza hook | identico |
| Documento giusto | i token `doc_tok` decodificati sono il documento (a meno dell'escaping del prompt) | 200 righe su 200 |
| PCA | proiezione nell'hook = proiezione dopo, sugli stati completi | differenza < 1e-2 (fp16) |

## 10. GPU e spazio

| Uso | Gruppo 4B | Gruppo 9B |
|---|---|---|
| Estrazione dei vettori | 24 GB | 45 GB per documenti lunghi |
| Training con il modello in memoria (fase 1b) | 24 GB | 45 GB |
| Training anche del modello (fase 2) | 24 GB | 45 GB |

Spazio su disco per riga, 2 layer: 1 KB per token di `h` + 256 B di `att`. Per le ~28.600 righe di SciFact, WiCE e HoVer: circa 60 GB.
