# Passo 1 — I modelli e gli hook per frase

## In breve

Serve un modulo, `evidence/models.py`, che per ciascun modello faccia **un forward** con il codice dei suoi autori e restituisca, dallo stesso forward:

1. i punteggi delle due opzioni nella posizione di risposta → P(supportato), come nel fine-tuning binario;
2. per ogni frase del documento, in **4 layer**: il vettore dell'ultimo token della frase (`h_end`) e la media dei vettori dei suoi token (`h_mean`);
3. per gli 8 layer ad attenzione piena, quanto la posizione di risposta "guarda" ogni frase (somma dei pesi di attenzione sui suoi token, per testa);
4. il vettore della posizione di risposta (`q`), negli stessi 4 layer.

I punti 2–4 si ottengono con degli *hook*: piccole funzioni agganciate ai layer, che copiano i dati mentre il forward passa, senza cambiarlo. Il riassunto per frase si calcola subito sulla GPU: non si conservano mai i vettori di tutti i token.

## 1. Pacchetti

Ambiente `factcheck` della repo (transformers 5.14, torch, accelerate, peft), più:

```sh
pip install flash-linear-attention causal-conv1d safetensors nltk rank_bm25 sentence-transformers scikit-learn
python -c "import nltk; nltk.download('punkt_tab')"      # dati del divisore in frasi
chmod -R go-w ~/nltk_data                                 # NLTK rifiuta una cartella dati scrivibile da altri utenti
```

`flash-linear-attention` e `causal-conv1d` servono ai 24 layer Gated DeltaNet di Qwen3.5: senza, transformers usa un'implementazione lenta in torch.

## 2. L'architettura comune: Qwen3.5

| | Gruppo 4B | Gruppo 9B |
|---|---|---|
| Layer | 32 | 32 |
| Layer ad attenzione piena | 3, 7, 11, 15, 19, 23, 27, 31 | idem |
| Altri layer | 24 Gated DeltaNet: portano avanti un riassunto del testo letto, sempre causali | idem |
| Teste di query / di chiavi (layer pieni) | 16 / 4 | 16 / 4 |
| Lunghezza del vettore (d) | 2.560 | 4.096 |
| Tokenizer | Qwen3.5 (comune a tutti) | idem |

Layer da cui si leggono i vettori delle frasi: **19, 23, 27, 31**, gli ultimi quattro ad attenzione piena (il 31 è anche l'ultimo layer). Perché questi e come verificarli con una sonda: passo [15](15_recap_architettura_layer_loss.md) §3.

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

## 5. Hook sui vettori delle frasi

Per ognuno dei 4 layer, un hook prende l'uscita del layer (un vettore per token) e calcola subito, sulla GPU:
- `q`: il vettore della posizione di risposta;
- `end`: il vettore dell'ultimo token di ogni frase;
- `mean`: la media dei vettori dei token di ogni frase, con le somme cumulative (un solo passaggio per tutte le frasi).

Gli intervalli di token delle frasi (`spans`) vengono dal rendering ([06](06_rendering_ed_estrazione.md) §3).

```python
import torch


class SentenceTap:
    """Per ogni layer scelto: vettore della posizione di risposta, dell'ultimo token di ogni frase, media dei token di ogni frase."""
    def __init__(self, layers, layer_idx):
        self.out, self.spans, self.answer_pos = {}, None, None       # spans: LongTensor [n, 2], token del prompt, estremi inclusi
        self.handles = [layers[i].register_forward_hook(self._hook(i)) for i in layer_idx]

    def _hook(self, i):
        def hook(module, args, output):
            h = (output[0] if isinstance(output, tuple) else output)[0]          # [T, D]
            s = self.spans
            csum = torch.cat([h.new_zeros(1, h.size(1), dtype=torch.float), h.float().cumsum(0)])   # somme cumulative
            mean = (csum[s[:, 1] + 1] - csum[s[:, 0]]) / (s[:, 1] - s[:, 0] + 1)[:, None]
            self.out[i] = {"q": h[self.answer_pos].float(), "end": h[s[:, 1]], "mean": mean.to(h.dtype)}
        return hook


def attention_per_sentence(rows, spans):
    """rows: {layer: [16, T]} righe di attenzione della posizione di risposta; spans: LongTensor [n, 2].
    -> [n, 8 × 16]: per ogni frase, la somma dei pesi sui suoi token, per layer e testa."""
    feats = []
    for layer in sorted(rows):
        csum = torch.cat([rows[layer].new_zeros(rows[layer].size(0), 1), rows[layer].float().cumsum(1)], 1)   # [16, T+1]
        feats.append((csum[:, spans[:, 1] + 1] - csum[:, spans[:, 0]]).T)                                     # [n, 16]
    return torch.cat(feats, 1)
```

Provato con un modello finto: la media e l'ultimo token coincidono con il calcolo diretto, e la somma dell'attenzione per frase con la somma sui token della frase, per ogni layer e testa.

## 6. Hook sull'attenzione

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

Dalle righe di pesi alla feature per frase: `attention_per_sentence` (§5) somma i pesi sui token di ogni frase, per ognuno degli 8 layer e delle 16 teste: 128 numeri per frase.

## 7. Che cosa restituisce `evidence/models.py`

```python
def forward_sentences(model, rendered):
    """rendered: input_ids, answer_pos, sent_tok_spans (primo e ultimo token di ogni frase nel prompt) -> dict:
       answer_logits [2]          punteggi delle opzioni sì/no (per P(supportato))
       q      [4, D]              vettore della posizione di risposta, layer 19, 23, 27, 31
       h_end  [n, 4, D]           vettore dell'ultimo token di ogni frase
       h_mean [n, 4, D]           media dei token di ogni frase
       att    [n, 128]            attenzione della posizione di risposta sulle frasi
       n_tok  [n]                 numero di token di ogni frase"""
```

## 8. Test

`python -m evidence.models --model knowline-4b --selftest`, poi per gli altri modelli:

| Test | Come | Deve dare |
|---|---|---|
| Verdetto invariato | 50 righe di dev, prompt senza preambolo: P(supportato) dal forward con gli hook contro l'adapter del benchmark | differenza media < 0,005, massima < 0,02 |
| Layer giusti | vettore dell'hook al layer i = `hidden_states[i + 1]` di transformers | identico |
| Attenzione giusta | 5 prompt da ~1.000 token, contro la modalità "eager" con la stessa maschera | differenza < 0,001 |
| Maschera conservata | per i modelli bidirezionali, il verdetto con e senza hook | identico |
| Riassunti delle frasi | `end` e `mean` contro il calcolo diretto sugli stati completi | differenza < 1e-3 |
| Memoria | forward da 8.000 e 16.000 token con gli hook | entra nella GPU (§9) |

## 9. GPU e spazio

| Uso | Gruppo 4B | Gruppo 9B |
|---|---|---|
| Estrazione dei vettori | 24 GB | 45 GB per documenti lunghi |
| Training con il modello in memoria (fase 1b) | 24 GB | 45 GB |
| Training anche del modello (fase 2) | 24 GB | 45 GB |

Spazio su disco, 4 layer, bf16: per frase 2 × 4 × D × 2 byte (82 KB su KnowLine-4B, 131 KB su elephant-1). Per le ~28.600 righe di SciFact, WiCE e HoVer (~52 frasi in media) circa 120 GB su KnowLine-4B; con 2 layer la metà.
