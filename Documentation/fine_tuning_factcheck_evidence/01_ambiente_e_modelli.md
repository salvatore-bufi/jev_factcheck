# Passo 1 — Caricare i modelli e leggere vettori e attenzione

## In breve

Serve un modulo, `evidence/models.py`, che per ciascuno dei tre modelli faccia **un forward** su un prompt e restituisca tre cose:

1. i punteggi di "sì" e "no" nella posizione di risposta → P(supportato), come nel piano binario;
2. alcuni **vettori** (stati nascosti) letti durante il forward: quelli dei token di ogni frase, in 4 layer, e quello della posizione di risposta;
3. per gli 8 layer che guardano tutto il testo, **quanto la posizione di risposta guarda ciascun token** (una riga di pesi di attenzione).

I punti 2 e 3 si ottengono con degli *hook*: piccole funzioni agganciate ai layer, che copiano i dati mentre il forward passa, senza cambiarlo.

## 1. Pacchetti

Lo stesso ambiente del benchmark (`factcheck`), più:

```sh
pip install nltk rank_bm25 safetensors sentence-transformers
python -c "import nltk; nltk.download('punkt_tab')"
pip install flash-linear-attention causal-conv1d     # solo per JPT-9B: senza, i suoi layer DeltaNet girano molto lenti
```

## 2. I tre modelli

| | d1-3B | JPT-9B | Winnow-12B |
|---|---|---|---|
| Pesi | `LiquidAI/d1-3B` @ `051bcc4` | `kirp/jpt-9b` @ `b447cc7` | `models/winnow-12b-bf16/model/` (creato da `convert.py`) |
| Verdetto | quello del piano binario | idem | idem |
| Caricamento | `AutoModel.from_pretrained(..., trust_remote_code=True)` | backend transformers di `llm2jev` | `AutoModelForCausalLM.from_pretrained(path)` |
| Prompt | renderer del piano binario | idem | idem |
| Risposta "sì" / "no" (id dei token) | {11683, 12447} / {2243, 4547, 19598} | 357 / 417 | 236799 / 236776 |
| Posizione di risposta | ultimo token del prompt | ultimo token | ultimo token |
| Lunghezza del vettore (d) | 2.048 | 4.096 | 3.840 |
| Layer totali | 30 | 32 | 48 |
| Layer che guardano tutto il testo | 2, 5, 9, 13, 17, 21, 24, 27 | 3, 7, 11, 15, 19, 23, 27, 31 | 5, 11, 17, 23, 29, 35, 41, 47 |
| Teste di attenzione in quei layer | 32 | 16 | 16 |
| 4 layer da cui leggere i vettori (partenza) | 21, 24, 27, 29 | 19, 23, 27, 31 | 29, 35, 41, 47 |

I renderer (le funzioni che costruiscono il prompt) e i token delle risposte sono quelli del piano binario ([../fine_tuning_factcheck/03_rendering_formato.md](../fine_tuning_factcheck/03_rendering_formato.md)): si importano, non si riscrivono. d1-3B e JPT-9B sono modelli anche per immagini: si usa solo la parte testuale.

Perché quei 4 layer: sono gli ultimi layer che guardano tutto il testo più l'uscita finale. Dopo l'ultimo di questi layer l'informazione fra frasi lontane non si mescola più. È solo il punto di partenza: l'esperimento H2a ([11](11_esperimenti.md)) sceglie quali servono davvero.

## 3. Trovare i layer nel codice del modello

I nomi interni dei moduli cambiano da modello a modello. Si cerca la lista dei layer testuali per lunghezza e si leggono i tipi di layer dal config:

```python
def text_layers(model, n_layers):
    found = [m for name, m in model.named_modules()
             if isinstance(m, torch.nn.ModuleList) and len(m) == n_layers and "vision" not in name]
    assert len(found) == 1
    return found[0]

def full_attention_layers(config):
    tc = getattr(config, "text_config", config)
    return [i for i, t in enumerate(tc.layer_types) if t == "full_attention"]
```

## 4. Hook sui vettori

Per ognuno dei 4 layer, un hook prende l'uscita del layer (un vettore per token) e calcola subito, sulla GPU:
- il vettore della posizione di risposta (`q`);
- il vettore dell'ultimo token di ogni frase (`h_end`);
- la media dei vettori dei token di ogni frase (`h_mean`).

Non si conservano mai i vettori di tutti i token (sarebbero decine di MB per riga): solo questi.

```python
class StateTap:
    def __init__(self, layers, mix_idx):
        self.out, self.spans, self.answer_pos = {}, None, None   # spans: [n, 2] = primo e ultimo token di ogni frase
        self.handles = [layers[i].register_forward_hook(self._hook(i)) for i in mix_idx]

    def _hook(self, i):
        def hook(module, args, output):
            h = (output[0] if isinstance(output, tuple) else output)[0]          # [T, D]
            s = self.spans
            csum = torch.cat([h.new_zeros(1, h.size(1)), h.float().cumsum(0)])  # somme cumulative → medie veloci
            mean = (csum[s[:, 1] + 1] - csum[s[:, 0]]) / (s[:, 1] - s[:, 0] + 1)[:, None]
            self.out[i] = {"q": h[self.answer_pos], "end": h[s[:, 1]], "mean": mean.to(h.dtype)}
        return hook
```

Si lavora con una riga per forward (o righe di lunghezza simile); con più righe insieme, posizioni e intervalli vanno spostati del padding di ciascuna.

## 5. Hook sull'attenzione

Il modello calcola l'attenzione con funzioni veloci che **non restituiscono i pesi**. Chiedere i pesi in modo standard ("eager") costa troppa memoria con documenti lunghi. Allora si registra una funzione di attenzione che:
1. calcola l'attenzione normale (stesso risultato di prima);
2. nei soli 8 layer che guardano tutto il testo, ricalcola **una riga** di pesi: quella della posizione di risposta verso tutti i token. Costa pochissimo.

```python
from transformers import AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward

CAPTURE = {"rows": {}, "full": set(), "bias": None}

def sdpa_capture(module, query, key, value, attention_mask, scaling=None, **kw):
    out = sdpa_attention_forward(module, query, key, value,
                                 _add_bias(attention_mask, CAPTURE["bias"], module), scaling=scaling, **kw)
    if module.layer_idx in CAPTURE["full"]:
        k = key.repeat_interleave(query.size(1) // key.size(1), dim=1)           # stesse teste di query e chiavi
        s = (query[:, :, -1:] @ k.transpose(-1, -2)).float() * scaling           # riga dell'ultima posizione
        if attention_mask is not None:
            s = s + attention_mask[:, :, -1:, : s.size(-1)].float()
        CAPTURE["rows"][module.layer_idx] = s.softmax(-1)[:, :, 0]               # [B, teste, T]
    return out

AttentionInterface.register("sdpa_capture", sdpa_capture)
# al caricamento: from_pretrained(..., attn_implementation="sdpa_capture")
```

- In transformers 5.14 (quello installato) LFM2, Qwen3.5 e Gemma 4 calcolano l'attenzione tutti attraverso questa interfaccia; `query` e `key` arrivano già pronte (dopo le rotazioni di posizione) e `scaling` arriva dal modello.
- `_add_bias` lascia la maschera com'è quando `CAPTURE["bias"]` è vuoto. Il bias serve solo all'esperimento H-FAITH ([09](09_training.md) §5), che "spegne" delle frasi.
- d1-3B usa codice proprio del repository del modello: se al primo test `CAPTURE["rows"]` resta vuoto, si fa lo stesso calcolo modificando il `forward` del suo modulo di attenzione.

Dalla riga di pesi alla feature `att[k]`: per ogni layer e testa, la somma dei pesi sui token della frase k.

## 6. Test

Comando: `python -m evidence.models --model d1-3b --selftest` (poi per JPT e Winnow).

| Test | Come | Deve dare |
|---|---|---|
| Verdetto invariato | 50 righe del dev del piano binario, prompt senza preambolo: P(supportato) dal nostro forward contro quella del piano binario | differenza media < 0,005, massima < 0,02 |
| Layer giusti | il vettore catturato dall'hook del layer i = `hidden_states[i + 1]` di transformers | identico |
| Attenzione giusta | su 5 prompt da ~1.000 token, confronto con la modalità "eager" | differenza < 0,001 |
| Memoria | forward da 8.000 e 16.000 token con gli hook | entra nella GPU (§7) |

## 7. GPU necessaria

| Uso | d1-3B | JPT-9B | Winnow-12B |
|---|---|---|---|
| Calcolo dei vettori da salvare | 24 GB | 24 GB | 48 GB |
| Training con il modello in memoria (fase 1b) | 24 GB | 48 GB | 48 GB |
| Training anche del modello (fase 2) | 24 GB | 48 GB | 48 GB |
