# Passo 1 — Ambiente, modelli, hook

## Obiettivo

Avere `evidence/models.py`, che per ciascuno dei tre modelli:
1. carica i pesi del verdetto del piano binario con transformers;
2. fa un forward su un prompt e restituisce i logit delle risposte in `answer_pos`;
3. raccoglie, nello stesso forward, gli stati di layer scelti in posizioni scelte e la riga di attenzione di `answer_pos` negli 8 layer di attenzione piena;
4. permette di aggiungere un bias di attenzione per frase (serve solo all'esperimento H-FAITH, [09](09_training.md) §5).

## 1. Ambiente

Lo stesso env del benchmark (`factcheck`, Python 3.12), più:

```sh
pip install nltk rank_bm25 safetensors sentence-transformers
python -c "import nltk; nltk.download('punkt_tab')"
# solo per JPT-9B (layer Gated DeltaNet di Qwen3.5): senza questi kernel transformers usa un'implementazione lenta in torch
pip install flash-linear-attention causal-conv1d
```

Controllo: `python -c "import fla, causal_conv1d"` deve riuscire sulla macchina con GPU. Su JPT, misurare il tempo di un forward da 4k token con e senza kernel e annotarlo.

I modelli si leggono dalla cache di Hugging Face (`HF_HUB_CACHE`, sul server `/mnt/data1/huggingface/hub` tramite `gpu-run`).

## 2. I tre modelli

| | d1-3B | JPT-9B | Winnow-12B |
|---|---|---|---|
| Pesi di partenza | `LiquidAI/d1-3B` @ `051bcc4` | `kirp/jpt-9b` @ `b447cc7` | `models/winnow-12b-bf16/model/` (da `convert.py`) |
| Pesi del verdetto | quelli del piano binario (adapter fuso) | idem | idem |
| Caricamento | `AutoModel.from_pretrained(..., trust_remote_code=True)` | il backend transformers di `llm2jev` (verificato uguale a vLLM nel benchmark) | `AutoModelForCausalLM.from_pretrained(path)` (`gemma4_text`) |
| Prompt | renderer del piano binario (`D1Renderer`) | renderer del piano binario (testo di `llm2jev`) | renderer del piano binario (`Scorer.render`, JSON con `safe_data`) |
| Risposte (`yes_ids` / `no_ids`) | {11683, 12447} / {2243, 4547, 19598} | 357 / 417 (A / B) | 236799 / 236776 (B / A) |
| `answer_pos` | ultimo token del prompt (a capo dopo `assistant`) | ultimo token (`:` di `Answer:`) | ultimo token (a capo dopo `Answer:`) |
| d | 2.048 | 4.096 | 3.840 |
| Layer totali | 30 | 32 | 48 |
| Layer di attenzione piena | 2, 5, 9, 13, 17, 21, 24, 27 | 3, 7, 11, 15, 19, 23, 27, 31 | 5, 11, 17, 23, 29, 35, 41, 47 |
| Teste di query (layer pieni) / teste KV | 32 / 8 | 16 / 4 | 16 / 1 (head_dim 512) |
| Layer del mix (punto di partenza) | 21, 24, 27, 29 | 19, 23, 27, 31 | 29, 35, 41, 47 |

I renderer, i token delle risposte e la domanda sono quelli del piano binario ([../fine_tuning_factcheck/03_rendering_formato.md](../fine_tuning_factcheck/03_rendering_formato.md), [../fine_tuning_factcheck/02d_varianti_domanda.md](../fine_tuning_factcheck/02d_varianti_domanda.md)): `evidence/render.py` li importa, non li riscrive.

d1-3B e JPT-9B sono modelli multimodali: si usa solo la torre testuale. Il layer del mix "29" di d1 è l'uscita finale (gli ultimi due layer sono convoluzioni dopo l'ultimo layer pieno).

## 3. Trovare i layer

Non si assumono i nomi dei moduli: si cerca la `nn.ModuleList` dei layer testuali e si controlla il tipo di ciascun layer contro il config.

```python
def text_layers(model, n_layers):
    """La ModuleList dei layer testuali: l'unica con n_layers elementi sotto la torre testuale."""
    found = [m for name, m in model.named_modules()
             if isinstance(m, torch.nn.ModuleList) and len(m) == n_layers and "vision" not in name]
    assert len(found) == 1, [type(m) for m in found]
    return found[0]

def full_attention_layers(config):
    tc = getattr(config, "text_config", config)
    return [i for i, t in enumerate(tc.layer_types) if t == "full_attention"]
```

Controllo da fare a mano una volta per modello: `print(type(layers[i].self_attn))` (o il nome equivalente) per un layer pieno e uno non pieno.

## 4. Hook sugli stati

Un forward hook per ogni layer del mix. Il pooling per frase si fa subito sulla GPU: non si conservano mai gli stati di tutti i token.

```python
class StateTap:
    """Per ogni layer del mix: stato in answer_pos, stato nell'ultimo token di ogni frase, media dei token di ogni frase."""
    def __init__(self, layers, mix_idx):
        self.out, self.spans, self.answer_pos = {}, None, None
        self.handles = [layers[i].register_forward_hook(self._hook(i)) for i in mix_idx]

    def _hook(self, i):
        def hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output       # [1, T, D]
            h = h[0]
            spans = self.spans                                            # LongTensor [n, 2], estremi inclusi
            csum = torch.cat([h.new_zeros(1, h.size(1)), h.float().cumsum(0)])
            mean = (csum[spans[:, 1] + 1] - csum[spans[:, 0]]) / (spans[:, 1] - spans[:, 0] + 1)[:, None]
            self.out[i] = {"q": h[self.answer_pos], "end": h[spans[:, 1]], "mean": mean.to(h.dtype)}
        return hook
```

Batch di più righe: con padding a sinistra, `spans` e `answer_pos` vanno spostati della lunghezza del padding di ciascuna riga. Per l'estrazione si usa batch 1 (o batch per lunghezza simile), per semplicità.

## 5. Hook sull'attenzione

FlashAttention e SDPA non restituiscono i pesi, e l'implementazione "eager" materializza T × T per testa (troppa memoria con documenti lunghi). Si registra una funzione di attenzione che chiama SDPA e, nei soli layer pieni, ricalcola **una riga** di pesi: quella della query in `answer_pos`.

```python
from transformers import AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward

CAPTURE = {"rows": {}, "full": set(), "bias": None}

def sdpa_capture(module, query, key, value, attention_mask, scaling=None, **kw):
    out = sdpa_attention_forward(module, query, key, value,
                                 _add_bias(attention_mask, CAPTURE["bias"], module),
                                 scaling=scaling, **kw)
    if module.layer_idx in CAPTURE["full"]:
        rep = query.size(1) // key.size(1)
        k = key.repeat_interleave(rep, dim=1)                         # [B, H, T, d]
        s = (query[:, :, -1:] @ k.transpose(-1, -2)).float() * scaling  # riga dell'ultima posizione
        if attention_mask is not None:
            s = s + attention_mask[:, :, -1:, : s.size(-1)].float()
        CAPTURE["rows"][module.layer_idx] = s.softmax(-1)[:, :, 0]    # [B, H, T]
    return out

AttentionInterface.register("sdpa_capture", sdpa_capture)
# caricamento: from_pretrained(..., attn_implementation="sdpa_capture")
```

- La query in `answer_pos` è l'ultima posizione del prompt in tutti e tre i modelli.
- `scaling` arriva dal modulo: non si assume (Gemma 4 ne ha uno proprio). In transformers 5.14 le classi di attenzione di LFM2, Qwen3.5 e Gemma 4 chiamano tutte `attention_interface(self, query, key, value, attention_mask, scaling=..., **kwargs)` e hanno `layer_idx`; Gemma 4 non usa softcap nell'attenzione (solo sui logit finali).
- `query` e `key` arrivano dopo RoPE e normalizzazioni, già nella forma `[B, H, T, d]`.
- `_add_bias` restituisce la maschera invariata quando `CAPTURE["bias"]` è `None`; serve solo a H-FAITH ([09](09_training.md) §5).
- d1-3B carica codice del repository del modello (`modeling_d1.py`): verificare che le sue classi passino per l'interfaccia. Se `CAPTURE["rows"]` resta vuoto al primo test (§6), si applica lo stesso calcolo con un piccolo patch del `forward` del suo modulo di attenzione.

Dalla riga di attenzione alle feature per frase: per ogni layer pieno e testa, somma dei pesi sui token della frase → tensore `[n, 8, H]` ([06](06_rendering_ed_estrazione.md) §4).

## 6. Test di questo passo

Per ciascun modello, in quest'ordine:

| Test | Come | Soglia |
|---|---|---|
| Identità del verdetto | prompt del piano binario (senza preambolo) su 50 righe di `dev_select`; P(supportato) dal nostro forward contro quella del piano binario | differenza media < 0,005, massima < 0,02 |
| Layer | `output_hidden_states=True` su un prompt: lo stato del layer i catturato dall'hook = `hidden_states[i + 1]` | uguale (stesso forward) |
| Riga di attenzione | su 5 prompt da ~1.000 token, confronto con `attn_implementation="eager"` e `output_attentions=True`, ultima riga, layer pieni | differenza massima < 1e-3 |
| Somma | ogni riga catturata somma a 1 (± 1e-3) | |
| Memoria | forward da 8k e 16k token con gli hook attivi | entra nella GPU prevista (§7) |

Script: `python -m evidence.models --model d1-3b --selftest` (stampa i risultati della tabella).

## 7. GPU

| Uso | d1-3B | JPT-9B | Winnow-12B |
|---|---|---|---|
| Estrazione delle feature (forward senza gradiente) | 24 GB | 24 GB | 48 GB |
| Fase 1b (forward in linea a ogni passo) | 24 GB | 48 GB | 48 GB |
| H-FAITH (backward fino alla maschera) | 24 GB, documenti ≤ 4k token, gradient checkpointing | — | 48 GB, documenti ≤ 4k token |
| Fase 2 (LoRA + testa) | 24 GB | 48 GB (o QLoRA su 24 GB) | 48 GB |

## Prodotto

- `evidence/models.py` con `load(model_name)`, `forward(prompt_ids, spans, answer_pos) -> (answer_logits, states, attn_rows)`.
- I test di §6 superati per i tre modelli, con l'output salvato in `data/evidence/runs/<modello>/selftest.txt`.
