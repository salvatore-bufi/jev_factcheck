# Passo 1 — Caricare i modelli e leggere vettori e attenzione

## In breve

Serve un modulo, `evidence/models.py`, che per ciascun modello faccia **un forward** su un prompt e restituisca tre cose:

1. i punteggi delle due opzioni nella posizione di risposta → P(supportato), come nel piano binario;
2. alcuni **vettori** (stati nascosti) letti durante il forward: quelli dei token di ogni frase, in 4 layer, e quello della posizione di risposta;
3. per gli 8 layer ad attenzione piena, **quanto la posizione di risposta guarda ciascun token** (una riga di pesi di attenzione).

I punti 2 e 3 si ottengono con degli *hook*: piccole funzioni agganciate ai layer, che copiano i dati mentre il forward passa, senza cambiarlo.

Il forward è quello che il piano binario usa già per il training (`forward_model` del renderer, [../fine_tuning_factcheck/03_rendering_formato.md](../fine_tuning_factcheck/03_rendering_formato.md) §4): stesso prompt, stessa maschera, stessa lettura degli autori. Qui si aggiungono solo gli hook.

## 1. Pacchetti

Lo stesso ambiente del piano binario, più:

```sh
pip install nltk rank_bm25 safetensors sentence-transformers
python -c "import nltk; nltk.download('punkt_tab')"
```

`flash-linear-attention` e `causal-conv1d` (per i layer DeltaNet) sono già richiesti dal piano binario.

## 2. I modelli

Tutti Qwen3.5, con la stessa struttura:

| | Gruppo 4B | Gruppo 9B |
|---|---|---|
| Layer | 32 | 32 |
| Layer ad attenzione piena | 3, 7, 11, 15, 19, 23, 27, 31 | idem |
| Altri layer | 24 Gated DeltaNet, sempre causali | idem |
| Teste di query / teste KV (layer pieni) | 16 / 4 | 16 / 4 |
| Lunghezza del vettore (d) | 2.560 | 4.096 |
| 4 layer da cui leggere i vettori (partenza) | 19, 23, 27, 31 | 19, 23, 27, 31 |

I 4 layer di partenza sono gli ultimi 4 ad attenzione piena (il 31 è anche l'ultimo layer). L'esperimento H2a ([11](11_esperimenti.md)) sceglie quali servono davvero.

Che cosa cambia da un modello all'altro, per l'estrazione:

| Modello | Forward con hook | Posizione di risposta | Note |
|---|---|---|---|
| **KnowLine-4B** | transformers (front end degli autori, backend `hf`) | ultimo token del prompt | richieste fino a 3.000 token scritte due volte: le frasi si leggono dalla **seconda** copia, che ha già visto tutto il prompt della prima |
| **elephant-1** | motore `qev` in transformers (legge la maschera a blocchi da `modeling_overrides`) | ultimo token del suffisso della domanda | prefisso (stato) e suffisso (domanda): le frasi stanno nel prefisso |
| Jeffan-4B, d3-flash | codice degli autori in transformers (maschera bidirezionale per tipo di layer) | ultimo token | |
| OneJev1.6-9B, Nimble-9B-v3 | transformers con il prompt degli autori, causale | ultimo token della domanda | il loro runtime (Flide, motore Nimble) divide prefisso e domanda: in transformers si usa una sequenza unica e si verifica che il verdetto coincida |
| StartLux 9B/4B | transformers con la maschera bidirezionale ricostruita (come nel training del piano binario) | ultimo token | gli autori usano solo vLLM: il verdetto va verificato contro il loro server |
| pplx-decider-flash | pacchetto `pplx_decider`, percorso di riferimento | la posizione di lettura della domanda | `q` = stato nella posizione di lettura (quella che la sua testa legge) |
| Clef-Flash | `joint_schema_model.py` | ultimo token prima della testa congiunta | `q` = stato finale in quella posizione |

## 3. Trovare i layer nel codice del modello

I nomi interni dei moduli possono cambiare (modelli caricati come multimodali o solo testo). Si cerca la lista dei layer testuali per lunghezza e si leggono i tipi di layer dal config:

```python
def text_layers(model, n_layers=32):
    found = [m for name, m in model.named_modules()
             if isinstance(m, torch.nn.ModuleList) and len(m) == n_layers and "visual" not in name and "vision" not in name]
    assert len(found) == 1
    return found[0]

def full_attention_layers(config):
    tc = getattr(config, "text_config", config)
    return [i for i, t in enumerate(tc.layer_types) if t == "full_attention"]      # [3, 7, …, 31]
```

## 4. Hook sui vettori

Per ognuno dei 4 layer, un hook prende l'uscita del layer (un vettore per token) e calcola subito, sulla GPU:
- il vettore della posizione di risposta (`q`);
- il vettore dell'ultimo token di ogni frase (`h_end`);
- la media dei vettori dei token di ogni frase (`h_mean`).

Non si conservano mai i vettori di tutti i token: solo questi.

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

Il modello calcola l'attenzione con funzioni veloci che **non restituiscono i pesi**, e chiederli in modo standard ("eager") costa troppa memoria con documenti lunghi. Si registra una funzione di attenzione che:
1. calcola l'attenzione normale, con la maschera del modello (causale, bidirezionale o a blocchi): stesso risultato di prima;
2. negli 8 layer pieni ricalcola **una riga** di pesi: quella della posizione di risposta verso tutti i token. Costa pochissimo.

```python
from transformers import AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward

CAPTURE = {"rows": {}, "full": {3, 7, 11, 15, 19, 23, 27, 31}, "query_pos": -1}

def sdpa_capture(module, query, key, value, attention_mask, scaling=None, **kw):
    out = sdpa_attention_forward(module, query, key, value, attention_mask, scaling=scaling, **kw)
    if module.layer_idx in CAPTURE["full"]:
        qp = CAPTURE["query_pos"]
        k = key.repeat_interleave(query.size(1) // key.size(1), dim=1)             # 16 teste di query, 4 di chiavi
        s = (query[:, :, qp:qp + 1 if qp != -1 else None] @ k.transpose(-1, -2)).float() * scaling
        if attention_mask is not None:                                             # la maschera del modello, anche bidirezionale
            s = s + attention_mask[:, :, qp:qp + 1 if qp != -1 else None, : s.size(-1)].float()
        CAPTURE["rows"][module.layer_idx] = s.softmax(-1)[:, :, 0]                 # [B, 16, T]
    return out

AttentionInterface.register("sdpa_capture", sdpa_capture)
```

- In transformers 5.14 (quello installato) la classe di attenzione di Qwen3.5 calcola l'attenzione attraverso questa interfaccia e ha `layer_idx`; `query` e `key` arrivano già pronte (dopo le rotazioni di posizione e le normalizzazioni) e `scaling` arriva dal modello.
- Se il codice degli autori installa la propria maschera bidirezionale con un hook o con un'altra funzione di attenzione (pplx, Jeffan, d3-flash, elephant-1), `sdpa_capture` va registrata **sopra** la loro (chiamando la loro invece di `sdpa_attention_forward`), così la maschera resta la loro. Il test di §6 lo verifica.
- Per elephant-1, OneJev e Nimble la posizione di risposta è la fine del suffisso della domanda; con una sola domanda per richiesta è l'ultimo token della sequenza.

Dalla riga di pesi alla feature `att[k]`: per ogni layer e testa, la somma dei pesi sui token della frase k → 8 × 16 = 128 numeri per frase, per tutti i modelli.

## 6. Test

Comando: `python -m evidence.models --model knowline-4b --selftest` (poi per gli altri modelli).

| Test | Come | Deve dare |
|---|---|---|
| Verdetto invariato | 50 righe del dev del piano binario, prompt senza preambolo: P(supportato) dal forward con gli hook contro l'adapter del benchmark | differenza media < 0,005, massima < 0,02 |
| Layer giusti | il vettore catturato dall'hook del layer i = `hidden_states[i + 1]` di transformers | identico |
| Attenzione giusta | su 5 prompt da ~1.000 token, confronto con la modalità "eager" **con la stessa maschera** del modello | differenza < 0,001 |
| Maschera conservata | per i modelli bidirezionali, il verdetto con gli hook è uguale a quello senza hook | identico |
| Memoria | forward da 8.000 e 16.000 token con gli hook | entra nella GPU (§7) |

## 7. GPU necessaria

| Uso | Gruppo 4B | Gruppo 9B |
|---|---|---|
| Calcolo dei vettori da salvare | 24 GB | 24 GB (~19 GB di pesi) o 45 GB per documenti lunghi |
| Training con il modello in memoria (fase 1b) | 24 GB | 45 GB |
| Training anche del modello (fase 2) | 24 GB | 45 GB |
