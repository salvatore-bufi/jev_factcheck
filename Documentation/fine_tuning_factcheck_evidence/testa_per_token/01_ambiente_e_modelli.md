# Passo 1 — Modelli e hook per token

## In breve

Come nel piano per frase ([../testa_per_frase/01_ambiente_e_modelli.md](../testa_per_frase/01_ambiente_e_modelli.md)): `evidence/models.py` fa un forward con il codice degli autori di ciascun modello e, con degli hook, copia i dati che servono alla testa. La differenza: si conservano i vettori di **tutti i token del documento**, non un riassunto per frase. Perché stiano in memoria e su disco si riducono subito a 256 numeri per layer con una **proiezione fissa** (§3).

Tutto il resto è identico al piano per frase e non si ripete qui:
- pacchetti (`flash-linear-attention`, `causal-conv1d` per i layer DeltaNet);
- tabella dei modelli, codice degli autori per ciascuno, posizione di risposta;
- come trovare i layer (`text_layers`, `full_attention_layers`);
- la funzione `sdpa_capture` che ricalcola la riga di attenzione della posizione di risposta negli 8 layer pieni.

## 1. Che cosa si cattura

| Dato | Forma per riga | Da dove |
|---|---|---|
| `h` | n_token × L × 256 | uscita dei L layer scelti (partenza: 23 e 31; H6 prova 19, 23, 27, 31), solo nei token del documento, proiettata (§3) |
| `att` | n_token × 128 | riga di attenzione della posizione di risposta, negli 8 layer pieni × 16 teste, solo nei token del documento |
| `q` | L × d | stato della posizione di risposta (non proiettato: è uno solo) |
| `doc_tok` | [primo, ultimo] | intervallo dei token del documento nel prompt |
| `answer_logits` | 2 | per ricalcolare P(supportato) |

`att` per token viene gratis: la riga di attenzione è già calcolata su tutti i token; nel piano per frase la si sommava per frase, qui si tiene com'è.

## 2. Hook sui vettori, per token

```python
class TokenTap:
    """Per ogni layer scelto: i vettori dei token del documento, proiettati a 256, e lo stato della posizione di risposta."""
    def __init__(self, layers, layer_idx, projections):
        self.out, self.doc, self.answer_pos = {}, None, None        # doc = (primo, ultimo) token del documento
        self.proj = projections                                      # {layer: matrice fissa [D, 256]} (§3)
        self.handles = [layers[i].register_forward_hook(self._hook(i)) for i in layer_idx]

    def _hook(self, i):
        def hook(module, args, output):
            h = (output[0] if isinstance(output, tuple) else output)[0]          # [T, D]
            a, b = self.doc
            self.out[i] = {"tok": (h[a:b + 1].float() @ self.proj[i]).half(),   # [n, 256], subito sulla GPU
                           "q": h[self.answer_pos]}
        return hook
```

La proiezione si fa sulla GPU dentro l'hook: dalla GPU escono 256 numeri per token invece di 2.560 o 4.096.

## 3. La proiezione fissa (PCA)

Per salvare su disco i vettori di migliaia di token servono vettori corti. Si usa una PCA calcolata una volta per modello e per layer:

1. 2.000 righe a caso del mix di training, forward con un hook che raccoglie 200 token a caso per riga e layer (400.000 vettori per layer);
2. media e prime 256 componenti principali (`torch.pca_lowrank` sui vettori centrati);
3. si salvano in `data/evidence_tok/pca/<modello>/layer<i>.safetensors` (media [D], matrice [D, 256]) con la quota di varianza spiegata.

La proiezione fissa serve solo alla fase 1a (vettori salvati). Nella fase 1b, con il modello in memoria, la testa impara la propria proiezione dai vettori completi ([09](09_training.md) §4); l'esperimento H9 dice se la PCA perde qualcosa.

## 4. Spazio e memoria

Per riga, con 2 layer, `att` in fp16, documento di n token:

| | Per token | HoVer (mediana ~1.500 token) | WiCE (95° percentile ~5.300 token) |
|---|---:|---:|---:|
| `h` (2 × 256, fp16) | 1 KB | ~1,5 MB | ~5,3 MB |
| `att` (128, fp16) | 256 B | ~0,4 MB | ~1,4 MB |

Per le ~28.600 righe di SciFact, WiCE e HoVer: ~60 GB. Con 4 layer ~100 GB: per questo si parte da 2 layer.

## 5. Test

Quelli del piano per frase (verdetto invariato, layer giusti, attenzione uguale alla modalità "eager", maschera conservata), più:

| Test | Deve dare |
|---|---|
| `doc_tok` decodificato è il documento (a meno dell'escaping del prompt) | identico, su 200 righe |
| PCA: varianza spiegata dalle 256 componenti | registrata; se sotto il 70% nel layer 31, provare 384 componenti |
| proiezione nell'hook = proiezione fatta dopo sugli stati completi | differenza < 1e-2 (fp16) |
