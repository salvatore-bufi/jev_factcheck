# Passo 9 — Training

## Obiettivo

Addestrare la testa sopra il verdetto senza cambiarlo. Tre fasi, nell'ordine:

| Fase | Backbone | Dati della testa | Script | Si usa per |
|---|---|---|---|---|
| **1a** | congelato, feature salvate | feature del passo [06](06_rendering_ed_estrazione.md) | `train_head.py --phase 1a` | quasi tutti gli esperimenti |
| **1b** | congelato, forward a ogni passo | stati di tutti i token | `train_head.py --phase 1b` | pooling attentivo (H2d), fedeltà (H-FAITH) |
| **2** | LoRA aggiornata con la testa | come 1a, dal forward con gradiente | `train_joint.py` | solo se 1a/1b non raggiungono gli obiettivi |

Ogni corsa scrive in `data/evidence/runs/<modello>/<id>/`: `config.json` (tutti i flag della testa, loss, iperparametri, hash di feature e mix), `head.safetensors`, `log.jsonl` (loss e metriche dev per epoca), `dev_metrics.json`.

## 1. Fase 1a — sulle feature salvate

Nessun LLM in memoria: si addestra su CPU o su una GPU qualsiasi.

| Iperparametro | Valore |
|---|---|
| Ottimizzatore | AdamW, lr 1e-3, weight decay 0,01, warmup 5%, cosine |
| Epoche | fino a 10, early stopping su `dev_evid_select` (pazienza 2) |
| Batch | 64 righe; righe con > 200 frasi in batch propri |
| Dropout | 0,1 nell'aggregatore; 0,1 sulle feature di attenzione (intera feature a zero per frase) |
| Criterio di checkpoint | F1 per frase + recall dell'insieme completo (media) su `dev_evid_select` |
| Loss | H0: `ref_loss`; poi quella scelta in H-LOSS ([11](11_esperimenti.md)) |

```sh
python -m evidence.train_head --phase 1a --model d1-3b --features data/evidence/features/d1-3b \
    --mix ev_head --config configs/h0.json --run h0
```

Le configurazioni (`configs/*.json`) contengono i flag di `EvidenceHead`, la loss e i suoi parametri. Ogni esperimento è un file di configurazione.

## 2. Selezione delle frasi in uscita

Si fissa dopo il training, su `dev_evid_select`, fra le righe supportate che il verdetto giudica supportate:

| Loss | Regola | Cosa si sceglie sul dev |
|---|---|---|
| `ref_loss` | `σ(s_k) > τ_e` | `τ_e` che massimizza l'F1 per frase |
| `zlpr_loss` | `s_k > s_0 + δ` | δ ∈ [−1, 1] a passi di 0,1; si tiene δ ≠ 0 solo se guadagna ≥ 1 punto di F1 |

Sempre: almeno 1 frase (la migliore) se il verdetto è "supportato", al massimo 5. Conferma su `dev_evid_confirm`: F1 entro 1 punto da quello del dev di scelta.

## 3. Controlli durante il training

| Metrica | Su | Allarme |
|---|---|---|
| F1 per frase, insieme completo | `dev_evid_select` | non sale sopra E-LEX dopo 2 epoche → controllare le posizioni ([06](06_rendering_ed_estrazione.md) §6) prima di toccare la testa |
| F1 per distanza fra frasi gold | righe sparse di `dev_evid_select` | crollo oltre 10 frasi |
| F1 per distanza in token dal preambolo | `dev_evid_select` | crollo oltre 1.024 (Winnow) o oltre 4.096 |
| Frasi mostrate per riga | `dev_evid_select` | media lontana dal numero gold (> +1 o < −0,5) |
| Righe con almeno una frase sopra `s_0` | `dev_evid_neg` | sopra il 30% |
| Recall dell'insieme completo con P(supportato) fra la soglia e soglia + 0,15 | `dev_evid_select` | molto sotto la media → μ_neg troppo alto |
| Pesi del mix | testa | tutto su un layer → il mix si può ridurre a quel layer |

## 4. Fase 1b — backbone congelato, forward a ogni passo

Il modello del verdetto è in memoria in bf16, `requires_grad=False`. Per H2d il forward è in `torch.no_grad()` e la testa riceve gli stati di tutti i token del documento.

| Parametro | d1-3B | JPT-9B, Winnow-12B |
|---|---|---|
| GPU | 24 GB | 48 GB |
| Batch | 4–8 righe, accumulo fino a 32 | 1–2 righe, accumulo fino a 32 |
| Lunghezza massima | 8.192 token | 8.192 token |
| lr della testa | 5e-4 | 5e-4 |
| Epoche | 2 | 1 |
| Inizializzazione | la testa migliore della fase 1a | idem |

Pooling attentivo (H2d): per ogni layer del mix, proiezione `d → 256` di tutti i token del documento (`nn.Linear` condiviso fra i layer, dopo il mix); query `u = W_u [c ; q]`, dove `c` è la media degli stati dei token del preambolo; per ogni frase, softmax di `uᵀ v_i` sui suoi token; il vettore pesato si concatena a `h_end` al posto di `h_mean`.

```sh
gpu-run --batch --name ev-h2d-d1 -- bash -c \
  'python -m evidence.train_head --phase 1b --model d1-3b --mix ev_head --config configs/h2d.json \
     --init data/evidence/runs/d1-3b/<migliore 1a>/head.safetensors --run h2d > logs/ev-h2d-d1.out 2>&1'
```

## 5. H-FAITH — fedeltà con maschera per frase (fase 1b)

Solo su d1-3B e Winnow-12B. Per ogni batch di righe positive giudicate supportate:

1. **Forward 1** (`no_grad`): feature, `s`, `s_0` dalla testa; margine `z_full` delle risposte.
2. `p_k = σ(s_k − s_0)`, con gradiente verso la testa.
3. **Bias di attenzione**: matrice `bias[b, i, j]` = `log(p_{frase(j)} + 1e-4)` se il token chiave j appartiene a una frase del documento e il token query i **non** appartiene alla stessa frase; 0 altrimenti (preambolo, stato fuori dal documento, domanda e token della stessa frase restano visibili). Si costruisce come `[B, 1, T, T]` e si somma alla maschera causale in `CAPTURE["bias"]` ([01](01_ambiente_e_modelli.md) §5).
   - d1-3B: il bias si applica negli 8 layer di attenzione;
   - Winnow-12B: in tutti i 48 layer (anche quelli a finestra).
4. **Forward 2** con il bias di sufficienza → `z_suff`; **forward 3** con `log(1 − p_k + 1e-4)` → `z_comp`. Gradiente attivo rispetto al bias (pesi congelati); gradient checkpointing sui layer.
5. Loss = `zlpr_loss` + 0,2 · `L_faith` ([08](08_testa_e_loss.md) §4).

Vincoli pratici:
- `[B, 1, T, T]` in fp32 con T = 4.096 occupa 64 MB per riga: batch 1, documenti ≤ 4.096 token;
- SDPA accetta una maschera additiva; FlashAttention no: per H-FAITH si carica il modello con `attn_implementation="sdpa_capture"` e la maschera esplicita;
- test prima di partire: con `p_k = 1` per tutte le frasi, `z_suff = z_full` entro 1e-3; con `p_k = 0` per tutte, `z_suff` uguale al margine ottenuto togliendo davvero tutte le frasi dal documento entro 0,5 (su Winnow) — su d1 la differenza è dovuta alle convoluzioni ai confini e si registra.

## 6. Fase 2 — LoRA + testa

Quando: dopo la scala di esperimenti su d1, F1 o insieme completo restano sotto gli obiettivi ([00](00_overview.md)).

| Parametro | Valore |
|---|---|
| Inizializzazione | LoRA del verdetto del piano binario (o quella riaddestrata col preambolo) + testa migliore della fase 1 |
| Moduli LoRA, rank | quelli del piano binario |
| lr | LoRA 5e-6 – 1e-5; testa 2e-4 |
| Loss | CE del verdetto sulle risposte (piano binario) + λ · loss di evidenza, λ = 0,3 |
| Dati | `ev_joint`: tutte le righe hanno il termine del verdetto; il termine di evidenza secondo `evidence_term` |
| Epoche | 1 |
| Ancoraggio (se la calibrazione si sposta) | + β · KL(p_verdetto_congelato ‖ p), β = 0,1, con le probabilità del verdetto congelato precalcolate sul mix |
| GPU | d1: 24 GB; JPT e Winnow: 48 GB (JPT anche con QLoRA su 24 GB) |

Nel forward: hook sui layer del mix (con gradiente), riga di attenzione degli 8 layer pieni, logit delle risposte solo in `answer_pos` con le righe del vocabolario delle risposte (come nel piano binario).

Se il termine `mil_neg` sposta la calibrazione del verdetto, nella fase 2 si applica con `stop_grad` sulle feature (agisce solo sulla testa).

```sh
gpu-run --batch --name ev-joint-d1 -- bash -c \
  'python -m evidence.train_joint --model d1-3b --mix ev_joint --config configs/<migliore>.json \
     --lambda 0.3 --run joint > logs/ev-joint-d1.out 2>&1'
```

## 7. Protezione del verdetto

| Controllo | Quando | Regola |
|---|---|---|
| Verdetto identico | fasi 1a e 1b | P(supportato) dalle feature = P del verdetto scelto al passo [05](05_formato_prompt.md), differenza < 0,005 |
| Passo zero | fase 2, prima del primo aggiornamento | verdetto identico al modello di partenza |
| Verdetto a ogni checkpoint | fase 2 | BAcc, AUC, ECE, soglia ottimale su `dev_select`: un checkpoint si accetta solo entro 0,3 punti di BAcc e 0,01 di ECE |

## 8. Test prima di ogni nuova configurazione

| Test | Atteso |
|---|---|
| Overfit di 64 righe (fase 1a) | F1 ≈ 1 sul training in poche centinaia di passi |
| H0 riprodotto con i flag spenti | stessi numeri di H0 (stesso seme) |
| Righe `none` | contributo zero alla loss e al gradiente |

## Prodotto

- Testa migliore per ogni modello, con `config.json`, regola di selezione (`τ_e` o δ) e metriche dev.
