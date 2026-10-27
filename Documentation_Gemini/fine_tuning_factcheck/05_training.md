# Step 5 — Il training: LoRA sul decision model, passo per passo

## Obiettivo

Addestrare un adapter LoRA su d1-3B (prove e candidato economico), JPT-9B (confronto) e Winnow-12B (massima accuratezza) con la loss di [04](04_loss.md), il curriculum di [02e](02e_mix_finale.md) e i dati renderizzati di [03](03_rendering_formato.md); controllare apprendimento e calibrazione durante il training; salvare checkpoint confrontabili.

## 1. LoRA in breve, con i numeri dei nostri modelli

Per una matrice congelata W (forma d_out × d_in) LoRA aggiunge B (d_out × r) e A (r × d_in): W' = W + (α/r)·B·A. Solo A e B si addestrano; B parte da zero, quindi **al passo 0 il modello è identico a quello di partenza**.

Esempio: in JPT `q_proj` di un layer di attenzione piena ha d_in = 4.096. Con r = 16 i parametri addestrati per quella matrice sono 16 · (4.096 + d_out): decine di migliaia di volte meno della matrice.

JPT-9B è già "Qwen3.5-9B + una LoRA fusa": i pesi pubblicati contengono W_jpt = W_qwen + ΔW_1. Si addestra un **nuovo** adapter sopra W_jpt (W_finale = W_jpt + ΔW_2); le matrici della LoRA originale non esistono più separatamente e non servono.

## 2. Moduli su cui mettere LoRA

Nomi verificati sugli indici dei pesi locali.

**d1-3B** (`lfm2` dentro `lfm2_vl`, 30 layer: 22 a convoluzione corta e 8 ad attenzione piena, ai layer 2, 5, 9, 13, 17, 21, 24, 27):

```python
D1_TARGETS = (r".*language_model\.layers\.\d+\."
              r"(self_attn\.(q_proj|k_proj|v_proj|out_proj)"
              r"|conv\.(in_proj|out_proj)"
              r"|feed_forward\.(w1|w2|w3))")
```

- `self_attn.*` esiste solo negli 8 layer ad attenzione; `conv.*` nei 22 a convoluzione (`conv.conv` è la convoluzione corta vera e propria: piccola, si lascia congelata);
- la MLP si chiama `feed_forward` con matrici `w1`, `w2`, `w3`;
- la normalizzazione finale si chiama `embedding_norm` (nome dell'implementazione LFM2), la testa del vocabolario è legata alle embedding;
- il prefisso `language_model` esclude il codificatore visivo (SigLIP2), che non si usa.

**JPT-9B** (`qwen3_5_text`, 32 layer: 24 di attenzione lineare DeltaNet e 8 di attenzione piena, ai layer 3, 7, …, 31):

```python
JPT_TARGETS = (r".*language_model\.layers\.\d+\."
               r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)"
               r"|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)"
               r"|mlp\.(gate_proj|up_proj|down_proj))")
```

- `self_attn.*` esiste solo nei layer ad attenzione piena; `linear_attn.*` solo in quelli DeltaNet.
- `linear_attn.in_proj_a` e `in_proj_b` sono proiezioni piccolissime (una per testa): si lasciano congelate.
- Il prefisso `language_model` esclude la torre visiva, che non si usa.

**Winnow-12B** (`gemma4_text`, 48 layer: 40 con finestra scorrevole di 1.024 token e 8 globali, ai layer 5, 11, …, 47):

```python
WINNOW_TARGETS = r"model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))"
```

## 3. Configurazione

| Parametro | Pilota (`fc_pilot`) | Stadio 1 | Stadio 2 | Perché |
|---|---|---|---|---|
| Rank r / α | 16 / 32 | 16 / 32 | stesso adapter | sufficiente per un compito stretto (F3 prova 8, 32, 64) |
| Dropout LoRA | 0,05 | 0,05 | 0,05 | |
| Learning rate | 2e-5 | **5e-5** | **1e-5** | come MiniCheck: più alto sul primo stadio, basso sul secondo |
| Warmup / schedule | 3% / cosine | 3% / cosine | 3% / cosine | |
| Epoche | 1 | 1 | 1 | dati in parte sintetici: la seconda epoca adatta lo stile |
| Batch effettivo | 32 esempi | 32 | 32 | |
| Ottimizzatore | AdamW (β 0,9 / 0,999, ε 1e-8), weight decay 0 | idem | **nuovo** AdamW | lo stato dello stadio 1 porterebbe il momento verso i dati sintetici |
| Lunghezza massima | 4.096 | 8.192 | 8.192 | |
| Precisione | bf16; gradient checkpointing | idem | idem | |
| Loss | CE sulle due risposte (sì/no) | idem | idem | [04](04_loss.md) |
| Checkpoint | ogni 20% | ogni 20% | ogni 10% | |
| Seed | 13 | 13 (3 seed per la configurazione finale: 13, 14, 15) | | |

## 4. Hardware e memoria

| Macchina | d1-3B | JPT-9B | Winnow-12B |
|---|---|---|---|
| Portatile RTX 5090, 24 GB | **LoRA in bf16** (~7 GB di pesi), 8k token, batch fisico 2–4 | QLoRA (pesi a 4 bit, ~6 GB), 4k token, batch fisico 1 | non realistico |
| L40S 45 GB | LoRA in bf16, 16k token o batch più grandi | LoRA in bf16 (~18 GB di pesi), fino a 8k token | LoRA in bf16 (~22 GB di pesi), 4–8k token |
| 2× L40S | data parallel | data parallel (ogni scheda una copia; gli adapter sono piccoli) | idem |

d1-3B non ha bisogno di QLoRA: i numeri del portatile sono già quelli definitivi in bf16.

Stima per Winnow su L40S a 8k token: pesi 22 GB + adapter e stati dell'ottimizzatore ~1 GB + attivazioni con gradient checkpointing ~8–12 GB ≈ 31–35 GB.

**Trucco di memoria**: non calcolare i logit del vocabolario intero. Con ~250.000 token di vocabolario e 8.192 posizioni, la proiezione completa costerebbe da sola ~4 GB in bf16 per esempio. Si prende lo stato finale **solo** in `answer_pos` e lo si moltiplica per le righe dei token di risposta (`margin` di [04](04_loss.md) §7).

QLoRA cambia leggermente i pesi base: va bene per iterare su JPT, ma i numeri definitivi si misurano sui pesi bf16 fusi ([08](08_merge_export.md)).

## 5. Installazione

```sh
conda run -n factcheck pip install peft
conda run -n factcheck python -c "import peft, transformers; print(peft.__version__, transformers.__version__)"
```

Se `peft` richiede una versione di transformers diversa da 5.14.1 (quella usata anche da vLLM nell'ambiente), creare un ambiente separato per il training (`conda create -n factcheck-train --clone factcheck`) invece di aggiornare transformers.

## 6. Lo script: `finetune/train.py`

Codice di riferimento completo nella struttura (PyTorch + PEFT + accelerate). I punti delicati sono commentati.

```python
"""LoRA training of a decision model on the logits of its yes/no answer tokens.
accelerate launch finetune/train.py --model jpt --data data/finetune/rendered/jpt/fc_pilot/train.jsonl \
    --out finetune/outputs/P0 --lr 2e-5 [--init-adapter finetune/outputs/S1/final] [--qlora]"""
import argparse, json, math, random, time
from pathlib import Path
import torch
import transformers
from accelerate import Accelerator
from peft import LoraConfig, PeftModel, get_peft_model
from losses import margin, binary_loss, anchor_kl

SPECS = {
    "d1":     dict(path="LiquidAI/d1-3B", revision="051bcc464b01b9f92942b364d9586b0ef5912432",
                   targets=D1_TARGETS, yes_ids=[11683, 12447], no_ids=[2243, 4547, 19598], T=1.0, softcap=None),
    "jpt":    dict(path="kirp/jpt-9b", revision="b447cc7ee105c0a76a22f8fde8ecf05074dc8be0",
                   targets=JPT_TARGETS, yes_ids=[357], no_ids=[417], T=1.087, softcap=None),
    "winnow": dict(path="winnow-12b-bf16/model", revision=None,
                   targets=WINNOW_TARGETS, yes_ids=[236799], no_ids=[236776], T=1.0, softcap=30.0),
}
TAKE = json.load(open("finetune/question.json")).get("take", "yes")   # "no" con Q4

def load_model(spec, qlora):
    cfg = transformers.AutoConfig.from_pretrained(spec["path"], revision=spec["revision"])
    cls = getattr(transformers, cfg.architectures[0])        # JPT: Qwen3_5ForConditionalGeneration; d1: Lfm2VlForConditionalGeneration
    kw = dict(revision=spec["revision"], torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    if qlora:
        kw["quantization_config"] = transformers.BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16)
    model = cls.from_pretrained(spec["path"], **kw)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    return model

def decoder_and_rows(model, spec):
    """-> (modulo che restituisce last_hidden_state normalizzato, righe dei token sì, righe dei token no)."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    decoder = base.model.language_model if hasattr(base.model, "language_model") else base.model
    head = base.get_output_embeddings().weight                # Winnow e d1: = embedding (legate)
    return decoder, head[spec["yes_ids"]], head[spec["no_ids"]]

def batches(path, micro_bs, max_tokens, seed):
    rows = [json.loads(l) for l in open(path)]
    random.Random(seed).shuffle(rows)
    # raggruppa per lunghezza simile (meno padding), poi rimescola i gruppi
    rows.sort(key=lambda r: len(r["input_ids"]) // 256)
    groups = [rows[i:i + micro_bs] for i in range(0, len(rows), micro_bs)]
    random.Random(seed + 1).shuffle(groups)
    return groups

def collate(group, pad_id, device):
    L = max(len(r["input_ids"]) for r in group)
    ids = torch.full((len(group), L), pad_id); mask = torch.zeros((len(group), L), dtype=torch.long)
    for i, r in enumerate(group):                             # padding a destra: answer_pos non cambia
        ids[i, :len(r["input_ids"])] = torch.tensor(r["input_ids"]); mask[i, :len(r["input_ids"])] = 1
    return (ids.to(device), mask.to(device),
            torch.tensor([r["answer_pos"] for r in group], device=device),
            torch.tensor([float(r["label"]) for r in group], device=device),
            torch.tensor([r.get("teacher_p") or 0.5 for r in group], device=device))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=SPECS, required=True)
    p.add_argument("--data", required=True); p.add_argument("--out", required=True)
    p.add_argument("--init-adapter"); p.add_argument("--qlora", action="store_true")
    p.add_argument("--lr", type=float, default=2e-5); p.add_argument("--rank", type=int, default=16)
    p.add_argument("--micro-bs", type=int, default=1); p.add_argument("--accum", type=int, default=32)
    p.add_argument("--loss", default="ce", choices=["ce", "brier"]); p.add_argument("--smoothing", type=float, default=0.0)
    p.add_argument("--beta", type=float, default=0.0); p.add_argument("--seed", type=int, default=13)
    p.add_argument("--ckpt-every", type=float, default=0.2); p.add_argument("--warmup", type=float, default=0.03)
    a = p.parse_args()

    torch.manual_seed(a.seed)
    acc = Accelerator(gradient_accumulation_steps=a.accum, mixed_precision="bf16")
    spec = SPECS[a.model]
    model = load_model(spec, a.qlora)
    if a.init_adapter:                                        # stadio 2: si continua lo stesso adapter
        model = PeftModel.from_pretrained(model, a.init_adapter, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05,
                                                 target_modules=spec["targets"], bias="none"))
    model.print_trainable_parameters()
    groups = batches(a.data, a.micro_bs, None, a.seed)
    steps = math.ceil(len(groups) / a.accum)
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=a.lr, weight_decay=0.0)
    sched = transformers.get_cosine_schedule_with_warmup(opt, int(a.warmup * steps), steps)
    model, opt, sched = acc.prepare(model, opt, sched)
    decoder, w_yes, w_no = decoder_and_rows(acc.unwrap_model(model), spec)
    pad = 0
    Path(a.out).mkdir(parents=True, exist_ok=True)
    log = open(Path(a.out) / "log.jsonl", "a") if acc.is_main_process else None
    model.train(); t0 = time.time()
    for i, group in enumerate(groups[acc.process_index::acc.num_processes]):
        ids, mask, pos, y, p0 = collate(group, pad, acc.device)
        with acc.accumulate(model):
            h = decoder(input_ids=ids, attention_mask=mask).last_hidden_state      # [B, L, d]
            h_ans = h[torch.arange(len(pos), device=pos.device), pos]               # [B, d]
            s = margin(h_ans, w_yes, w_no, spec["T"], spec["softcap"], TAKE)
            loss = binary_loss(s, y, a.loss, a.smoothing)
            if a.beta:
                loss = loss + a.beta * anchor_kl(s, p0)
            acc.backward(loss)
            if acc.sync_gradients:
                acc.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
        if log and i % 50 == 0:
            log.write(json.dumps({"step": i, "loss": loss.item(), "lr": sched.get_last_lr()[0],
                                  "p_mean": torch.sigmoid(s).mean().item(), "sec": time.time() - t0}) + "\n"); log.flush()
        if acc.is_main_process and (i + 1) % max(1, int(a.ckpt_every * len(groups) / acc.num_processes)) == 0:
            acc.unwrap_model(model).save_pretrained(Path(a.out) / f"ckpt-{i + 1}")
    if acc.is_main_process:
        acc.unwrap_model(model).save_pretrained(Path(a.out) / "final")
        (Path(a.out) / "config.json").write_text(json.dumps(vars(a), indent=2))

if __name__ == "__main__":
    main()
```

Punti da controllare quando si scrive il codice vero:

- `decoder_and_rows`: il modulo che restituisce `last_hidden_state` **dopo** la normalizzazione finale. Per JPT è `model.model.language_model` (i pesi si chiamano `model.language_model.layers.*` e `model.language_model.norm`); per Winnow è `model.model`. Si verifica con il test §8.
- con `get_peft_model` i nomi dei moduli acquisiscono il prefisso `base_model.model.`: l'espressione regolare dei target funziona perché inizia con `.*` (JPT) — per Winnow aggiungere `.*` all'inizio se PEFT non trova moduli (`print_trainable_parameters` deve dare un numero > 0).
- il padding a destra non sposta `answer_pos`; con batch fisico 1 non c'è padding.
- per JPT il token di padding: usare `tok.pad_token_id` (con `attention_mask = 0` il valore non conta).
- con più GPU, accelerate fa avanzare lo scheduler una volta per processo a ogni passo dell'ottimizzatore: controllare nel log che il learning rate arrivi a ~0 alla fine dell'epoca, e se scende troppo presto moltiplicare `steps` per `num_processes` nella creazione dello scheduler.
- l'ottimizzatore preparato da accelerate salta `opt.step()` durante l'accumulo: la chiamata a ogni micro-batch è corretta.

## 7. Comandi

```sh
# render
conda run -n factcheck python finetune/render.py --model d1 --mix fc_pilot
# pilota d1-3B sul portatile (LoRA in bf16, niente QLoRA)
conda run -n factcheck accelerate launch --num_processes 1 finetune/train.py --model d1 \
    --data data/finetune/rendered/d1/fc_pilot/train.jsonl --out finetune/outputs/P0-d1 --lr 2e-5 --micro-bs 2 --accum 16
# pilota JPT sul portatile (QLoRA)
conda run -n factcheck accelerate launch --num_processes 1 finetune/train.py --model jpt --qlora \
    --data data/finetune/rendered/jpt/fc_pilot/train.jsonl --out finetune/outputs/P0 --lr 2e-5
# stadio 1 e 2 su 2 L40S
conda run -n factcheck accelerate launch --num_processes 2 finetune/train.py --model winnow \
    --data data/finetune/rendered/winnow/fc_stage1/train.jsonl --out finetune/outputs/W1/stage1 --lr 5e-5 --accum 16
conda run -n factcheck accelerate launch --num_processes 2 finetune/train.py --model winnow \
    --data data/finetune/rendered/winnow/fc_stage2/train.jsonl --out finetune/outputs/W1/stage2 --lr 1e-5 --accum 16 \
    --init-adapter finetune/outputs/W1/stage1/final
```

Con 2 GPU e `--accum 16` il batch effettivo resta 32 (2 × 1 × 16).

## 8. Test prima del primo training (obbligatori)

| Test | Come | Atteso |
|---|---|---|
| **Margine ridotto = forward completo** | 50 prompt: `margin(h_ans, …)` contro `(max logits[:, gruppo sì] − max logits[:, gruppo no]) / T` dal forward con la testa del vocabolario, in fp32 | differenza < 1e-3 |
| **Passo zero = baseline** | adapter appena creato (B = 0): P(supportato) su 200 righe di dev_select | uguale (entro ~0,01 per il bf16) alle probabilità di `finetune/score.py` con vLLM |
| **Overfit di 64 esempi** | 300 passi su 64 esempi, lr 1e-4 | loss < 0,05: la pipeline impara |
| **Parametri addestrabili** | `print_trainable_parameters` | > 0, ordine di decine di milioni per r = 16 |

Se il passo zero non coincide con la baseline: prompt, token di risposta o `take` sbagliati; correggere prima di addestrare.

**Solo per d1-3B**: il training carica il modello con la classe standard di transformers (`Lfm2VlForConditionalGeneration`, senza `trust_remote_code`), mentre la baseline usa l'API del repository (`system_one`, con un'implementazione propria dei layer). Il test del passo zero confronta quindi le due implementazioni: le P(sì) devono coincidere entro ~0,01 su 200 righe. Se non coincidono, addestrare caricando il modello con `trust_remote_code=True` e verificare che il suo forward supporti i gradienti.

## 9. Valutazione durante il training — `finetune/quick_eval.py`

A ogni checkpoint, su un campione fisso di dev_select (al massimo 200 righe per fonte, ~1.400 righe):

```python
@torch.no_grad()
def quick_eval(model, rows, spec):
    ps = []
    for r in rows:                                    # batch fisico 1 per semplicità
        ids = torch.tensor([r["input_ids"]], device="cuda")
        h = decoder(input_ids=ids).last_hidden_state[0, r["answer_pos"]]
        ps.append(torch.sigmoid(margin(h[None], w_yes, w_no, spec["T"], spec["softcap"], TAKE)).item())
    recs = [{"dataset": r["source"], "label": r["label"]} for r in rows]
    tuned = tune_threshold(recs, ps); auc = metrics(recs, ps, tuned["threshold"])
    return {"bacc": tuned["macro_balanced_accuracy"], "thr": tuned["threshold"],
            "bacc_05": metrics(recs, ps, 0.5)["macro_balanced_accuracy"], "ece": ece(ps, [r["label"] for r in rows])}
```

| Metrica | Andamento atteso | Allarme |
|---|---|---|
| Loss di training | scende e si appiattisce | oscillazioni forti → lr troppo alto |
| BAcc macro su dev_select | sopra la baseline a una domanda | una fonte crolla mentre le altre salgono |
| Soglia ottimale | scende da 0,90–0,95 verso 0,4–0,6 | resta > 0,85 dopo metà dello stadio 1 → pochi negativi "non stabilito" |
| BAcc a 0,5 | vicina a quella a soglia ottimale | distanza > 1 punto |
| ECE | stabile o in calo | sale molto → sovra-confidenza |
| Solo claim (200 righe con documento vuoto) | P media bassa | sale → il modello guarda solo il claim |

## 10. Scelta del checkpoint

1. Fra i checkpoint dello stadio 2 (o del pilota): BAcc macro più alta su dev_select; a parità (entro 0,3) ECE più bassa.
2. Valutazione completa del checkpoint scelto con vLLM sui pesi fusi ([08](08_merge_export.md)) su dev_select e conferma su dev_confirm.
3. AggreFact non entra nella scelta.

## 11. Tempi (da misurare nei primi 100 passi)

| Corsa | Esempi | Token (≈) | 1 L40S a ~1.000 token/s |
|---|---:|---:|---|
| `fc_pilot`, d1-3B | ~27k | ~15M | ~1–1,5 h (stima: ~3 volte meno calcolo per token di JPT); sul portatile 2–3 h |
| stadi 1 + 2, d1-3B | ~90k | ~70M | ~6–8 h |
| `fc_pilot`, JPT | ~27k | ~15M | ~4 h (QLoRA sul portatile: ~2× più lento) |
| stadi 1 + 2, JPT | ~90k | ~70M | ~20 h |
| stadi 1 + 2, Winnow | ~90k | ~70M | ~30 h; ~15 h su 2 L40S |

## 12. Problemi tipici

| Sintomo | Causa probabile | Cosa fare |
|---|---|---|
| La loss non scende | risposte o `take` invertiti; `answer_pos` sbagliata | test di [03](03_rendering_formato.md) §8 e di questo file §8 |
| BAcc giù dopo pochi passi | lr troppo alto; prompt diverso dal benchmark | lr 1e-5; confronto token per token |
| Out of memory | sequenze lunghe; logit del vocabolario calcolati | batch fisico 1, gradient checkpointing, solo `answer_pos` |
| Soglia ottimale bloccata in alto | pochi negativi `not_established` | più `remove_evidence` e C2D |
| Probabilità tutte 0/1 | sovra-confidenza | smoothing 0,05, Brier, β·KL, temperatura rifittata |
| Nessun parametro addestrabile | regex dei target non corrisponde ai nomi PEFT | stampare `[n for n, _ in model.named_modules()][:50]` e correggere |

## 13. Riproducibilità

In `finetune/outputs/<esperimento>/config.json`: argomenti, SHA-256 del file renderizzato e del mix, revisione dei pesi, `question.json`, versioni di torch/transformers/peft, macchina, durata, seed.

## 14. Appendice — L'encoder stile FactCG (controllo A5 e studente della distillazione)

| Parametro | Valore |
|---|---|
| Backbone | `microsoft/deberta-v3-large` (alternativa: `answerdotai/ModernBERT-large`, contesto 8.192) |
| Classe | `AutoModelForSequenceClassification`, `num_labels = 2` (testa nuova) |
| Input | stringa unica, template FLAN di FactCG: `{chunk}\n\nChoose your answer: based on the paragraph above can we conclude that "{claim}"?\n\nOPTIONS:\n- Yes\n- No\nI think the answer is ` |
| Lunghezza | 2.048 token (DeBERTa); documenti divisi in chunk di 550 parole (`FactCG/chunking.py`), positivi oltre il budget scartati come in [03](03_rendering_formato.md) §7 |
| Training | tutti i pesi; 2 stadi; lr 1e-5; warmup 6%; decadimento lineare; AdamW ε = 1e-6; batch effettivo 32–64; 1 epoca per stadio |
| Precisione | bf16 |
| Hardware | portatile 24 GB: batch fisico 2 a 2.048 token con gradient checkpointing |
| Inferenza | softmax, classe 1 = supportato; `max` sui chunk; soglia 0,5 e fittata |

Si può usare `transformers.Trainer` standard (la loss è la CE della classe di classificazione). Per la distillazione il target è la probabilità dell'insegnante ([08](08_merge_export.md) §6).

## Criteri per passare allo step successivo

- Test §8 superati.
- Un checkpoint che batte la baseline a una domanda su dev_select, confermato su dev_confirm, con soglia ottimale fra 0,3 e 0,7 ed ECE non peggiore.
