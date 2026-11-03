# Step 5 — Il training: LoRA sul decision model, passo per passo

## Obiettivo

Addestrare un adapter LoRA prima su KnowLine-4B (prove), poi su elephant-1 (modello principale a 9B), poi sugli altri decision model e sui due backbone (B-ctrl), con la loss di [04](04_loss.md), il curriculum di [02e](02e_mix_finale.md) e i dati renderizzati di [03](03_rendering_formato.md); controllare apprendimento e calibrazione durante il training; salvare checkpoint confrontabili.

## 1. LoRA in breve

Per una matrice congelata W (forma d_out × d_in) LoRA aggiunge B (d_out × r) e A (r × d_in): W' = W + (α/r)·B·A. Solo A e B si addestrano; B parte da zero, quindi **al passo 0 il modello è identico a quello di partenza**.

Si aggiunge sempre una **nuova** LoRA sopra i pesi pubblicati, qualunque sia il training originale:
- modelli pubblicati come pesi completi (8 su 10): W_finale = W_pubblicato + ΔW_nostro;
- **Nimble-9B-v3** è pubblicato come LoRA (r = 64) su Qwen3.5-9B: prima si fonde la sua LoRA nel backbone (con PEFT, `merge_and_unload`), si verifica che le risposte non cambino, poi si aggiunge la nostra come per gli altri.

## 2. Moduli su cui mettere LoRA

Tutti i 12 modelli sono Qwen3.5 (`qwen3_5_text`, 32 layer: 24 Gated DeltaNet e 8 ad attenzione piena, ai layer 3, 7, …, 31), quindi i nomi dei moduli sono gli stessi:

```python
QWEN35_TARGETS = (r".*layers\.\d+\."
                  r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)"
                  r"|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)"
                  r"|mlp\.(gate_proj|up_proj|down_proj))")
```

- `self_attn.*` esiste solo negli 8 layer pieni; `linear_attn.*` solo nei 24 DeltaNet;
- `linear_attn.in_proj_a` e `in_proj_b` sono proiezioni piccolissime (una per testa): si lasciano congelate;
- la torre visiva (presente nei modelli caricati come multimodali, es. StartLux, d3-flash) non si tocca: l'espressione regolare non la prende se i suoi moduli hanno nomi diversi; si controlla con `print_trainable_parameters` e con l'elenco dei moduli adattati;
- **teste proprie**: la testa a 255 vie di pplx-decider-flash (~0,65 M parametri) e la testa congiunta di Clef-Flash si addestrano per intero (`modules_to_save` di PEFT), non con LoRA.

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
| Lunghezza massima | 8.192 | 8.192 | 8.192 | |
| Precisione | bf16; gradient checkpointing | idem | idem | |
| Loss | CE sulle due opzioni | idem | idem | [04](04_loss.md) |
| Checkpoint | ogni 20% | ogni 20% | ogni 10% | |
| Seed | 13 | 13 (3 seed per la configurazione finale: 13, 14, 15) | | |

Stessa configurazione per tutti i modelli, decision model e backbone: è la condizione di B-ctrl ("stessa ricetta").

## 4. Hardware e memoria

| Macchina | Gruppo 4B (KnowLine, StartLux-4B, pplx, Jeffan, Qwen3.5-4B) | Gruppo 9B (elephant-1, StartLux-9B, d3-flash, OneJev, Nimble, Clef, Qwen3.5-9B) |
|---|---|---|
| Portatile RTX 5090, 24 GB | **LoRA in bf16** (~9 GB di pesi), 8k token, batch fisico 1–2 | QLoRA (pesi a 4 bit), 4k token: solo per prove |
| L40S 45 GB | LoRA in bf16, 16k token o batch più grandi | **LoRA in bf16** (~19 GB di pesi), 8k token, batch fisico 1 |

Stima per un 9B su L40S a 8k token: pesi 19 GB + adapter e stati dell'ottimizzatore ~1 GB + attivazioni con gradient checkpointing ~8–12 GB ≈ 28–32 GB.

**Trucco di memoria**: non calcolare i logit del vocabolario intero (~248.000 token × 8.192 posizioni ≈ 4 GB in bf16 per esempio). Si prende lo stato finale **solo** nella posizione di lettura e lo si moltiplica per le righe delle due opzioni (`margin` di [04](04_loss.md) §7); per pplx e Clef si chiama solo la loro testa.

QLoRA cambia leggermente i pesi base: va bene per le prove, ma i numeri definitivi si misurano sui pesi bf16 fusi ([08](08_merge_export.md)).

## 5. Installazione

```sh
pip install peft flash-linear-attention causal-conv1d
python -c "import peft, transformers, fla; print(peft.__version__, transformers.__version__)"
```

Senza `flash-linear-attention` i layer DeltaNet usano un'implementazione in torch molto più lenta. Se `peft` richiede una versione di transformers diversa da quella dell'ambiente, creare un ambiente separato per il training (`conda create -n factcheck-train --clone factcheck`) invece di aggiornare transformers.

## 6. Lo script: `finetune/train.py`

Lo script è uguale per tutti i modelli: ciò che cambia (caricamento, maschera, lettura) sta nel renderer del modello ([03](03_rendering_formato.md) §4).

```python
"""LoRA training of a decision model on the two option scores it reads.
accelerate launch finetune/train.py --model knowline-4b --data data/finetune/rendered/knowline-4b/fc_pilot/train.jsonl \
    --out finetune/outputs/P0-knowline --lr 2e-5 [--init-adapter finetune/outputs/S1/final] [--qlora]"""
import argparse, json, math, random, time
from pathlib import Path
import torch
import transformers
from accelerate import Accelerator
from peft import LoraConfig, PeftModel, get_peft_model
from losses import binary_loss, anchor_kl
from render import RENDERERS, QWEN35_TARGETS          # un renderer per modello, con load() e forward_model()

TAKE = json.load(open("finetune/question.json")).get("take", "yes")

def batches(path, micro_bs, seed):
    rows = [json.loads(l) for l in open(path)]
    random.Random(seed).shuffle(rows)
    rows.sort(key=lambda r: len(r["input_ids"]) // 256)        # lunghezze simili: meno padding
    groups = [rows[i:i + micro_bs] for i in range(0, len(rows), micro_bs)]
    random.Random(seed + 1).shuffle(groups)
    return groups

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=RENDERERS, required=True)
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
    r = RENDERERS[a.model]()
    model = r.load(qlora=a.qlora)                              # pesi pubblicati (Nimble: LoRA degli autori già fusa)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    if a.init_adapter:                                         # stadio 2: si continua lo stesso adapter
        model = PeftModel.from_pretrained(model, a.init_adapter, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, bias="none",
                                                 target_modules=QWEN35_TARGETS,
                                                 modules_to_save=r.own_head_modules))   # pplx, Clef: la loro testa
    model.print_trainable_parameters()
    groups = batches(a.data, a.micro_bs, a.seed)
    steps = math.ceil(len(groups) / a.accum)
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=a.lr, weight_decay=0.0)
    sched = transformers.get_cosine_schedule_with_warmup(opt, int(a.warmup * steps), steps)
    model, opt, sched = acc.prepare(model, opt, sched)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    log = open(Path(a.out) / "log.jsonl", "a") if acc.is_main_process else None
    model.train(); t0 = time.time()
    for i, group in enumerate(groups[acc.process_index::acc.num_processes]):
        y = torch.tensor([float(x["label"]) for x in group], device=acc.device)
        p0 = torch.tensor([x.get("teacher_p") or 0.5 for x in group], device=acc.device)
        with acc.accumulate(model):
            s = r.margin(model, group, TAKE)                   # forward con maschera e lettura del modello -> s [B]
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

`r.margin(model, group, take)` fa, per ogni modello, il forward con la sua maschera e restituisce il margine:
- lettura dalla testa del vocabolario: stato finale nella posizione di lettura × righe delle due opzioni (`margin` di [04](04_loss.md) §7);
- pplx e Clef: la loro testa sullo stato finale, poi `margin_from_logits`;
- backbone: le due lettere del prompt con l'ordine scelto per quell'esempio ([03](03_rendering_formato.md) §2.4).

Punti da controllare quando si scrive il codice vero:
- il modulo che restituisce lo stato **dopo** la normalizzazione finale (per KnowLine la temperatura è incorporata proprio lì: non va saltata);
- con `get_peft_model` i nomi dei moduli acquisiscono il prefisso `base_model.model.`: l'espressione regolare inizia con `.*` per questo;
- con più GPU, accelerate fa avanzare lo scheduler una volta per processo: controllare nel log che il learning rate arrivi a ~0 alla fine dell'epoca.

## 7. Comandi

```sh
# render
python finetune/render.py --model knowline-4b --mix fc_pilot
# pilota KnowLine-4B (portatile, o server)
accelerate launch --num_processes 1 finetune/train.py --model knowline-4b \
    --data data/finetune/rendered/knowline-4b/fc_pilot/train.jsonl --out finetune/outputs/P0-knowline --lr 2e-5 --accum 32
# stadio 1 e 2 su elephant-1, sul server
gpu-run --batch --name ft-elephant-s1 -- bash -c 'accelerate launch --num_processes 1 finetune/train.py --model elephant-1 \
    --data data/finetune/rendered/elephant-1/fc_stage1/train.jsonl --out finetune/outputs/E1/stage1 --lr 5e-5 \
    > logs/ft-elephant-s1.out 2>&1'
gpu-run --batch --name ft-elephant-s2 -- bash -c 'accelerate launch --num_processes 1 finetune/train.py --model elephant-1 \
    --data data/finetune/rendered/elephant-1/fc_stage2/train.jsonl --out finetune/outputs/E1/stage2 --lr 1e-5 \
    --init-adapter finetune/outputs/E1/stage1/final > logs/ft-elephant-s2.out 2>&1'
```

Se una corsa supera le 24 ore, aggiungere `--time` a `gpu-run`. Per i backbone: `--model qwen3.5-9b` / `qwen3.5-4b`, stessi dati e iperparametri.

## 8. Test prima del primo training (obbligatori, per ogni modello)

| Test | Come | Atteso |
|---|---|---|
| **Identità con il benchmark** | [03](03_rendering_formato.md) §8: P(supportato) da `r.margin` (in eval) contro l'adapter del benchmark | differenza media < 0,005 |
| **Passo zero = baseline** | adapter appena creato (B = 0): P(supportato) su 200 righe di dev_select | uguale (entro ~0,01 per il bf16) alla baseline di [01](01_baseline.md) |
| **Gradienti con la maschera** | per i modelli bidirezionali: un passo di training con il codice degli autori; i gradienti della LoRA nei layer pieni sono diversi da zero e la maschera resta bidirezionale (test della maschera di [03](03_rendering_formato.md) §8) | |
| **Overfit di 64 esempi** | 300 passi su 64 esempi, lr 1e-4 | loss < 0,05: la pipeline impara |
| **Parametri addestrabili** | `print_trainable_parameters` | > 0, decine di milioni per r = 16; per pplx e Clef anche la testa |

Se il passo zero non coincide con la baseline: prompt, maschera, opzioni o `take` sbagliati; correggere prima di addestrare.

## 9. Valutazione durante il training — `finetune/quick_eval.py`

A ogni checkpoint, su un campione fisso di dev_select (al massimo 200 righe per fonte, ~1.400 righe), con `r.margin` in eval:

| Metrica | Andamento atteso | Allarme |
|---|---|---|
| Loss di training | scende e si appiattisce | oscillazioni forti → lr troppo alto |
| BAcc macro su dev_select | sopra la baseline a una domanda | una fonte crolla mentre le altre salgono |
| Soglia ottimale | si avvicina a 0,4–0,6 | resta > 0,85 dopo metà dello stadio 1 → pochi negativi "non stabilito" |
| BAcc a 0,5 | vicina a quella a soglia ottimale | distanza > 1 punto |
| ECE | stabile o in calo | sale molto → sovra-confidenza |
| Solo claim (200 righe con documento vuoto) | P media bassa | sale → il modello guarda solo il claim |

## 10. Scelta del checkpoint

1. Fra i checkpoint dello stadio 2 (o del pilota): BAcc macro più alta su dev_select; a parità (entro 0,3) ECE più bassa.
2. Valutazione completa del checkpoint scelto sui pesi fusi, con l'adapter del benchmark ([08](08_merge_export.md)), su dev_select e conferma su dev_confirm.
3. AggreFact non entra nella scelta.

## 11. Problemi tipici

| Sintomo | Causa probabile | Cosa fare |
|---|---|---|
| La loss non scende | opzioni o `take` invertiti; posizione di lettura sbagliata | test di [03](03_rendering_formato.md) §8 e di questo file §8 |
| Passo zero diverso dalla baseline solo per i modelli bidirezionali | maschera non applicata in training | test della maschera; usare il forward degli autori |
| BAcc giù dopo pochi passi | lr troppo alto; prompt diverso dal benchmark | lr 1e-5; confronto token per token |
| Out of memory | sequenze lunghe; logit del vocabolario calcolati | batch fisico 1, gradient checkpointing, solo la posizione di lettura |
| Training molto lento | kernel DeltaNet non installati | `flash-linear-attention`, `causal-conv1d` |
| Soglia ottimale bloccata in alto | pochi negativi `not_established` | più `remove_evidence` e C2D |
| Probabilità tutte 0/1 | sovra-confidenza | smoothing 0,05, Brier, β·KL, temperatura rifittata |
| Nessun parametro addestrabile | regex dei target non corrisponde ai nomi | stampare `[n for n, _ in model.named_modules()][:50]` e correggere |

## 12. Riproducibilità

In `finetune/outputs/<esperimento>/config.json`: argomenti, SHA-256 del file renderizzato e del mix, modello e revisione dei pesi, `question.json`, versioni di torch/transformers/peft, macchina, durata, seed.

## 13. Appendice — L'encoder stile FactCG (controllo A5 e studente della distillazione)

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

- Test §8 superati per il modello da addestrare.
- Un checkpoint che batte la baseline a una domanda su dev_select, confermato su dev_confirm, con soglia ottimale fra 0,3 e 0,7 ed ECE non peggiore.
