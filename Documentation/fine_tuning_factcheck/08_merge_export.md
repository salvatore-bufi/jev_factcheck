# Step 8 — Fusione, soglia finale, congelamento, test, export

## Obiettivo

Trasformare l'adapter scelto in un modello pronto all'uso: pesi fusi, verifica che le probabilità non cambino nel passaggio, temperatura e soglia definitive, congelamento, **una** valutazione sul test, pacchetto finale. In più due opzioni: estrarre la testa binaria e distillare un encoder.

## 1. Fusione dell'adapter — `finetune/merge.py`

Fondere significa calcolare W' = W + (α/r)·B·A una volta e salvare pesi normali: in inferenza non serve più PEFT e la velocità è quella del modello originale.

```python
import shutil, torch, transformers
from pathlib import Path
from peft import PeftModel

def merge(base_path, revision, adapter_dir, out_dir, copy_from=None):
    cfg = transformers.AutoConfig.from_pretrained(base_path, revision=revision)
    cls = getattr(transformers, cfg.architectures[0])
    base = cls.from_pretrained(base_path, revision=revision, torch_dtype=torch.bfloat16)   # sempre bf16
    model = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    model.save_pretrained(out_dir, safe_serialization=True, max_shard_size="4GB")
    if copy_from:                                   # Winnow: tokenizer e template dalla cartella convertita
        for f in Path(copy_from).iterdir():
            if f.suffix in (".json", ".jinja", ".model", ".txt") and "safetensors" not in f.name and f.name != "config.json":
                shutil.copy(f, Path(out_dir) / f.name)
    else:                                           # JPT: processor e tokenizer della revisione pinnata
        transformers.AutoProcessor.from_pretrained(base_path, revision=revision).save_pretrained(out_dir)

# d1-3B: pesi fusi + i file di codice del repository (l'API system_one serve anche in produzione)
D1_REV = "051bcc464b01b9f92942b364d9586b0ef5912432"
merge("LiquidAI/d1-3B", D1_REV, "finetune/outputs/D1/stage2/final", "finetune/release/d1-3b-fc-v1")
from huggingface_hub import snapshot_download
src = Path(snapshot_download("LiquidAI/d1-3B", revision=D1_REV))
for name in ("api.py", "runner.py", "prompt.py", "hybrid.py", "lfm2_vl.py", "modeling_d1.py",
             "chat_template.jinja", "processor_config.json", "tokenizer.json", "tokenizer_config.json", "LICENSE"):
    shutil.copy(src / name, Path("finetune/release/d1-3b-fc-v1") / name)
# config.json: tenere quello originale (contiene "auto_map", che fa caricare D1Model con trust_remote_code)
shutil.copy(src / "config.json", Path("finetune/release/d1-3b-fc-v1") / "config.json")

# JPT
merge("kirp/jpt-9b", "b447cc7ee105c0a76a22f8fde8ecf05074dc8be0", "finetune/outputs/J1/stage2/final",
      "finetune/release/jpt-9b-fc-v1")
# Winnow
merge("winnow-12b-bf16/model", None, "finetune/outputs/W1/stage2/final",
      "finetune/release/winnow-12b-fc-v1", copy_from="winnow-12b-bf16/model")
```

Regole:

- fondere **sempre su pesi bf16**, anche se il training era QLoRA (fondere su pesi a 4 bit degrada il modello);
- copiare tokenizer, chat template e processor identici a quelli di partenza;
- controllare che `config.json` del modello fuso abbia la stessa architettura e lo stesso `model_type` della partenza;
- per d1-3B, verificare che i nomi dei tensori salvati dalla classe standard di transformers coincidano con quelli del `model.safetensors` originale (stesso elenco di chiavi): l'API del repository li carica per nome.

## 2. Verifica dopo la fusione

Su 200 righe di dev_select (5 fonti), confrontare P(supportato) del modello **con adapter** (PyTorch, `quick_eval` di [05](05_training.md) §9) e del modello **fuso** servito come nei benchmark (`finetune/score.py --model-dir ...`, vLLM):

| Controllo | Soglia |
|---|---|
| Differenza media di P | < 0,005 |
| Decisioni diverse (alla soglia di dev) | al massimo 1 su 200 |

Differenze maggiori indicano un errore nella fusione o nei file copiati (tipicamente tokenizer o template).

## 3. Temperatura e soglia finali

Dopo il fine-tuning le probabilità sono cambiate: scala e soglia della baseline non valgono più.

1. `finetune/score.py` con il modello fuso su dev_select e dev_confirm.
2. Scala a (minimo log-loss), poi soglia (BAcc macro, griglia 0,01–0,99) su dev_select ([01](01_baseline.md) §4).
3. Conferma su dev_confirm (calo ≤ 1 punto).
4. Se la BAcc a **0,5** (con la scala fittata) è entro 0,5 punti da quella alla soglia fittata, **in produzione si usa 0,5**: è più robusta ai cambi di dominio (obiettivo di MiniCheck). La soglia fittata si riporta nel report.
5. Soglia **secondaria**, dichiarata: fittata sulle 4.520 righe del dev di AggreFact (`dev_split.py`), applicata solo al test di AggreFact, per confronto con i numeri storici della repo.

Il file `decision.json`:

```json
{"model": "winnow-12b-fc-v1",
 "question_file": "finetune/question.json", "question_sha256": "<...>",
 "read": "P(B)", "letters": {"A": 236776, "B": 236799},
 "base_temperature": 1.0, "fitted_scale": 1.12, "threshold": 0.50,
 "chunking": {"max_prompt_tokens": 16384, "aggregate": "max"},
 "fitted_on": "dev_select", "confirmed_on": "dev_confirm",
 "manifest_sha256": "<sha256 di data/finetune/splits/manifest.json>"}
```

(I numeri sono un esempio di formato, non un risultato.)

## 4. Opzionale: estrarre la testa binaria

Il fact checker legge solo σ(s) con s calcolato da due righe della testa del vocabolario. Per **JPT** (nessun softcapping) si può salvare una testa lineare equivalente:

```python
W = model.lm_head.weight                                      # [V, 4096]
u = (W[357] - W[417]) / (1.087 * fitted_scale)                # P(supportato) = σ(u · h)
score_weight = torch.stack([torch.zeros_like(u), u])          # testa a 2 classi: logit [0, u·h]
```

Con `score_weight` come pesi di una testa `ForSequenceClassification` a 2 classi (classe 1 = supportato) il modello si può servire come classificatore, senza proiettare sul vocabolario: utile se il server supporta classificatori per l'architettura (vLLM ha un modo di pooling per la classificazione; per l'architettura Qwen3.5 va verificato). Il classificatore legge l'**ultimo** token: il prompt deve terminare esattamente come in training.

Per **Winnow** il softcapping (30·tanh(z/30)) rende la funzione non lineare in h: la testa estratta non è esattamente equivalente. Si continua a leggere i logprob delle due lettere.

Verifica in ogni caso: 200 righe, stessa P del modello generativo (differenza < 0,005).

## 5. Congelamento — `frozen.json`

Prima di guardare qualunque test:

```python
import hashlib, json, glob
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
frozen = {
    "weights": {f: sha(f) for f in sorted(glob.glob("finetune/release/winnow-12b-fc-v1/*.safetensors"))},
    "tokenizer": {f: sha(f) for f in sorted(glob.glob("finetune/release/winnow-12b-fc-v1/tokenizer*"))},
    "decision": json.load(open("finetune/release/winnow-12b-fc-v1/decision.json")),
    "question_sha256": sha("finetune/question.json"),
    "splits_manifest_sha256": sha("data/finetune/splits/manifest.json"),
    "mix_sha256": {m: sha(f"data/finetune/{m}/train.jsonl") for m in ("mix_fc_stage1", "mix_fc_stage2")},
    "training_config": json.load(open("finetune/outputs/W1/stage2/config.json")),
    "eval_code_sha256": {f: sha(f) for f in ("finetune/score.py", "finetune/report.py", "metrics.py")},
}
json.dump(frozen, open("finetune/release/winnow-12b-fc-v1/frozen.json", "w"), indent=2)
```

Dopo il congelamento nessuna modifica in risposta ai risultati del test (stesso schema già usato per il risultato di Jev in `results/frozen.json`).

## 6. Valutazione finale

Una sola esecuzione di [06](06_valutazione.md) §6 su test_id, test_ood, test_long e AggreFact, con il report nel formato di [06](06_valutazione.md) §7, la tabella AggreFact per fonte e accanto: baseline, controllo C1, encoder A5, FactCG, MiniCheck, Bespoke.

## 7. Servizio in produzione

### 7.0 d1-3B con la sua API (consigliato per d1-3B)

Il modello fuso si serve come l'originale, con l'API del repository (stesse classi, `trust_remote_code`):

```python
import json, torch
from transformers import AutoModel
model = AutoModel.from_pretrained("finetune/release/d1-3b-fc-v1", trust_remote_code=True,
                                  dtype=torch.bfloat16).to("cuda")
q = json.load(open("finetune/question.json"))
res = model.system_one_batch([({"document": doc, "claim": claim}, {q["question_id"]: q["question"]})
                              for doc, claim in pairs])
p_yes = [r["answers"][q["question_id"]]["noul"] for r in res]
p_supported = [1 - p if q.get("take") == "no" else p for p in p_yes]   # Q4: 1 - P(sì)
```

La scala di temperatura fittata si applica al margine: s = ln(p/(1−p)) / scala. Gli autori indicano `model.compile(mode="reduce-overhead")` per le domande singole su GPU NVIDIA (CUDA graph). Misurato nei run di ottobre (L40S, tre domande, senza compile): latenza p50 38 ms, ~24 righe/s, 11,6 GB di memoria di picco.

### 7.1 JPT e Winnow con vLLM (consigliato)

Come nei benchmark: il modello fuso si carica con vLLM, si genera **un token** con temperatura 0 e si leggono i logprob delle due lettere:

```python
from vllm import LLM, SamplingParams
llm = LLM(model="finetune/release/jpt-9b-fc-v1", dtype="bfloat16", max_model_len=32768,
          enable_prefix_caching=True, max_logprobs=20)
params = SamplingParams(max_tokens=1, temperature=0.0, logprobs=2, logprob_token_ids=[357, 417])
out = llm.generate([{"prompt_token_ids": ids}], params)[0]       # ids da render.py
lp = out.outputs[0].logprobs[0]
la, lb = lp[357].logprob, lp[417].logprob                        # logprob = logit − costante: la softmax la elimina
s = (la - lb) / (1.087 * fitted_scale)
p_supported = 1 / (1 + math.exp(-s))
decision = p_supported > threshold
```

Per Winnow: lettere 236776 (A, falso) e 236799 (B, vero), `s = (lb − la) / (1.0 · fitted_scale)`. I logprob restituiti da vLLM sono già dopo il softcapping.

Documenti più lunghi di C token: chunking per frasi e `max` ([03](03_rendering_formato.md) §7.3).

### 7.2 Winnow in formato GGUF (facoltativo)

Il server ufficiale di Winnow (llama.cpp modificato, `winnow-12b/serve.sh`) legge solo GGUF e verifica gli hash dei modelli dichiarati nei manifest (`winnow-12b/src/manifests/models.json`). Per usare il modello fine-tunato:

1. Conversione con il convertitore di llama.cpp incluso nel runtime di Winnow:
   ```sh
   conda run -n factcheck python winnow-12b/src/.runtime/llama.cpp/convert_hf_to_gguf.py \
       finetune/release/winnow-12b-fc-v1 --outtype bf16 --outfile finetune/release/winnow-12b-fc-v1/Winnow-12B-FC-BF16.gguf
   ```
   La cartella fusa ha una config solo testo (`gemma4_text`, prodotta da `winnow-12b-bf16/convert.py`): se il convertitore non la riconosce, usare la `config.json` multimodale di `google/gemma-4-12B-it` adattata, o aggiungere la mappatura inversa di `convert.py`. Da verificare.
2. Quantizzazione in Q8_0 con `llama-quantize` (da compilare nello stesso albero: `cmake --build <build dir> --target llama-quantize`; lo script `scripts/build.py` di Winnow oggi compila solo `llama-server` e `winnow-unit`).
3. Registrare il nuovo file per il server: il server accetta modelli dichiarati nei manifest con il loro SHA-256; serve una voce o un manifest locale con l'hash del nuovo GGUF (procedura da verificare in `winnow-12b/src/docs/INSTALL.md` e `scripts/winnow.py`).
4. **Verifica**: le stesse 63 righe usate per la conversione originale, probabilità del server contro vLLM sui pesi bf16. Riferimento della conversione originale (Q8_0): differenze medie 0,007 / 0,019 / 0,015 sulle tre domande, 62 decisioni uguali su 63.

Se non serve il server di Winnow, la strada 7.1 è più semplice e non richiede nessuno di questi passi.

## 8. Opzionale: distillazione in un encoder

Quando il costo d'inferenza è il vincolo principale.

1. **Dati per lo studente**: tutte le righe dei mix più un insieme grande di coppie **solo dal train**, anche senza etichetta verificata: VitaminC e FEVER interi, tutte le coppie D2C generate prima del filtro del giudice, chunk di documenti con i loro riassunti. Obiettivo: 200.000–500.000 coppie.
2. **Etichette dell'insegnante**: P(supportato) del modello finale fuso, calcolata **sui chunk che vedrà lo studente** (550 parole, `FactCG/chunking.py`), così insegnante e studente vedono lo stesso testo. Costo: un forward breve per chunk; ~1 giorno su una L40S per 500k chunk.
3. **Training dello studente** (configurazione in [05](05_training.md) §14), loss:
   ```text
   L = KL(p_insegnante ‖ p_studente) + 0,5 · CE(etichetta vera)       (il secondo termine solo dove l'etichetta esiste)
   ```
4. **Valutazione** sulla stessa suite. Obiettivo indicativo: studente ≥ 75,6 su AggreFact (FactCG) e a non più di 1,5 punti dall'insegnante sulla media dei blocchi.

## 9. Pacchetto finale

```text
finetune/release/<modello>-fc-<versione>/
  *.safetensors, config.json, tokenizer*, chat_template*   pesi fusi
  decision.json        domanda, lettere, scala, soglia, chunking
  frozen.json          hash e provenienza
  REPORT.md            risultati (06 §7), limiti, dati usati, licenze
  MODEL_CARD.md        uso previsto e limiti
```

Contenuto minimo della model card:

- **uso previsto**: dato un documento e un claim in inglese, probabilità che il documento supporti l'intero claim;
- **non fa**: verifica nel mondo reale senza documento; retrieval; distinzione fra contraddetto e non stabilito; altre domande da decision model (non più valutate);
- **dati**: elenco delle fonti e delle famiglie generate, con licenze;
- **risultati**: tabella di [06](06_valutazione.md) §7.

**Licenze.** d1-3B è sotto LFM Open License v1.0: il derivato si può usare commercialmente solo da entità con fatturato annuo sotto 10 milioni di dollari (e la licenza va distribuita con il modello). JPT-9B è CC BY-NC 4.0: anche il derivato è non commerciale. Winnow-12B è Apache 2.0, ma il derivato eredita i vincoli dei dati: ANLI, FactKG, AVeriTeC sono non commerciali; MiniCheck (MIT), SummEdits (CC BY), HoVer/FEVER/VitaminC (CC BY-SA), CNN/DailyMail (Apache 2.0) e i dati generati con modelli aperti non pongono vincoli aggiuntivi. Un modello commerciale va addestrato senza ANLI, FactKG e AVeriTeC.

## Criteri di completamento

- Verifica della fusione superata (e del GGUF, se usato).
- Scala e soglia da dev_select, confermate su dev_confirm; scelta 0,5 vs soglia fittata documentata.
- `frozen.json` prima del test; test eseguito una volta; report e model card scritti.
