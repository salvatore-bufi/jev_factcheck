# Step 8 — Fusione, soglia finale, congelamento, test, export

## Obiettivo

Trasformare l'adapter scelto in un modello pronto all'uso: pesi fusi **caricabili dal runtime degli autori**, verifica che le probabilità non cambino nel passaggio, temperatura e soglia definitive, congelamento, **una** valutazione sul test, pacchetto finale. In più due opzioni: estrarre la testa binaria e distillare un encoder.

## 1. Fusione dell'adapter — `finetune/merge.py`

Fondere significa calcolare W' = W + (α/r)·B·A una volta e salvare pesi normali: in inferenza non serve più PEFT e la velocità è quella del modello originale.

Il punto delicato: ogni modello si serve con il **codice dei suoi autori**, che legge dalla cartella dei pesi anche file propri. La cartella fusa deve quindi essere una **copia della cartella originale** in cui cambiano solo i tensori.

```python
import shutil, torch
from pathlib import Path
from huggingface_hub import snapshot_download
from peft import PeftModel

def merge(repo, revision, adapter_dir, out_dir, load_fn):
    src = Path(snapshot_download(repo, revision=revision))
    out = Path(out_dir)
    shutil.copytree(src, out, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("*.safetensors", "*.safetensors.index.json"))
    base = load_fn(src, dtype=torch.bfloat16)                    # sempre bf16, con lo stesso caricamento del training
    model = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    model.save_pretrained(out, safe_serialization=True, max_shard_size="4GB")
    shutil.copy(src / "config.json", out / "config.json")        # il config originale (modeling_overrides, auto_map, …)
```

File da conservare, per modello (copiati da `copytree`; da controllare uno per uno):

| Modello | File oltre ai tensori |
|---|---|
| elephant-1 | `config.json` con `modeling_overrides` (la maschera a blocchi), tokenizer, template |
| StartLux 9B / 4B | `decision_config.json` (temperature per tipo), tokenizer, template |
| d3-flash | codice del modello (`trust_remote_code`), `decision_config.json` |
| OneJev1.6-9B | `decision_config.json` (`onejev-slot-v2`, per Flide) |
| Nimble-9B-v3 | il motore pretende `schema_config.json` e la revisione del backbone che fissa: si rilasciano i pesi fusi **più** un `schema_config.json` aggiornato, oppure un nuovo adapter LoRA sul backbone (la LoRA degli autori fusa con la nostra, salvata come adapter) |
| Clef-Flash | `joint_schema_model.py` e i pesi della testa congiunta (aggiornati dal training) |
| KnowLine-4B | `knowline_server.py`, `knowline_layout.py`, template (la temperatura resta nei pesi della normalizzazione finale) |
| pplx-decider-flash | il pacchetto `source/src/pplx_decider`, `readout.safetensors` (aggiornato dal training), `decision_config.json` |
| Jeffan-4B | `inference/jeffan_engine`, `dm_config.json` |

Regole:
- fondere **sempre su pesi bf16**, anche se il training era QLoRA;
- le teste proprie addestrate (pplx, Clef) si salvano nel loro file, con lo stesso nome e la stessa forma;
- controllare che l'elenco delle chiavi dei tensori salvati sia identico a quello originale: i codici degli autori caricano per nome.

## 2. Verifica dopo la fusione

Su 200 righe di dev_select (5 fonti), confrontare P(supportato) del modello **con adapter** (PyTorch, `quick_eval` di [05](05_training.md) §9) e del modello **fuso servito dall'adapter del benchmark** (che usa il codice degli autori, con i pesi della cartella fusa al posto di quelli del Hub):

| Controllo | Soglia |
|---|---|
| Differenza media di P | < 0,005 |
| Decisioni diverse (alla soglia di dev) | al massimo 1 su 200 |

Differenze maggiori indicano un errore nella fusione o nei file copiati (tipicamente config, tokenizer o template). Per StartLux il controllo passa dal server vLLM con il plugin bidirezionale: è anche la verifica che la maschera usata in training fosse quella giusta.

Gli adapter della repo leggono i pesi dalla cache del Hub a una revisione fissata: per servire la cartella fusa serve un'opzione per il percorso locale dei pesi (per alcuni adapter c'è già, per gli altri va aggiunta in `models/<modello>/adapter.py`).

## 3. Temperatura e soglia finali

Dopo il fine-tuning le probabilità sono cambiate: scala e soglia della baseline non valgono più.

1. `benchmark.py` con il modello fuso (adapter del modello, pesi locali) su dev_select e dev_confirm.
2. Scala a (minimo log-loss), poi soglia (BAcc macro, griglia 0,01–0,99) su dev_select ([01](01_baseline.md) §4).
3. Conferma su dev_confirm (calo ≤ 1 punto).
4. Se la BAcc a **0,5** (con la scala fittata) è entro 0,5 punti da quella alla soglia fittata, **in produzione si usa 0,5**: è più robusta ai cambi di dominio (obiettivo di MiniCheck). La soglia fittata si riporta nel report.
5. Soglia **secondaria**, dichiarata: fittata su AF-dev (`--split dev-jev4520`, 4.520 righe), applicata solo al test di AggreFact, per confronto con lo studio focalizzato (che usa la stessa soglia, T-AF).

Il file `decision.json`:

```json
{"model": "elephant-1-fc-v1",
 "question_file": "finetune/question.json", "question_sha256": "<...>",
 "read": "opzione 'true' del codice degli autori", "runtime": "qev@81ce62f + serve.patch",
 "base_temperature": 1.0, "fitted_scale": 1.12, "threshold": 0.50,
 "chunking": {"max_prompt_tokens": 16384, "aggregate": "max"},
 "fitted_on": "dev_select", "confirmed_on": "dev_confirm",
 "manifest_sha256": "<sha256 di data/finetune/splits/manifest.json>"}
```

(I numeri sono un esempio di formato, non un risultato.)

## 4. Opzionale: estrarre la testa binaria

Per i modelli con lettura dalla testa del vocabolario il fact checker legge solo σ(s), con s calcolato da due righe della testa. Si può salvare una testa lineare equivalente:

```python
W = model.lm_head.weight                                      # [V, d]
u = (W[id_sì] - W[id_no]) / (T * fitted_scale)                # P(supportato) = σ(u · h)  (take = yes)
score_weight = torch.stack([torch.zeros_like(u), u])          # testa a 2 classi: logit [0, u·h]
```

Con `score_weight` come pesi di una testa a 2 classi il modello si serve come classificatore, senza proiettare sul vocabolario. Funziona solo se il runtime degli autori accetta un classificatore; altrimenti non serve. Per pplx e Clef la testa propria fa già questo lavoro.

Verifica: 200 righe, stessa P del modello originale (differenza < 0,005).

## 5. Congelamento — `frozen.json`

Prima di guardare qualunque test:

```python
import hashlib, json, glob
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
frozen = {
    "weights": {f: sha(f) for f in sorted(glob.glob("finetune/release/elephant-1-fc-v1/*.safetensors"))},
    "tokenizer": {f: sha(f) for f in sorted(glob.glob("finetune/release/elephant-1-fc-v1/tokenizer*"))},
    "decision": json.load(open("finetune/release/elephant-1-fc-v1/decision.json")),
    "question_sha256": sha("finetune/question.json"),
    "splits_manifest_sha256": sha("data/finetune/splits/manifest.json"),
    "mix_sha256": {m: sha(f"data/finetune/{m}/train.jsonl") for m in ("mix_fc_stage1", "mix_fc_stage2")},
    "training_config": json.load(open("finetune/outputs/E1/stage2/config.json")),
    "eval_code_sha256": {f: sha(f) for f in ("finetune/report.py", "core/metrics.py")},
}
json.dump(frozen, open("finetune/release/elephant-1-fc-v1/frozen.json", "w"), indent=2)
```

Dopo il congelamento nessuna modifica in risposta ai risultati del test (stesso schema già usato per il risultato di Jev in `results/frozen.json`).

## 6. Valutazione finale

Una sola esecuzione di [06](06_valutazione.md) §6 su test_id, test_ood, test_long e AggreFact, con il report nel formato di [06](06_valutazione.md) §7, la tabella AggreFact per fonte e accanto: baseline, backbone fine-tunato (B-ctrl), encoder A5, FactCG, MiniCheck, Bespoke.

## 7. Servizio in produzione

Ogni modello fine-tunato si serve **come l'originale**, con il runtime degli autori: lo stesso che usa l'adapter del benchmark.

| Modello | Runtime |
|---|---|
| elephant-1 | motore `qev` in-process (transformers), cartella locale (il motore legge `modeling_overrides` solo da una cartella) |
| StartLux 9B / 4B | `startlux-decide serve --launch-vllm` (vLLM 0.31 con il plugin bidirezionale) |
| d3-flash | `AutoModel` con `trust_remote_code`, `prepare` / `run` |
| OneJev1.6-9B | `flide serve --model <cartella>` |
| Nimble-9B-v3 | `NimbleEngine` (copia in `models/nimble-9b/nimble`) |
| Clef-Flash | `joint_schema_model.py` (`load_release_model`) |
| KnowLine-4B | front end `knowline_server.py` (backend transformers o SGLang) |
| pplx-decider-flash | `pplx_decider`, percorso di riferimento |
| Jeffan-4B | `jeffan_engine` (percorso transformers) |

In produzione si fa una sola domanda (quella di `question.json`) e si legge P(supportato) con la scala e la soglia di `decision.json`: s = ln(p / (1 − p)) / scala, P = σ(s).

Documenti più lunghi di C token: chunking per frasi e `max` ([03](03_rendering_formato.md) §7.3). Tutti i runtime rifiutano richieste oltre 32.768 token (alcuni 65.536): non troncano.

## 8. Opzionale: distillazione in un encoder

Quando il costo d'inferenza è il vincolo principale.

1. **Dati per lo studente**: tutte le righe dei mix più un insieme grande di coppie **solo dal train**, anche senza etichetta verificata: VitaminC e FEVER interi, tutte le coppie D2C generate prima del filtro del giudice, chunk di documenti con i loro riassunti. Obiettivo: 200.000–500.000 coppie.
2. **Etichette dell'insegnante**: P(supportato) del modello finale fuso, calcolata **sui chunk che vedrà lo studente** (550 parole, `FactCG/chunking.py`), così insegnante e studente vedono lo stesso testo. Costo: un forward breve per chunk; ~1 giorno su una L40S per 500k chunk.
3. **Training dello studente** (configurazione in [05](05_training.md) §13), loss:
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

**Licenze.** Il derivato eredita la licenza dei pesi di partenza: MIT (elephant-1), Apache 2.0 (KnowLine-4B, d3-flash, OneJev, Clef-Flash, pplx-decider-flash, Jeffan-4B, Qwen3.5), **CC BY-NC 4.0 (StartLux 9B e 4B, Nimble-9B-v3: derivati non commerciali)**. In più valgono i vincoli dei dati: ANLI, FactKG, AVeriTeC sono non commerciali; MiniCheck (MIT), SummEdits (CC BY), HoVer/FEVER/VitaminC (CC BY-SA), CNN/DailyMail (Apache 2.0) e i dati generati con modelli aperti non pongono vincoli aggiuntivi. Un modello commerciale va addestrato senza ANLI, FactKG e AVeriTeC.

## Criteri di completamento

- Verifica della fusione superata con l'adapter del benchmark.
- Scala e soglia da dev_select, confermate su dev_confirm; scelta 0,5 vs soglia fittata documentata.
- `frozen.json` prima del test; test eseguito una volta; report e model card scritti.
