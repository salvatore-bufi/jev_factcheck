# KnowLine-4B-Gen5.5

[PelaAI/KnowLine-4B-Gen5.5](https://huggingface.co/PelaAI/KnowLine-4B-Gen5.5): a 4B decision model fine-tuned from
Qwen3.5-4B (two LoRA SFT models averaged and merged). Gen5.5 has the same weights as Gen5 and a new way of serving them:
short requests are read twice. The card reports 66.75 on the Decision Index 0.3 public suite (self-run) and lists Gen5 at
53.68 on the 0.4.1 board. Licence: Apache-2.0 (weights); the front-end files are MIT.

## How the benchmark runs it

The model repository ships its own front end, `knowline_server.py` with `knowline_layout.py`, and `adapter.py` imports
both from the snapshot, so prompt and scoring are the authors':
- **Prompt**, one per question:
  1. the state as one user turn, rendered as indented text (keys and values, not JSON);
  2. a user turn with the instruction ("Evaluate the conversation or state above using the question below. ..."),
     `Question: ...` and `Options:` with `A. Yes: <true description>` and `B. No: <false description>`;
  3. the assistant turn, opened with `Answer:`, in the chat template with thinking off.
- **Read twice** (Gen5.5, on): a text-only request of at most 3,000 tokens, counted over the whole request read once,
  is written a second time after "Read the same input again before answering:". Longer requests are read once. On the
  test split most rows are short enough: 31 of the first 32 test rows were read twice.
- **Read-out**: one prefill per question, a softmax over the label tokens. The calibration temperature T = 1.50 is folded
  into the final norm of the weights, so the front end uses temperature 1.
- **Backends**, as in the repo's `knowline_engine.py`:
  - `hf` (default): transformers in process, bf16. The authors report it gives the same answer as their SGLang run on
    81 of 82 test answers;
  - `sglang`: the authors' setting. The adapter starts an SGLang server with the flags of `serve_knowline.sh` (FP8 at
    load) and stops it at the end. It needs `sglang==0.5.21` in this environment, or `sglang_python` pointing to one.
    This backend has not been run here.

`knowline_server.KnowLine.run` turns the label log-probabilities into probabilities and drops them. The adapter does
the same steps with the front end's own functions (`render_request`, `warm`, `score`) and keeps the log-probabilities,
so answers can be re-tempered afterwards. Its probabilities are identical to `run()`'s (checked on three rows: a short
document, a median one, and the longest, which is read once).

What is stored:
- `logits`, `temperature`: the label log-probabilities in option-key order (`false`, `true`), temperature 1;
- `probs`: computed from them;
- `read_twice`: whether the row was written twice;
- `prompt_tokens`: tokens of each question's prompt.

The three Python files of the repo only render, score and serve. `knowline_engine.py` (not used here) starts SGLang
with `subprocess`; the adapter does the same with the same flags.

## Run

```sh
hf download PelaAI/KnowLine-4B-Gen5.5
python benchmark.py --model knowline-4b --gpu 0 --limit 32
python benchmark.py --model knowline-4b --gpu 0
python benchmark.py --model knowline-4b --gpu 0 --opt knowline-4b:read_twice=false     # the Gen5 prompts
```

Options, in [config/models/knowline-4b.json](../../config/models/knowline-4b.json) (or `--opt knowline-4b:KEY=VALUE`):

| option | default | meaning |
|---|---|---|
| `revision` | `779944d` | snapshot in the Hugging Face cache |
| `backend` | hf | `hf` (transformers, bf16) or `sglang` (FP8 at load, the authors' runs) |
| `read_twice`, `twice_max_tokens` | true, 3000 | the Gen5.5 layout, and its cut |
| `workers` | 16 | threads scoring the questions of a row |
| `mem` | 0.72 | sglang: `--mem-fraction-static` |
| `sglang_python` | "" | sglang: an interpreter that has SGLang, if not this one |
| `startup_timeout` | 1800 s | sglang: time for the server to come up |
| `batch` | 32 | rows per call |

**GPU.** About 9 GB of bf16 weights (the vision tower included). The `hf` backend computes the full-vocabulary logits at
every prompt position, which is about 9 GB more on the longest test prompt (about 18,000 tokens): it fits on 24 GB, with
headroom to spare on 48 GB.

## Verified

32 test rows on an RTX 5090 Laptop GPU (24 GB), `hf` backend, transformers 5.14.1, without the linear-attention kernels:
- 0 failures, 1.35 rows/s;
- ROC AUC of P(yes) against the label: 0.955 for `support_simple`, 0.08 and 0.26 for `unsupported_detail` and
  `contradicted_detail` (their "yes" means not supported, so values under 0.5 are the expected direction);
- the set rule gives balanced accuracy 0.80 and ROC AUC 0.92;
- the adapter's probabilities equal those of the front end's own `KnowLine.run` (difference 0 on three rows).
