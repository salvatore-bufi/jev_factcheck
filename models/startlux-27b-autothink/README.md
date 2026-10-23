# StartLux-Decision-v1.1-27B-AutoThink

[startlux-models/StartLux-Decision-v1.1-27B-AutoThink](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-27B-AutoThink):
a 27B decision model (Qwen3.5 architecture with a vision tower) that reads the prompt causally and thinks only where it is
unsure. It reads the option letters once; when the top probability is below the gate (0.6 by default), it writes a
thought, reads the letters again and answers `0.25 × first + 0.75 × after`. The card reports 73.37 on the Decision
Index 0.3 public subset (with gate 0.9).

Licences:
- weights: **CC BY-NC 4.0** (non-commercial only; commercial use needs a licence from StartLux Labs);
- inference package ([StartLuxLabs/StartLux-Decision](https://github.com/StartLuxLabs/StartLux-Decision)): Apache-2.0.

## How the benchmark runs it

The weights repository holds no inference code. The package `startlux-decide` holds what makes the weights a
decision model:
- **Prompt**: a fixed system line, then "Evidence:" with the state as compact JSON, "Question:", and lettered options
  (`A) yes: <true description>`, `B) no: <false description>`), in the chat template with thinking off.
- **Read-out**: the option-letter logits at the last position, divided by a temperature per question type (from
  `decision_config.json`: noul 1.376, choice 0.9503, score 1.0108).
- **Thinking**:
  - the gate;
  - the thought: sampled (temperature 0.6, top-p 0.95, top-k 20), up to 4,096 tokens, cut by a loop detector. On
    vLLM 0.31 it is streamed with multi-token prediction;
  - the read-out after the thought, and the mixture `(1 - mix) * first + mix * after`.

So the benchmark runs the package itself, as it runs Winnow's server. `adapter.py` starts
`startlux-decide serve --launch-vllm` from this folder's own environment (`.venv`). That starts one vLLM 0.31.0 server
on the benchmark GPU with the package's flags, and the `/v1/systemone` server in front of it. The adapter then:
- sends one request per row, with all its questions, the thinking controls (`think_mode`, `gate`, `mix`, `think_budget`)
  and `return_details`;
- stops both servers at the end.

The server's log goes to `server.log` in this folder, vLLM's to `vllm_logs/`.

What is stored:
- `probs`: the server's final answer, per option key (`true`/`false` for yes/no questions);
- `logits`, `temperature`: for an answer read in one pass, `T * log(p)` of its read-out (the letter logits up to a
  constant) and the type's temperature T. An answer mixed with a read-out after a thought has no logits, so a run in
  which any answer thought cannot be re-tempered (`grid_search.py` keeps only its default temperature);
- `first_pass`: the first read-out of every answer;
- `thought`: `used`, `tokens`, and `ended` (`closed`, `loop`, `budget` or `skipped`);
- `prompt_tokens`: the request's input tokens as the server counts them (all three prompts of the row).

## Setup (once)

```sh
bash models/startlux-27b-autothink/setup.sh
```

It creates `models/startlux-27b-autothink/.venv` and installs the package at a pinned commit (`a99977d`) with vLLM
0.31.0 (about 5 GB). A separate environment, because:
- the benchmark's environment runs another vLLM version than the package validates;
- the package's transformers engine needs transformers 5.8, and the benchmark's environment has a newer one.

## Run

```sh
hf download startlux-models/StartLux-Decision-v1.1-27B-AutoThink
python benchmark.py --model startlux-27b-autothink --gpu 0 --limit 32
python benchmark.py --model startlux-27b-autothink --gpu 0 --opt startlux-27b-autothink:quantization=fp8   # 48 GB GPU
python benchmark.py --model startlux-27b-autothink --gpu 0 --opt startlux-27b-autothink:gate=0.9           # the card's setting
```

Options, in [config/models/startlux-27b-autothink.json](../../config/models/startlux-27b-autothink.json) (or
`--opt startlux-27b-autothink:KEY=VALUE`):

| option | default | meaning |
|---|---|---|
| `revision` | `dd62d5c` | snapshot in the Hugging Face cache (empty: refs/main) |
| `port`, `vllm_port` | 8092, 8101 | decision server and vLLM server |
| `max_model_len` | 32768 | vLLM context; the longest LLM-AggreFact test prompt is 23,224 tokens |
| `gpu_util` | 0.88 | vLLM `--gpu-memory-utilization` (the package default) |
| `quantization` | "" | `fp8`: weights quantized at load (the card reports 73.20 against 73.52 in bf16) |
| `workers` | 8 | concurrent requests |
| `timeout`, `startup_timeout` | 900, 1800 s | per request, and for the servers to come up |
| `url` | "" | use a decision server that is already running instead of starting one |
| `batch` | 64 | rows per call |
| `think_mode` | auto | `auto`: think when unsure; `never`: one pass only; `always` |
| `gate`, `mix`, `think_budget` | 0.6, 0.75, 4096 | the package defaults (the card's published results use gate 0.9) |

Each value of `quantization`, `gate`, `mix`, `think_mode` or `think_budget` gets its own result folder, through the
params hash.

**GPU.** About 54 GB of bf16 weights: an 80 GB GPU in bf16. On a 48 GB GPU (L40S) use `quantization=fp8`, about 27 GB of
weights.

**Caveats.**
- Thoughts are sampled, so an answer that thinks can differ on another run. The seed comes from the prompt.
- Latency includes the local HTTP round trip. On vLLM 0.31 the server keeps a prefix cache, so a document read before
  is not read cold again.
- If the benchmark is killed hard, the vLLM server can outlive it: check `nvidia-smi` and stop it.

## Verified

Offline only:
- the package's prompt renders the three questions;
- the tokenizer passes the package's letter check;
- the longest test prompt is 23,224 tokens;
- the answer conversion stores logits only for answers read in one pass.

Not yet run: it needs the environment from `setup.sh`.
