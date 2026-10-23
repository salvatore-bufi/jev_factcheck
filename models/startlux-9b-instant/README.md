# StartLux-Decision-v1.1-9B-Instant

[startlux-models/StartLux-Decision-v1.1-9B-Instant](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-9B-Instant):
a 9B decision model (Qwen3.5 architecture with a vision tower) that answers every question with one read-out of the
option letters and never thinks. Its full-attention layers read the prompt bidirectionally. The card reports 69.96 on
the Decision Index 0.3 public subset.

Licences:
- weights: **CC BY-NC 4.0** (non-commercial only; commercial use needs a licence from StartLux Labs);
- inference package ([StartLuxLabs/StartLux-Decision](https://github.com/StartLuxLabs/StartLux-Decision)): Apache-2.0.

## How the benchmark runs it

The weights repository holds no inference code. The package `startlux-decide` holds what makes the weights a
decision model:
- **Prompt**: a fixed system line, then "Evidence:" with the state as compact JSON, "Question:", and lettered options
  (`A) yes: <true description>`, `B) no: <false description>`), in the chat template with thinking off.
- **Read-out**: the option-letter logits at the last position, divided by a temperature per question type (from
  `decision_config.json`: noul 1.3396, choice 1.1014, score 1.1374).
- **Bidirectional prompt attention** in the full-attention layers. Plain vLLM or transformers read the prompt causally
  and give a different model without an error. The package's vLLM plugin does it, and only on vLLM 0.19.1 or 0.31.0.

So the benchmark runs the package itself, as it runs Winnow's server. `adapter.py` starts
`startlux-decide serve --launch-vllm` from this folder's own environment (`.venv`). That starts one vLLM 0.31.0 server
on the benchmark GPU with the package's flags and the plugin (the launcher refuses to go on unless the vLLM log shows
the plugin active), and the `/v1/systemone` server in front of it. The adapter then:
- sends one request per row, with all its questions and `return_details`;
- stops both servers at the end.

The server's log goes to `server.log` in this folder, vLLM's to `vllm_logs/`.

What is stored:
- `probs`: the server's answer, per option key (`true`/`false` for yes/no questions);
- `logits`, `temperature`: `T * log(p)` of the read-out, which is the letter logits up to a constant, and the type's
  temperature T. Every answer is one read-out, so `grid_search.py` can re-temper them all;
- `prompt_tokens`: the request's input tokens as the server counts them (all three prompts of the row).

## Setup (once)

```sh
bash models/startlux-9b-instant/setup.sh
```

It creates `models/startlux-9b-instant/.venv` and installs the package at a pinned commit (`a99977d`) with vLLM 0.31.0
(about 5 GB). A separate environment is needed for two reasons:
- the benchmark's environment runs another vLLM, which the plugin refuses;
- the package's transformers engine needs transformers 5.8, and the benchmark's environment has a newer one.

## Run

```sh
hf download startlux-models/StartLux-Decision-v1.1-9B-Instant
python benchmark.py --model startlux-9b-instant --gpu 0 --limit 32
python benchmark.py --model startlux-9b-instant --gpu 0
```

Options, in [config/models/startlux-9b-instant.json](../../config/models/startlux-9b-instant.json) (or
`--opt startlux-9b-instant:KEY=VALUE`):

| option | default | meaning |
|---|---|---|
| `revision` | `18a17f7` | snapshot in the Hugging Face cache (empty: refs/main) |
| `port`, `vllm_port` | 8092, 8101 | decision server and vLLM server |
| `max_model_len` | 32768 | vLLM context; the longest LLM-AggreFact test prompt is 23,224 tokens |
| `gpu_util` | 0.88 | vLLM `--gpu-memory-utilization` (the package default) |
| `quantization` | "" | `fp8`: weights quantized at load (the card reports 69.84 against 69.98 in bf16) |
| `workers` | 8 | concurrent requests |
| `timeout`, `startup_timeout` | 900, 1800 s | per request, and for the servers to come up |
| `url` | "" | use a decision server that is already running instead of starting one |
| `batch` | 64 | rows per call |

**GPU.** About 19 GB of bf16 weights. The plugin needs each prompt prefilled whole in one step, so vLLM also reserves
the activations of a 32,768-token step: a 40 GB+ GPU is comfortable (e.g. an L40S).

**Caveats.**
- Latency includes the local HTTP round trip.
- If the benchmark is killed hard, the vLLM server can outlive it: check `nvidia-smi` and stop it.

## Verified

Offline only:
- the package's prompt renders the three questions;
- the tokenizer passes the package's letter check, and the letter ids equal `decision_config.json`;
- the longest test prompt is 23,224 tokens;
- the answer conversion reproduces the server's probability from the stored logits.

Not yet run: it needs the environment from `setup.sh`.
