# OneJev1.6-9B

[OmniJev/OneJev1.6-9B](https://huggingface.co/OmniJev/OneJev1.6-9B): the 9B model of OneJev 1.6, a full fine-tune of
Qwen3.5-9B on about 440,000 decision questions (card). Licence: Apache-2.0. Decision Index 0.4.1: Full 53.05, Public
65.30 (snapshot `0501b8c`).

## How the benchmark runs it

The card serves the model with **Flide** (`flide serve --model OmniJev/OneJev1.6-9B`), and the checkpoint's
`decision_config.json` says `{"format_id": "onejev-slot-v2", "package": "flide"}`. It does **not** run on the `qev`
engine that `models/elephant-1` copies. So the benchmark runs Flide itself, in this folder's own environment (as
StartLux runs its own server):

- **Prompt** (Flide's OneJev family, `onejev-labels-v2`):
  - a system turn and the state block form a shared prefix;
  - each question is a suffix: `Question: …`, `Options:` with `A. yes: <true description>` and
    `B. no: <false description>`, then `Answer with one letter: A, B.`;
  - Flide forks the prefix into every question inside one forward pass, so each question gets the answer it would get
    if asked alone.
- **Attention**: causal. The checkpoint has no `readout.json` and no `modeling_overrides` in `config.json`.
- **Read-out**: the answer-slot logits of the LM head at the last token of each question (`slot_lm`), softmax at
  temperature 1 (there is no `calibration.json`). Flide's default `debias=1` renders one option order.
- **Limits**: a state plus one question above 32,768 tokens, or a request above 65,536, is refused (logged in
  `failures.jsonl`), never shortened.

`adapter.py` starts `flide serve` on the benchmark GPU with the snapshot folder, sends one `/v1/systemone?debug=1`
request per row with all its questions, and stops the server at the end. The log goes to `server.log`.

What is stored:

- `probs`: the server's answer;
- `logits`, `temperature`: the raw slot logits from the response's diagnostics, in option-key order (`false`, `true`
  for yes/no), and the temperature (1), so answers can be re-tempered;
- `prompt_tokens`: the request's input tokens as the server counts them.

## Setup (once)

```sh
bash models/onejev-9b/setup.sh        # .venv with flide @ 08b87d0 (0.1.4), torch 2.12+, flash-linear-attention
```

## Run

```sh
hf download OmniJev/OneJev1.6-9B --revision 259e5470b0abfd795a0bbb4598d6e319fa224a38
python benchmark.py --model onejev-9b --gpu 0 --limit 32 --no-profile --tag smoke
```

Options: [config/models/onejev-9b.json](../../config/models/onejev-9b.json).

| option | default | meaning |
|---|---|---|
| `revision` | `259e547` | snapshot in the Hugging Face cache |
| `port` | 8094 | the Flide server |
| `dtype` | bfloat16 | weights at load |
| `max_branch_tokens`, `max_request_tokens` | 32768, 65536 | Flide's limits (requests above are refused) |
| `quantize` | "" | `fp8`: Flide's FP8 at load |
| `workers`, `timeout`, `startup_timeout` | 8, 900, 1800 | concurrent requests; seconds per request and for start-up |
| `url` | "" | use a Flide server that is already running |

**GPU.** About 19 GB of bf16 weights: a 24 GB GPU holds it only with short documents; use 40 GB+ for the full test, or
`quantize=fp8` (state it in the results).
