# pplx-decider-flash-v1.2

[perplexity-ai/pplx-decider-flash-v1.2](https://huggingface.co/perplexity-ai/pplx-decider-flash-v1.2): a 4B decision
model on Qwen3.5-4B, **distilled from Perplexity's 27B decision models** (the student learns the teachers' full
distributions) and combined by weight averaging. Licence: Apache-2.0. Decision Index 0.4.1: Full 54.29, Public 60.06
(snapshot `0501b8c`).

## How the benchmark runs it

The weights repository ships its package, `source/src/pplx_decider` (0.3.0). `adapter.py` imports it from the snapshot
and runs the **reference path** (`fast=False`, the evaluated computation):

- **Prompt**: protocol v3, `joint-question-inline-readout-v1`. All questions of a request share one sequence, and each
  question is read at its own inline readout position. Options are shown in request order under the codes `A`, `B`, …
  (a yes/no question is `A: No / false`, `B: Yes / true`).
- **Attention**: the full-attention layers are non-causal over the whole sequence (`noncausal_full_attention`); the
  linear-attention layers stay causal. The package installs this mask with a forward hook.
- **Read-out**: a 255-way decision head (`readout.safetensors`, `[255, 2560]`) on the hidden state at each readout
  position, restricted to the question's valid options. Calibrated temperature 1.554 (`decision_config.json`).
- **Limits**: `max_length` 32,768 tokens per sequence (training used up to about 32k). A longer request is split by the
  package into question chunks that each keep the full state; a single question that does not fit fails the row.
  Nothing is truncated.

`Decider.logits` returns the raw readout logits before temperature. The adapter stores them with the saved temperature,
so `probs` equal what `Decider.decide` returns.

**Joint vs one question per request.** The card reports that this model is more accurate with one question per request
on most multi-question tasks. In the focused study, the three-question file (`claim_support.json`) is a joint request
(composition C2) and `q1_only.json` is one question per request (C1); the difference is measured, not assumed.

## Run

```sh
hf download perplexity-ai/pplx-decider-flash-v1.2 --revision 9ff7cf9838e59df461c276b7699d56b244d2af58
python benchmark.py --model pplx-decider-flash --gpu 0 --limit 32 --no-profile --tag smoke
```

Options: [config/models/pplx-decider-flash.json](../../config/models/pplx-decider-flash.json) (`revision`,
`max_length`, `batch_tokens`, `batch`).

**Versions.** The card pins torch 2.14.0 and transformers 5.17.0. The package uses `Qwen3_5Model` and
`create_recurrent_attention_mask`, which transformers 5.14 has; record the version used (`run.json` holds it). About
9 GB of bf16 weights.
