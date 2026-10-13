# Rune 26B-A4B v3

[surogate/rune-26b-a4b-GGUF](https://huggingface.co/surogate/rune-26b-a4b-GGUF): a Gemma 4 mixture-of-experts decision
model (26.5B parameters, 8 of 128 experts active per token). Despite the repository name, v3 holds bf16 safetensors.
Licence: Apache 2.0.

## How the benchmark runs it

`adapter.py` runs the model in-process with **vLLM** and reproduces surogate's *decisions v1* prompt (its documentation
and golden-test generator define it byte for byte): a fixed system prompt, a user turn
`SHARED STATE (JSON string): <json> / QUESTION: ... / OPTIONS: A: ... B: ... / Answer with one option letter only.`,
the chat template with thinking off, and a softmax over the option-letter logits divided by a decision temperature of
**2** (the card's recommendation for calibrated probabilities). For a yes/no question option A is the `false`
description and B the `true` one. `noul`, `choice` and `score` with up to 26 options. Raw logits are saved.
surogate's own engine was not run: this is a re-implementation of its documented protocol.

## Run

```sh
hf download surogate/rune-26b-a4b-GGUF
python benchmark.py --model rune-26b --gpu 0 --limit 32
python benchmark.py --model rune-26b --gpu 0
```

The weights take **51.6 GB**. With `cpu_offload_gb=auto` (default) the part that does not fit the GPU stays in system
RAM and is streamed over PCIe: exact, but slow. The automatic value keeps 9 GiB of the GPU for KV cache and activations:
about 36 GB on a 24 GB card (measured about 0.4 rows/s, roughly 20 hours for the full split, about 40 GB of free RAM
needed), about 17 GB on a 45 GB L40S. If start-up fails with "KV cache is needed, which is larger than the available KV
cache memory", raise it (`--opt rune-26b:cpu_offload_gb=20`) or lower `max_model_len`.

Options, in [config/models/rune-26b.json](../../config/models/rune-26b.json) (or `--opt rune-26b:KEY=VALUE`): `revision` (`c6b360d`), `temperature` (2.0), `max_model_len` (24576), `cpu_offload_gb` (auto),
`gpu_util` (0.92), `max_num_seqs` (32), `max_batched_tokens` (16384), `tensor_parallel` (1), `batch` (128).

## Verified

On test row `test:5075` (supported) P(yes) was 1.000 / 0.173 / 0.042 at temperature 2 (support / unsupported detail /
contradiction), so the claim is accepted; option letters are single tokens after the prompt, checked in context as
surogate does. After the move to this benchmark, 32 test rows match the earlier full run: mean |ΔP(yes)| 0.009,
all 32 decisions equal.
