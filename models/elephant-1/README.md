# elephant-1

[baselabs/elephant-1](https://huggingface.co/baselabs/elephant-1): a 9B decision model post-trained from Qwen3.5-9B by
Base Labs (Baseten). The card reports 62.65 on the Jev Decision Index 0.4. Licence: MIT (weights); its runtime, OneJev's
`qev`, is Apache-2.0.

## How the benchmark runs it

The checkpoint is served by its own runtime: OneJev's `qev` engine at commit `81ce62f`, with the `serve.patch` shipped in
the model repository. The card's `setup.sh` installs exactly that. `./qev` holds a copy of the six modules the engine
uses, byte-identical to that source (see `qev/__init__.py`), so there is nothing to install or build. `adapter.py` calls
`DecisionEngine.decide` in-process with plain transformers, so prompt, attention and read-out are the authors':
- **Prompt**: one per question.
  - The system turn plus `<state>` with the state as indented JSON `</state>` form a shared prefix, prefilled once per
    row.
  - Each question is a suffix: `Question: ...`, then `Options:` with `A. yes: <true description>` and
    `B. no: <false description>`, then `Answer with one letter: A, B.`.
  - The three suffixes run as one batched forward on a fork of the prefix cache.
- **Attention**: block-bidirectional in the full-attention layers (the prefix sees itself; each suffix sees the prefix
  and itself), as declared under `modeling_overrides` in the checkpoint's `config.json`. The linear-attention layers
  stay causal. The engine reads the declaration only from a local folder, so the adapter passes the snapshot folder,
  not the hub id. Given the hub id, the engine would silently run the model causally.
- **Read-out**: softmax over the LM-head logits of the option letters (`slot_lm`), with the LM head in float32. The
  checkpoint has no `calibration.json`, so the temperature is 1.
- **Limits**: requests over 32,768 tokens (state plus one question) or 65,536 (state plus all questions) are refused,
  never truncated.

What is stored:
- `logits`, `temperature`: the raw slot logits of every answer, from `decide(debug=True)`, in option-key order
  (`false`, `true`), and temperature 1;
- `probs`: computed from those logits. The engine's own answers are the same values rounded to 6 decimals;
- `slot_mass`: the share of the next-token probability on the option letters (a sanity check: low values mean the
  model wanted to answer something else);
- `prompt_tokens`: the request's input tokens (prefix plus the three suffixes).

## Run

```sh
hf download baselabs/elephant-1
python benchmark.py --model elephant-1 --gpu 0 --limit 32
python benchmark.py --model elephant-1 --gpu 0
```

Options, in [config/models/elephant-1.json](../../config/models/elephant-1.json) (or `--opt elephant-1:KEY=VALUE`):

| option | default | meaning |
|---|---|---|
| `revision` | `b3f8a93` | snapshot in the Hugging Face cache |
| `dtype`, `head_dtype` | bfloat16, float32 | model weights, and the LM head (the engine's defaults) |
| `fork_mode` | auto | `auto`: batched suffixes if they match sequential ones at load (tolerance 0.05); `sequential` |
| `max_branch_tokens`, `max_request_tokens` | 32768, 65536 | the engine's length limits |
| `batch` | 16 | rows per call; each row is its own prefill |

**GPU.** 18 GB of bf16 weights: a 24 GB GPU runs the 32-row check; the longest test documents need more headroom (the
batched path falls back to sequential suffixes on out of memory).

**Kernels.** `flash-linear-attention` and `causal-conv1d` speed up the Gated DeltaNet layers, as for d3; without them
transformers uses its PyTorch reference.

**Environment.** transformers 5.14 or newer: the block mask uses `create_causal_mask(block_sequence_ids=...)`, present
in 5.14, 5.17, 5.18 and 5.19 (the card pins 5.19.0). No separate environment is needed.

## Verified

32 test rows on an RTX 5090 Laptop GPU (24 GB), transformers 5.14.1, without the linear-attention kernels:
- 0 failures, 3.3 rows/s;
- ROC AUC of P(yes) against the label: 0.92 for `support_simple`, 0.08 and 0.14 for `unsupported_detail` and
  `contradicted_detail` (their "yes" means not supported, so values under 0.5 are the expected direction);
- the set rule gives balanced accuracy 0.78 and ROC AUC 0.91;
- the engine loaded with `attention: noncausal`, and its batched suffixes matched the sequential path within 0.0026 in
  probability (`fork_mode: batched`);
- the block mask is active: on one request, turning it off changes the slot logits by 0.3 to 1.0.
