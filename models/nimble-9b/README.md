# Bespoke-Nimble-9B-v3

[bespokelabs/Bespoke-Nimble-9B-v3](https://huggingface.co/bespokelabs/Bespoke-Nimble-9B-v3): a decision model by
Bespoke Labs, a **LoRA adapter** (rank 64) on [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B). The card
reports 56.88 on the Decision Index 0.2.1. Licences:
- adapter and engine: **CC BY-NC 4.0** (non-commercial);
- base model: Apache-2.0.

## How the benchmark runs it

The adapter repository holds only the LoRA weights and the prompt contract (`schema_config.json`). The engine of the
authors' Decision Index run is published in the dataset repository
[bespokelabs/bespoke-nimble-9b-v3-decision-index](https://huggingface.co/datasets/bespokelabs/bespoke-nimble-9b-v3-decision-index)
(`code/nimble`). `./nimble` is a byte-identical copy of it at `654b124`. It can't run from the Hugging Face cache: the
engine hashes its prompt files next to its own resolved path, which in the cache is the hash-named `blobs` folder.

At load, `NimbleEngine` checks the contract:
- its two prompt files have the SHA-256 that `schema_config.json` pins;
- the tokenizer gives the contract's 255 candidate codes and token ids.

It then loads Qwen3.5-9B at the revision the contract pins (`c202236`, bf16, transformers, SDPA) and applies the LoRA
with PEFT. Everything the model sees is the authors':
- **Prompt**, one per question:
  - a system prompt ("Classify the context using the supplied schema. ... Return only that choice's one-letter code");
  - one JSON user turn with the context (the state as JSON) and the schema of all the row's questions. A yes/no
    question is a boolean field with choices `A` = false and `B` = true and their descriptions;
  - then `Requested field: "<question id>"`, in the chat template with thinking off.
- **Shared prefix**: the three prompts of a row differ only in the requested field, so their common prefix (the context
  and the whole schema) runs once and each question's short tail runs on a copy of its cache.
- **Read-out**: softmax over the candidate code tokens, temperature 1.
- **Limits**: prompts over 32,768 tokens are refused, never truncated. The engine splits batches on out-of-memory and
  refuses only a single prompt that doesn't fit.

The engine is written for the Decision Index kit, which isn't installed here. The adapter provides the kit's two names
the engine imports (`Engine`, `Unsupported`). It then calls the steps of `NimbleEngine.__call__` with the engine's own
functions, keeping the probabilities of both options; `__call__` returns only P(true).

What is stored:
- `probs` per option key (`false`, `true`);
- `logits` = log(p) (the candidate logits up to a constant; temperature 1), so answers can be re-tempered;
- `prompt_tokens`: each question's prompt length.

## Run

The base model has to be in the cache too, at the contract's revision:

```sh
hf download bespokelabs/Bespoke-Nimble-9B-v3
hf download Qwen/Qwen3.5-9B --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a      # about 19 GB
python benchmark.py --model nimble-9b --gpu 0 --limit 32
python benchmark.py --model nimble-9b --gpu 0
```

Options, in [config/models/nimble-9b.json](../../config/models/nimble-9b.json) (or `--opt nimble-9b:KEY=VALUE`):

| option | default | meaning |
|---|---|---|
| `revision` | `8e927b9` | the LoRA adapter's snapshot |
| `max_tokens` | 32768 | the engine's limit per prompt; longer prompts are refused |
| `batch_tokens` | 65536 | padded tokens per batch when the prefix is not shared |
| `shared_prefix` | true | the row's common prefix runs once (the engine's default and fastest path) |
| `cache_tokens` | 131072 | prefix tokens times tails per batch of the shared path |
| `batch` | 16 | rows per call; each row is its own request |

**GPU.** About 20 GB in bf16 (the card's figure): a 24 GB GPU is tight for the longest documents; 48 GB is comfortable.

**Environment.** It needs `peft` (the authors used 0.21.0). The card lists transformers 5.17.0, torch 2.8,
flash-linear-attention 0.5.2 and causal-conv1d 1.7.0. The kernels are optional and only speed up the Gated DeltaNet
layers. transformers 5.14 or newer has the Qwen3.5 classes.

## Verified

Offline only (the base weights and `peft` are not on this machine):
- the engine's contract check passes with the base model's tokenizer at `c202236`: prompt files' SHA-256 and the 255
  candidate codes;
- the prompts render the three questions, with the yes/no choices as `A` = false and `B` = true (token ids 32 and 33);
- the longest LLM-AggreFact test prompt is 23,920 tokens.

Not yet run on a GPU.
