# d3-flash

[vllm-sr/d3-flash](https://huggingface.co/vllm-sr/d3-flash): the 8.4B multimodal decision model of Decision 3.0
(vLLM Semantic Router), a Qwen3.5-9B backbone (Gated DeltaNet + full attention) with a 255-way answer-code readout.
Licence: Apache-2.0. Its larger sibling is [d3-turbo](../d3-turbo/README.md).

## How the benchmark runs it

The model ships its own code (`trust_remote_code=True`) and answers typed questions with `system_one`, with no generated
tokens. `adapter.py` loads it with `AutoModel` and calls the runtime's `prepare` and `run`, the same two steps as
`system_one`, so prompt, readout and batching are the authors' own:
- a fixed system prompt, then "State:" and the state as JSON, "Question:" and the question, "Options:" with one answer
  code per option (a yes/no question gets `A: No / false`, `B: Yes / true`), and "Reply with only the code of the best
  option.", in the chat template with thinking off;
- a softmax over the answer codes at the last prompt position, at the package's temperature (1, in `decision_config.json`);
- `noncausal_full_attention`: the full-attention layers see the whole prompt, and each question is its own prompt, so
  the state is read once per question. d3-turbo reads it once per row instead.

Calling `prepare` and `run` instead of `system_one` gives the probabilities of every option in option order (`system_one`
reports only P(yes) for a yes/no question). The runtime reports probabilities only; the adapter stores
`T * log(p)` as logits, which equal the readout logits up to a constant, so answers can be re-tempered afterwards (e.g. by
`grid_search.py`). The shipped Python files are the d3-turbo ones in an older version: the only network call downloads
images given as http(s) URLs (never used here), and nothing starts processes, calls `eval` or writes files.

## Run

```sh
hf download vllm-sr/d3-flash
python benchmark.py --model d3-flash --gpu 0 --limit 32
python benchmark.py --model d3-flash --gpu 0
```

Options, in [config/models/d3-flash.json](../../config/models/d3-flash.json) (or `--opt d3-flash:KEY=VALUE`):
- `revision` (`13dc1cc`);
- `max_length` (32,768 tokens per question);
- `question_batch` (8 questions per forward pass);
- `verify` (`fast`: the files are checked against `MODEL_MANIFEST.json`);
- `batch` (32 rows per call; each row is still its own request).

**GPU.** 16.8 GB of bf16 weights on one device (the runtime has no offload); a 24 GB GPU is enough for the test split.

**Context.** The package sets no length cap. The adapter caps prompts at 32,768 tokens per question, the length d3-turbo
validates; longer prompts fail and are logged, never truncated. On the LLM-AggreFact test split the longest prompt is
about 23,250 tokens (after the benchmark's 80,000-character document cap).

**Kernels.** `flash-linear-attention` (see `requirements.txt`) gives the fast GPU kernels for the Gated DeltaNet
layers; without it transformers runs its slower PyTorch reference. The kernels in use are recorded in `run.json`
(`model.kernels`).

## Verified

32 test rows on an RTX 5090 Laptop GPU (24 GB), without flash-linear-attention (1.25 rows/s):
- 0 failures;
- ROC AUC of P(yes) against the label: 0.91 for `support_simple`, 0.12 and 0.29 for `unsupported_detail` and
  `contradicted_detail` (their "yes" means not supported, so values under 0.5 are the expected direction);
- the set rule gives balanced accuracy 0.84.
