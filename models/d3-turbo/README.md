# d3-turbo

[vllm-sr/d3-turbo](https://huggingface.co/vllm-sr/d3-turbo): the 26B multimodal decision model of Decision 3.0
(vLLM Semantic Router), a Qwen3.5 backbone (Gated DeltaNet + full attention) with a 255-way answer-code readout.
Licence: Apache-2.0.

## How the benchmark runs it

The model ships its own code (`trust_remote_code=True`) and answers typed questions with `system_one`, with no generated
tokens. `adapter.py` loads it with `AutoModel` and calls the runtime's `prepare` and `run`, the same two steps as
`system_one`, so prompt, readout and batching are the authors' own:
- a fixed system prompt, then "State:" and the state as JSON, "Question:" and the question, "Options:" with one answer
  code per option (a yes/no question gets `A: No / false`, `B: Yes / true`), and "Reply with only the code of the best
  option.", in the chat template with thinking off;
- a softmax over the answer codes at the last prompt position, at the package's temperature (1, in `decision_config.json`);
- `prefix_bidir` attention: the state block is computed once per row and its three questions run on it.

Calling `prepare` and `run` instead of `system_one` gives the probabilities of every option in option order (`system_one`
reports only P(yes) for a yes/no question). The runtime reports probabilities only; the adapter stores
`T * log(p)` as logits, which equal the readout logits up to a constant, so answers can be re-tempered afterwards (e.g. by
`grid_search.py`). I read the shipped Python files: the only network call downloads images given as http(s) URLs
(never used here), and nothing starts processes, calls `eval` or writes files (`d3_server.py` is not imported).

## Run

```sh
hf download vllm-sr/d3-turbo
python benchmark.py --model d3-turbo --gpu 0 --limit 32
python benchmark.py --model d3-turbo --gpu 0
```

Options, in [config/models/d3-turbo.json](../../config/models/d3-turbo.json) (or `--opt d3-turbo:KEY=VALUE`):
- `revision` (`a0f3e9e`);
- `long_context` (`false`);
- `question_batch` (8 questions per forward pass);
- `verify` (`fast`: the files are checked against `MODEL_MANIFEST.json`);
- `batch` (32 rows per call; each row is still its own request).

**GPU.** The bf16 weights are 52.2 GB and the runtime puts the whole model on one device, with no CPU offload. You need
one GPU with about 56 GiB or more (e.g. 80 GB A100/H100); `load` stops with a message on smaller GPUs, including a
48 GB L40S.

**Context.** 32,768 tokens per question (the validated length). `long_context=true` lifts it to the package's limit;
the card says this is slower and less accurate. Longer prompts fail and are logged, never truncated. On the
LLM-AggreFact test split the longest prompt is 23,247 tokens (after the benchmark's 80,000-character document cap).

**Kernels.** `flash-linear-attention` (see `requirements.txt`) gives the fast GPU kernels for the Gated DeltaNet
layers; without it transformers runs its slower PyTorch reference. The kernels in use are recorded in `run.json`
(`model.kernels`).

## Verified

Offline, with the tokenizer only:
- the prompts of the three questions match the runtime's own rendering;
- the yes/no options are in the order the benchmark expects (false, true);
- the three questions of a row share one state block;
- the longest test prompts fit the 32,768-token cap.

Not yet run on a GPU large enough for the weights.
