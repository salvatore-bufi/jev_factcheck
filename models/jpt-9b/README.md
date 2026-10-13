# JPT-9B

[kirp/jpt-9b](https://huggingface.co/kirp/jpt-9b): an open 9B decision model (Qwen3.5-9B with a merged LoRA) that implements
the Jev typed-decision interface. Licence: CC BY-NC 4.0 (non-commercial).

## How the benchmark runs it

`adapter.py` runs the model in-process with **vLLM**. Prompts and readout come from
[`llm2jev`](https://github.com/tic-top/llm2jev), the reference named on the model card: the chat template with thinking off,
the state, the question with all its options labelled A, B, ..., an `Answer:` prefill, and a softmax over the option-label
logits at the card's fixed temperature **1.087**. One forward pass per question; the questions of a row share the document
through vLLM's prefix cache. For a yes/no question option A is "Yes" (true). `noul`, `choice` and `score` are supported.
The raw option logits are saved, so `report.py --temperature` can re-temper the answers.

## Run

```sh
pip install -r models/jpt-9b/requirements.txt          # llm2jev
hf download kirp/jpt-9b                                # into the Hugging Face cache
python benchmark.py --model jpt-9b --gpu 0 --limit 32  # quick check
python benchmark.py --model jpt-9b --gpu 0             # full test split
```

Options, in [config/models/jpt-9b.json](../../config/models/jpt-9b.json) (or `--opt jpt-9b:KEY=VALUE`): `revision` (pinned to `b447cc7`, the weights of the published runs; the next Hub
revision only changed the model card), `temperature` (1.087), `max_model_len` (32768 tokens; longer prompts fail and are
logged, never shortened), `gpu_util` (0.92), `max_num_seqs` (32: Qwen3.5's DeltaNet layers need one state block per
sequence), `max_batched_tokens` (8192), `batch` (64 rows per vLLM call). About 18 GB of GPU memory at the defaults.

## Verified

- On test row `test:5075`, vLLM and llm2jev's own transformers backend gave the same probabilities to four decimals
  (0.9900 / 0.0344 / 0.0157).
- After moving to this benchmark, 32 test rows match the earlier full run (mean |ΔP(yes)| 0.003, all decisions equal;
  the small differences are bf16 batch-shape effects).
- `VLLM_USE_FLASHINFER_SAMPLER=0` (set by `benchmark.py`): without `nvcc` FlashInfer's sampler would try to compile CUDA.
  Nothing is sampled, so nothing is lost.
