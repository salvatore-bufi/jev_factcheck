# d1-3B

[LiquidAI/d1-3B](https://huggingface.co/LiquidAI/d1-3B): a 3B decision model built on LFM2.5-VL-3B. Licence: LFM Open
License v1.0 (check it before commercial use).

## How the benchmark runs it

The model ships its own code (`trust_remote_code=True`) and answers typed questions with `system_one`, in one forward pass
with no generated tokens. `adapter.py` calls `model.system_one_batch` through plain transformers, so prompt, readout and
batching are the authors' own: the state as indented JSON, `QUESTION:` and the question (a yes/no question ends with
"Reply with yes or no only."), a softmax over the option tokens; a request with several questions reads the state once.
The engine reports probabilities only, so d1-3B answers cannot be re-tempered afterwards, and its logits are read in bf16,
so probabilities take quantized values (e.g. exactly 0.0474 = sigmoid(-3)). I read the shipped Python files and found no
network calls, subprocesses, `eval` or file writes.

## Run

```sh
hf download LiquidAI/d1-3B
python benchmark.py --model d1-3b --gpu 0 --limit 32
python benchmark.py --model d1-3b --gpu 0
```

Options, in [config/models/d1-3b.json](../../config/models/d1-3b.json) (or `--opt d1-3b:KEY=VALUE`): `revision` (`051bcc4`), `batch` (64 rows per call; each row is still its own pass). A 3B model: any 8 GB+ GPU.
Context 32,768 tokens; longer prompts fail and are logged.

## Verified

On 400 development rows the three questions carry signal (ROC AUC 0.90 / 0.90 / 0.77); after the move to this benchmark,
32 test rows reproduce the earlier full run exactly.
