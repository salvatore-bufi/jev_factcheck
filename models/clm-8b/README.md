# CLM-v0.1-8B

[Contrastive-LM/CLM-v0.1-8B](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B): two small projection heads (a 73 MB
checkpoint) on a frozen **Qwen3-8B** encoder. It does not generate. Licence: Apache 2.0.

## How the benchmark runs it

The state text (the state fields as prose — `document: ...`, blank line, `claim: ...` — then a blank line and the
question) and each option text (`false: ...` / `true: ...` for a yes/no question, the option's own text for a choice or
score) are embedded by Qwen3-8B (last-token pooling, through vLLM's pooling runner), projected by the state head and the
action head, and the answer is `softmax(scale × cosine)`. `adapter.py` imports the text layout and the heads from the
`contrastive-lm` package itself, so the inputs are the reference ones. Option texts do not depend on the row and are
embedded once. Texts are not truncated to the reference server's default 2,048 tokens. Raw logits (`scale × cosine`)
are saved.

The answer is a *relative similarity* between option texts, so the wording of the descriptions matters a lot. With the
default question set two of the three questions carry almost no signal (P(yes) about 0.94 and 0.77 for every row) and the
`min` rule rejects every claim: the first full run scored 49.98% (every source at 50%). `support_simple` alone has
ROC AUC 0.76. Question sets written for CLM (short answer-style options) would suit it better.

## Run

```sh
pip install --no-deps contrastive-lm==0.1.0
hf download Contrastive-LM/CLM-v0.1-8B && hf download Qwen/Qwen3-8B
python benchmark.py --model clm-8b --gpu 0 --limit 32
python benchmark.py --model clm-8b --gpu 0
```

Options, in [config/models/clm-8b.json](../../config/models/clm-8b.json) (or `--opt clm-8b:KEY=VALUE`): `encoder_revision` (`b968826`, the one CLM was trained on), `heads` (path to `CLM_v0.1-8B.pt`; default the
cache), `temperature` (1.0), `max_model_len` (32768), `cpu_offload_gb` (0), `gpu_util` (0.92), `max_num_seqs` (32),
`max_batched_tokens` (8192), `batch` (128). The 16 GB encoder fits a 24 GB card.

## Notes

- vLLM's engine must start before any CPU torch work in the process (the heads are loaded afterwards), and
  `benchmark.py` sets `VLLM_WORKER_MULTIPROC_METHOD=spawn`: a fork after OpenMP has run hangs or crashes the engine.
- If the shell sets an `http_proxy`, the pooling start-up can hang on it; unset the proxy variables if it stalls.
- Verified: on 16 test rows the in-process answers matched the official path (`vllm serve` + the package's `Engine`)
  within 0.006–0.015 on average, all decisions equal; after the move to this benchmark 32 rows match the earlier run
  (mean |ΔP(yes)| 0.014, bf16 batch effects).
