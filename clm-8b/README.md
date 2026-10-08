# CLM-v0.1-8B on LLM-AggreFact

Evaluates [Contrastive-LM/CLM-v0.1-8B](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B) on the 29,320-row LLM-AggreFact test split with **vLLM**, and produces a row for the results table in the top-level README.

## What CLM is (and why it behaves differently)

CLM is not a generative model. It is two small projection heads (a 73 MB checkpoint, `CLM_v0.1-8B.pt`) on top of a frozen **Qwen3-8B** encoder. A question is answered by encoding two kinds of text with Qwen3-8B (last-token pooling):

- the **state text**: the state fields written as prose (`document: ...`, blank line, `claim: ...`), a blank line, then the question's instructions;
- each **option text**: for a yes/no question, `false: <description>` and `true: <description>`.

The state goes through the state head, each option through the action head, and the answer is `softmax(scale * cosine(state, option))` over the options. The yes-probability is the `true` option. So the answer is a *relative similarity between two option texts*: how the option descriptions are worded matters a lot. Question 1 (`support_simple`) has no descriptions, so CLM falls back to `false: No. This is false: <question>` and `true: Yes. This is true: <question>`; questions 2 and 3 use the descriptions from the pack.

## How this script runs it

`benchmark.py` does in-process what the reference setup does with `vllm serve Qwen/Qwen3-8B --runner pooling` plus `clm-serve`: Qwen3-8B embeddings from vLLM, then the heads from the `contrastive-lm` package. It imports that package's own text layout (`clm.schema`) and head code (`clm.heads`), so the inputs are the reference ones. The six option texts are identical for every row, so they are embedded once.

The frozen Jev checker is applied unchanged ([../packs/claim_support.json](../packs/claim_support.json)):

```
score = min(p_support, 1 - p_unsupported_detail, 1 - p_contradicted_detail)
supported = score > 0.30
```

This is a **fixed transfer evaluation**, like the JPT-9B, Rune and Winnow runs: same questions, rule, threshold and 80,000-character document cap. Nothing is tuned for CLM, and the 0.30 threshold was fitted to Jev's probabilities. Unlike the reference server's default, texts are **not** truncated to 2,048 tokens (that would cut long documents); the encoder runs at `--max-model-len` (default 32768), and a longer text is reported as a failure and excluded, never shortened silently. The cache stores the raw logits (`scale * cosine`), so `--temperature` (default 1, the client's) and `--threshold` only rescale the report for free. Do not pick either from test results.

## Run

From the repository root, in the `factcheck` conda environment (see `requirements.txt`). The Hugging Face cache must hold `Contrastive-LM/CLM-v0.1-8B` and `Qwen/Qwen3-8B` (revision `b968826`).

```sh
pip install --no-deps contrastive-lm==0.1.0          # once
python clm-8b/benchmark.py --limit 1                 # one claim vs its document (smoke test; separate output folder)
python clm-8b/benchmark.py                           # full test split; resumable
python clm-8b/benchmark.py --offline                 # rebuild the report from cached answers (no GPU)
python clm-8b/test_benchmark.py                      # offline unit checks
```

If interrupted (Ctrl-C), rerun the same command: finished rows are skipped. The cache is tied to the head checkpoint, document cap, pack, data hash and `--limit`; a changed configuration is refused instead of mixed in.

- **GPU:** one card, GPU 0 unless you pass `--gpu ID`. The 16 GB bf16 encoder fits a 24 GB card (about 4.8 GB left for the KV cache); the defaults (`--max-num-seqs 32`, `--max-batched-tokens 8192`, `--gpu-util 0.92`) are the ones that worked there. On a 45 GB card you can raise them.
- **Data:** `data/test.parquet` if present, else `./lytang___llm-aggre_fact`; override with `--data`.
- **Speed:** a 16-row test ran at about 4.5 rows/s including warm-up, which would be roughly 2 hours for the full split; this is an extrapolation, not a measured full run.

## Output

In `results/runs/clm-8b/` (git-ignored): `answers.jsonl` (the cache), `report.json` and `table_row.md`, a ready-made row such as `| CLM-v0.1-8B (ours; fixed Jev rule) | 8B | <avg> | <11 sources> |`. Smoke runs go to `results/runs/clm-8b-limit<N>/` and are labelled as non-benchmark.

## What was verified

- On 16 test rows, the in-process answers match the official path (`vllm serve` pooling server + the package's `Engine`) to within 0.006 / 0.007 / 0.015 on average for the three questions (largest single difference 0.044), and all 16 final decisions agree.
- Offline unit checks cover the text layout, the scoring rule, the report and the cache identity.
- **Not checked:** the full benchmark. Be aware that on the one smoke-test row (`test:5075`, label supported) the frozen rule *rejects* the claim (support 0.66, unsupported-detail 0.96, contradiction 0.94, so the score is 0.04): CLM rates the descriptions "at least one detail is absent..." and "evidence incompatible..." as similar to the state even when it is supported. That is how CLM behaves with these descriptions, not a bug I could find, and it suggests this transfer may score low. One row proves nothing, so wait for the full result.

## Notes

- `VLLM_USE_FLASHINFER_SAMPLER=0` and `VLLM_WORKER_MULTIPROC_METHOD=spawn` are set by the script. The first because there may be no `nvcc`. The second, and loading the heads only after the encoder has started, because forking vLLM's engine after OpenMP has run in the parent makes it hang or segfault in `libgomp`.
- If your shell sets an `http_proxy`, vLLM's pooling start-up can hang waiting on it; the local addresses are covered by `no_proxy`, but if it stalls, run with the proxy variables unset.
- License of the weights: Apache 2.0.
