# Winnow-12B in bf16 (no quantization) on LLM-AggreFact

Evaluates [EldanRing/Winnow-12B](https://huggingface.co/EldanRing/Winnow-12B), a Gemma 4 12B fine-tune for typed decisions, at full bf16 precision with **vLLM**, on the 29,320-row LLM-AggreFact test split, and produces a row for the results table in the top-level README. The sibling folder [../winnow-12b/](../winnow-12b/README.md) runs the NVFP4 quantization on the authors' llama.cpp server instead.

## How it works

Winnow is published only as GGUF files, and vLLM 0.28 cannot load GGUF. But `Winnow-12B-BF16.gguf` contains only BF16 and F32 tensors, so it is a lossless copy of the model, and llama.cpp's Gemma 4 converter only renamed tensors (no scaling or permutation). Two steps follow:

1. **`convert.py`** turns the GGUF into a normal bf16 safetensors checkpoint in `winnow-12b-bf16/model/` (24 GB, about 3 minutes). It renames each tensor to its Hugging Face / vLLM name and refuses to finish unless all 666 tensors are mapped, none is missing or unexpected, and every shape matches the architecture. It writes a plain text-only `gemma4_text` config (the repository's "unified" multimodal config is not needed and is rejected by this transformers version) and copies the tokenizer and chat template. The vision and audio towers live in a separate projector file and are not used.
2. **`benchmark.py`** runs the model in-process with vLLM and reproduces the prompt protocol of the Winnow server (read from its source, `native/protocol.h` and `engine.h`): the fixed system prompt, `State:` plus the compact JSON state with `<` escaped, one suffix per question (`Question: ... Options: A: ... B: ... Return the correct letter label.`), the chat boundary with the thinking-off block and `Answer:`, the prefix and each suffix tokenized separately (BOS only at the start), and a softmax over the option-letter logits at temperature 1. For a noul question option A is `false`, option B is `true`, and the answer is P(B).

For each (document, claim) pair it asks the three questions of the frozen Jev checker ([../packs/claim_support.json](../packs/claim_support.json)) and applies the frozen rule:

```
score = min(p_support, 1 - p_unsupported_detail, 1 - p_contradicted_detail)
supported = score > 0.30
```

This is a **fixed transfer evaluation**, like the CLM, JPT-9B, Rune and Winnow-NVFP4 runs: same questions, rule, threshold and 80,000-character document cap (75% head, 25% tail). Nothing is tuned for Winnow, and the 0.30 threshold was fitted to Jev's probabilities. The cache stores the raw option logits, so `--temperature` (default 1, Winnow's own) and `--threshold` only rescale the report for free. Do not pick either from test results.

## Run

From the repository root, in the `factcheck` conda environment (see `requirements.txt`). The Winnow GGUF files must be in the Hugging Face cache (`hf download EldanRing/Winnow-12B`; only `gguf/Winnow-12B-BF16.gguf` and the small root files are used here).

```sh
python winnow-12b-bf16/convert.py                    # once: GGUF -> safetensors in winnow-12b-bf16/model/
python winnow-12b-bf16/benchmark.py --limit 1        # one claim vs its document (smoke test; separate output folder)
python winnow-12b-bf16/benchmark.py                  # full test split; resumable
python winnow-12b-bf16/benchmark.py --offline        # rebuild the report from cached answers (no GPU)
python winnow-12b-bf16/test_benchmark.py             # offline unit checks, no GPU
```

If interrupted (Ctrl-C), rerun the same command: finished rows are skipped. The cache is tied to the converted weights, document cap, prompt, pack, data hash and `--limit`; a changed configuration is refused instead of mixed in. A prompt longer than `--max-model-len` is printed as a failure and excluded, and the report is then marked incomplete; it is never shortened silently or given an invented score.

## Hardware

The bf16 weights take 22.2 GiB, and the KV cache for a long context adds about 8 GiB at 24K tokens and 10 GiB at 32K.

- **45 GB GPU (for example an L40S):** the defaults work as they are (`--cpu-offload-gb 0`, `--max-model-len 32768`). `--gpu ID` picks the card. I have not run it on one, so I can't give a measured speed; I'd expect a few hours for the full split.
- **24 GB GPU (the laptop used for development):** it only works with CPU offload and a short context: `--cpu-offload-gb 8 --max-model-len 8192` (or a longer context with more offload). That was fast enough to validate but slow, about 0.2 rows/s, and 8K tokens excludes the longest documents, which show up as failures. Use a larger GPU for the real run.

Other options: `--max-num-seqs` (32), `--max-batched-tokens` (16384), `--gpu-util` (0.92), `--window` (128 rows per batch). The script sets `VLLM_USE_FLASHINFER_SAMPLER=0`, because without `nvcc` FlashInfer's sampler would try to compile CUDA at start-up; it never samples (one greedy token, label logprobs only).

## Output

In `results/runs/winnow-12b-bf16/` (git-ignored): `answers.jsonl` (the cache), `report.json` and `table_row.md`, a ready-made row such as `| Winnow-12B-BF16 (ours; fixed Jev rule) | 12B | <avg> | <11 sources> |`. Smoke runs go to `results/runs/winnow-12b-bf16-limit<N>/` and are labelled as non-benchmark.

## What was verified

- **Conversion:** all 666 tensors mapped and shape-checked; spot checks of seven tensors (embeddings, attention, MLP, norms, layer scale, final norm) are bit-identical to the GGUF; the shared K/V global layers have no `v_proj`, as the architecture requires.
- **Prompts:** for 8 rows from 7 different sources (254 to 1,286 tokens), all 24 prompts are token-for-token identical to the ones the Winnow server builds itself (compared through its `/v1/winnow/inspect` endpoint).
- **Probabilities:** on 63 test rows, these bf16 answers match the Q8_0 GGUF on the authors' server closely (mean absolute difference 0.007 / 0.019 / 0.015 for support / unsupported / contradicted; final decision agrees on 62 of 63 rows). They differ more from the NVFP4 file (mean 0.04 to 0.08; 58 of 63 decisions agree), as expected from 4-bit quantization; Q8 and NVFP4 differ from each other about as much.
- Not checked: the full benchmark, and the L40S speed.

## Notes

- Thinking, MTP, vision and the experimental reasoning mode are not used.
- License of the weights: Apache 2.0.
