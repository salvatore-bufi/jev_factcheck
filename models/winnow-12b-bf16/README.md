# Winnow-12B in bf16 (vLLM)

[EldanRing/Winnow-12B](https://huggingface.co/EldanRing/Winnow-12B) at full precision. The authors publish only GGUF
files; vLLM 0.28 cannot load GGUF, but `gguf/Winnow-12B-BF16.gguf` holds only BF16 and F32 tensors, so it is a lossless
copy of the weights. Licence: Apache 2.0.

## How the benchmark runs it

1. **`convert.py`** (once) turns the BF16 GGUF into a bf16 safetensors checkpoint in `models/winnow-12b-bf16/model/`
   (24 GB, about 3 minutes, git-ignored). llama.cpp's Gemma 4 converter only renames tensors, so the inverse is a rename:
   it refuses to finish unless all 666 tensors map, none is missing or unexpected, and every shape matches; it writes a
   text-only `gemma4_text` config and copies the tokenizer and chat template.
2. **`adapter.py`** runs it in-process with **vLLM** and reproduces the Winnow server's prompt (read from its source,
   `native/protocol.h` and `engine.h`): fixed system prompt, `State:` + compact JSON with `<` escaped, one suffix per
   question (`Question: ... Options: A: ... Return the correct letter label.`), the chat boundary with the thinking-off
   block and `Answer:`, prefix and suffix tokenized separately (BOS only at the start), softmax over the option-letter
   logits at temperature 1. Yes/no: A = false, B = true. `noul`, `choice` and `score` with up to 26 options.

## Run

```sh
hf download EldanRing/Winnow-12B --include "gguf/Winnow-12B-BF16.gguf" "*.json" "*.jinja"
python models/winnow-12b-bf16/convert.py
python benchmark.py --model winnow-12b-bf16 --gpu 0 --limit 32
python benchmark.py --model winnow-12b-bf16 --gpu 0
```

The weights take 22.2 GiB. `cpu_offload_gb=auto` keeps in RAM what does not fit: 0 on a 45 GB card, about 15 GB on a
24 GB card at a 32K context (slow; a 24 GB card is fine for checks, not for the full run). Options, in
[config/models/winnow-12b-bf16.json](../../config/models/winnow-12b-bf16.json) (or `--opt winnow-12b-bf16:KEY=VALUE`): `temperature` (1.0),
`max_model_len` (32768), `cpu_offload_gb` (auto), `gpu_util` (0.92), `max_num_seqs` (32), `max_batched_tokens` (16384),
`batch` (128).

## Verified

- Conversion: all 666 tensors mapped and shape-checked; seven spot-checked tensors bit-identical to the GGUF.
- Prompts: on 8 rows from 7 sources, all 24 prompts token-for-token identical to those the Winnow server builds itself
  (`/v1/winnow/inspect`).
- Probabilities: on 63 test rows within 0.007 / 0.019 / 0.015 (mean) of the Q8_0 file on the authors' server, 62 of 63
  decisions equal; further from NVFP4 (58 of 63), as expected from 4-bit quantization.
- After the move to this benchmark, 32 test rows match the earlier full run (run without offload on another GPU): mean
  |ΔP(yes)| 0.005, 31 of 32 decisions equal (one near-threshold row moved; largest single difference 0.117).
