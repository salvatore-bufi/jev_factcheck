# Clef-Flash

[Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash): a 9B decision model post-trained from
Qwen3.5-9B (full fine-tune) with a **joint schema head**. Licence: Apache-2.0. Decision Index 0.4.1: Full 44.43, Public
56.15 (snapshot `0501b8c`). It is in the focused study because it has a published LLM-AggreFact score (76.6) to
reproduce.

## How the benchmark runs it

The weights repository ships its code, `joint_schema_model.py`. `adapter.py` imports it from the snapshot and follows
the card's usage example (`load_release_model`, `encode_record`, `collate_records`, then the model's forward):

- **Prompt**: one request per row with all its questions:
  - a fixed system turn ("Read the complete state and schema. Decide every field jointly. …");
  - `STATE:` with the state as compact JSON;
  - `SCHEMA FIELDS:`, with for each question its `ID`, `TYPE`, `INSTRUCTION` and `ALLOWED OPTIONS` (each option as
    `{"option_id", "description"}`; yes/no options are `true` / `false` with the repo's default descriptions);
  - an empty thinking block and `JOINT SCHEMA DECISIONS:`.
- **Attention**: the backbone as loaded by `Qwen3_5ForConditionalGeneration` (the card does not state a non-causal
  mask).
- **Read-out**: the joint schema head (4 layers, 2 evidence-routing layers, width 1024). It reads the final hidden
  states and gives one logit per allowed option of every question, scored jointly. The card applies a softmax per
  question, so the temperature is 1.
- **Limits**: `encode_record` silently cuts a state that does not fit `max_length`. The adapter sets
  `max_length = 32768` and fails such a row (logged in `failures.jsonl`) instead.

What is stored: `logits` (the head's option logits in option-key order, `false`, `true` for yes/no), `temperature` 1,
`probs` the softmax, and `prompt_tokens` (the request's tokens).

## Run

```sh
hf download Cloudflare/clef-flash --revision f0b7d3cd65296abfe459e6c228b618f5d8b26d2f
python benchmark.py --model clef-flash --gpu 0 --limit 32 --no-profile --tag smoke
```

Options: [config/models/clef-flash.json](../../config/models/clef-flash.json) (`revision`, `max_length`, `batch`).

**GPU.** About 19 GB of bf16 weights plus activations of the longest request: use a 40 GB+ GPU for the full test.

**Note.** The joint head sees all questions of a request at once. So its answer to Q1 in the three-question file
(`claim_support.json`) can differ from its answer to Q1 asked alone (`q1_only.json`). That is exactly the C1/C2
comparison of the focused study.
