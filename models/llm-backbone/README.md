# llm-backbone

A plain LLM with **no decision training**, read as a decision model. It is the counterpart of the decision models
post-trained from it: on the same questions, the difference between a TDM and its backbone measures what the decision
training added. Two backbones are configured:

| Config | Repo | Revision | Counterpart of |
|---|---|---|---|
| `config/models/llm-backbone-qwen3.5-9b.json` (BB9) | `Qwen/Qwen3.5-9B` | `c2022362…` | elephant-1, StartLux-9B-Instant, d3-flash, OneJev1.6-9B, Nimble-9B, clef-flash |
| `config/models/llm-backbone-qwen3.5-4b.json` (BB4) | `Qwen/Qwen3.5-4B` | `851bf6e8…` | KnowLine-4B, StartLux-4B-Instant, pplx-decider-flash, Jeffan-4B |

`config/models/llm-backbone.json` is the default (same as BB9). Always pass one of the two named files with
`--model-config`, so the two backbones never share a result folder.

## How the benchmark runs it

`adapter.py` runs the model in-process with **vLLM**, as `models/jpt-9b` does.

- **Prompt** (neutral, the same for every backbone; it never names the right answer):

  ```
  Evaluate the state below and answer the question.

  document:
  <document>

  claim:
  <claim>

  Question: <the question's instructions>
  A. yes
  B. no
  Answer with the letter of the correct option only.
  ```

  The prompt goes inside the chat template with thinking off (`enable_thinking=False`). With `chat: false` it is a
  completion prompt ending in `Answer:`, for a pre-trained (Base) checkpoint.
- **Both option orders.** Every yes/no question is asked twice: once with `A. yes / B. no` and once with
  `A. no / B. yes`. The two prompts share the document prefix through vLLM's prefix cache. Averaging the two orders
  removes the position bias of the letters (readout R1 of the focused study; AnyJev's training-free L0 readout).
- **Read-out**: the next-token log-probabilities of the letter tokens `A` and `B`. They are requested with
  `logprob_token_ids`, so they are returned even when the letters are not in the top tokens.
- **Limits**: prompts above `max_model_len` (32,768 tokens) fail and are logged in `failures.jsonl`, never shortened.
  Only yes/no (`noul`) questions are supported.

## What is stored

For each answer:

| Field | Content |
|---|---|
| `probs`, `logits` | P(yes) = the mean of the two orders (`readout: r1`) or the first order only (`r0`); logits `[0, logit(p)]`, temperature 1 |
| `orders` | `{"yes_first": [lp_A, lp_B], "no_first": [lp_A, lp_B]}`, the raw letter log-probabilities of both prompts |
| `letter_mass` | mean probability the model put on the two letters. Values well below 0.5 mean the model wanted to answer something else (a thinking tag, a word): check the chat template |

`analysis/focused_10.py` re-reads `orders` as the three readouts of the focused study:

- R0, the yes-first order alone;
- R1, both orders with the label prior divided out;
- R2, R1 with a temperature fitted on the dev split.

## Run

```sh
hf download Qwen/Qwen3.5-9B --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a
hf download Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
python benchmark.py --model llm-backbone --model-config config/models/llm-backbone-qwen3.5-4b.json \
    --gpu 0 --limit 32 --no-profile --tag smoke
```
