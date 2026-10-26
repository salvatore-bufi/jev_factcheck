# Bespoke-MiniCheck-7B (baseline)

[bespokelabs/Bespoke-MiniCheck-7B](https://huggingface.co/bespokelabs/Bespoke-MiniCheck-7B): a 7B grounded fact-checker
(InternLM2.5-7B fine-tuned) by Bespoke Labs and the authors of LLM-AggreFact. It is the top model of the MiniCheck series
and was state of the art on the LLM-AggreFact leaderboard when released. Weights licence: CC BY-NC 4.0; commercial use
needs a licence from Bespoke Labs (company@bespokelabs.ai).

Like [FactCG](../factcg/README.md), it is a **baseline, not a decision model**: it ignores the question set, checks the
claim against the document, and gives one support probability, judged with its own rule (> 0.5).

## How the benchmark runs it

`adapter.py` reproduces the authors' scorer, `minicheck.inference.LLMCheck.score`
([Liyan06/MiniCheck](https://github.com/Liyan06/MiniCheck) @`b58b9fa`, Apache-2.0), step for step, on vLLM in process:
- **Chunks**: the document is split into sentences (NLTK, newlines kept) and packed greedily into chunks of at most
  `max_model_len - 300` = 32,468 tokens. On the test split every document is one chunk.
- **Claim sentences**: the claim is split into sentences (NLTK). The model checks one sentence at a time.
- **Prompt**: for every (chunk, sentence) pair, the authors' fixed system prompt ("Determine whether the provided claim is
  consistent with the corresponding document. ... responding with either "Yes" or "No".") and the user turn
  `Document: <chunk>\nClaim: <sentence>`, in the model's chat template.
- **Read-out**: one greedy token with the top 5 log-probabilities. P(supported) is the summed probability of the top
  tokens that read "yes" (any case).
- **SentenceFusion**: for each claim sentence the best chunk, then the weakest sentence:
  `min over sentences of max over chunks`. Supported when > 0.5.

The authors' package itself is not installed: it pulls in `openai`, `datasets` and `pandas`, and loads the model from
the hub id. The prompts the adapter builds are identical to the package's (checked on 50 test rows), and its scores
match `LLMCheck.score` run on the same engine within 0.006. That difference comes from vLLM batching the two runs
differently.

What is stored, as the pseudo-question `minicheck_support`:
- `probs`: the final score (`true`) and its complement;
- `chunks`, `sentences`: how many document chunks and claim sentences were checked;
- `per_sentence`: the best-chunk score of each claim sentence (the final score is their minimum);
- no `logits`: the score is a sum over a top-5 list, not a softmax over options, so it cannot be re-tempered.

## Run

```sh
hf download bespokelabs/Bespoke-MiniCheck-7B
python -c "import nltk; nltk.download('punkt_tab')"     # once; behind a proxy: NLTK_ALLOW_PROXIED_URLOPEN=1
python benchmark.py --model minicheck-7b --gpu 0 --limit 32
python benchmark.py --model minicheck-7b --gpu 0
```

Options, in [config/models/minicheck-7b.json](../../config/models/minicheck-7b.json) (or `--opt minicheck-7b:KEY=VALUE`):

| option | default | meaning |
|---|---|---|
| `revision` | `1ed7786` | snapshot in the Hugging Face cache |
| `max_model_len` | 32768 | the authors' default; chunks are 300 tokens shorter |
| `gpu_util` | 0.90 | vLLM `gpu_memory_utilization` (its default, as upstream) |
| `prefix_caching` | true | an upstream option (off by default there): the sentences of a claim share the document prefix |
| `eager` | true | no CUDA graphs (see below) |
| `batch` | 64 | rows per vLLM call |

**vLLM bug.** In vLLM 0.28, 0.30 and 0.31, `InternLM2ForCausalLM.forward` requires `intermediate_tensors`, and the new
model runner leaves it out when it captures CUDA graphs, so the engine fails at start-up
(`forward() missing 1 required positional argument: 'intermediate_tensors'`). Normal execution passes it, so the
adapter runs eagerly (`enforce_eager`). Each prompt generates a single token, so graphs would save almost nothing.

**GPU.** 15.5 GB of bf16 weights; a 24 GB GPU runs it (vLLM keeps about 45,000 tokens of KV cache there).

## Verified

32 test rows on an RTX 5090 Laptop GPU (24 GB), vLLM 0.28.0:
- 0 failures, 6.4 rows/s;
- balanced accuracy 0.84, ROC AUC 0.87 with its own rule (> 0.5).

On 50 rows (48 random and the 2 longest documents):
- the prompts are identical to those of the authors' `LLMCheck`;
- the scores match `LLMCheck.score` within 0.006;
- the longest documents fit in one chunk.
