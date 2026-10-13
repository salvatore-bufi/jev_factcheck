# FactCG-DeBERTa-v3-Large (baseline)

[yaxili96/FactCG-DeBERTa-v3-Large](https://huggingface.co/yaxili96/FactCG-DeBERTa-v3-Large), from
[FactCG](https://github.com/derenlei/FactCG) (NAACL 2025): a 0.4B DeBERTa classifier of factual consistency.
**It is not a decision model** and ignores the question set; it is kept as a reference point.

## How the benchmark runs it

The document is split into chunks of at most 550 words (sentence packing, `chunking.py`, as upstream); each chunk is paired
with the claim in the upstream prompt template (`factcg_client.py`) and scored; the row's score is the highest support
probability. It is stored as one pseudo-question `factcg_support` and judged with the model's own rule, `> 0.5`
(upstream and LLM-AggreFact convention). The 80,000-character document cap applies, as for the other models. The
published leaderboard row FactCG-DeBERTa-L (75.6) is the same model.

## Run

```sh
pip install -r models/factcg/requirements.txt
hf download yaxili96/FactCG-DeBERTa-v3-Large
python benchmark.py --model factcg --gpu 0
```

Options, in [config/models/factcg.json](../../config/models/factcg.json) (or `--opt factcg:KEY=VALUE`): `chunk_size` (550 words), `max_length` (2048 tokens), `pair_batch` (8 pairs per pass), `batch` (32 rows).
`nltk`'s sentence tokenizer (`punkt_tab`) is downloaded on first use if missing. The checkpoint in `ckpt/` (git-ignored)
is the original Lightning file; the Hub repository holds the same weights in transformers format, which is what is loaded.
The module's original standalone CLI and notes are in `legacy/factcg_original/`.

## Verified

After the move to this benchmark, 32 test rows reproduce the earlier full run exactly.
