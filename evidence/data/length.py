"""Prompt length of a row in a model's tokens, for the build's length filter (--max-tokens).

A row is kept only if its whole prompt fits the budget:

    document tokens + 2 × claim tokens + PROMPT_MARGIN  <=  max_tokens

The claim counts twice because the evidence prompt carries it in the preamble and in the state. PROMPT_MARGIN
covers everything else (instructions, question with its option descriptions, preamble words, chat tokens):
KnowLine-4B's own front end adds 157 tokens around document and claim for the longest question of the study
(unsupported_detail), measured with knowline_layout.render_request on a document read once. Short requests that
KnowLine writes twice (under 3,000 tokens) never reach a limit in the tens of thousands, so they need no care here.

The tokenizer is the model's, read from the local Hugging Face cache at the revision of config/models/<model>.json.
All ten decision models of the study are Qwen3.5 and share its tokenizer, so the count holds for them too.
"""
import json
from pathlib import Path

from core.datasets import hub_cache

ROOT = Path(__file__).resolve().parents[2]
PROMPT_MARGIN = 256
REPOS = {"knowline-4b": "PelaAI/KnowLine-4B-Gen5.5"}


class PromptLength:
    def __init__(self, model="knowline-4b"):
        from transformers import AutoTokenizer

        if model not in REPOS:
            raise SystemExit(f"length filter: no tokenizer known for {model!r} (known: {sorted(REPOS)})")
        revision = json.loads((ROOT / "config" / "models" / f"{model}.json").read_text())["revision"]
        folder = hub_cache() / f"models--{REPOS[model].replace('/', '--')}" / "snapshots" / revision
        if not (folder / "tokenizer.json").is_file():
            raise SystemExit(f"length filter: tokenizer of {REPOS[model]}@{revision} not in the Hugging Face cache "
                             f"(hf download {REPOS[model]} --revision {revision})")
        self.tokenizer = AutoTokenizer.from_pretrained(str(folder))
        self.describe = {"model": model, "repo": REPOS[model], "revision": revision, "prompt_margin": PROMPT_MARGIN}

    def count(self, text):
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def prompt_tokens(self, row):
        """-> (document tokens, estimated prompt tokens) of a built row."""
        doc = self.count(row["document"])
        return doc, doc + 2 * self.count(row["claim"]) + PROMPT_MARGIN
