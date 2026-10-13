"""FactCG-DeBERTa-v3-Large (yaxili96/FactCG-DeBERTa-v3-Large): a baseline, NOT a decision model.

It ignores the question set: it scores the claim against each chunk of the document (sentence-packed,
at most 550 words), with the upstream FactCG prompt, and keeps the highest support probability. Its
answer is stored as one pseudo-question `factcg_support`, judged with its own rule (> 0.5).
"""
import glob
import sys
from pathlib import Path

from core.adapter import BaseAdapter, make_answer
from core.datasets import hub_cache

HERE = Path(__file__).resolve().parent
MODEL_ID = "yaxili96/FactCG-DeBERTa-v3-Large"


class Adapter(BaseAdapter):
    NAME = "FactCG-DeBERTa-v3-Large"
    SIZE = "0.4B"
    KIND = "baseline"
    DEFAULT_DECISION = {"combine": "single", "inputs": [{"question": "factcg_support", "take": "yes"}], "threshold": 0.5}
    DEFAULTS = {
        "chunk_size": 550,      # words per chunk (upstream default)
        "max_length": 2048,     # tokenizer max length (upstream)
        "pair_batch": 8,        # chunk/claim pairs per forward pass
        "batch": 32,            # rows per call
    }

    def load(self):
        sys.path.insert(0, str(HERE))
        from chunking import chunk_document
        from factcg_client import FactCGModel

        cache = hub_cache()
        if not glob.glob(str(cache / "models--yaxili96--FactCG-DeBERTa-v3-Large/snapshots/*/model.safetensors")):
            raise SystemExit(f"{MODEL_ID} is not in the Hugging Face cache {cache}; run `hf download {MODEL_ID}`")
        self.chunk_document = chunk_document
        self.model = FactCGModel(MODEL_ID, cache, self.opt["max_length"])

    def describe(self):
        return {"repo": MODEL_ID, "kind": "baseline (ignores the questions)", "chunk_size": self.opt["chunk_size"],
                "max_length": self.opt["max_length"], "rule": "max support probability over chunks > 0.5"}

    def score(self, states, questions):
        pairs = []   # (row, chunk text)
        for i, state in enumerate(states):
            for chunk in self.chunk_document(state["document"], self.opt["chunk_size"]):
                pairs.append((i, chunk))
        order = sorted(range(len(pairs)), key=lambda j: -len(pairs[j][1]))   # longest first: less padding
        probs = [None] * len(pairs)
        for b in range(0, len(order), self.opt["pair_batch"]):
            idx = order[b:b + self.opt["pair_batch"]]
            support, _ = self.model.score_pairs([pairs[j][1] for j in idx], [states[pairs[j][0]]["claim"] for j in idx])
            for j, p in zip(idx, support):
                probs[j] = p
        per_row = {}
        for (i, _), p in zip(pairs, probs):
            per_row.setdefault(i, []).append(p)
        results = []
        for i in range(len(states)):
            chunk_probs = per_row.get(i) or [0.0]    # empty document: nothing supports the claim
            best = max(chunk_probs)
            answer = make_answer("noul", ["false", "true"], probs={"true": best, "false": 1 - best})
            results.append(({"factcg_support": answer}, {"prompt_tokens": None, "chunks": len(chunk_probs)}))
        return results
