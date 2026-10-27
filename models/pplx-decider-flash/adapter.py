"""pplx-decider-flash-v1.2 (perplexity-ai/pplx-decider-flash-v1.2) through the model repository's own package, pplx_decider.

A Qwen3.5-4B backbone distilled from Perplexity's 27B decision models (and souped), plus a 255-way decision head
(readout.safetensors, [255, 2560]). Protocol v3 (joint_config.json, "joint-question-inline-readout-v1"): all questions of
a request share one sequence, each question is read at its own inline readout position, and every question attends to
the state and to the other questions. The full-attention layers are non-causal over the whole sequence
(decision_config.json: "noncausal_full_attention"); the linear-attention layers stay causal. Options are shown in request
order under the codes A, B, ... (a yes/no question is A: No / false, B: Yes / true). The calibrated temperature is
1.554 (decision_config.json).

The adapter imports pplx_decider from the snapshot (source/src), loads Decider.from_pretrained on the reference path
(fast=False: the evaluated computation) and calls Decider.logits, which returns the raw readout logits of each
question's valid options before temperature. They are stored with the saved temperature, so `probs` equal what
Decider.decide returns. A request longer than max_length is split by the package into question chunks that each keep the
full state (never truncated); a single question that does not fit fails the row.
"""
import sys
from pathlib import Path

from core.adapter import BaseAdapter, make_answer, option_keys
from core.datasets import hub_cache

REPO = "perplexity-ai/pplx-decider-flash-v1.2"


def snapshot(revision):
    path = hub_cache() / f"models--{REPO.replace('/', '--')}" / "snapshots" / revision
    need = ["config.json", "decision_config.json", "joint_config.json", "readout.safetensors", "tokenizer.json",
            "model.safetensors.index.json", "source/src/pplx_decider/model.py"]
    if not all((path / f).exists() for f in need):
        raise SystemExit(f"{REPO} (revision {revision}) is not complete in the Hugging Face cache {hub_cache()}; "
                         f"run `hf download {REPO}`")
    return path


class Adapter(BaseAdapter):
    NAME = "pplx-decider-flash-v1.2"
    SIZE = "4B"
    DEFAULTS = {
        "revision": "9ff7cf9838e59df461c276b7699d56b244d2af58",
        "max_length": 32768,           # tokens per sequence (training used up to about 32k); see the docstring
        "batch_tokens": 32768,         # padded tokens per forward batch on the reference path (the package default)
        "batch": 16,                   # rows per call
    }

    def load(self):
        import torch

        if not torch.cuda.is_available():
            raise SystemExit(f"{self.NAME} needs a GPU (about 9 GB of bf16 weights)")
        self.torch = torch
        self.folder = snapshot(self.opt["revision"])
        sys.path.insert(0, str(self.folder / "source" / "src"))
        import pplx_decider

        self.pkg = pplx_decider
        self.decider = pplx_decider.Decider.from_pretrained(self.folder, device="cuda:0", fast=False,
                                                            max_length=self.opt["max_length"],
                                                            batch_tokens=self.opt["batch_tokens"])

    def describe(self):
        return {"repo": REPO, "revision": self.opt["revision"], "runtime": f"pplx_decider {self.pkg.__version__} "
                "(source/src in the snapshot), reference path", "temperature": self.decider.temperature,
                "max_length": self.opt["max_length"], "attention_mode": "noncausal_full_attention"}

    def row(self, state, questions, logits):
        t = self.decider.temperature
        answers = {qid: make_answer(q["type"], option_keys(q), logits=list(logits[qid]), temperature=t)
                   for qid, q in questions.items()}
        return answers, {"prompt_tokens": None}

    def score(self, states, questions):
        requests = [{"state": state, "questions": questions} for state in states]
        try:
            return [self.row(s, questions, lg) for s, lg in zip(states, self.decider.logits(requests))]
        except Exception:      # one row too long, or out of memory: retry row by row so only that row fails
            self.torch.cuda.empty_cache()
        results = []
        for state, request in zip(states, requests):
            try:
                results.append(self.row(state, questions, self.decider.logits([request])[0]))
            except (self.pkg.LengthError, self.pkg.RequestError) as exc:
                results.append(f"{type(exc).__name__}: {str(exc)[:200]}")
            except Exception as exc:
                results.append(f"{type(exc).__name__}: {str(exc)[:200]}")
                self.torch.cuda.empty_cache()
        return results
