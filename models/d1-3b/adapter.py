"""d1-3B (LiquidAI/d1-3B) through its own one-pass System One API (`model.system_one_batch`).

The model ships its own code (trust_remote_code): the state as indented JSON, `QUESTION:` and the
question (a noul ends with "Reply with yes or no only."), and a softmax over the option tokens at the
answer slot. A request with several questions reads the state once (one tree-shaped pass). Plain
transformers on one GPU; the engine reports probabilities only (no logits), so answers cannot be
re-tempered afterwards.
"""
import glob
import sys
from pathlib import Path

from core.adapter import BaseAdapter, make_answer, option_keys
from core.datasets import hub_cache

MODEL_ID = "LiquidAI/d1-3B"
CONTEXT = 32768


def check_model():
    need = ["config.json", "model.safetensors", "tokenizer.json", "modeling_d1.py", "runner.py", "prompt.py"]
    for snap in sorted(glob.glob(str(hub_cache() / "models--LiquidAI--d1-3B/snapshots/*"))):
        if all((Path(snap) / f).exists() for f in need):
            return
    raise SystemExit(f"{MODEL_ID} (6 GB, with its code files) is not complete in the Hugging Face cache {hub_cache()}; "
                     "run `hf download LiquidAI/d1-3B` or copy models--LiquidAI--d1-3B into that cache.")


class Adapter(BaseAdapter):
    NAME = "d1-3B"
    SIZE = "3B"
    DEFAULTS = {
        "revision": "051bcc464b01b9f92942b364d9586b0ef5912432",
        "batch": 64,           # rows per call; each row is still its own forward pass (its questions share it)
    }

    def load(self):
        import torch
        from transformers import AutoModel

        check_model()
        self.torch = torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        self.model = AutoModel.from_pretrained(MODEL_ID, revision=self.opt["revision"], trust_remote_code=True,
                                               dtype=dtype).to(device).eval()
        self.engine = self.model.engine
        package = type(self.model).__module__.rsplit(".", 1)[0]
        self.as_question = sys.modules[package + ".prompt"].as_question
        self.device, self.dtype = device, str(dtype)

    def describe(self):
        return {"repo": MODEL_ID, "revision": self.opt["revision"], "api": "system_one_batch",
                "device": self.device, "dtype": self.dtype, "calibration": None}

    def score(self, states, questions):
        objs = {qid: self.as_question(q) for qid, q in questions.items()}
        results = []
        for state in states:
            lengths = {qid: self.engine.tokens(state, [q]) for qid, q in objs.items()}
            if max(lengths.values()) > CONTEXT:
                results.append(f"prompt of {max(lengths.values())} tokens exceeds the model's {CONTEXT}-token context")
                continue
            try:
                out = self.model.system_one_batch([(state, questions)])[0]
            except Exception as exc:   # e.g. out of memory on a very long document
                results.append(f"{type(exc).__name__}: {str(exc)[:200]}")
                if self.device == "cuda":
                    self.torch.cuda.empty_cache()
                continue
            answers = {}
            for qid, a in out["answers"].items():
                kind = questions[qid]["type"]
                probs = {"true": a["noul"], "false": 1 - a["noul"]} if kind == "noul" else a["probabilities"]
                answers[qid] = make_answer(kind, option_keys(questions[qid]), probs=probs)
            results.append((answers, {"prompt_tokens": lengths, "request_input_tokens": out["usage"]["input_tokens"]}))
        return results
