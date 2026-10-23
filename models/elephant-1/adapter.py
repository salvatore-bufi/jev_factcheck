"""elephant-1 (baselabs/elephant-1) on its own serving runtime, OneJev's qev engine (copied in ./qev).

A Qwen3.5-9B decision model. The engine renders one prompt per question: the system turn and "<state>...</state>"
form a shared prefix, prefilled once per row; the question suffixes ("Question: ... Options: A. yes: ... B. no: ...
Answer with one letter: A, B.") run as one batched forward on a fork of the prefix cache. The answer is a softmax
over the LM-head logits of the option letters (slot_lm readout, head in float32). The full-attention layers are
block-bidirectional (prefix block, then each suffix block), as declared in the checkpoint's config.json; the engine
reads that declaration only from a local folder, so the adapter passes the snapshot path, never the hub id.

The checkpoint has no calibration.json, so the temperature is 1. The adapter keeps the raw slot logits of every
answer (decide(debug=True)) and computes the probabilities from them (the engine's own answers are the same values
rounded to 6 decimals), so answers can be re-tempered afterwards.
"""
from pathlib import Path
import sys

from core.adapter import BaseAdapter, make_answer, option_keys
from core.datasets import hub_cache

HERE = Path(__file__).resolve().parent
REPO = "baselabs/elephant-1"
RUNTIME = "OneJev qev @81ce62f + serve.patch"


def snapshot(revision):
    path = hub_cache() / f"models--{REPO.replace('/', '--')}" / "snapshots" / revision
    if not all((path / f).exists() for f in ("config.json", "model.safetensors", "tokenizer.json", "readout.json")):
        raise SystemExit(f"{REPO} (revision {revision}) is not complete in the Hugging Face cache {hub_cache()}; "
                         f"run `hf download {REPO}`")
    return path


class Adapter(BaseAdapter):
    NAME = "elephant-1"
    SIZE = "9B"
    DEFAULTS = {
        "revision": "b3f8a93462fb1051a9de031e94a7ff380e254072",
        "dtype": "bfloat16",            # the model; the LM head runs in head_dtype
        "head_dtype": "float32",        # the engine's default: bf16 logits would quantize the probabilities
        "fork_mode": "auto",            # auto: batched suffixes if they match sequential ones at load; or sequential
        "max_branch_tokens": 32768,     # state plus one question; longer requests are refused, never truncated
        "max_request_tokens": 65536,    # state plus all questions
        "batch": 16,                    # rows per call; each row is its own prefill
    }

    def load(self):
        import torch

        sys.path.insert(0, str(HERE))
        from qev.engine import DecisionEngine
        from qev.schema import SystemOneRequest

        self.torch, self.Request = torch, SystemOneRequest
        if not torch.cuda.is_available():
            raise SystemExit(f"{self.NAME} needs a GPU (18 GB of bf16 weights)")
        self.folder = snapshot(self.opt["revision"])
        self.engine = DecisionEngine(str(self.folder), device="cuda:0", dtype=self.opt["dtype"],
                                     head_dtype=self.opt["head_dtype"], fork_mode=self.opt["fork_mode"],
                                     max_branch_tokens=self.opt["max_branch_tokens"],
                                     max_request_tokens=self.opt["max_request_tokens"])
        info = self.engine.info
        if info.extra["attention"] != "noncausal":
            raise SystemExit(f"{self.NAME}: the engine reads attention {info.extra['attention']!r}, expected noncausal")

    def describe(self):
        info = self.engine.info
        return {"repo": REPO, "revision": self.opt["revision"], "runtime": RUNTIME, "prompt": info.prompt_version,
                "dtype": info.dtype, "head_dtype": self.opt["head_dtype"], "attention": info.extra["attention"],
                "readout": info.extra["readout"], "calibration": info.extra["calibration"],
                "fork_mode": info.fork_mode, "fork_max_abs_diff": info.fork_max_abs_diff,
                "transformers": info.transformers_version, "torch": info.torch_version}

    def decide(self, request):
        """The engine's decide(), with the fallback of the checkpoint's engine.py: the batched prefix fork can
        overflow 32-bit indexing on the largest requests, and the sequential branches are the same computation."""
        try:
            return self.engine.decide(request, debug=True)
        except RuntimeError as exc:
            if "32BitIndexMath" not in str(exc):
                raise
            mode, self.engine.info.fork_mode = self.engine.info.fork_mode, "sequential"
            try:
                return self.engine.decide(request, debug=True)
            finally:
                self.engine.info.fork_mode = mode

    def score(self, states, questions):
        results = []
        for state in states:
            try:
                resp, meta = self.decide(self.Request(state=state, questions=questions))
            except ValueError as exc:            # over the length limits: refused, never truncated
                results.append(f"ValueError: {exc}")
                continue
            except Exception as exc:             # e.g. out of memory on a very long document
                results.append(f"{type(exc).__name__}: {str(exc)[:200]}")
                self.torch.cuda.empty_cache()
                continue
            answers = {}
            for qid, q in questions.items():
                d = meta["diagnostics"][qid]
                (logits,) = d["slot_logits"]            # one branch per question (no option-order debiasing)
                if q["type"] == "noul":                 # the engine's slots are A = yes, B = no
                    yes, no = logits
                    logits = [no, yes]                  # choice and score slots already follow option_keys
                ans = make_answer(q["type"], option_keys(q), logits=logits, temperature=d["temperature"])
                ans["slot_mass"] = d["allowed_token_mass"]   # share of the next-token mass on the option letters
                answers[qid] = ans
            results.append((answers, {"prompt_tokens": resp.usage.input_tokens}))
        return results
