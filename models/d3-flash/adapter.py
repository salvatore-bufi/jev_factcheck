"""d3-flash (vllm-sr/d3-flash) through its own System One runtime (`d3_runtime.D3`, loaded by `AutoModel`).

The model ships its own code (trust_remote_code): a fixed system prompt, a user turn "State: <state> /
Question: ... / Options: A: No / false, B: Yes / true / Reply with only the code of the best option.",
and a 255-way readout over the answer codes at the last prompt position, softmax at the package's
temperature (1, decision_config.json). noul options are A = false, B = true. The full-attention layers see
the whole prompt (noncausal_full_attention); each question is its own prompt, the state is read again for each.

The adapter calls the runtime's `prepare` and `run` (what `system_one` does) to get each question's option
probabilities in option order. The runtime reports probabilities only; at temperature T the readout
logits are T * log(p) up to a constant, so they are stored and answers can be re-tempered afterwards.
The package sets no length cap; the adapter applies 32,768 tokens per question, as d3-turbo validates.
"""
import glob
import math
from pathlib import Path

from core.adapter import BaseAdapter, make_answer, option_keys
from core.datasets import hub_cache

MODEL_ID = "vllm-sr/d3-flash"


def check_model():
    need = ["config.json", "decision_config.json", "readout.safetensors", "tokenizer.json", "MODEL_MANIFEST.json",
            "model.safetensors.index.json", "modeling_d3.py", "d3_runtime.py", "d3_format.py"]
    for snap in sorted(glob.glob(str(hub_cache() / "models--vllm-sr--d3-flash/snapshots/*"))):
        if all((Path(snap) / f).exists() for f in need):
            return
    raise SystemExit(f"{MODEL_ID} (17 GB, with its code files) is not complete in the Hugging Face cache {hub_cache()}; "
                     "run `hf download vllm-sr/d3-flash` or copy models--vllm-sr--d3-flash into that cache.")


def logits_of(probs, temperature):
    """Readout logits up to a constant from probabilities at `temperature` (an underflowed 0 stays -inf)."""
    return [temperature * math.log(p) if p > 0 else float("-inf") for p in probs]


class Adapter(BaseAdapter):
    NAME = "d3-flash"
    SIZE = "8.4B"
    DEFAULTS = {
        "revision": "13dc1cc8ef23caefa765907e1d764751e5629f7d",
        "max_length": 32768,       # tokens per question; longer prompts fail (never truncated)
        "question_batch": 8,       # questions per forward pass (the runtime's batch_size)
        "verify": "fast",          # check the files against MODEL_MANIFEST.json: fast, full or none
        "batch": 32,               # rows per call; each row is still its own request
    }

    def load(self):
        import torch
        from transformers import AutoModel

        check_model()
        self.torch = torch
        if not torch.cuda.is_available():
            raise SystemExit(f"{self.NAME} needs a GPU (17 GB of bf16 weights)")
        self.model = AutoModel.from_pretrained(MODEL_ID, revision=self.opt["revision"], trust_remote_code=True,
                                               device="cuda:0", batch_size=self.opt["question_batch"],
                                               verify=self.opt["verify"])
        self.runtime = self.model.runtime
        self.runtime.max_length = self.opt["max_length"] or None

    def describe(self):
        rt = self.runtime
        return {"repo": MODEL_ID, "revision": self.opt["revision"], "api": "D3.prepare + D3.run (the system_one path)",
                "device": str(rt.device), "dtype": f"bfloat16 backbone, {rt.readout_dtype} readout",
                "temperature": rt.temperature, "attention_mode": rt.attention_mode, "max_length": rt.max_length,
                "fast_path": rt.fast is not None, "kernels": rt.kernels}

    def score(self, states, questions):
        rt = self.runtime
        results = []
        for state in states:
            try:
                prepared = rt.prepare(state, questions)
                if prepared.errors:
                    qid, err = next(iter(prepared.errors.items()))
                    results.append(f"{qid}: {err['error']}: {err['message']}")
                    continue
                probs, tokens = rt.run(prepared)
            except Exception as exc:   # e.g. out of memory on a very long document
                results.append(f"{type(exc).__name__}: {str(exc)[:200]}")
                self.torch.cuda.empty_cache()
                continue
            answers = {}
            for qid, q in questions.items():
                keys = option_keys(q)
                if list(prepared.questions[qid].keys) != keys:
                    raise RuntimeError(f"{qid}: runtime option order {prepared.questions[qid].keys} != {keys}")
                answers[qid] = make_answer(q["type"], keys, probs=probs[qid],
                                           logits=logits_of(probs[qid], rt.temperature), temperature=rt.temperature)
            lengths = {qid: len(prepared.sequences[qid]) for qid in questions}
            results.append((answers, {"prompt_tokens": lengths, "request_input_tokens": tokens}))
        return results
