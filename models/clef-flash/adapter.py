"""Clef-Flash (Cloudflare/clef-flash) through the model repository's own code, joint_schema_model.py.

A full fine-tune of Qwen3.5-9B with a joint schema head: a small transformer (4 layers, 2 evidence-routing layers) that
reads the backbone's final hidden states and scores every option of every question of a request jointly, in one
forward pass. The prompt is the repo's: a fixed system turn, "STATE:" with the state as compact JSON, "SCHEMA FIELDS:"
with each question (ID, TYPE, INSTRUCTION, ALLOWED OPTIONS with their descriptions), and an empty thinking block before
"JOINT SCHEMA DECISIONS:". The output is one logit per allowed option; the card applies a softmax per question
(temperature 1).

The adapter imports joint_schema_model.py from the snapshot and calls load_release_model, encode_record,
collate_records and the model's forward, as the card's usage example does, one row per forward. encode_record cuts a
state that does not fit max_length silently; the adapter checks the length first and fails such a row instead.
"""
import importlib.util
import sys
from pathlib import Path

from core.adapter import BaseAdapter, make_answer, option_keys
from core.datasets import hub_cache

REPO = "Cloudflare/clef-flash"


def snapshot(revision):
    path = hub_cache() / f"models--{REPO.replace('/', '--')}" / "snapshots" / revision
    need = ["config.json", "model.safetensors.index.json", "joint_head.safetensors", "joint_head_config.json",
            "joint_schema_model.py", "tokenizer.json"]
    if not all((path / f).exists() for f in need):
        raise SystemExit(f"{REPO} (revision {revision}) is not complete in the Hugging Face cache {hub_cache()}; "
                         f"run `hf download {REPO}`")
    return path


class Adapter(BaseAdapter):
    NAME = "Clef-Flash"
    SIZE = "9B"
    DEFAULTS = {
        "revision": "f0b7d3cd65296abfe459e6c228b618f5d8b26d2f",
        "max_length": 32768,           # tokens per request (state + schema); longer rows fail, never cut
        "batch": 16,                   # rows per call; each row is its own forward pass
    }

    def load(self):
        import torch

        if not torch.cuda.is_available():
            raise SystemExit(f"{self.NAME} needs a GPU (about 19 GB of bf16 weights)")
        self.torch = torch
        self.folder = snapshot(self.opt["revision"])
        spec = importlib.util.spec_from_file_location("clef_joint_schema_model", self.folder / "joint_schema_model.py")
        self.jsm = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.jsm          # its dataclasses look their module up while being defined
        spec.loader.exec_module(self.jsm)
        self.model, self.processor = self.jsm.load_release_model(self.folder, device="cuda:0", dtype=torch.bfloat16)
        self.tok = self.processor.tokenizer
        self.device = next(self.model.parameters()).device

    def describe(self):
        return {"repo": REPO, "revision": self.opt["revision"], "runtime": "joint_schema_model.py (in the snapshot)",
                "max_length": self.opt["max_length"], "dtype": "bfloat16", "temperature": 1.0}

    def score(self, states, questions):
        jsm, results = self.jsm, []
        for state in states:
            record = {"state": state, "questions": questions}
            try:
                encoded = jsm.encode_record(self.tok, record, max_length=self.opt["max_length"])
                if len(encoded.input_ids) >= self.opt["max_length"]:     # a cut state fills max_length exactly
                    results.append(f"request does not fit max_length {self.opt['max_length']} (the state would be cut)")
                    continue
                batch = jsm.collate_records([encoded], self.tok.pad_token_id, self.device)
                with self.torch.inference_mode():
                    logits = self.model(batch)[0]
            except Exception as exc:   # e.g. out of memory on a very long document
                results.append(f"{type(exc).__name__}: {str(exc)[:200]}")
                self.torch.cuda.empty_cache()
                continue
            answers = {}
            for question, question_logits in zip(encoded.questions, logits):
                q = questions[question.question_id]
                raw = dict(zip(question.option_ids, question_logits.float().tolist()))
                keys = option_keys(q)
                answers[question.question_id] = make_answer(q["type"], keys, logits=[raw[k] for k in keys],
                                                            temperature=1.0)
            results.append((answers, {"prompt_tokens": len(encoded.input_ids)}))
        return results
