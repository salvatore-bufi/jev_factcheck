"""Bespoke-Nimble-9B-v3 (bespokelabs/Bespoke-Nimble-9B-v3): a LoRA adapter on Qwen/Qwen3.5-9B, through the authors' engine.

The adapter repository holds only the LoRA weights and the prompt contract (schema_config.json). The engine that
produced the authors' Decision Index run is published separately, in the dataset repository
bespokelabs/bespoke-nimble-9b-v3-decision-index @654b124 (code/nimble, CC BY-NC 4.0); ./nimble is a byte-identical copy
(it cannot run from the Hugging Face cache: it hashes its prompt files next to its own resolved path, which there is the
blobs folder). NimbleEngine checks at load that its two prompt files have the SHA-256 the adapter's contract pins, and
that the tokenizer gives the contract's candidate codes.

The engine loads Qwen3.5-9B (bf16, transformers) with the LoRA through PEFT. Each question is one prompt: a system
prompt ("Classify the context using the supplied schema. ..."), then one JSON user turn holding the context (the state as
JSON) and the schema of every question of the row (yes/no as a boolean field with its false/true descriptions), then
"Requested field: <name>"; the answer is a softmax over the candidate code tokens (A = false, B = true for a boolean).
The questions of a row share their prompt up to the field name: that prefix runs once and each question's short tail runs
on a copy of its cache (shared_prefix).

The engine is a Decision Index kit engine. The kit is not installed here: its two names the engine imports (Engine,
Unsupported) are stood in for, and the adapter calls the steps of NimbleEngine.__call__ with the engine's own functions,
keeping the probabilities of both options (__call__ returns only P(true)). The readout temperature is 1, so log(p) is
stored as logits (the candidate logits up to a constant) and answers can be re-tempered afterwards.
"""
import json
import math
from pathlib import Path
import sys
import types

from core.adapter import BaseAdapter, make_answer, option_keys
from core.datasets import hub_cache

HERE = Path(__file__).resolve().parent
REPO = "bespokelabs/Bespoke-Nimble-9B-v3"
ENGINE = "bespokelabs/bespoke-nimble-9b-v3-decision-index@654b124 (code/nimble, copied in ./nimble)"


def snapshot(kind, repo, revision, need, hint):
    path = hub_cache() / f"{kind}--{repo.replace('/', '--')}" / "snapshots" / revision
    missing = [f for f in need if not (path / f).exists()]
    if missing:
        raise SystemExit(f"{repo} (revision {revision[:7]}) is missing {missing[:3]} in the Hugging Face cache "
                         f"{hub_cache()}; run `{hint}`")
    return path


def base_weights(revision):
    """The base model's snapshot, with every weight shard its index lists."""
    hint = f"hf download Qwen/Qwen3.5-9B --revision {revision}"
    path = snapshot("models", "Qwen/Qwen3.5-9B", revision, ["config.json", "tokenizer.json", "model.safetensors.index.json"], hint)
    shards = set(json.loads((path / "model.safetensors.index.json").read_text())["weight_map"].values())
    return snapshot("models", "Qwen/Qwen3.5-9B", revision, sorted(shards), hint)


def kit_stand_in():
    """The two names decision_index_engine.py imports from the Decision Index kit, when the kit is not installed."""
    try:
        import decision_index.engines  # noqa: F401
        return
    except ImportError:
        pass

    class Engine:
        def __init__(self, **options):
            self.options = options

    class Unsupported(Exception):
        pass

    kit, engines = types.ModuleType("decision_index"), types.ModuleType("decision_index.engines")
    engines.Engine, engines.Unsupported, kit.engines = Engine, Unsupported, engines
    sys.modules.update({"decision_index": kit, "decision_index.engines": engines})


class Adapter(BaseAdapter):
    NAME = "Bespoke-Nimble-9B-v3"
    SIZE = "9B"
    DEFAULTS = {
        "revision": "8e927b9b4afdbb14479fac10a7364d1a695be208",        # the LoRA adapter
        "max_tokens": 32768,           # the engine's default: longer prompts are refused, never truncated
        "batch_tokens": 65536,         # the engine's default: padded tokens per batch without the shared prefix
        "shared_prefix": True,         # the engine's default: the row's common prompt prefix runs once
        "cache_tokens": 131072,        # the engine's default: prefix tokens times tails per batch
        "batch": 16,                   # rows per call; each row is its own request
    }

    def load(self):
        import torch

        self.torch = torch
        adapter = snapshot("models", REPO, self.opt["revision"],
                           ["adapter_config.json", "adapter_model.safetensors", "schema_config.json"], f"hf download {REPO}")
        contract = json.loads((adapter / "schema_config.json").read_text())
        base_weights(contract["revision"])
        kit_stand_in()
        sys.path.insert(0, str(HERE))
        from nimble.evaluation import decision_index_engine as de

        self.de = de
        self.engine = de.NimbleEngine(adapter=str(adapter), max_tokens=self.opt["max_tokens"],
                                      batch_tokens=self.opt["batch_tokens"], shared_prefix=self.opt["shared_prefix"],
                                      cache_tokens=self.opt["cache_tokens"])

    def describe(self):
        import peft
        import transformers

        return {"repo": REPO, "revision": self.opt["revision"], "engine": ENGINE,
                **self.engine.provenance, "temperature": 1.0, "transformers": transformers.__version__,
                "peft": peft.__version__}

    def run(self, state, questions):
        """NimbleEngine.__call__ up to the scores, with its own functions: -> (prepared prompts, per-question results)."""
        de, e = self.de, self.engine
        if isinstance(state, str) and not state.strip():
            state = json.dumps(state)
        body = {"state": state, "questions": {k: de.native_question(q) for k, q in questions.items()}}
        context, schema = de.native_schema(body)
        try:
            prompt = de.extended_schema.prepare_prompts(e.tokenizer, context, schema, e.max_tokens)
        except ValueError as exc:
            if "limit is" in str(exc):
                raise de.Unsupported(f"maximum context length: {exc}") from None
            raise
        rows = [dict(input_ids=prompt.full_ids[i], candidate_ids=prompt.candidate_ids[i], choices=prompt.choices[i])
                for i in range(len(prompt.names))]
        if e.shared_prefix and len(rows) > 1 and len(prompt.prefix_ids) >= 256:
            return prompt, e.score_shared(prompt, rows)
        return prompt, e.score(rows)

    def score(self, states, questions):
        results = []
        for state in states:
            try:
                prompt, scored = self.run(state, questions)
            except Exception as exc:            # Unsupported (too long, out of memory) or another failure
                results.append(f"{type(exc).__name__}: {str(exc)[:200]}")
                self.torch.cuda.empty_cache()
                continue
            answers = {}
            for name, out in zip(prompt.names, scored):
                probs = out["probabilities"]           # keyed "false"/"true" for yes/no, option names otherwise
                keys = option_keys(questions[name])
                answers[name] = make_answer(questions[name]["type"], keys, probs={k: probs[k] for k in keys}, temperature=1.0,
                                            logits=[math.log(probs[k]) if probs[k] > 0 else float("-inf") for k in keys])
            results.append((answers, {"prompt_tokens": {n: len(ids) for n, ids in zip(prompt.names, prompt.full_ids)}}))
        return results
