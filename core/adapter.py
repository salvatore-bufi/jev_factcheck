"""The interface every model implements (models/<name>/adapter.py defines `class Adapter(BaseAdapter)`).

score(states, questions) gets a batch of states ({"document", "claim"}) and the question set, and
returns one item per state: either (answers, info) or an error string (the row then fails; it never
gets an invented score). `answers` maps each question id to an answer built with `make_answer`;
`info` holds at least `prompt_tokens` ({question id: tokens} or the request total).
"""
import importlib.util
import json
import math
from pathlib import Path
import sys

import core


def make_answer(qtype, keys, probs=None, logits=None, temperature=None):
    """Store one answer: option keys in order, their probabilities, and the raw logits when available.

    noul answers always use the keys "true" and "false" (in whatever order the model reads them).
    """
    keys = [str(k) for k in keys]
    if probs is None:
        if logits is None:
            raise ValueError("an answer needs probs or logits")
        t = temperature or 1.0
        finite = [x for x in logits if math.isfinite(x)]
        if not finite:
            raise ValueError("no finite option logit")
        peak = max(finite)
        weights = [math.exp((x - peak) / t) if math.isfinite(x) else 0.0 for x in logits]
        total = sum(weights)
        probs = [w / total for w in weights]
    if isinstance(probs, dict):
        probs = [probs[k] for k in keys]
    answer = {"type": qtype, "keys": keys, "probs": {k: float(p) for k, p in zip(keys, probs)},
              "logits": [float(x) for x in logits] if logits is not None else None,
              "temperature": temperature}
    if qtype == "noul" and set(keys) != {"true", "false"}:
        raise ValueError("noul answers must use the keys 'true' and 'false'")
    return answer


def option_keys(question):
    """The answer keys of a question, in the order the question defines them."""
    kind = question["type"]
    if kind == "choice":
        return [str(k) for k in question["criteria"]]
    if kind == "score":
        return [str(i) for i in range(len(question["criteria"]))]
    return ["false", "true"]


class BaseAdapter:
    NAME = ""             # display name, e.g. "JPT-9B"
    SIZE = ""             # parameter count shown in tables
    KIND = "decision"     # "decision", or "baseline" for a model that ignores the questions
    DEFAULTS = {}         # tunable options, overridable with --opt key=value
    DEFAULT_DECISION = None   # a baseline's own decision rule (used instead of the question set's)

    def __init__(self, options=None):
        unknown = set(options or {}) - set(self.DEFAULTS)
        if unknown:
            raise SystemExit(f"{self.NAME}: unknown options {sorted(unknown)}; known: {sorted(self.DEFAULTS)}")
        self.opt = {**self.DEFAULTS, **(options or {})}

    def load(self):
        raise NotImplementedError

    def score(self, states, questions):
        raise NotImplementedError

    def reset_cache(self):
        """Drop any prefix/state cache, so a latency measurement starts cold for each document."""

    def close(self):
        pass

    def describe(self):
        """Model identity for run.json (repository ids, revisions, files)."""
        return {}

    @property
    def batch(self):
        return int(self.opt.get("batch", 32))


def coerce(value, default):
    if isinstance(default, bool):
        return str(value).lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    if default is None:
        for cast in (int, float):
            try:
                return cast(value)
            except ValueError:
                pass
    return value


def parse_options(pairs, defaults):
    options = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--opt expects key=value, got {pair!r}")
        key, value = pair.split("=", 1)
        options[key] = coerce(value, defaults.get(key))
    return options


def load_model_config(name, path=None):
    """Options from config/models/<name>.json (or `path`); keys starting with "_" are comments."""
    path = Path(path) if path else core.CONFIG / "models" / f"{name}.json"
    if not path.exists():
        if path != core.CONFIG / "models" / f"{name}.json":
            raise SystemExit(f"model config {path} not found")
        return {}, None
    cfg = {k: v for k, v in json.loads(path.read_text()).items() if not k.startswith("_")}
    return cfg, str(path)


def effective_options(cls, config=None, overrides=None):
    """DEFAULTS of the adapter < the model config file < command-line --opt / --batch."""
    options = {**(config or {}), **(overrides or {})}
    unknown = set(options) - set(cls.DEFAULTS)
    if unknown:
        raise SystemExit(f"{cls.NAME}: unknown options {sorted(unknown)}; known: {sorted(cls.DEFAULTS)}")
    return {**cls.DEFAULTS, **options}


def available():
    return sorted(p.parent.name for p in core.MODELS.glob("*/adapter.py"))


def load_adapter_class(name):
    path = core.MODELS / name / "adapter.py"
    if not path.exists():
        raise SystemExit(f"Unknown model {name!r}; available: {', '.join(available())}")
    module_name = f"models_{name.replace('-', '_')}_adapter"
    if module_name in sys.modules:
        return sys.modules[module_name].Adapter
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.Adapter
