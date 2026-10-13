"""Question sets: loading, validation, a short hash, and the per-split hash registry.

A question set (config/questions/*.json) holds `name`, `questions` (Decision Index schema: `noul`,
`choice` or `score`) and an optional `decision` rule. The hash covers the questions only: the same
questions always give the same hash, while the decision rule is applied at report time and can
change without new inference.
"""
import json
from pathlib import Path

from . import registry
from .registry import short_hash

TYPES = ("noul", "choice", "score")


def validate(questions):
    if not isinstance(questions, dict) or not questions:
        raise SystemExit("questions must be a non-empty object of named questions")
    for qid, q in questions.items():
        kind = q.get("type")
        if kind not in TYPES:
            raise SystemExit(f"question {qid!r}: type must be one of {TYPES}")
        if q.get("instructions") is None:
            raise SystemExit(f"question {qid!r}: instructions are required")
        crit = q.get("criteria")
        if kind == "noul" and crit is not None and (not isinstance(crit, dict) or set(crit) - {"true", "false"}):
            raise SystemExit(f"question {qid!r}: noul criteria may only define 'true' and 'false'")
        if kind == "choice" and (not isinstance(crit, dict) or len(crit) < 2):
            raise SystemExit(f"question {qid!r}: choice needs at least two named criteria")
        if kind == "score" and (not isinstance(crit, list) or not 2 <= len(crit) <= 10):
            raise SystemExit(f"question {qid!r}: score needs an ordered list of 2 to 10 levels")


def load(path):
    path = Path(path)
    cfg = json.loads(path.read_text())
    validate(cfg.get("questions"))
    cfg.setdefault("name", path.stem)
    return cfg


def questions_hash(questions):
    """8 hex characters of sha256 over the canonical JSON of the questions only."""
    return short_hash(questions)


def register(split_dir, cfg):
    """Record the question set in <split_dir>/hash_questions.json; append only when the hash is new."""
    digest = questions_hash(cfg["questions"])
    return registry.register(Path(split_dir) / "hash_questions.json", digest,
                             {"name": cfg["name"], "questions": cfg["questions"]}, "questions")
