"""Deterministic identity for fact-checking runs.

Like MiniCheck (and unlike AICCTU), FactCG has no persisted index to reuse
across runs -- it compares claims against documents directly -- so there is
only one identity here: run_identity, covering every hyperparameter that can
affect the final verdicts (model + chunking + generation). Same config ->
same folder, so re-running an identical configuration lands back on the
same outputs/ directory.

The identity is a short, readable slug (codebase/model names) plus a short
hash of the full parameter dict, so folder names stay short while the full
configuration is always recoverable from the params dict itself (also
persisted as a JSON manifest via write_manifest).
"""
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, Tuple


def slugify(name: str) -> str:
    """Turn a model identifier into a short, filesystem-safe slug.

    'yaxili96/FactCG-DeBERTa-v3-Large' -> 'FactCG-DeBERTa-v3-Large'
    """
    name = name.rsplit("/", 1)[-1]
    name = re.sub(r"[^A-Za-z0-9.\-]+", "-", name)
    return name.strip("-")


def short_hash(params: Dict, length: int = 8) -> str:
    """Deterministic short hash of a params dict (order-independent)."""
    blob = json.dumps(params, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:length]


def run_identity(codebase: str, dataset: str, params: Dict) -> Tuple[str, Dict]:
    """Identity for a fact-checking run: every hyperparameter affecting results.

    `params` must include at least "llm_model".
    """
    full_params = {
        "dataset": dataset,
        "codebase": codebase,
        **params,
    }
    folder = f"{codebase}__{slugify(params['llm_model'])}--{short_hash(full_params)}"
    return folder, full_params


def write_manifest(directory: Path, params: Dict, filename: str) -> None:
    """Persist a params dict as a readable JSON manifest inside directory."""
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / filename, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False, sort_keys=True)
