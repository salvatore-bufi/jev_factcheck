"""STEP 1b — Build a training mix: which training rows the evidence loss sees, and with which weights.

Where it sits in the workflow:
    evidence.data.make_splits  ->  [evidence.data.make_mix]  ->  evidence.extract  ->  evidence.train_head

Run:
    python -m evidence.data.make_mix --recipe evidence/recipes/ev_head.json
Reads:   dataset/evidence/splits/train.jsonl          (and the dev/test sets, only to check that no document leaks)
         the recipe (json): families of rows -> loss term, row weight, omega
Writes:  dataset/evidence/mixes/<recipe name>.jsonl   the training rows, each with three extra fields:
             evidence_term   supervised | mil_neg | mil_pos  (which loss the row gets, doc 08 §6)
             row_weight      how much the row counts in the batch average
             omega           weight of the NON-gold sentences in the loss (low where annotations are incomplete)
         dataset/evidence/mixes/<recipe name>.manifest.json   counts per family, constraint checks, hashes

Why separate from the split: the split is frozen forever (dev/test numbers must stay comparable); the mix changes
from experiment to experiment (omega, weight of db_split rows, negatives or not...). One experiment = one recipe.
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evidence import config  # noqa: E402
from evidence.data import io  # noqa: E402

EVAL_SETS = ("dev_evid_select", "dev_evid_confirm", "dev_evid_neg", "test_evid_id", "test_evid_ood")


def matches(row, match):
    """Does the row fit a family's "match" rule?

    Input:  row (dict), match (dict key -> required value). A key is looked up in the row, then in row["meta"].
    Output: True if every key has the required value.
    """
    for key, value in match.items():
        actual = row[key] if key in row else row["meta"].get(key)
        if actual != value:
            return False
    return True


def apply_recipe(rows, recipe):
    """Give every row the first family that matches it.

    Input:  rows (training rows), recipe (dict with "families").
    Output: (kept rows with evidence_term / row_weight / omega / family set, Counter of rows per family,
             Counter of rows that matched no family by (source, label, evidence_term)).
    Rows of a family with "drop": true are not kept. Rows that match no family are not kept either (and are
    reported, so a forgotten family is visible).
    """
    kept, per_family, unmatched = [], Counter(), Counter()
    for r in rows:
        family = next((f for f in recipe["families"] if matches(r, f["match"])), None)
        if family is None:
            unmatched[(r["source"], r["label"], r["evidence_term"])] += 1
            continue
        per_family[family["name"]] += 1
        if family.get("drop"):
            continue
        out = dict(r)
        out["evidence_term"] = family["term"]
        out["row_weight"] = float(family["row_weight"])
        out["omega"] = float(family["omega"])
        out["family"] = family["name"]
        kept.append(out)
    return kept, per_family, unmatched


def subsample_negatives(rows, neg_fraction, seed):
    """Keep only enough negatives (mil_neg rows) so that they are `neg_fraction` of the mix.

    Input:  rows (mix rows), neg_fraction (float in (0, 1) or None = keep all), seed.
    Output: the rows (same order), with some negatives removed at random (reproducible with the seed).
    """
    if neg_fraction is None:
        return rows
    neg = [i for i, r in enumerate(rows) if r["evidence_term"] == "mil_neg"]
    n_pos = len(rows) - len(neg)
    want = int(round(neg_fraction * n_pos / (1 - neg_fraction)))     # neg / (pos + neg) = neg_fraction
    if want >= len(neg):
        return rows
    keep = set(random.Random(seed).sample(neg, want))
    return [r for i, r in enumerate(rows) if r["evidence_term"] != "mil_neg" or i in keep]


def constraints(rows, eval_doc_ids):
    """The checks of doc 04 §3, measured on the mix.

    Input:  rows (mix rows), eval_doc_ids (set of doc_id of every dev/test set).
    Output: list of dicts {name, value, target, ok}. "Evidence beyond 1,024 tokens" is not here: it needs token
            positions, which exist only after rendering (it is reported by evidence.extract).
    """
    n = len(rows)
    pos = [r for r in rows if r["evidence_term"] == "supervised"]
    neg = [r for r in rows if r["evidence_term"] == "mil_neg"]
    share = lambda k, total: k / total if total else 0.0
    neg_share = share(len(neg), n)
    multi = share(sum(len(r["evidence"]) >= 2 for r in pos), len(pos))
    far = share(sum(r["evidence"][-1] - r["evidence"][0] >= 10 for r in pos), len(pos))
    shared = sum(r["doc_id"] in eval_doc_ids for r in rows)
    wice_leak = sum(r["source"] == "wice" and r["meta"].get("aggrefact_overlap") is not False for r in rows)
    return [
        {"name": "negatives share", "value": round(neg_share, 3), "target": "0.20-0.45", "ok": 0.20 <= neg_share <= 0.45},
        {"name": "positives with >= 2 gold", "value": round(multi, 3), "target": ">= 0.40", "ok": multi >= 0.40},
        {"name": "positives with gold >= 10 sentences apart", "value": round(far, 3), "target": ">= 0.25", "ok": far >= 0.25},
        {"name": "documents shared with dev/test", "value": shared, "target": "0", "ok": shared == 0},
        {"name": "WiCE rows overlapping AggreFact", "value": wice_leak, "target": "0", "ok": wice_leak == 0},
    ]


def main(argv=None):
    """Command line entry point: read train + recipe, build the mix, check it, write it."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--recipe", type=Path, default=config.RECIPES / "ev_head.json")
    p.add_argument("--splits", type=Path, default=config.SPLITS)
    p.add_argument("--out", type=Path, default=config.MIXES)
    opts = p.parse_args(argv)

    recipe = json.loads(opts.recipe.read_text())
    train = io.read_jsonl(opts.splits / "train.jsonl")
    eval_doc_ids = {r["doc_id"] for name in EVAL_SETS for r in io.read_jsonl(opts.splits / f"{name}.jsonl")}

    rows, per_family, unmatched = apply_recipe(train, recipe)
    rows = subsample_negatives(rows, recipe.get("neg_fraction"), recipe.get("seed", config.SEED))
    checks = constraints(rows, eval_doc_ids)

    path = opts.out / f"{recipe['name']}.jsonl"
    io.write_jsonl(path, rows)
    io.write_json(opts.out / f"{recipe['name']}.manifest.json", {
        "recipe": recipe, "recipe_sha256": io.sha256(opts.recipe),
        "train_sha256": io.sha256(opts.splits / "train.jsonl"),
        "rows": len(rows), "per_family": dict(per_family),
        "unmatched": {str(k): v for k, v in unmatched.items()},
        "by_term": dict(Counter(r["evidence_term"] for r in rows)),
        "constraints": checks, "sha256": io.sha256(path), "git_commit": io.git_commit(),
    })

    print(f"{recipe['name']}: {len(rows)} rows -> {path}")
    print("  per family:", dict(per_family))
    if unmatched:
        print("  WARNING rows matching no family (left out):", dict(unmatched))
    for c in checks:
        print(f"  [{'ok' if c['ok'] else 'NO'}] {c['name']}: {c['value']} (target {c['target']})")
    hard = [c for c in checks if not c["ok"] and c["target"] == "0"]
    if hard:                                       # leaks are errors; the share targets are only warnings
        raise SystemExit("leak between the mix and the dev/test sets")


if __name__ == "__main__":
    main()
