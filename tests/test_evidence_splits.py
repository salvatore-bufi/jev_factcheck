"""Offline checks of the split and the mix rules on hand-written rows (no data files, no GPU).

    python tests/test_evidence_splits.py          # or: python -m pytest tests

What is checked:
    - unit() is deterministic and roughly uniform;
    - HoVer: dev/test keep only "exact" rows; WiCE: rows overlapping LLM-AggreFact never train;
    - a document present in two stages stays only in the latest one;
    - dev positives are halved by document, negatives go to dev_evid_neg;
    - the mix: first matching family wins, "drop" families are removed, negatives subsampled to the asked share.
"""
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evidence.data.make_mix import apply_recipe, subsample_negatives  # noqa: E402
from evidence.data.make_splits import (assign_hover, assign_wice, make_sets, remove_shared_documents,  # noqa: E402
                                       unit)


def row(id, source, label, doc_id, orig_split="train", term=None, **meta):
    """A minimal row with the fields the split and the mix read."""
    return {"id": id, "source": source, "label": label, "doc_id": doc_id, "orig_split": orig_split,
            "evidence": [0, 3] if label else [], "evidence_sets": [[0, 3]] if label else [],
            "evidence_term": term or ("supervised" if label else "mil_neg"), "meta": meta}


def test_unit():
    assert unit(13, "x", "doc") == unit(13, "x", "doc") != unit(14, "x", "doc")
    share = sum(unit(13, "x", f"d{i}") < 0.1 for i in range(5000)) / 5000
    assert 0.08 < share < 0.12


def test_hover_and_wice_rules():
    dropped = Counter()
    rows = [row("t1", "hover", 1, "h1", "dev", gold_alignment="exact"),
            row("t2", "hover", 1, "h2", "dev", gold_alignment="db_split")]
    out = assign_hover(rows, 13, 0.1, dropped)
    assert [(s, r["id"]) for s, r in out] == [("test", "t1")] and dropped["test_db_split"] == 1
    dropped = Counter()
    rows = [row("w1", "wice", 1, "w1", "train", aggrefact_overlap=False),
            row("w2", "wice", 1, "w2", "train", aggrefact_overlap="test"),
            row("w3", "wice", 1, "w3", "train", aggrefact_overlap=None),
            row("w4", "wice", 0, "w4", "dev", aggrefact_overlap="dev")]
    out = assign_wice(rows, 13, None, dropped)
    assert [(s, r["id"]) for s, r in out] == [("train", "w1"), ("test", "w4")]
    assert dropped["train_aggrefact_overlap"] == 2


def test_shared_documents_and_sets():
    dropped = {"scifact": Counter()}
    assigned = [("train", row("a", "scifact", 1, "D")), ("test", row("b", "scifact", 0, "D")),
                ("dev", row("c", "scifact", 1, "E")), ("dev", row("d", "scifact", 0, "E"))]
    kept = remove_shared_documents(assigned, dropped)
    assert [r["id"] for _, r in kept] == ["b", "c", "d"]
    assert dropped["scifact"]["train_document_in_later_split"] == 1
    sets = make_sets(kept, 13)
    assert [r["id"] for r in sets["test_evid_id"]] == ["b"] and [r["id"] for r in sets["dev_evid_neg"]] == ["d"]
    assert [r["id"] for r in sets["dev_evid_select"] + sets["dev_evid_confirm"]] == ["c"]


def test_mix_recipe():
    recipe = {"families": [
        {"name": "hover_exact", "match": {"source": "hover", "label": 1, "gold_alignment": "exact"},
         "term": "supervised", "row_weight": 1.0, "omega": 0.5},
        {"name": "partial", "match": {"evidence_term": "none"}, "drop": True},
        {"name": "neg", "match": {"label": 0}, "term": "mil_neg", "row_weight": 1.0, "omega": 1.0}]}
    rows = [row("p", "hover", 1, "a", gold_alignment="exact"), row("x", "wice", 0, "b", term="none"),
            row("n", "hover", 0, "c", gold_alignment="exact"), row("u", "hover", 1, "d", gold_alignment="db_split")]
    kept, per_family, unmatched = apply_recipe(rows, recipe)
    assert [r["id"] for r in kept] == ["p", "n"] and kept[0]["omega"] == 0.5
    assert per_family == {"hover_exact": 1, "partial": 1, "neg": 1} and sum(unmatched.values()) == 1
    many = [dict(kept[0], id=f"p{i}") for i in range(70)] + [dict(kept[1], id=f"n{i}") for i in range(70)]
    sub = subsample_negatives(many, 0.3, 13)
    share = sum(r["evidence_term"] == "mil_neg" for r in sub) / len(sub)
    assert abs(share - 0.3) < 0.01


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
