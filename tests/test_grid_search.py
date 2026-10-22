"""Offline checks of grid_search.py on synthetic run folders (no GPU, no model).

    python tests/test_grid_search.py          # or: python -m pytest tests
"""
import csv
import json
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import core  # noqa: E402
import grid_search as gs  # noqa: E402
from core.metrics import metrics  # noqa: E402

QUESTIONS = json.loads((ROOT / "config/questions/claim_support.json").read_text())


def answer(z_true, z_false, temperature, with_logits=True):
    """A noul answer stored as the runner stores it (keys false/true, probs at `temperature`)."""
    t = temperature or 1.0
    a, b = math.exp(z_false / t), math.exp(z_true / t)
    return {"type": "noul", "keys": ["false", "true"], "probs": {"false": a / (a + b), "true": b / (a + b)},
            "logits": [z_false, z_true] if with_logits else None, "temperature": temperature if with_logits else None}


def make_run(split_dir, name, temperature, with_logits=True, status="complete", limit=0):
    run_dir = split_dir / name
    run_dir.mkdir(parents=True)
    rows = []
    for i in range(40):
        label = i % 2
        source = "A" if i < 20 else "B"
        noise = (i % 7) / 7 - 0.5                                   # makes some rows hard
        support = (1.5 if label else -1.5) + 2 * noise
        rows.append({"id": f"test:{i}", "source": source, "label": label, "claim_chars": 10, "doc_chars": 100,
                     "prompt_tokens": {}, "batch": 0,
                     "answers": {"support_simple": answer(support, 0.0, temperature, with_logits),
                                 "unsupported_detail": answer(-support, 0.0, temperature, with_logits),
                                 "contradicted_detail": answer(-support - 1, 0.0, temperature, with_logits)}})
    identity = {"model": name.split("_")[0], "params_hash": "p" * 8, "questions_hash": "q" * 8, "limit": limit}
    (run_dir / "answers.jsonl").write_text(json.dumps({"identity": identity}) + "\n"
                                           + "".join(json.dumps(r) + "\n" for r in rows))
    (run_dir / "run.json").write_text(json.dumps({"identity": identity, "model_name": name, "model_kind": "decision",
                                                  "default_decision": None, "questions": QUESTIONS, "status": status}))
    return run_dir


def test_parse_values():
    assert gs.parse_values("0.1,0.2") == [0.1, 0.2]
    assert gs.parse_values("0.1:0.3:0.1") == [0.1, 0.2, 0.3]
    assert gs.parse_values("default,2", allow_default=True) == [None, 2.0]


def test_scores_temperatures_and_metrics():
    with TemporaryDirectory() as tmp:
        run = gs.Run(make_run(Path(tmp), "logits_x", temperature=2.0))
        rule = run.rules()["rule"]
        # T equal to the saved temperature reproduces the saved probabilities
        assert max(abs(a - b) for a, b in zip(run.score(rule, None), run.score(rule, 2.0))) < 1e-12
        assert any(abs(a - b) > 1e-6 for a, b in zip(run.score(rule, None), run.score(rule, 0.5)))
        # vectorised metrics equal core.metrics for every threshold
        scores = run.score(rule, 1.0)
        rows = [{"label": int(l), "dataset": s} for l, s in zip(run.labels, run.sources)]
        for row in gs.evaluate(run, scores, [0.2, 0.3, 0.6]):
            ref = metrics(rows, scores.tolist(), row["threshold"])
            assert abs(row["macro_balanced_accuracy"] - ref["macro_balanced_accuracy"]) < 1e-12
            assert abs(row["accuracy"] - ref["accuracy"]) < 1e-12 and abs(row["roc_auc"] - ref["roc_auc"]) < 1e-12
        assert set(run.rules()) == {"rule", "support_simple", "unsupported_detail", "contradicted_detail"}


def test_models_without_logits_only_have_default():
    with TemporaryDirectory() as tmp:
        run = gs.Run(make_run(Path(tmp), "nologits_x", temperature=None, with_logits=False))
        rows, skipped = gs.search(run, None, [None, 1.0], [0.3, 0.5])
        assert {r["temperature"] for r in rows} == {None} and len(skipped) == 4   # 4 rules x one T


def test_command_line_writes_tables():
    with TemporaryDirectory() as tmp:
        split_dir = Path(tmp) / "toy" / "test"
        make_run(split_dir, "alpha_x", temperature=1.0)
        make_run(split_dir, "beta_x", temperature=None, with_logits=False)
        make_run(split_dir, "gamma_x", temperature=1.0, status="incomplete")       # skipped
        make_run(split_dir, "delta_x", temperature=1.0, limit=8)                  # skipped
        saved, core.RESULTS, gs.RESULTS = core.RESULTS, Path(tmp), Path(tmp)
        argv = sys.argv
        try:
            sys.argv = ["grid_search.py", "--dataset", "toy", "--thresholds", "0.2:0.6:0.2", "--temperatures", "default,1"]
            gs.main()
        finally:
            sys.argv, core.RESULTS, gs.RESULTS = argv, saved, saved
        md = (split_dir / "grid_qqqqqqqq.md").read_text()
        assert "## Summary" in md and "alpha_x" in md and "beta_x" in md and "gamma_x" not in md and "delta_x" not in md
        rows = list(csv.DictReader(open(split_dir / "grid_qqqqqqqq.csv")))
        assert len(rows) == 4 * 2 * 3 + 4 * 1 * 3      # alpha: 4 rules x 2 T x 3 thresholds; beta: default T only


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("grid search checks passed")
