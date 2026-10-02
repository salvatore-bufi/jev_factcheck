"""Offline checks of rune-26b/benchmark.py: synthetic data only; no GPU, model or network."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import math
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark as bm  # noqa: E402

# Softmax over [A=false, B=true] letter logits: equal logits -> 0.5; temperature flattens the answer.
assert bm.p_yes([1.0, 1.0], 1.0) == 0.5
assert math.isclose(bm.p_yes([0.0, 2.0], 1.0), 1 / (1 + math.exp(-2)))
assert bm.p_yes([0.0, 2.0], 2.0) < bm.p_yes([0.0, 2.0], 1.0)
assert bm.p_yes([-1e9, 0.0], 2.0) == 1.0 and bm.p_yes([float("-inf"), 0.0], 1.0) == 1.0

# Decision rule: min(support, 1 - unsupported, 1 - contradicted).
good = {"support_simple": [-9.0, 0.0], "unsupported_detail": [0.0, -9.0], "contradicted_detail": [0.0, -9.0]}
bad = {"support_simple": [0.0, -9.0], "unsupported_detail": [-9.0, 0.0], "contradicted_detail": [0.0, -9.0]}
assert bm.decide(good, 1.0) > 0.99 and bm.decide(bad, 1.0) < 0.01

args = type("A", (), {"threshold": 0.30, "temperature": 2.0, "max_doc_chars": 80000})()
rows = [{"id": f"test:{i}", "dataset": ds, "label": lab}
        for i, (ds, lab) in enumerate([("Wice", 1), ("Wice", 0), ("Reveal", 1), ("Reveal", 0)])]
done = {r["id"]: {"id": r["id"], "logits": good if r["label"] else bad, "prompt_tokens": 10} for r in rows}
report, row = bm.build_report(rows, done, args, {"x": 1}, True)
assert report["macro_balanced_accuracy"] == 1.0 and "100.00" in row and report["complete"]

with TemporaryDirectory() as tmp:   # cache identity protects against mixing configurations
    path = Path(tmp) / "answers.jsonl"
    path.write_text(json.dumps({"identity": {"x": 1}}) + "\n" + json.dumps(done["test:0"]) + "\n")
    assert list(bm.read_cache(path, {"x": 1})) == ["test:0"]
    try:
        bm.read_cache(path, {"x": 2})
        raise AssertionError("identity mismatch not detected")
    except SystemExit:
        pass
print("Rune benchmark checks passed: synthetic data only.")
