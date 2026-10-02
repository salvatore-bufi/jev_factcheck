"""Offline checks of jpt-9b/benchmark.py: synthetic data only; no GPU, model or network."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark as bm  # noqa: E402

# Decision rule: min(support, 1 - unsupported, 1 - contradicted), strict > threshold.
assert abs(bm.decide({"support_simple": 0.9, "unsupported_detail": 0.1, "contradicted_detail": 0.2}) - 0.8) < 1e-12
assert bm.decide({"support_simple": 0.9, "unsupported_detail": 0.9, "contradicted_detail": 0.0}) < 0.30

args = type("A", (), {"threshold": 0.30, "max_doc_chars": 80000})()
rows = [{"id": f"test:{i}", "dataset": ds, "label": lab}
        for i, (ds, lab) in enumerate([("Wice", 1), ("Wice", 0), ("Reveal", 1), ("Reveal", 0)])]
good = {"support_simple": 0.9, "unsupported_detail": 0.1, "contradicted_detail": 0.1}
bad = {"support_simple": 0.1, "unsupported_detail": 0.9, "contradicted_detail": 0.1}
done = {r["id"]: {"id": r["id"], "answers": good if r["label"] else bad, "prompt_tokens": 10} for r in rows}
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
print("JPT-9B benchmark checks passed: synthetic data only.")
