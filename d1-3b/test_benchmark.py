"""Offline checks of d1-3b/benchmark.py: scoring rule, report, cache identity, cache check. No GPU, model or network."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark as bm  # noqa: E402

# Frozen rule: min(support, 1 - unsupported detail, 1 - contradicted detail).
assert abs(bm.decide({"support_simple": 0.9, "unsupported_detail": 0.1, "contradicted_detail": 0.2}) - 0.8) < 1e-12
assert bm.decide({"support_simple": 0.9, "unsupported_detail": 0.9, "contradicted_detail": 0.0}) < 0.30
assert json.loads(bm.PACK.read_text())["decision"]["threshold"] == bm.THRESHOLD

args = type("A", (), {"threshold": 0.30, "max_doc_chars": 80000})()
rows = [{"id": f"test:{i}", "dataset": ds, "label": lab}
        for i, (ds, lab) in enumerate([("Wice", 1), ("Wice", 0), ("Reveal", 1), ("Reveal", 0)])]
good = {"support_simple": 0.9, "unsupported_detail": 0.1, "contradicted_detail": 0.1}
bad = {"support_simple": 0.1, "unsupported_detail": 0.9, "contradicted_detail": 0.1}
done = {r["id"]: {"id": r["id"], "answers": good if r["label"] else bad, "prompt_tokens": 10} for r in rows}
report, row = bm.build_report(rows, done, args, {"x": 1}, True)
assert report["macro_balanced_accuracy"] == 1.0 and "100.00" in row and report["complete"]
assert row.startswith("| d1-3B (ours; fixed Jev rule) | 3B |")

with TemporaryDirectory() as tmp:   # cache identity protects against mixing configurations
    path = Path(tmp) / "answers.jsonl"
    path.write_text(json.dumps({"identity": {"x": 1}}) + "\n" + json.dumps(done["test:0"]) + "\n")
    assert list(bm.read_cache(path, {"x": 1})) == ["test:0"]
    try:
        bm.read_cache(path, {"x": 2})
        raise AssertionError("identity mismatch not detected")
    except SystemExit:
        pass

with TemporaryDirectory() as tmp:   # the run is offline: a missing or partial model gives a clear message
    os.environ["HF_HUB_CACHE"] = tmp
    try:
        bm.check_model()
        raise AssertionError("missing model not detected")
    except SystemExit as exc:
        assert "hf download LiquidAI/d1-3B" in str(exc)
print("d1-3B benchmark checks passed: no GPU, model or network used.")
