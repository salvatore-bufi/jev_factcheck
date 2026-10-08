"""Offline checks of clm-8b/benchmark.py: text layout, scoring rule, cache identity. No GPU, model or network."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import math
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark as bm  # noqa: E402
from clm.schema import candidates, state_text  # noqa: E402

pack = json.loads(bm.PACK.read_text())["questions"]

# What the encoder sees: state fields as prose, a blank line, then the question; options are "<key>: <description>".
state = {"document": "The cat sat.", "claim": "A cat sat."}
q1 = pack["support_simple"]
assert state_text(state, q1["instructions"]) == "document: The cat sat.\n\nclaim: A cat sat.\n\n" + q1["instructions"]
keys, texts = candidates(q1)
assert keys == ["false", "true"]
assert texts[0] == "false: No. This is false: " + q1["instructions"] and texts[1] == "true: Yes. This is true: " + q1["instructions"]
keys, texts = candidates(pack["unsupported_detail"])
assert texts == ["false: " + pack["unsupported_detail"]["criteria"]["false"], "true: " + pack["unsupported_detail"]["criteria"]["true"]]

# Softmax over [false, true] logits; the frozen rule.
assert bm.p_yes([1.0, 1.0], 1.0) == 0.5 and math.isclose(bm.p_yes([0.0, 2.0], 1.0), 1 / (1 + math.exp(-2)))
good = {"support_simple": [-9.0, 0.0], "unsupported_detail": [0.0, -9.0], "contradicted_detail": [0.0, -9.0]}
bad = {"support_simple": [0.0, -9.0], "unsupported_detail": [-9.0, 0.0], "contradicted_detail": [0.0, -9.0]}
assert bm.decide(good, 1.0) > 0.99 and bm.decide(bad, 1.0) < 0.01

args = type("A", (), {"threshold": 0.30, "temperature": 1.0, "max_doc_chars": 80000})()
rows = [{"id": f"test:{i}", "dataset": ds, "label": lab}
        for i, (ds, lab) in enumerate([("Wice", 1), ("Wice", 0), ("Reveal", 1), ("Reveal", 0)])]
done = {r["id"]: {"id": r["id"], "logits": good if r["label"] else bad, "prompt_tokens": 10} for r in rows}
report, row = bm.build_report(rows, done, args, {"x": 1}, True)
assert report["macro_balanced_accuracy"] == 1.0 and "100.00" in row and report["complete"]

with TemporaryDirectory() as tmp:
    path = Path(tmp) / "answers.jsonl"
    path.write_text(json.dumps({"identity": {"x": 1}}) + "\n" + json.dumps(done["test:0"]) + "\n")
    assert list(bm.read_cache(path, {"x": 1})) == ["test:0"]
    try:
        bm.read_cache(path, {"x": 2})
        raise AssertionError("identity mismatch not detected")
    except SystemExit:
        pass
print("CLM benchmark checks passed: no GPU, model or network used.")
