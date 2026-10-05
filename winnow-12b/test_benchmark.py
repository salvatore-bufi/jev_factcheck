"""Offline checks of winnow-12b/benchmark.py: synthetic data and a fake server; no GPU or network."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark as bm  # noqa: E402

# Decision rule: min(support, 1 - unsupported, 1 - contradicted).
assert abs(bm.decide({"support_simple": 0.9, "unsupported_detail": 0.1, "contradicted_detail": 0.2}) - 0.8) < 1e-12
assert bm.decide({"support_simple": 0.9, "unsupported_detail": 0.9, "contradicted_detail": 0.0}) < 0.30

# Client: parses a good response, reports HTTP errors as strings (never invents a score).
class Resp:
    def __init__(self, code, body): self.status_code, self._b, self.text = code, body, json.dumps(body)
    def json(self): return self._b

class FakeSession:
    def __init__(self, resp): self.resp = resp
    def post(self, *a, **k): return self.resp

client = bm.Client("http://x", 1)
ok = {"answers": {q: {"type": "noul", "noul": 0.25} for q in client.pack}, "usage": {"input_tokens": 7}}
client.local.session = FakeSession(Resp(200, ok))
assert client.ask({"document": "d", "claim": "c"}) == ({q: 0.25 for q in client.pack}, 7)
client.local.session = FakeSession(Resp(422, {"error": "too long"}))
assert isinstance(client.ask({"document": "d", "claim": "c"}), str)

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
print("Winnow benchmark checks passed: synthetic data and fake responses only.")
