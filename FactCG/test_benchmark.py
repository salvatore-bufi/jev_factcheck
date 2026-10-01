"""Offline check of benchmark.py: synthetic parquet, fake model, no network or GPU."""
from pathlib import Path
from tempfile import TemporaryDirectory
import sys

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark as bm  # noqa: E402


class FakeModel:
    device = "cpu"

    def score_pairs(self, docs, claims):
        p = [0.9 if "yes" in d else 0.1 for d in docs]
        return p, [[1 - x, x] for x in p]


def chunker(text, size):
    return text.split("|")


with TemporaryDirectory() as tmp:
    rows = [{"dataset": ds, "doc": doc, "claim": "c", "label": lab, "contamination_identifier": "x"}
            for ds in ["Wice", "Reveal"]
            for doc, lab in [("no|yes", 1), ("no|no", 0), ("yes", 1), ("no", 0)]]
    path = Path(tmp) / "t.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    loaded = bm.load_rows(path, 0)
    assert len(loaded) == 8 and "contamination_identifier" not in loaded[0]
    args = type("A", (), {"window": 3, "batch_size": 2, "chunk_size": 550})()
    identity = {"x": 1}
    cache = Path(tmp) / "scores.jsonl"
    done = {}
    bm.score_rows(loaded, FakeModel(), chunker, args, cache, identity, done)
    assert len(done) == 8 and done["test:0"]["score"] == 0.9 and done["test:1"]["score"] == 0.1
    assert bm.read_cache(cache, identity).keys() == done.keys()      # resume reads everything
    try:
        bm.read_cache(cache, {"x": 2})
        raise AssertionError("identity mismatch not detected")
    except SystemExit:
        pass
    report, row = bm.build_report(loaded, done, args, identity, True)
    assert report["macro_balanced_accuracy"] == 1.0 and "100.00" in row and report["complete"]
print("FactCG benchmark checks passed: synthetic data and fake model only.")
