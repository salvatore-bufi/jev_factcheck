"""Run with python test_dev_split.py. Synthetic data only, plus the frozen-hash check when local data exists."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json

from bench import ROOT
import dev_split


def synthetic(n=600):
    # 60 distinct documents, so several rows share a document and must stay in one pool.
    return [{"dataset": f"S{i % 3}", "doc": f"document {i % 60}\n text", "claim": f"claim {i}",
             "label": i % 2, "contamination_identifier": "x"} for i in range(n)]


def check_split():
    first, second = dev_split.split_pools(synthetic()), dev_split.split_pools(synthetic())
    assert first == second, "pool construction must be deterministic"
    tune, selection = first["tune"], first["selection"]
    assert not {r["group"] for r in tune} & {r["group"] for r in selection}, "a document crossed pools"
    for rows in (tune, selection):
        counts = {}
        for row in rows:
            counts[(row["dataset"], row["label"])] = counts.get((row["dataset"], row["label"]), 0) + 1
            assert "contamination_identifier" not in row and row["id"].startswith("dev:")
        assert counts and max(counts.values()) <= dev_split.PER_STRATUM
        assert [r["id"] for r in rows] == sorted(r["id"] for r in rows)


def check_verify_refuses_other_rows():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "results").mkdir()
        (root / "results/frozen.json").write_text(json.dumps(
            {"data_hashes": {"tune.jsonl": "0" * 64, "selection.jsonl": "0" * 64}}))
        (root / "data").mkdir()
        for phase in dev_split.PHASES:
            (root / f"data/{phase}.jsonl").write_text("{}\n")
        try:
            dev_split.verify(root)
            raise AssertionError("foreign development rows were accepted")
        except SystemExit:
            pass


def check_real_rows():
    if not list((ROOT / "lytang___llm-aggre_fact").rglob("llm-aggre_fact-dev.arrow")):
        print("skipped: no local dev split")
        return
    rows, data_hash = dev_split.load_dev_rows()
    assert len(rows) == 4520 and len({r["id"] for r in rows}) == 4520 and len(data_hash) == 64


if __name__ == "__main__":
    check_split()
    check_verify_refuses_other_rows()
    check_real_rows()
    print("dev_split checks passed")
