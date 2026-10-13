#!/usr/bin/env python3
"""Rebuild Jev's development pools (tune + selection) from the local dev split and verify them.

bench.prepare() wrote data/tune.jsonl and data/selection.jsonl from data/dev.parquet. This
replicates its logic from the Hugging Face cache's llm-aggre_fact-dev.arrow and refuses the
result unless both files hash to the values frozen in results/frozen.json, so other models are
calibrated on exactly the 4,520 rows Jev's threshold was fitted on (combined_dev_fit).

    python dev_split.py      # rebuild if missing, then verify
"""
import hashlib
from pathlib import Path

from bench import ROOT, digest, read_json, read_rows, write_rows

PHASES = ("tune", "selection")
SEED = "jev-check-20260928"
PER_STRATUM = 110


def find_dev_arrow(root=ROOT):
    found = sorted((root / "lytang___llm-aggre_fact").rglob("llm-aggre_fact-dev.arrow"))
    if len(found) != 1:
        raise SystemExit(f"Expected exactly one llm-aggre_fact-dev.arrow, found {len(found)}")
    return found[0]


def read_dev(path):
    path = Path(path)
    if path.suffix == ".arrow":
        import pyarrow as pa
        with path.open("rb") as handle:
            return pa.ipc.open_stream(handle).read_all().to_pylist()
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()


def split_pools(rows):
    """Same rule as bench.prepare(): document-hash 60/40 pools, up to 110 rows per source/label."""
    pools = {phase: {} for phase in PHASES}
    for i, row in enumerate(rows):
        row["id"] = f"dev:{i}"
        group = digest(" ".join(row["doc"].split()))
        phase = "tune" if int(group[:8], 16) % 10 < 6 else "selection"
        row["group"] = group
        row.pop("contamination_identifier", None)
        pools[phase].setdefault((row["dataset"], row["label"]), []).append(row)
    selected = {}
    for phase, buckets in pools.items():
        rows_out = []
        for _, bucket in sorted(buckets.items()):
            bucket.sort(key=lambda row: digest([SEED, row["id"]]))
            rows_out.extend(bucket[:PER_STRATUM])
        rows_out.sort(key=lambda row: row["id"])
        selected[phase] = rows_out
    return selected


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(root=ROOT):
    expected = read_json(root / "results/frozen.json")["data_hashes"]
    for phase in PHASES:
        path = root / f"data/{phase}.jsonl"
        if file_sha256(path) != expected[f"{phase}.jsonl"]:
            raise SystemExit(f"{path} does not match the frozen hash; not Jev's development rows")


def rebuild(root=ROOT, source=None):
    paths = [root / f"data/{phase}.jsonl" for phase in PHASES]
    if not all(p.exists() for p in paths):
        source = source or (root / "data/dev.parquet" if (root / "data/dev.parquet").exists()
                            else find_dev_arrow(root))
        for phase, rows in split_pools(read_dev(source)).items():
            write_rows(root / f"data/{phase}.jsonl", rows)
    verify(root)


def load_dev_rows(root=ROOT):
    """-> (tune + selection rows sorted by id, sha256 of the two verified files)."""
    rebuild(root)
    rows, hashes = [], []
    for phase in PHASES:
        path = root / f"data/{phase}.jsonl"
        rows.extend(read_rows(path))
        hashes.append(file_sha256(path))
    rows.sort(key=lambda row: int(row["id"].split(":")[1]))
    return rows, hashlib.sha256("".join(hashes).encode()).hexdigest()


if __name__ == "__main__":
    rows, data_hash = load_dev_rows()
    print(f"Verified {len(rows)} development rows (tune + selection) against results/frozen.json; "
          f"sha256={data_hash}")
