"""Dataset resolution and loading.

`--dataset NAME --split SPLIT` is looked up in this order:

1. ./dataset/NAME/ : the file named after the split (SPLIT.arrow, .parquet, .jsonl or .json),
   or the exact file when SPLIT already has an extension (e.g. --split test.json);
2. the Hugging Face hub cache (honours HF_HUB_CACHE / HF_HOME): datasets--ORG--NAME, whose
   snapshot holds data/SPLIT-*.parquet. NAME is the folder name, with "--" between organisation
   and dataset (lytang--LLM-AggreFact is the Hub dataset lytang/LLM-AggreFact).

A row needs `doc` (or `document`), `claim` and a binary `label` (1 = supported). `dataset` (or
`source`) names the source group used for the per-source metrics; it defaults to "all". An `id`
column is kept, otherwise rows are numbered `SPLIT:index`.
"""
import glob
import hashlib
import json
import os
from pathlib import Path

from . import DATASETS as _DEFAULT_ROOT

DATASETS = _DEFAULT_ROOT   # replaceable (tests)

EXTENSIONS = (".arrow", ".parquet", ".jsonl", ".json")


def hub_cache():
    """The Hugging Face hub cache directory (honours HF_HUB_CACHE and HF_HOME)."""
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser()
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")).expanduser() / "hub"


def split_name(split):
    """test.json -> test (the name used in result paths)."""
    base = Path(split).name
    for ext in EXTENSIONS:
        if base.endswith(ext):
            return base[: -len(ext)]
    return base


def resolve(dataset, split, root=None):
    """-> list of data files for this dataset split (one local file, or the hub cache's parquet shards)."""
    tried = []
    local = Path(root or DATASETS) / dataset
    if Path(split).suffix in EXTENSIONS:
        candidates = [local / split]
    else:
        candidates = [local / f"{split}{ext}" for ext in EXTENSIONS]
    for path in candidates:
        tried.append(str(path))
        if path.is_file():
            return [path]
    hub_name = dataset.replace("/", "--")
    pattern = str(hub_cache() / f"datasets--{hub_name}" / "snapshots" / "*" / "data" / f"{split_name(split)}-*.parquet")
    tried.append(pattern)
    shards = sorted(glob.glob(pattern))
    if shards:
        snapshots = sorted({Path(s).parents[1] for s in shards})
        if len(snapshots) > 1:   # several revisions in the cache: refuse instead of mixing them
            raise SystemExit(f"Several cached revisions of {dataset}: {[s.name for s in snapshots]}; "
                             "copy the one you want into ./dataset/" + dataset)
        return [Path(s) for s in shards]
    raise SystemExit(f"Dataset {dataset!r} split {split!r} not found. Looked for:\n  " + "\n  ".join(tried))


def read_table(path):
    path = Path(path)
    if path.suffix == ".arrow":
        import pyarrow as pa
        with path.open("rb") as handle:
            try:
                return pa.ipc.open_stream(handle).read_all().to_pylist()
            except pa.ArrowInvalid:
                handle.seek(0)
                return pa.ipc.open_file(handle).read_all().to_pylist()
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        return data["rows"] if isinstance(data, dict) else data
    raise SystemExit(f"Unsupported data file {path}")


def normalize(raw, split):
    rows = []
    for i, row in enumerate(raw):
        doc = row.get("doc", row.get("document"))
        if doc is None or "claim" not in row or "label" not in row:
            raise SystemExit(f"Row {i} lacks doc/document, claim or label")
        label = int(row["label"])
        if label not in (0, 1):
            raise SystemExit(f"Row {i}: label must be 0 or 1, got {row['label']!r}")
        rows.append({"id": str(row.get("id", f"{split_name(split)}:{i}")), "doc": doc, "claim": row["claim"],
                     "label": label, "dataset": row.get("dataset", row.get("source", "all"))})
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit("Row ids must be unique")
    return rows


def subset(rows, limit):
    """A deterministic pseudo-random subset (hash order of the ids), kept in original order."""
    if not limit:
        return rows
    order = sorted(range(len(rows)), key=lambda i: hashlib.sha256(rows[i]["id"].encode()).hexdigest())
    return [rows[i] for i in sorted(order[:limit])]


def fingerprint(files):
    digest = hashlib.sha256()
    for path in files:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def load(dataset, split, limit=0, root=None):
    """-> (rows, info). info records where the data came from and its fingerprint."""
    root = Path(root or DATASETS)
    files = resolve(dataset, split, root)
    raw = [row for path in files for row in read_table(path)]
    rows = normalize(raw, split)
    revision = None
    rev_file = Path(root) / dataset / "REVISION"
    if rev_file.is_file() and str(files[0]).startswith(str(Path(root) / dataset)):
        revision = rev_file.read_text().strip()
    elif "snapshots" in str(files[0]):
        revision = files[0].parents[1].name
    info = {"dataset": dataset, "split": split_name(split), "files": [str(f) for f in files],
            "revision": revision, "sha256": fingerprint(files), "rows_total": len(rows), "limit": limit}
    return subset(rows, limit), info
