"""The evidence sources. Each module exposes:

    NAME
    load_raw(opts, prov) -> iterator of raw records (everything `convert` needs, already joined); fills `prov`
                            with the files and revisions it read (written to the manifest)
    convert(raw, opts)   -> a row (evidence.data.schema) or a Skip(reason)
"""
import glob
from pathlib import Path

from core.datasets import hub_cache


def hub_files(repo, pattern):
    """Files of a Hugging Face dataset in the local hub cache: -> (sorted paths, snapshot revision).

    pattern is relative to the snapshot folder (e.g. "default/train/*.parquet"). Several cached revisions holding
    the files are refused rather than mixed.
    """
    root = hub_cache() / f"datasets--{repo.replace('/', '--')}" / "snapshots"
    files = sorted(Path(p) for p in glob.glob(str(root / "*" / pattern)))
    if not files:
        raise SystemExit(f"{repo}: no file matching {pattern} in {root}. Download it first "
                         f"(hf download {repo} --repo-type dataset)")
    snapshots = sorted({f.relative_to(root).parts[0] for f in files})
    if len(snapshots) > 1:
        raise SystemExit(f"{repo}: several cached revisions hold {pattern}: {snapshots}")
    return files, snapshots[0]


def registry():
    from . import hover, scifact, wice
    return {m.NAME: m for m in (scifact, wice, hover)}
