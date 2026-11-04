"""Print built rows with their evidence marked, for reading by hand.

    python -m evidence.data.show --source hover --n 5                 # random positives
    python -m evidence.data.show --source wice --label 0 --n 3 --full # negatives, whole document
    python -m evidence.data.show --source scifact --id scifact:train:2:13734012

Evidence sentences are prefixed with ">>", contradicting ones with "xx". Without --full only those sentences and
their neighbours are printed.
"""
import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evidence.data import io  # noqa: E402


def show(row, full=False, width=160):
    ev, contra = set(row["evidence"]), set(row["contra"])
    print("=" * width)
    print(f"{row['id']}  label={row['label']}  term={row['evidence_term']}  sentences={len(row['sentences'])}  "
          f"evidence={row['evidence']}  sets={row['evidence_sets']}")
    print(f"CLAIM: {row['claim']}")
    keep = set(range(len(row["sentences"]))) if full else {j for k in ev | contra for j in (k - 1, k, k + 1)}
    last = -2
    for k, (s, e) in enumerate(row["sentences"]):
        if k not in keep:
            continue
        if k != last + 1:
            print("   …")
        mark = ">>" if k in ev else "xx" if k in contra else "  "
        text = row["document"][s:e].replace("\n", " ⏎ ")
        print(f"{mark} {k:3d}  {text[:width - 10]}")
        last = k


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True)
    p.add_argument("--dir", type=Path, default=io.OUT)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--label", type=int, choices=[0, 1], default=1)
    p.add_argument("--id")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--full", action="store_true")
    a = p.parse_args(argv)
    rows = io.read_jsonl(a.dir / f"{a.source}.jsonl")
    if a.id:
        picked = [r for r in rows if r["id"] == a.id]
    else:
        pool = [r for r in rows if r["label"] == a.label]
        picked = random.Random(a.seed).sample(pool, min(a.n, len(pool)))
    for row in picked:
        show(row, a.full)


if __name__ == "__main__":
    main()
