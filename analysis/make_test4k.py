"""AF-4k: 364 rows per LLM-AggreFact test source (all 358 of Wice), as class-balanced as possible; seed 2026."""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, ".")
from core import datasets

rows, _ = datasets.load("lytang--LLM-AggreFact", "test")
rng = random.Random(2026)
by_source = defaultdict(lambda: {0: [], 1: []})
for r in sorted(rows, key=lambda r: int(r["id"].split(":")[1])):
    by_source[r["dataset"]][r["label"]].append(r)

picked = []
for source in sorted(by_source):
    groups = by_source[source]
    if len(groups[0]) + len(groups[1]) <= 364:
        picked += groups[0] + groups[1]
        continue
    minority = 0 if len(groups[0]) <= len(groups[1]) else 1
    n_min = min(182, len(groups[minority]))
    picked += rng.sample(groups[minority], n_min) + rng.sample(groups[1 - minority], 364 - n_min)

out = Path("dataset/lytang--LLM-AggreFact/test4k.jsonl")
with out.open("w") as f:
    for r in sorted(picked, key=lambda r: int(r["id"].split(":")[1])):
        f.write(json.dumps({"id": r["id"], "dataset": r["dataset"], "doc": r["doc"],
                            "claim": r["claim"], "label": r["label"]}) + "\n")
print(out, len(picked))
