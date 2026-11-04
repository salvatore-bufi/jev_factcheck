"""STEP 9b — Latency: does the evidence keep the speed of a decision model? (doc 10 §5)

Three conditions, same GPU, same requests (real rows of different lengths), the first --warmup not counted:
    A  no preamble,   verdict only
    B  preamble,      verdict only
    C  preamble,      verdict + evidence (hooks, attention rows, head, selection)   <- what the server does
Target: C at most +3% over B (4B models: +3% or +2 ms). B - A is the cost of the preamble, reported apart.
Also timed separately: sentence splitting (CPU) and rendering, the head alone.

Run (GPU):
    python -m evidence.latency --run data/evidence/runs/knowline-4b/<id> --n 1000
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evidence import config  # noqa: E402
from evidence.data import io  # noqa: E402


def percentiles(values):
    """p50 and p95 in milliseconds. Input: list of seconds. Output: dict."""
    v = sorted(values)
    return {"p50_ms": round(1000 * statistics.median(v), 2), "p95_ms": round(1000 * v[int(0.95 * (len(v) - 1))], 2),
            "mean_ms": round(1000 * statistics.mean(v), 2)}


def main(argv=None):
    """Command line entry point (see the file header)."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True, help="run or release folder with the head")
    p.add_argument("--rows", type=Path, default=config.SPLITS / "test_evid_id.jsonl")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--warmup", type=int, default=50)
    opts = p.parse_args(argv)

    from evidence import serve
    from evidence.render import render
    from evidence.sentences import split
    app = serve.Service(opts.run)
    rows = io.read_jsonl(opts.rows)[:opts.n + opts.warmup]
    times = {"A": [], "B": [], "C": [], "split": [], "head": []}
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    from evidence.model import forward_sentences
    for i, row in enumerate(rows):
        t0 = time.perf_counter()
        spans = split(row["document"])
        t_split = time.perf_counter() - t0
        for cond, preamble in (("A", None), ("B", app.preamble)):
            sync()
            t0 = time.perf_counter()
            ex = render(row["document"], row["claim"], spans, preamble)
            forward_sentences(app.model, ex, features=False)
            sync()
            if i >= opts.warmup:
                times[cond].append(time.perf_counter() - t0)
        sync()
        t0 = time.perf_counter()
        app.answer(row["document"], row["claim"], spans=spans, timings=times["head"] if i >= opts.warmup else None)
        sync()
        if i >= opts.warmup:
            times["C"].append(time.perf_counter() - t0 + t_split)
            times["A"][-1] += t_split
            times["B"][-1] += t_split
            times["split"].append(t_split)
    report = {k: percentiles(v) for k, v in times.items() if v}
    b, c = report["B"]["p50_ms"], report["C"]["p50_ms"]
    report["C_vs_B"] = {"extra_ms": round(c - b, 2), "extra_pct": round(100 * (c - b) / b, 2),
                        "within_target": (c - b) <= max(0.03 * b, 2.0)}
    report["B_vs_A_preamble_ms"] = round(report["B"]["p50_ms"] - report["A"]["p50_ms"], 2)
    out = Path(opts.run) / "latency.json"
    io.write_json(out, report)
    for k, v in report.items():
        print(k, v)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
