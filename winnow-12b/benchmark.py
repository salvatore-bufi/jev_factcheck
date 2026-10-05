#!/usr/bin/env python3
"""Evaluate Winnow-12B (NVFP4 GGUF) on the pinned LLM-AggreFact test split.

Run from the repository root (or anywhere), with the Winnow server running (see serve.sh):

    python winnow-12b/benchmark.py --limit 1       # one claim against its document (smoke test)
    python winnow-12b/benchmark.py                 # full test split, resumable
    python winnow-12b/benchmark.py --offline       # rebuild the report from cached answers

Winnow-12B (EldanRing/Winnow-12B, a Gemma 4 12B fine-tune) is a typed-decision model that
ships only as GGUF files and runs on its own llama.cpp-based server, which exposes the same
`/v1/systemone` API as Jev (a noul answer is the probability of "true"). vLLM cannot load
this NVFP4 GGUF, so unlike the other models this script is an HTTP client: the server renders
the prompts and reads the answer logits itself, as its authors intend.

The experiment is a fixed transfer evaluation, like the CLM, JPT-9B and Rune runs in the
README: the frozen three-question pack (packs/claim_support.json), the minimum rule and the
threshold > 0.30, with the same 80,000-character document cap. Nothing is tuned for Winnow.
The server's decision temperature is its default of 1. The raw yes-probabilities are cached,
so --threshold only rescores the cache; do not pick a value from test results.

Data: data/test.parquet if present, else the Hugging Face cache directory
./lytang___llm-aggre_fact; override with --data.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))   # shared metrics.py, evaluate.state_for

from metrics import metrics  # noqa: E402

MODEL = "Winnow-12B"
NAME = "Winnow-12B-NVFP4"
SIZE = "12B"
URL = "http://127.0.0.1:8091"
THRESHOLD = 0.30              # frozen Jev rule, applied unchanged
MAX_DOC_CHARS = 80000
PACK = ROOT / "packs/claim_support.json"
INVERT = {"support_simple": False, "unsupported_detail": True, "contradicted_detail": True}
SHORT = {
    "AggreFact-CNN": "CNN", "AggreFact-XSum": "XSum", "TofuEval-MediaS": "MediaS",
    "TofuEval-MeetB": "MeetB", "Wice": "WiCE", "Reveal": "REVEAL", "ClaimVerify": "ClaimVerify",
    "FactCheck-GPT": "FactCheck", "ExpertQA": "ExpertQA", "Lfqa": "LFQA", "RAGTruth": "RAGTruth",
}
COLUMNS = ["CNN", "XSum", "MediaS", "MeetB", "WiCE", "REVEAL", "ClaimVerify", "FactCheck",
           "ExpertQA", "LFQA", "RAGTruth"]


# ---------------------------------------------------------------- data

def resolve_data(path):
    """Accept test.parquet, a Hugging Face `datasets` cache .arrow file, or the cache directory."""
    path = Path(path)
    if path.is_dir():
        found = sorted(path.rglob("llm-aggre_fact-test.arrow"))
        if len(found) != 1:
            raise SystemExit(f"Expected exactly one llm-aggre_fact-test.arrow under {path}, found {len(found)}")
        path = found[0]
    if not path.exists():
        raise SystemExit(f"{path} not found")
    return path


def read_table(path):
    if path.suffix == ".arrow":
        import pyarrow as pa
        with path.open("rb") as handle:
            return pa.ipc.open_stream(handle).read_all()
    import pyarrow.parquet as pq
    return pq.read_table(path)


def load_rows(path, limit):
    rows = read_table(Path(path)).to_pylist()
    for i, row in enumerate(rows):
        row["id"] = f"test:{i}"
        row.pop("contamination_identifier", None)
    if limit:
        # Deterministic pseudo-random subset (hash order) so a smoke test spans several sources.
        order = sorted(range(len(rows)), key=lambda i: hashlib.sha256(rows[i]["id"].encode()).hexdigest())
        rows = [rows[i] for i in sorted(order[:limit])]
    return rows


# ---------------------------------------------------------------- cache

def read_cache(path, identity):
    done = {}
    if not path.exists():
        return done
    with path.open() as handle:
        header = json.loads(handle.readline())
        if header.get("identity") != identity:
            raise SystemExit(f"{path} was produced with a different configuration; use another --output")
        for line in handle:
            if line.strip():
                record = json.loads(line)
                done[record["id"]] = record
    return done


# ---------------------------------------------------------------- scoring rule

def decide(answers):
    """Frozen rule: min(support, 1 - unsupported detail, 1 - contradicted detail)."""
    return min((1 - p) if INVERT[q] else p for q, p in answers.items())


# ---------------------------------------------------------------- inference

class Client:
    """Posts one (document, claim) state with the three frozen questions to the Winnow server."""

    def __init__(self, url, timeout):
        self.url, self.timeout = url.rstrip("/"), timeout
        self.pack = json.loads(PACK.read_text())["questions"]
        self.local = threading.local()

    def check(self):
        try:
            health = requests.get(self.url + "/health", timeout=5)
            health.raise_for_status()
            models = requests.get(self.url + "/v1/models", timeout=5).json()
        except (requests.RequestException, ValueError) as exc:
            raise SystemExit(f"Winnow server not reachable at {self.url} ({exc}). Start it with winnow-12b/serve.sh")
        names = [m.get("name") for m in models.get("models", [])]
        if MODEL not in names:
            raise SystemExit(f"Server at {self.url} serves {names}, expected {MODEL!r}")

    def ask(self, state):
        """-> ({qid: P(yes)}, input tokens) or an error string; a failed row never gets an invented score."""
        session = getattr(self.local, "session", None) or self.local.__dict__.setdefault("session", requests.Session())
        try:
            response = session.post(self.url + "/v1/systemone", json={"state": state, "questions": self.pack},
                                    timeout=self.timeout)
        except requests.RequestException as exc:
            return f"request failed: {exc}"
        if response.status_code != 200:
            return f"HTTP {response.status_code}: {response.text[:200]}"
        body = response.json()
        try:
            answers = {q: float(body["answers"][q]["noul"]) for q in self.pack}
        except (KeyError, TypeError, ValueError):
            return f"unexpected response: {str(body)[:200]}"
        return answers, int(body.get("usage", {}).get("input_tokens", 0))


def score_rows(rows, client, args, cache_path, identity, done):
    from evaluate import state_for
    todo = [r for r in rows if r["id"] not in done]
    new_file = not cache_path.exists()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    started, failures = time.time(), 0
    with cache_path.open("a") as out, ThreadPoolExecutor(args.workers) as pool:
        if new_file:
            out.write(json.dumps({"identity": identity}) + "\n")
        for start in range(0, len(todo), args.window):
            window = todo[start:start + args.window]
            results = list(pool.map(lambda r: client.ask(state_for(r, args.max_doc_chars)), window))
            for row, result in zip(window, results):
                if isinstance(result, str):
                    failures += 1
                    print(f"  {row['id']}: {result}", flush=True)
                    continue
                answers, tokens = result
                record = {"id": row["id"], "answers": answers, "prompt_tokens": tokens}
                done[row["id"]] = record
                out.write(json.dumps(record) + "\n")
            out.flush()
            finished = min(start + args.window, len(todo))
            print(f"{finished}/{len(todo)} new rows ({finished / max(time.time() - started, 1e-9):.1f} rows/s, "
                  f"{failures} failed)", flush=True)


# ---------------------------------------------------------------- report

def build_report(rows, done, args, identity, complete):
    scored = [r for r in rows if r["id"] in done]
    scores = [decide(done[r["id"]]["answers"]) for r in scored]
    result = metrics(scored, scores, args.threshold)
    by = {SHORT.get(k, k): v for k, v in result["by_source"].items()}
    avg = result["macro_balanced_accuracy"]
    report = {
        "model": MODEL, "threshold": args.threshold,
        "score_rule": "min(support, 1 - unsupported_detail, 1 - contradicted_detail) > threshold",
        "pack": "packs/claim_support.json", "max_doc_chars": args.max_doc_chars,
        "rows_total": len(rows), "rows_scored": len(scored), "complete": complete,
        "setting": "fixed transfer of the frozen Jev checker; no tuning for this model",
        "identity": identity, "reported_at": datetime.now(timezone.utc).isoformat(),
        "macro_balanced_accuracy": avg,
        "pooled": {k: result[k] for k in ["n", "balanced_accuracy", "accuracy", "roc_auc",
                                          "supported_recall", "unsupported_recall"]},
        "by_source": by,
        "prompt_tokens_max_per_row_sum": sum(done[r["id"]]["prompt_tokens"] for r in scored),
    }
    cells = [f"{by[c]['balanced_accuracy'] * 100:.2f}" if c in by and by[c]["balanced_accuracy"] is not None else "n/a"
             for c in COLUMNS]
    avg_cell = f"{avg * 100:.2f}" if avg is not None else "n/a"
    row_md = f"| {NAME} (ours; fixed Jev rule) | {SIZE} | {avg_cell} | " + " | ".join(cells) + " |"
    return report, row_md


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_data = ROOT / "data/test.parquet"
    if not default_data.exists():
        default_data = ROOT / "lytang___llm-aggre_fact"
    p.add_argument("--data", default=str(default_data),
                   help="test.parquet, a HF datasets cache llm-aggre_fact-test.arrow, or the cache directory")
    p.add_argument("--output", default=str(ROOT / "results/runs/winnow-12b"))
    p.add_argument("--limit", type=int, default=0, help="score a deterministic subset (0 = all rows)")
    p.add_argument("--threshold", type=float, default=THRESHOLD,
                   help="decision threshold (default: the frozen 0.30); rescoring the cache is free, but do not "
                        "pick a value from test results")
    p.add_argument("--max-doc-chars", type=int, default=MAX_DOC_CHARS, help="75%% head + 25%% tail; 0 = no cap")
    p.add_argument("--url", default=URL, help="Winnow server base URL")
    p.add_argument("--workers", type=int, default=4, help="concurrent requests")
    p.add_argument("--window", type=int, default=64, help="rows submitted between cache writes")
    p.add_argument("--timeout", type=float, default=300, help="per-request timeout, seconds")
    p.add_argument("--offline", action="store_true", help="only rebuild the report from cached answers")
    args = p.parse_args()

    data = resolve_data(args.data)
    out_dir = Path(args.output)
    if args.limit:   # a smoke test must never share a cache or report with a full run
        out_dir = out_dir.with_name(out_dir.name + f"-limit{args.limit}")
    identity = {
        "model": MODEL, "quantization": "NVFP4", "max_doc_chars": args.max_doc_chars,
        "pack_sha256": hashlib.sha256(PACK.read_bytes()).hexdigest(), "limit": args.limit,
        "data_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
    }
    rows = load_rows(data, args.limit)
    cache_path = out_dir / "answers.jsonl"
    done = read_cache(cache_path, identity)

    if not args.offline and len(done) < len(rows):
        client = Client(args.url, args.timeout)
        client.check()
        print(f"{len(rows) - len(done)} of {len(rows)} rows to score", flush=True)
        score_rows(rows, client, args, cache_path, identity, done)

    if not done:
        raise SystemExit("No cached answers; run without --offline")
    complete = all(r["id"] in done for r in rows)
    report, row_md = build_report(rows, done, args, identity, complete)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    header = "| Model | Size | Average | " + " | ".join(COLUMNS) + " |"
    status = "" if complete else (f"\n\n**Incomplete: {report['rows_scored']}/{report['rows_total']} rows scored; "
                                  "not a benchmark score.**")
    if args.limit:
        status += "\n\n**Subset run (--limit): not a benchmark score.**"
    (out_dir / "table_row.md").write_text(header + "\n|---|---:|---:|" + "---:|" * len(COLUMNS) + "\n" + row_md + status + "\n")
    print(header)
    print(row_md + status)
    print(f"Report: {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
