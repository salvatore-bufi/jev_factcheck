#!/usr/bin/env python3
"""Evaluate d1-3B (LiquidAI/d1-3B) on the pinned LLM-AggreFact test split.

Run from the repository root (or anywhere):

    python d1-3b/benchmark.py --limit 1       # one claim against its document (smoke test)
    python d1-3b/benchmark.py                 # full test split, resumable
    python d1-3b/benchmark.py --offline       # rebuild the report from cached answers

d1-3B (Liquid AI, built on LFM2.5-VL-3B) is a decision model that ships its own code: it is loaded with
`trust_remote_code=True` and answers typed questions with `model.system_one(state, questions)` in a single
forward pass with no generated tokens. This script calls that official API, so the prompt (the state as
indented JSON, then `QUESTION:` and the question; a yes/no question ends with "Reply with yes or no only."),
the readout (a softmax over the yes/no tokens) and the batching (a request with several questions reads the
document once) are the model authors' own. It runs with plain transformers on one GPU; vLLM is not needed
for a 3B model.

The experiment is a fixed transfer evaluation, like the JPT-9B, Rune, Winnow and CLM runs in the README:
the frozen three-question pack (packs/claim_support.json), the minimum rule and the threshold > 0.30, with
the same 80,000-character document cap. Nothing is tuned for d1-3B. The cache stores the raw yes-probabilities,
so --threshold only rescores it; do not pick a value from test results.

Data: data/test.parquet if present, else the Hugging Face cache directory
./lytang___llm-aggre_fact; override with --data. The model is read from the local Hugging Face cache.
"""
import argparse
from datetime import datetime, timezone
import glob
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))   # shared metrics.py, evaluate.state_for

from metrics import metrics  # noqa: E402

MODEL_ID = "LiquidAI/d1-3B"
NAME = "d1-3B"
SIZE = "3B"
THRESHOLD = 0.30              # frozen Jev rule, applied unchanged
MAX_DOC_CHARS = 80000
CONTEXT = 32768               # the model's context length
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


def load_rows(path, limit, split="test"):
    if split == "dev":   # Jev's verified tune + selection pools (dev_split.py); ids are dev:<row>
        from dev_split import load_dev_rows
        rows = load_dev_rows()[0]
    else:
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

def hub_cache():
    """The Hugging Face hub cache directory (honours HF_HUB_CACHE and HF_HOME)."""
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser()
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")).expanduser() / "hub"


def check_model():
    """The run is offline, so d1-3B must already be complete in the cache: fail with a clear message."""
    need = ["config.json", "model.safetensors", "tokenizer.json", "modeling_d1.py", "runner.py", "prompt.py"]
    for snap in sorted(glob.glob(str(hub_cache() / "models--LiquidAI--d1-3B/snapshots/*"))):
        if all((Path(snap) / f).exists() for f in need):
            return
    raise SystemExit(f"{MODEL_ID} (6 GB, with its code files) is not complete in the Hugging Face cache {hub_cache()}; "
                     "run `hf download LiquidAI/d1-3B` or copy models--LiquidAI--d1-3B into that cache.")


class Scorer:
    """The model's own `system_one_batch`: one forward pass per row; the three questions share the state."""

    def __init__(self, args):
        import torch
        from transformers import AutoModel

        self.torch = torch
        self.pack = json.loads(PACK.read_text())["questions"]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        self.model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, dtype=dtype).to(device).eval()
        self.engine = self.model.engine
        # The model's own question classes live in its downloaded code (transformers_modules.<...>.prompt).
        package = type(self.model).__module__.rsplit(".", 1)[0]
        as_question = sys.modules[package + ".prompt"].as_question
        self.questions = [as_question(q) for q in self.pack.values()]
        print(f"device={device} dtype={dtype}", flush=True)

    def score(self, states):
        """states -> per row ({qid: P(yes)}, prompt tokens), or an error string; never an invented score."""
        results = []
        for state in states:
            n = self.engine.tokens(state, self.questions)   # longest prompt of the three questions
            if n > CONTEXT:
                results.append(f"prompt of {n} tokens exceeds the model's {CONTEXT}-token context")
                continue
            try:
                out = self.model.system_one_batch([(state, self.pack)])[0]
                answers = {q: float(a["noul"]) for q, a in out["answers"].items()}
                results.append((answers, int(out["usage"]["input_tokens"])))
            except Exception as exc:   # e.g. out of memory on a very long document
                results.append(f"{type(exc).__name__}: {str(exc)[:160]}")
                self.torch.cuda.empty_cache()
        return results


def score_rows(rows, scorer, args, cache_path, identity, done):
    from evaluate import state_for
    todo = [r for r in rows if r["id"] not in done]
    new_file = not cache_path.exists()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    started, failures = time.time(), 0
    with cache_path.open("a") as out:
        if new_file:
            out.write(json.dumps({"identity": identity}) + "\n")
        for start in range(0, len(todo), args.window):
            window = todo[start:start + args.window]
            results = scorer.score([state_for(r, args.max_doc_chars) for r in window])
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
        "model": "LiquidAI/d1-3B", "threshold": args.threshold,
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
    p.add_argument("--output", default=str(ROOT / "results/runs/d1-3b"))
    p.add_argument("--limit", type=int, default=0, help="score a deterministic subset (0 = all rows)")
    p.add_argument("--threshold", type=float, default=THRESHOLD,
                   help="decision threshold (default: the frozen 0.30); rescoring the cache is free, but do not "
                        "pick a value from test results")
    p.add_argument("--max-doc-chars", type=int, default=MAX_DOC_CHARS, help="75%% head + 25%% tail; 0 = no cap")
    p.add_argument("--window", type=int, default=64, help="rows between cache writes")
    p.add_argument("--gpu", default=None, help="physical GPU id(s), sets CUDA_VISIBLE_DEVICES")
    p.add_argument("--offline", action="store_true", help="only rebuild the report from cached answers")
    args = p.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")   # the model and its code come from the local cache

    data = resolve_data(args.data)
    out_dir = Path(args.output)
    if args.limit:   # a smoke test must never share a cache or report with a full run
        out_dir = out_dir.with_name(out_dir.name + f"-limit{args.limit}")
    snapshots = sorted(glob.glob(str(hub_cache() / "models--LiquidAI--d1-3B/snapshots/*")))
    weights = Path(snapshots[0]) / "model.safetensors" if snapshots else None
    identity = {
        "model": MODEL_ID, "snapshot": Path(snapshots[0]).name if snapshots else None,
        "max_doc_chars": args.max_doc_chars,
        "pack_sha256": hashlib.sha256(PACK.read_bytes()).hexdigest(), "limit": args.limit,
        "data_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
    }
    rows = load_rows(data, args.limit)
    cache_path = out_dir / "answers.jsonl"
    done = read_cache(cache_path, identity)

    if not args.offline and len(done) < len(rows):
        check_model()
        scorer = Scorer(args)
        print(f"{len(rows) - len(done)} of {len(rows)} rows to score", flush=True)
        score_rows(rows, scorer, args, cache_path, identity, done)

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
