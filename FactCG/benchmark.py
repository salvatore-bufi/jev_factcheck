#!/usr/bin/env python3
"""Evaluate FactCG-DeBERTa-v3-Large on the pinned LLM-AggreFact test split.

Run from the repository root (or anywhere):

    python FactCG/benchmark.py                 # full test split, resumable
    python FactCG/benchmark.py --limit 200     # quick smoke test (not a benchmark score)
    python FactCG/benchmark.py --offline       # rebuild the report from cached scores

This is a fixed, zero-shot evaluation: the checkpoint, chunking (<=550 words per
chunk), score rule (maximum support probability over a document's chunks) and the
0.5 threshold follow the upstream FactCG / MiniCheck convention. Nothing is
tuned on development or test data. It reuses factcg_client.py and chunking.py
unchanged; the per-claim pipeline in main.py is not used. The results never
touch the frozen Jev experiment files.

Data: data/test.parquet if present, else the Hugging Face cache directory
./lytang___llm-aggre_fact; override with --data (parquet, .arrow file or cache dir).
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))   # FactCG modules
sys.path.insert(1, str(ROOT))   # shared metrics

import config  # noqa: E402
from metrics import metrics  # noqa: E402

NAME = "FactCG-DeBERTa-v3-Large"
SIZE = "0.4B"
THRESHOLD = 0.5
SHORT = {  # dataset field -> leaderboard column
    "AggreFact-CNN": "CNN", "AggreFact-XSum": "XSum", "TofuEval-MediaS": "MediaS",
    "TofuEval-MeetB": "MeetB", "Wice": "WiCE", "Reveal": "REVEAL", "ClaimVerify": "ClaimVerify",
    "FactCheck-GPT": "FactCheck", "ExpertQA": "ExpertQA", "Lfqa": "LFQA", "RAGTruth": "RAGTruth",
}
COLUMNS = ["CNN", "XSum", "MediaS", "MeetB", "WiCE", "REVEAL", "ClaimVerify", "FactCheck",
           "ExpertQA", "LFQA", "RAGTruth"]


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
        order = sorted(range(len(rows)), key=lambda i: (hashlib.sha256(rows[i]["id"].encode()).hexdigest()))
        rows = [rows[i] for i in sorted(order[:limit])]
    return rows


def resolve_cache(cache_dir):
    if cache_dir:
        return Path(cache_dir).expanduser()
    shared = Path.home() / ".cache/huggingface/hub"
    if (shared / "models--yaxili96--FactCG-DeBERTa-v3-Large").exists():
        return shared
    return config.CACHE_DIR


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


def score_rows(rows, model, chunk_document, args, cache_path, identity, done):
    todo = [r for r in rows if r["id"] not in done]
    new_file = not cache_path.exists()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with cache_path.open("a") as out:
        if new_file:
            out.write(json.dumps({"identity": identity}) + "\n")
        for start in range(0, len(todo), args.window):
            window = todo[start:start + args.window]
            pairs = []  # (row index in window, chunk index, chunk text)
            for w, row in enumerate(window):
                for c, chunk in enumerate(chunk_document(row["doc"], args.chunk_size)):
                    pairs.append((w, c, chunk))
            # Longest first within the window keeps padding small and surfaces OOM early.
            order = sorted(range(len(pairs)), key=lambda i: -len(pairs[i][2]))
            probs = [None] * len(pairs)
            for b in range(0, len(order), args.batch_size):
                idx = order[b:b + args.batch_size]
                support, _ = model.score_pairs([pairs[i][2] for i in idx],
                                               [window[pairs[i][0]]["claim"] for i in idx])
                for i, p in zip(idx, support):
                    probs[i] = p
            per_row = {}
            for (w, c, _), p in zip(pairs, probs):
                per_row.setdefault(w, []).append(p)
            for w, row in enumerate(window):
                chunk_probs = per_row.get(w) or [0.0]  # empty document: nothing supports the claim
                record = {"id": row["id"], "score": max(chunk_probs), "n_chunks": len(chunk_probs)}
                done[row["id"]] = record
                out.write(json.dumps(record) + "\n")
            out.flush()
            finished = min(start + args.window, len(todo))
            rate = finished / max(time.time() - started, 1e-9)
            print(f"{finished}/{len(todo)} new rows ({rate:.1f} rows/s)", flush=True)


def build_report(rows, done, args, identity, complete):
    scored = [r for r in rows if r["id"] in done]
    result = metrics(scored, [done[r["id"]]["score"] for r in scored], THRESHOLD)
    by = {SHORT.get(k, k): v for k, v in result["by_source"].items()}
    report = {
        "model": config.MODEL_ID, "threshold": THRESHOLD, "chunk_size": args.chunk_size,
        "max_length": config.MAX_LENGTH, "score_rule": "max support probability over document chunks",
        "rows_total": len(rows), "rows_scored": len(scored), "complete": complete,
        "setting": "zero-shot: no tuning on development or test data",
        "identity": identity, "reported_at": datetime.now(timezone.utc).isoformat(),
        "macro_balanced_accuracy": result["macro_balanced_accuracy"],
        "pooled": {k: result[k] for k in ["n", "balanced_accuracy", "accuracy", "roc_auc",
                                          "supported_recall", "unsupported_recall"]},
        "by_source": by,
    }
    avg = result["macro_balanced_accuracy"]
    cells = [f"{by[c]['balanced_accuracy'] * 100:.2f}" if c in by and by[c]["balanced_accuracy"] is not None else "n/a"
             for c in COLUMNS]
    avg_cell = f"{avg * 100:.2f}" if avg is not None else "n/a"
    row_md = f"| {NAME} (ours; zero-shot) | {SIZE} | {avg_cell} | " + " | ".join(cells) + " |"
    return report, row_md


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_data = ROOT / "data/test.parquet"
    if not default_data.exists():
        default_data = ROOT / "lytang___llm-aggre_fact"
    p.add_argument("--data", default=str(default_data),
                   help="test.parquet, a HF datasets cache llm-aggre_fact-test.arrow, or the cache directory")
    p.add_argument("--output", default=str(ROOT / "results/runs/factcg-deberta-v3-large"))
    p.add_argument("--limit", type=int, default=0, help="score a deterministic subset (0 = all rows)")
    p.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--window", type=int, default=32, help="rows whose chunks are batched together")
    p.add_argument("--cache-dir", default=None,
                   help="Hugging Face cache holding the model (default: ~/.cache/huggingface/hub if it "
                        "already has FactCG, else FactCG/cache)")
    p.add_argument("--device", default=None, help="cpu or cuda (default: auto)")
    p.add_argument("--gpu", default=None, help="physical GPU id(s), sets CUDA_VISIBLE_DEVICES")
    p.add_argument("--offline", action="store_true", help="only rebuild the report from cached scores")
    args = p.parse_args()

    if args.gpu is not None:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    data = resolve_data(args.data)

    out_dir = Path(args.output)
    # A smoke test must never share a cache/report with a full run.
    if args.limit:
        out_dir = out_dir.with_name(out_dir.name + f"-limit{args.limit}")
    identity = {
        "model": config.MODEL_ID, "chunk_size": args.chunk_size, "max_length": config.MAX_LENGTH,
        "threshold": THRESHOLD, "limit": args.limit,
        "data_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
    }
    rows = load_rows(data, args.limit)
    cache_path = out_dir / "scores.jsonl"
    done = read_cache(cache_path, identity)

    if not args.offline and len(done) < len(rows):
        from chunking import chunk_document
        from factcg_client import FactCGModel
        model = FactCGModel(config.MODEL_ID, resolve_cache(args.cache_dir), config.MAX_LENGTH, args.device)
        print(f"device={model.device}; {len(rows) - len(done)} of {len(rows)} rows to score", flush=True)
        score_rows(rows, model, chunk_document, args, cache_path, identity, done)

    if not done:
        raise SystemExit("No cached scores; run without --offline")
    complete = all(r["id"] in done for r in rows)
    report, row_md = build_report(rows, done, args, identity, complete)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    header = "| Model | Size | Average | " + " | ".join(COLUMNS) + " |"
    status = "" if complete else f"\n\n**Incomplete: {report['rows_scored']}/{report['rows_total']} rows scored; not a benchmark score.**"
    if args.limit:
        status += "\n\n**Subset run (--limit): not a benchmark score.**"
    (out_dir / "table_row.md").write_text(header + "\n|---|---:|---:|" + "---:|" * len(COLUMNS) + "\n" + row_md + status + "\n")
    print(header)
    print(row_md + status)
    print(f"Report: {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
