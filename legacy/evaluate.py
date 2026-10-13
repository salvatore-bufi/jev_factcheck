#!/usr/bin/env python3
"""Compare System One models on labeled JSONL/Parquet; defaults to development data."""
import argparse
import concurrent.futures
from decimal import Decimal
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import threading
import time
from urllib.parse import urlsplit

import requests

from bench import digest, load_key, read_rows, write_json
from metrics import metrics

ROOT = Path(__file__).resolve().parent
API = "https://api.typesafe.ai/v1"


def api_url(value):
    value = value.rstrip("/")
    url = urlsplit(value)
    if url.scheme not in ("http", "https") or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError("API base URL must be HTTP(S), without credentials, query or fragment")
    return value


def extract(answers, pack):
    """Decimal arithmetic keeps 1 - 0.70 equal to 0.30 at strict boundaries."""
    scores = {}
    for name, spec in pack["extractions"].items():
        value = answers[spec["answer_key"]][spec["field"]]
        if "option" in spec:
            value = value[spec["option"]]
        value = Decimal(str(value)) / Decimal(str(spec.get("divisor", 1)))
        if spec.get("invert"):
            value = 1 - value
        if not value.is_finite() or not 0 <= value <= 1:
            raise ValueError(f"Invalid extracted score: {name}")
        scores[name] = value
    for name, spec in pack.get("combinations", {}).items():
        values = [scores[k] for k in spec["inputs"]]
        if not values:
            raise ValueError(f"Empty combination: {name}")
        operation = spec["operation"]
        if operation == "min":
            scores[name] = min(values)
        elif operation == "mean":
            scores[name] = sum(values) / len(values)
        elif operation == "product":
            scores[name] = math.prod(values)
        else:
            raise ValueError(f"Unknown combination: {operation}")
    return scores


def validate_pack(pack):
    """Check question references and scoring recipes before spending any API calls."""
    answers = {}
    if not pack.get("questions") or not pack.get("extractions"):
        raise ValueError("Pack requires questions and extractions")
    for name, q in pack["questions"].items():
        if not q.get("instructions"):
            raise ValueError(f"Missing instructions: {name}")
        if q["type"] == "noul":
            answers[name] = {"noul": .5}
        elif q["type"] == "choice" and isinstance(q.get("criteria"), dict) and 1 <= len(q["criteria"]) <= 255:
            answers[name] = {"probabilities": {key: 1 / len(q["criteria"]) for key in q["criteria"]}}
        elif q["type"] == "score" and isinstance(q.get("criteria"), list) and 2 <= len(q["criteria"]) <= 10:
            answers[name] = {"score": 0, "probabilities": {str(i): 1 / len(q["criteria"]) for i in range(len(q["criteria"]))}}
        else:
            raise ValueError(f"Invalid question type or criteria: {name}")
    extract(answers, pack)


def load_data(path, seed):
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        raw = pq.read_table(path).to_pylist()
    else:
        raw = read_rows(path)
    rows = []
    for i, row in enumerate(raw):
        doc = row.get("doc", row.get("document"))
        claim = row.get("claim")
        if not isinstance(doc, str) or not doc.strip() or not isinstance(claim, str) or not claim.strip():
            raise ValueError(f"Row {i}: nonempty doc/document and claim strings required")
        if type(row.get("label")) not in (int, bool) or row["label"] not in (0, 1):
            raise ValueError(f"Row {i}: binary label required")
        rows.append({"id": str(row.get("id", f"row:{i}")), "doc": doc, "claim": claim,
                     "label": int(row["label"]), "dataset": str(row.get("dataset", "all"))})
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Dataset must be nonempty with unique IDs")
    # Interleave source/label strata so a small pilot is not just the first source.
    groups = {}
    for row in rows:
        groups.setdefault((row["dataset"], row["label"]), []).append(row)
    buckets = [sorted(groups[k], key=lambda r: digest([seed, r["id"]])) for k in sorted(groups)]
    return [bucket[i] for i in range(max(map(len, buckets))) for bucket in buckets if i < len(bucket)]


def state_for(row, cap):
    document = row["doc"]
    if cap and len(document) > cap:
        head = 3 * cap // 4
        document = document[:head] + "\n[Middle of long document omitted]\n" + document[-(cap-head):]
    return {"document": document, "claim": row["claim"]}


class Client:
    def __init__(self, rps, base_url=API, api_key_env=None, timeout=90):
        self.base_url, self.timeout = api_url(base_url), timeout
        if api_key_env:
            self.key = os.environ.get(api_key_env)
            if not self.key:
                raise RuntimeError(f"API key environment variable is unset: {api_key_env}")
        elif self.base_url == API:
            self.key = os.environ.get("TYPESAFE_API_KEY") or load_key()
        else:
            self.key = None  # Never forward TypeSafe credentials to a different server.
        self.rps, self.next_start = rps, 0.0
        self.lock, self.local = threading.Lock(), threading.local()

    def request(self, method, route, payload=None):
        if not hasattr(self.local, "session"):
            self.local.session = requests.Session()
        for attempt in range(5):
            with self.lock:
                delay = max(0, self.next_start - time.monotonic())
                self.next_start = max(time.monotonic(), self.next_start) + 1 / self.rps
            time.sleep(delay)
            try:
                response = self.local.session.request(method, self.base_url + route, json=payload,
                    headers={"Authorization": f"Bearer {self.key}"} if self.key else {},
                    timeout=(15, self.timeout), allow_redirects=False)
            except requests.RequestException:
                if attempt == 4:
                    raise RuntimeError("System One connection failed after five attempts") from None
                time.sleep(2 ** attempt)
                continue
            if response.status_code == 429 or response.status_code >= 500:
                if attempt == 4:
                    raise RuntimeError(f"System One HTTP {response.status_code} after five attempts")
                try:
                    wait = float(response.headers.get("Retry-After", 0))
                except ValueError:
                    wait = 0
                time.sleep(min(60, max(2 ** attempt, wait)))
                continue
            if response.status_code != 200:
                raise RuntimeError(f"System One HTTP {response.status_code}")
            try:
                return response.json(), attempt + 1
            except ValueError:
                raise RuntimeError("System One returned invalid JSON") from None

    def ask(self, model, state, questions):
        started = time.monotonic()
        body, attempts = self.request("POST", "/systemone", {"model": model, "state": state, "questions": questions})
        validate_response(body, questions)
        return {"model": body["model"], "answers": body["answers"], "usage": body["usage"],
                "seconds": time.monotonic() - started, "attempts": attempts}


def validate_response(body, questions):
    try:
        if not isinstance(body["model"], str) or not body["model"]:
            raise ValueError
        for name, question in questions.items():
            if body["answers"][name]["type"] != question["type"]:
                raise ValueError
        for field in ("input_tokens", "output_tokens"):
            if type(body["usage"][field]) is not int or body["usage"][field] < 0:
                raise ValueError
    except (ValueError, KeyError, TypeError):
        raise RuntimeError("Invalid System One response schema") from None


def report_run(rows, cache, pack, args, model, directory):
    available = [r for r in rows if r["id"] in cache]
    scores = [float(extract(cache[r["id"]]["answers"], pack)[args.score]) for r in available]
    usage = {key: sum(cache[r["id"]]["usage"][key] for r in available) for key in ("input_tokens", "output_tokens")}
    pack_path = directory / f"pack-{digest(pack)[:12]}.json"
    write_json(pack_path, pack)
    result = {"requested_model": model, "base_url": args.base_url,
              "resolved_models": sorted({cache[r["id"]]["model"] for r in available}),
              "directory": str(directory), "data": str(args.data.resolve()), "seed": args.seed,
              "requested_rows": len(rows), "completed_rows": len(available), "complete": len(available) == len(rows),
              "score": args.score, "threshold": args.threshold, "score_arithmetic": "decimal-v1",
              "pack_hash": digest(pack), "pack_path": str(pack_path), "usage": usage,
              "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "truncated_documents": sum(bool(args.max_doc_chars and len(r["doc"]) > args.max_doc_chars) for r in available),
              "median_request_seconds_including_pacing": statistics.median([cache[r["id"]]["seconds"] for r in available]) if available else None,
              "metrics": metrics(available, scores, args.threshold) if available else None}
    if result["metrics"]:
        for m in [result["metrics"], *result["metrics"]["by_source"].values()]:
            m["tpr"] = m["supported_recall"]
            m["fpr"] = 1-m["unsupported_recall"] if m["unsupported_recall"] is not None else None
    if args.input_price is not None and args.output_price is not None:
        result["estimated_cost_usd"] = (usage["input_tokens"] * args.input_price + usage["output_tokens"] * args.output_price) / 1e6
    else:
        result["estimated_cost_usd"] = None
    result["cost_note"] = "Successful cached responses only; retries or interrupted requests may add unrecorded charges. Null means prices were not supplied."
    key = digest([digest(pack), args.score, args.threshold, args.seed, len(rows), result["evaluator_sha256"]])[:12]
    write_json(directory / f"report-{key}.json", result)
    write_json(directory / "report.json", result)
    return result


def run_model(model, rows, pack, args, data_hash):
    identity = {"cache_version": 1, "requested_model": model, "questions": pack["questions"],
                "data_sha256": data_hash, "max_doc_chars": args.max_doc_chars, "run_name": args.run_name}
    if args.base_url != API:
        identity["base_url"] = args.base_url  # Preserve existing TypeSafe cache identities.
    slug = re.sub(r"[^a-zA-Z0-9_.-]", "_", model)[:80]
    directory = args.output / f"{slug}-{digest(identity)[:12]}"
    if args.dry_run:
        return {"requested_model": model, "base_url": args.base_url,
                "directory": str(directory), "requested_rows": len(rows), "dry_run": True}
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Run is already active: {directory}") from None
        manifest_path, cache_path = directory / "run.json", directory / "responses.jsonl"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"identity": identity, "resolved_model": None}
        if manifest["identity"] != identity:
            raise RuntimeError("Run configuration mismatch")
        stored = read_rows(cache_path) if cache_path.exists() else []
        cache = {r["id"]: r for r in stored}
        if len(cache) != len(stored):
            raise RuntimeError("Duplicate cached IDs")
        if any(p.get("cache_key") != digest(identity) for p in stored):
            raise RuntimeError("Cache belongs to another configuration")
        resolved = {p["model"] for p in stored}
        if len(resolved) > 1 or (resolved and manifest["resolved_model"] not in (None, next(iter(resolved)))):
            raise RuntimeError("Cached model versions disagree")
        manifest["resolved_model"] = next(iter(resolved), manifest["resolved_model"])
        for row in rows:
            if row["id"] in cache:
                p = cache[row["id"]]
                if p["input_hash"] != digest(state_for(row, args.max_doc_chars)):
                    raise RuntimeError("Cached input mismatch")
                validate_response(p, pack["questions"])
                extract(p["answers"], pack)
        write_json(manifest_path, manifest)
        pending = [r for r in rows if r["id"] not in cache]
        print(json.dumps({"model": model, "cached": len(rows)-len(pending), "pending": len(pending), "directory": str(directory)}), flush=True)
        if args.offline or not pending:
            return report_run(rows, cache, pack, args, model, directory)
        client = Client(args.rps, args.base_url, args.api_key_env, args.timeout)
        last_report_n = len(rows) - len(pending)
        with cache_path.open("a") as output, concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            while pending:
                # Resolve aliases with one response, then pin every later request/resume.
                batch, pending = pending[:args.workers if manifest["resolved_model"] else 1], pending[args.workers if manifest["resolved_model"] else 1:]
                requested = manifest["resolved_model"] or model
                futures = {pool.submit(client.ask, requested, state_for(r, args.max_doc_chars), pack["questions"]): r for r in batch}
                failures = []
                for future in concurrent.futures.as_completed(futures):
                    row = futures[future]
                    try:
                        prediction = future.result()
                        if manifest["resolved_model"] and prediction["model"] != manifest["resolved_model"]:
                            raise RuntimeError("Resolved model changed during run")
                        if re.search(r"-\d+(?:\.\d+)+$", requested) and prediction["model"] != requested:
                            raise RuntimeError("Response differs from requested pinned model")
                        extract(prediction["answers"], pack)
                    except Exception as error:
                        # Never print response bodies, exception text, documents, or credentials.
                        message = str(error) if isinstance(error, RuntimeError) else "Invalid response or extraction"
                        failures.append({"id": row["id"], "error": message})
                        continue
                    manifest["resolved_model"] = prediction["model"]
                    prediction.update(id=row["id"], input_hash=digest(state_for(row, args.max_doc_chars)), cache_key=digest(identity))
                    output.write(json.dumps(prediction) + "\n")
                    output.flush()
                    cache[row["id"]] = prediction
                write_json(manifest_path, manifest)
                completed = sum(r["id"] in cache for r in rows)
                if last_report_n == 0 or completed - last_report_n >= args.report_every or not pending or failures:
                    report = report_run(rows, cache, pack, args, model, directory)
                    last_report_n = completed
                    print(json.dumps({"model": model, "completed": completed, "total": len(rows),
                                      "macro_balanced_accuracy": report["metrics"]["macro_balanced_accuracy"] if report["metrics"] else None}), flush=True)
                if failures:
                    with (directory / "errors.jsonl").open("a") as handle:
                        handle.write("".join(json.dumps(e) + "\n" for e in failures))
                    raise RuntimeError(f"{len(failures)} request(s) failed; successful responses saved in {directory}")
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    models = commands.add_parser("models", help="List model names from the selected endpoint (no inference)")
    run = commands.add_parser("run", help="Evaluate a fixed configuration; never tune thresholds automatically")
    for command in (models, run):
        command.add_argument("--base-url", type=api_url, default=API, help="System One API base URL, including /v1")
        command.add_argument("--api-key-env", help="Environment variable containing this endpoint's key; custom endpoints default to no auth")
        command.add_argument("--timeout", type=float, default=90, help="Per-request read timeout in seconds")
    run.add_argument("--model", nargs="+", required=True)
    run.add_argument("--data", type=Path, default=ROOT / "data/selection.jsonl")
    run.add_argument("--pack", type=Path, default=ROOT / "packs/claim_support.json")
    run.add_argument("--score")
    run.add_argument("--threshold", type=float)
    run.add_argument("--limit", type=int, default=100, help="Balanced deterministic pilot; 0 means all rows")
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--workers", type=int, default=8)
    run.add_argument("--rps", type=float, default=8)
    run.add_argument("--report-every", type=int, default=100)
    run.add_argument("--max-doc-chars", type=int, default=80000, help="Retain 75%% head + 25%% tail; 0 disables shortening")
    run.add_argument("--output", type=Path, default=ROOT / "results/runs")
    run.add_argument("--run-name", default="default", help="Use a new name to refresh an alias or force a fresh cache")
    run.add_argument("--input-price", type=float, help="USD per million input tokens; never assumed for an unknown model")
    run.add_argument("--output-price", type=float, help="USD per million output tokens")
    run.add_argument("--offline", action="store_true", help="Report from cache only, no key or network required")
    run.add_argument("--dry-run", action="store_true", help="Show planned run paths and row counts, no writes or requests")
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("Timeout must be finite and positive")
    if args.command == "models":
        body, _ = Client(1, args.base_url, args.api_key_env, args.timeout).request("GET", "/models")
        print(json.dumps(body, indent=2))
        return
    pack = json.loads(args.pack.read_text())
    validate_pack(pack)
    decision = pack.get("decision", {})
    args.score = args.score or decision.get("score")
    args.threshold = args.threshold if args.threshold is not None else decision.get("threshold", .5)
    names = set(pack["extractions"]) | set(pack.get("combinations", {}))
    if not args.score or args.score not in names:
        parser.error("Choose --score from: " + ", ".join(sorted(names)))
    if not math.isfinite(args.threshold) or not 0 <= args.threshold <= 1:
        parser.error("Threshold must be finite and between 0 and 1")
    if args.limit < 0 or args.workers < 1 or args.report_every < 1 or not math.isfinite(args.rps) or args.rps <= 0 or args.max_doc_chars < 0:
        parser.error("Invalid limit, workers, rate, or context cap")
    if any(p is not None and (not math.isfinite(p) or p < 0) for p in (args.input_price, args.output_price)):
        parser.error("Prices must be finite and nonnegative")
    data_hash = hashlib.sha256(args.data.read_bytes()).hexdigest()
    rows = load_data(args.data, args.seed)
    if args.limit:
        rows = rows[:args.limit]
    reports = [run_model(model, rows, pack, args, data_hash) for model in dict.fromkeys(args.model)]
    if not args.dry_run:
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output / "comparison.json", reports)
        lines = ["# System One comparison", "", "Evaluation results; inspect each report's data path, endpoint and coverage.", "",
                 "| Requested model | Resolved model | Rows | Macro balanced accuracy | TPR | FPR |", "|---|---|---:|---:|---:|---:|"]
        for report in reports:
            m = report["metrics"] or {}
            percent = lambda value: f"{100*value:.2f}%" if value is not None else "—"
            fpr = 1-m["unsupported_recall"] if m.get("unsupported_recall") is not None else None
            lines.append(f"| {report['requested_model']} | {', '.join(report['resolved_models']) or '—'} | {report['completed_rows']}/{report['requested_rows']} | {percent(m.get('macro_balanced_accuracy'))} | {percent(m.get('supported_recall'))} | {percent(fpr)} |")
        (args.output / "comparison.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, KeyError, OSError) as error:
        raise SystemExit(str(error)) from None
