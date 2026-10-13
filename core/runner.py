"""Run ONE model on one dataset split (benchmark.py starts this in its own process, on one GPU).

Phases, all timed and tagged in the telemetry:

1. load       model start-up (cold), measured once per run
2. warmup     a few rows, not counted; the first call is timed separately (kernel selection, graphs)
3. latency    a fixed sample scored ONE row at a time, cache reset before each row (cold document)
4. throughput the same kind of sample at several batch sizes (rows/s against batch size)
5. main       every row of the split in batches; resumable; per-question answers saved for each row

Outputs in results/<dataset>/<split>/<model>_<questions hash>[-limitN]/:
run.json, answers.jsonl, batches.jsonl, latency.jsonl, throughput.jsonl, telemetry.jsonl,
failures.jsonl and report.json (rebuilt by report.py at any time without inference).
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core  # noqa: E402
from core import datasets, questions as qs, registry  # noqa: E402
from core.adapter import effective_options, load_adapter_class, load_model_config, parse_options  # noqa: E402
from core.profiling import Telemetry, environment, other_gpu_processes, summarize_telemetry  # noqa: E402
from core.state import DEFAULT_MAX_DOC_CHARS, state_for, truncated  # noqa: E402

RUNNER_VERSION = 1


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    tmp = Path(path).with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def row_features(row, cap):
    return {"source": row["dataset"], "label": row["label"],
            "claim_chars": len(row["claim"]), "claim_words": len(row["claim"].split()),
            "doc_chars": len(row["doc"]), "doc_words": len(row["doc"].split()),
            "truncated": truncated(row, cap)}


def total_tokens(info):
    tokens = (info or {}).get("prompt_tokens")
    if isinstance(tokens, dict):
        return sum(v for v in tokens.values() if v)
    return tokens


def stratified_sample(rows, n, seed="benchmark-latency-v1"):
    """n rows, round-robin over sources in a seeded hash order (every source represented)."""
    by_source = {}
    for row in sorted(rows, key=lambda r: hashlib.sha256(f"{seed}:{r['id']}".encode()).hexdigest()):
        by_source.setdefault(row["dataset"], []).append(row)
    queues = [by_source[k] for k in sorted(by_source)]
    sample = []
    while len(sample) < n and any(queues):
        for q in queues:
            if q and len(sample) < n:
                sample.append(q.pop(0))
    return sample


def read_answer_cache(path, identity):
    path = Path(path)
    done = {}
    if not path.exists():
        return done
    with path.open() as handle:
        header = json.loads(handle.readline())
        if header.get("identity") != identity:
            raise SystemExit(f"{path} was produced with a different configuration:\n  cached:  "
                             f"{json.dumps(header.get('identity'), sort_keys=True)}\n  current: "
                             f"{json.dumps(identity, sort_keys=True)}\nUse another --tag or delete the folder.")
        for line in handle:
            if line.strip():
                rec = json.loads(line)
                done[rec["id"]] = rec
    return done


def call(adapter, states, questions):
    """adapter.score with a guard: an exception fails the whole batch (each row gets the error)."""
    try:
        out = adapter.score(states, questions)
        if len(out) != len(states):
            raise RuntimeError(f"adapter returned {len(out)} results for {len(states)} rows")
        return out
    except Exception as exc:   # e.g. CUDA out of memory: never invent a score
        message = f"{type(exc).__name__}: {str(exc)[:300]}"
        return [message] * len(states)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--questions", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--opt", action="append", default=[])
    p.add_argument("--model-config", default=None, help="model options file (default config/models/<model>.json)")
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--max-doc-chars", type=int, default=DEFAULT_MAX_DOC_CHARS)
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--latency-sample", type=int, default=200)
    p.add_argument("--throughput-sizes", default="1,8,32")
    p.add_argument("--throughput-sample", type=int, default=64)
    p.add_argument("--no-profile", action="store_true")
    p.add_argument("--allow-busy-gpu", action="store_true")
    p.add_argument("--telemetry-interval", type=float, default=0.2)
    p.add_argument("--tag", default="")
    p.add_argument("--gpu", default=None)
    args = p.parse_args(argv)

    cfg = qs.load(args.questions)
    qhash = qs.questions_hash(cfg["questions"])
    rows, data_info = datasets.load(args.dataset, args.split, args.limit)
    split_dir = core.RESULTS / args.dataset / data_info["split"]
    qs.register(split_dir, cfg)

    cls = load_adapter_class(args.model)
    model_config, model_config_path = load_model_config(args.model, args.model_config)
    overrides = parse_options(args.opt, cls.DEFAULTS)
    if args.batch:
        overrides["batch"] = args.batch
    options = effective_options(cls, model_config, overrides)
    phash = registry.short_hash(options)
    registry.register(split_dir / f"hash_{args.model}.json", phash,
                      {"model": args.model, "model_name": cls.NAME, "options": options}, "options")
    adapter = cls(options)

    name = f"{args.model}_{phash}_{qhash}" + (f"-limit{args.limit}" if args.limit else "") + (f"-{args.tag}" if args.tag else "")
    run_dir = split_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    identity = {"model": args.model, "options": options, "params_hash": phash, "questions_hash": qhash,
                "dataset": args.dataset, "split": data_info["split"], "data_sha256": data_info["sha256"],
                "limit": args.limit, "max_doc_chars": args.max_doc_chars, "runner_version": RUNNER_VERSION}
    answers_path = run_dir / "answers.jsonl"
    done = read_answer_cache(answers_path, identity)

    busy = other_gpu_processes()
    if busy["compute"] and not args.allow_busy_gpu:
        raise SystemExit(f"GPU {os.environ.get('CUDA_VISIBLE_DEVICES')} is used by other compute processes {busy['compute']}: "
                         "latency and memory would be distorted. Free it, pick another --gpu, or pass --allow-busy-gpu.")

    run_path = run_dir / "run.json"
    previous = json.loads(run_path.read_text()) if run_path.exists() else {}
    run = {"identity": identity, "model_name": cls.NAME, "model_size": cls.SIZE, "model_kind": cls.KIND,
           "default_decision": cls.DEFAULT_DECISION, "questions": cfg, "data": data_info,
           "model_config": {"file": model_config_path, "values": model_config, "command_line": overrides},
           "environment": environment(), "gpu_busy_at_start": busy, "status": "running",
           "sessions": previous.get("sessions", [])}
    session = {"started_at": now(), "argv": sys.argv, "gpu": args.gpu}
    run["sessions"].append(session)
    write_json(run_path, run)

    telemetry = Telemetry(run_dir / "telemetry.jsonl", args.telemetry_interval).start()
    failures = (run_dir / "failures.jsonl").open("a")
    questions = cfg["questions"]

    def fail(row_id, phase, error):
        failures.write(json.dumps({"id": row_id, "phase": phase, "error": error, "at": now()}) + "\n")
        failures.flush()

    try:
        # 1. load
        telemetry.phase("load")
        t0 = time.perf_counter()
        adapter.load()
        session["load_seconds"] = round(time.perf_counter() - t0, 3)
        run["model"] = adapter.describe()
        print(f"[{args.model}] loaded in {session['load_seconds']:.1f}s", flush=True)

        # 2. warmup (first call timed separately)
        telemetry.phase("warmup")
        warm = stratified_sample(rows, args.warmup, seed="benchmark-warmup-v1")
        times = []
        for row in warm:
            t = time.perf_counter()
            call(adapter, [state_for(row, args.max_doc_chars)], questions)
            times.append(time.perf_counter() - t)
        session["first_call_seconds"] = round(times[0], 4) if times else None
        session["warmup_seconds"] = [round(x, 4) for x in times]

        # 3. latency: one row at a time, cold prefix cache
        latency_path = run_dir / "latency.jsonl"
        if not args.no_profile and len(read_jsonl(latency_path)) < min(args.latency_sample, len(rows)):
            latency_path.unlink(missing_ok=True)
            telemetry.phase("latency")
            sample = stratified_sample(rows, args.latency_sample)
            with latency_path.open("w") as out:
                for i, row in enumerate(sample):
                    adapter.reset_cache()
                    t = time.perf_counter()
                    result = call(adapter, [state_for(row, args.max_doc_chars)], questions)[0]
                    seconds = time.perf_counter() - t
                    rec = {"id": row["id"], **row_features(row, args.max_doc_chars), "seconds": round(seconds, 5),
                           "ok": not isinstance(result, str)}
                    if isinstance(result, str):
                        rec["error"] = result
                    else:
                        rec["prompt_tokens"] = result[1].get("prompt_tokens")
                    out.write(json.dumps(rec) + "\n")
                    out.flush()
            print(f"[{args.model}] latency probe: {len(sample)} rows", flush=True)

        # 4. throughput against batch size
        throughput_path = run_dir / "throughput.jsonl"
        sizes = [int(s) for s in args.throughput_sizes.split(",") if s.strip()]
        if not args.no_profile and sizes and not throughput_path.exists():
            sample = stratified_sample(rows, args.throughput_sample, seed="benchmark-throughput-v1")
            states = [state_for(r, args.max_doc_chars) for r in sample]
            with throughput_path.open("w") as out:
                for size in sizes:
                    telemetry.phase(f"throughput_bs{size}")
                    adapter.reset_cache()
                    t = time.perf_counter()
                    failed = 0
                    for start in range(0, len(states), size):
                        failed += sum(isinstance(r, str) for r in call(adapter, states[start:start + size], questions))
                    seconds = time.perf_counter() - t
                    out.write(json.dumps({"batch_size": size, "rows": len(states), "failed": failed,
                                          "seconds": round(seconds, 4),
                                          "rows_per_s": round(len(states) / seconds, 3)}) + "\n")
                    out.flush()
            print(f"[{args.model}] throughput sweep: sizes {sizes}", flush=True)

        # 5. main pass (resumable)
        telemetry.phase("main")
        adapter.reset_cache()
        todo = [r for r in rows if r["id"] not in done]
        batches_path = run_dir / "batches.jsonl"
        first_batch = len(read_jsonl(batches_path))
        new_file = not answers_path.exists()
        started, n_done, n_failed = time.perf_counter(), 0, 0
        with answers_path.open("a") as out, batches_path.open("a") as bout:
            if new_file:
                out.write(json.dumps({"identity": identity}) + "\n")
            size = adapter.batch
            for b, start in enumerate(range(0, len(todo), size), start=first_batch):
                window = todo[start:start + size]
                t_start = time.time()
                t = time.perf_counter()
                results = call(adapter, [state_for(r, args.max_doc_chars) for r in window], questions)
                seconds = time.perf_counter() - t
                ok = failed = tokens = 0
                for row, result in zip(window, results):
                    if isinstance(result, str):
                        failed += 1
                        fail(row["id"], "main", result)
                        continue
                    answers, info = result
                    rec = {"id": row["id"], **row_features(row, args.max_doc_chars),
                           "prompt_tokens": info.get("prompt_tokens"), "answers": answers, "batch": b}
                    tokens += total_tokens(info) or 0
                    done[row["id"]] = rec
                    out.write(json.dumps(rec) + "\n")
                    ok += 1
                out.flush()
                bout.write(json.dumps({"batch": b, "rows": len(window), "ok": ok, "failed": failed,
                                       "prompt_tokens": tokens, "seconds": round(seconds, 4),
                                       "t_start": t_start, "t_end": time.time()}) + "\n")
                bout.flush()
                n_done += ok
                n_failed += failed
                elapsed = time.perf_counter() - started
                print(f"[{args.model}] {min(start + size, len(todo))}/{len(todo)} rows "
                      f"({(start + len(window)) / max(elapsed, 1e-9):.2f} rows/s, {n_failed} failed)", flush=True)
        session["main_rows_scored"] = n_done
        session["main_rows_failed"] = n_failed
        session["main_seconds"] = round(time.perf_counter() - started, 3)
        complete = all(r["id"] in done for r in rows)
        run["status"] = "complete" if complete else "incomplete"
    except BaseException as exc:   # record the failure (also Ctrl-C), then re-raise
        run["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "error"
        session["error"] = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        raise
    finally:
        telemetry.stop()
        failures.close()
        session["finished_at"] = now()
        session["telemetry"] = summarize_telemetry(telemetry.samples)
        write_json(run_path, run)
        try:
            adapter.close()
        except Exception:
            pass

    from core.report import build_run_report
    report = build_run_report(run_dir)
    print(f"[{args.model}] status {run['status']}; report: {run_dir / 'report.json'}", flush=True)
    if report.get("accuracy"):
        acc = report["accuracy"]
        print(f"[{args.model}] macro balanced accuracy {acc['macro_balanced_accuracy']}", flush=True)


if __name__ == "__main__":
    main()
