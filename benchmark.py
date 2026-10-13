#!/usr/bin/env python3
"""Benchmark decision models on a dataset split, with the same questions for every model.

    python benchmark.py --model d1-3b --gpu 2                       # LLM-AggreFact test split
    python benchmark.py --model d1-3b jpt-9b clm-8b --gpu 2         # several models, one after another
    python benchmark.py --dataset prova --split test.json --model d1-3b --gpu 0
    python benchmark.py --model jpt-9b --gpu 0 --limit 32           # quick check on 32 rows
    python benchmark.py --list                                      # available models

Each model runs in its own process, strictly one after the other (never two models on the GPU at
the same time, which would distort latency and memory), on the single GPU given by --gpu
(the nvidia-smi index). Results go to results/<dataset>/<split>/<model>_<params hash>_<questions hash>/;
the question set is recorded in results/<dataset>/<split>/hash_questions.json and the model's options in
results/<dataset>/<split>/hash_<model>.json. A run can be interrupted and resumed with the same command.
Reports are rebuilt without inference by report.py.

Model options (temperature, context, offload, batch, revisions, ...) come from config/models/<model>.json;
the command line overrides them:
  --opt key=value           for every model that has the option (e.g. --opt batch=16)
  --opt MODEL:key=value     for one model only (e.g. --opt rune-26b:cpu_offload_gb=18)
  --model-config FILE       another options file (one model only)
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.adapter import available, load_adapter_class  # noqa: E402

DEFAULT_DATASET = "lytang--LLM-AggreFact"
DEFAULT_QUESTIONS = ROOT / "config/questions/claim_support.json"


def options_for(model, pairs, defaults):
    out = []
    for pair in pairs:
        scope, _, rest = pair.partition(":") if ":" in pair.split("=", 1)[0] else ("", "", pair)
        key = rest.split("=", 1)[0]
        if scope and scope != model:
            continue
        if not scope and key not in defaults:
            continue   # a global option this model does not have
        out.append(rest)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", nargs="+", help="one or more model folders under models/")
    p.add_argument("--gpu", help="nvidia-smi index of the single GPU to use (required)")
    p.add_argument("--dataset", default=DEFAULT_DATASET, help="folder under dataset/ (or a cached Hub dataset)")
    p.add_argument("--split", default="test", help="split name (test, dev, ...) or a file in the dataset folder")
    p.add_argument("--questions", default=str(DEFAULT_QUESTIONS), help="question set (JSON)")
    p.add_argument("--limit", type=int, default=0, help="deterministic subset of N rows (0 = all)")
    p.add_argument("--model-config", help="model options file (one model only; default config/models/<model>.json)")
    p.add_argument("--opt", action="append", default=[], help="model option, key=value or MODEL:key=value (overrides the config file)")
    p.add_argument("--batch", type=int, help="rows per batch in the main pass (default: each model's own)")
    p.add_argument("--max-doc-chars", type=int, default=80000, help="document cap (75%% head + 25%% tail; 0 = none)")
    p.add_argument("--latency-sample", type=int, default=200, help="rows timed one at a time")
    p.add_argument("--throughput-sizes", default="1,8,32", help="batch sizes of the throughput sweep ('' = skip)")
    p.add_argument("--throughput-sample", type=int, default=64, help="rows per throughput measurement")
    p.add_argument("--no-profile", action="store_true", help="skip the latency probe and throughput sweep")
    p.add_argument("--allow-busy-gpu", action="store_true", help="run even if other processes use the GPU")
    p.add_argument("--tag", default="", help="suffix for the result folder, to keep a variant apart")
    p.add_argument("--list", action="store_true", help="list available models and exit")
    args = p.parse_args()

    if args.list:
        for name in available():
            cls = load_adapter_class(name)
            print(f"{name:18s} {cls.NAME} ({cls.SIZE}, {cls.KIND})")
        return
    if not args.model or args.gpu is None:
        p.error("--model and --gpu are required")
    if args.model_config and len(args.model) > 1:
        p.error("--model-config applies to one model; edit config/models/<model>.json or use --opt MODEL:key=value")
    unknown = [m for m in args.model if m not in available()]
    if unknown:
        p.error(f"unknown models {unknown}; available: {', '.join(available())}")

    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID")
    env.setdefault("HF_HUB_OFFLINE", "1")                     # everything comes from local caches
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")        # no nvcc needed; nothing is sampled
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")   # fork after OpenMP hangs vLLM's engine

    outcome = {}
    for model in args.model:
        defaults = load_adapter_class(model).DEFAULTS
        cmd = [sys.executable, str(ROOT / "core/runner.py"), "--model", model, "--dataset", args.dataset,
               "--split", args.split, "--questions", args.questions, "--limit", str(args.limit),
               "--max-doc-chars", str(args.max_doc_chars), "--latency-sample", str(args.latency_sample),
               "--throughput-sizes", args.throughput_sizes, "--throughput-sample", str(args.throughput_sample),
               "--gpu", str(args.gpu), "--tag", args.tag]
        for pair in options_for(model, args.opt, defaults):
            cmd += ["--opt", pair]
        if args.batch:
            cmd += ["--batch", str(args.batch)]
        if args.model_config:
            cmd += ["--model-config", args.model_config]
        if args.no_profile:
            cmd.append("--no-profile")
        if args.allow_busy_gpu:
            cmd.append("--allow-busy-gpu")
        print(f"\n=== {model} on GPU {args.gpu}: {args.dataset}/{args.split} ===", flush=True)
        started = time.time()
        code = subprocess.run(cmd, env=env, cwd=ROOT).returncode
        outcome[model] = (code, time.time() - started)
        if code in (130, -2):   # Ctrl-C: stop the whole sequence
            break

    print("\n=== summary ===")
    for model, (code, seconds) in outcome.items():
        print(f"{model:18s} {'ok' if code == 0 else f'exit {code}'}  ({seconds / 60:.1f} min)")
    subprocess.run([sys.executable, str(ROOT / "report.py"), "--dataset", args.dataset, "--split", args.split,
                    "--questions", args.questions], cwd=ROOT)
    sys.exit(0 if all(code == 0 for code, _ in outcome.values()) else 1)


if __name__ == "__main__":
    main()
