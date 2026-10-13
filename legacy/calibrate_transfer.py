#!/usr/bin/env python3
"""Fit each transfer model's own threshold on Jev's development rows, then apply it once to its test cache.

The transfer runs (JPT-9B, Rune-26B, Winnow-12B NVFP4 and bf16) use the threshold 0.30, fitted to
Jev's probabilities. This script gives each model the treatment Jev had, in two separate phases:

    python calibrate_transfer.py fit      # dev caches only -> results/runs/<model>-dev/threshold.json
    python calibrate_transfer.py apply    # frozen thresholds -> test caches, once

`fit` reads nothing from the test split. It scores each model's complete development cache
(`<model>/benchmark.py --split dev`) with that model's own rule and default temperature, and fits
the threshold with metrics.tune_threshold: the grid, objective and tie-break used for Jev, on the
same 4,520 rows (Jev's combined_dev_fit). Only the threshold is fitted; temperatures stay fixed.

`apply` refuses a model without a threshold file, or whose development cache changed after the fit.
It scores the existing complete test cache (no new inference) at 0.30 and at the fitted threshold,
writes results/runs/<model>/report_dev_threshold.json and results/runs/dev_threshold_summary.md,
and leaves the original report.json untouched. Models without a complete cache are skipped.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from metrics import metrics, tune_threshold  # noqa: E402

MODELS = ["jpt-9b", "rune-26b", "winnow-12b", "winnow-12b-bf16"]   # folder = default run name
FROZEN_THRESHOLD = 0.30


def load_module(folder, root=ROOT):
    spec = importlib.util.spec_from_file_location(f"bench_{folder.replace('-', '_')}",
                                                  root / folder / "benchmark.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_cache(path):
    """-> (header identity, {id: record}); the benchmarks' own format."""
    with path.open() as handle:
        identity = json.loads(handle.readline())["identity"]
        records = {}
        for line in handle:
            if line.strip():
                record = json.loads(line)
                records[record["id"]] = record
    return identity, records


def score(module, record):
    """The model's own frozen rule at its default temperature (logit caches) or as cached (probabilities)."""
    if "logits" in record:
        return module.decide(record["logits"], module.TEMPERATURE)
    return module.decide(record["answers"])


def scored(module, path, rows, check):
    """-> (scores, identity, cache sha256) or a reason string when the cache is missing, foreign or incomplete."""
    if not path.exists():
        return f"no cache at {path}"
    identity, records = read_cache(path)
    problem = check(identity)
    if problem:
        return f"{path}: {problem}"
    missing = sum(r["id"] not in records for r in rows)
    if missing:
        return f"{path}: incomplete, {missing} of {len(rows)} rows missing"
    return ([score(module, records[r["id"]]) for r in rows], identity,
            hashlib.sha256(path.read_bytes()).hexdigest())


def summary_metrics(result, short):
    return {
        "macro_balanced_accuracy": result["macro_balanced_accuracy"],
        "macro_roc_auc": sum(v["roc_auc"] for v in result["by_source"].values()) / len(result["by_source"]),
        "pooled": {k: result[k] for k in ["n", "balanced_accuracy", "accuracy", "roc_auc",
                                          "supported_recall", "unsupported_recall"]},
        "by_source": {short.get(k, k): v for k, v in result["by_source"].items()},
    }


def fit(runs, models):
    from dev_split import load_dev_rows
    rows, dev_hash = load_dev_rows()
    for folder in models:
        module = load_module(folder)
        run_dir = runs / f"{folder}-dev"
        out = run_dir / "threshold.json"

        def check(identity):
            if identity.get("split") != "dev" or identity.get("data_sha256") != dev_hash:
                return "not a development cache of the verified rows"
            if identity.get("limit"):
                return "subset (--limit) cache"
        result = scored(module, run_dir / "answers.jsonl", rows, check)
        if isinstance(result, str):
            print(f"{folder}: skipped ({result})")
            continue
        scores, identity, cache_hash = result
        if out.exists():
            previous = json.loads(out.read_text())
            if previous["dev_cache_sha256"] != cache_hash:
                raise SystemExit(f"{out} was fitted on a different development cache; refusing to refit silently")
            print(f"{folder}: already fitted, threshold {previous['threshold']:.2f}")
            continue
        tuned = tune_threshold(rows, scores)
        record = {
            "model": folder, "threshold": tuned["threshold"],
            "temperature": getattr(module, "TEMPERATURE", None),
            "objective": "development macro balanced accuracy; grid .01-.99; ties: nearest .5, then larger",
            "rows": len(rows), "dev_data_sha256": dev_hash, "dev_cache_sha256": cache_hash,
            "cache_identity": identity, "fitted_at": datetime.now(timezone.utc).isoformat(),
            "dev_at_fitted": summary_metrics(tuned, module.SHORT),
            "dev_at_frozen_030": summary_metrics(metrics(rows, scores, FROZEN_THRESHOLD), module.SHORT),
        }
        out.write_text(json.dumps(record, indent=2) + "\n")
        print(f"{folder}: threshold {tuned['threshold']:.2f}, dev macro BAcc "
              f"{tuned['macro_balanced_accuracy'] * 100:.2f} (at 0.30: "
              f"{record['dev_at_frozen_030']['macro_balanced_accuracy'] * 100:.2f})")


def apply(runs, models, test_data):
    lines, per_source = [], []
    jev = json.loads((ROOT / "results/test_metrics.json").read_text())["selected"]
    frozen = json.loads((ROOT / "results/frozen.json").read_text())
    jev_dev = frozen["combined_dev_fit"]
    for folder in models:
        module = load_module(folder)
        threshold_path = runs / f"{folder}-dev" / "threshold.json"
        if not threshold_path.exists():
            print(f"{folder}: skipped (no {threshold_path}; run `fit` first)")
            continue
        fitted = json.loads(threshold_path.read_text())
        dev_cache = runs / f"{folder}-dev" / "answers.jsonl"
        if hashlib.sha256(dev_cache.read_bytes()).hexdigest() != fitted["dev_cache_sha256"]:
            raise SystemExit(f"{dev_cache} changed after the threshold was fitted")
        data = module.resolve_data(test_data)
        data_hash = hashlib.sha256(data.read_bytes()).hexdigest()
        rows = module.load_rows(data, 0)

        def check(identity):
            if identity.get("split") or identity.get("data_sha256") != data_hash or identity.get("limit"):
                return "not a full test cache of the pinned data"
        result = scored(module, runs / folder / "answers.jsonl", rows, check)
        if isinstance(result, str):
            print(f"{folder}: skipped ({result})")
            continue
        scores, identity, cache_hash = result
        at_frozen = summary_metrics(metrics(rows, scores, FROZEN_THRESHOLD), module.SHORT)
        at_fitted = summary_metrics(metrics(rows, scores, fitted["threshold"]), module.SHORT)
        report = {
            "model": folder, "threshold": fitted["threshold"], "threshold_source": str(threshold_path),
            "temperature": fitted["temperature"],
            "setting": "threshold fitted on Jev's development rows for this model, applied once to the test cache",
            "test_cache_sha256": cache_hash, "cache_identity": identity,
            "applied_at": datetime.now(timezone.utc).isoformat(),
            "test_at_fitted": at_fitted, "test_at_frozen_030": at_frozen,
        }
        (runs / folder / "report_dev_threshold.json").write_text(json.dumps(report, indent=2) + "\n")
        name = f"{module.NAME}" + (f" (T={fitted['temperature']:g})" if fitted["temperature"] else "")
        frozen_bacc, fitted_bacc = at_frozen["macro_balanced_accuracy"], at_fitted["macro_balanced_accuracy"]
        lines.append(f"| {name} | {fitted['threshold']:.2f} | "
                     f"{fitted['dev_at_fitted']['macro_balanced_accuracy'] * 100:.2f} | {frozen_bacc * 100:.2f} | "
                     f"**{fitted_bacc * 100:.2f}** | {(fitted_bacc - frozen_bacc) * 100:+.2f} | "
                     f"{at_fitted['macro_roc_auc'] * 100:.2f} |")
        per_source.append((name, at_frozen["by_source"], at_fitted["by_source"]))
        print(f"{folder}: test macro BAcc {frozen_bacc * 100:.2f} at 0.30 -> {fitted_bacc * 100:.2f} "
              f"at {fitted['threshold']:.2f}")
    if not lines:
        raise SystemExit("Nothing applied")

    jev_auc = sum(v["roc_auc"] for v in jev["by_source"].values()) / len(jev["by_source"])
    columns = list(per_source[0][1])
    text = [
        "# Transfer models with their own development threshold",
        "",
        "Threshold fitted per model on Jev's 4,520 development rows (`calibrate_transfer.py fit`), then applied "
        "once to the existing test cache. Temperatures are each model's default; only the threshold is fitted. "
        "Macro = unweighted mean over the 11 sources.",
        "",
        "| Model | Dev threshold | Dev macro BAcc | Test BAcc @0.30 | Test BAcc @dev threshold | Δ | Test macro AUC |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| jev-1.13.0 (reference) | {frozen['threshold']:.2f} | "
        f"{jev_dev['macro_balanced_accuracy'] * 100:.2f} | — | **{jev['macro_balanced_accuracy'] * 100:.2f}** | — | "
        f"{jev_auc * 100:.2f} |",
        *lines,
        "",
        "## Test balanced accuracy per source (0.30 → development threshold)",
        "",
        "| Model | " + " | ".join(columns) + " |",
        "|---|" + "---:|" * len(columns),
        *[f"| {name} | " + " | ".join(f"{a[c]['balanced_accuracy'] * 100:.1f} → {b[c]['balanced_accuracy'] * 100:.1f}"
                                      for c in columns) + " |" for name, a, b in per_source],
        "",
    ]
    (runs / "dev_threshold_summary.md").write_text("\n".join(text))
    print(f"Summary: {runs / 'dev_threshold_summary.md'}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("phase", choices=["fit", "apply"])
    p.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    p.add_argument("--runs", type=Path, default=ROOT / "results/runs", help="folder holding the run caches")
    default_test = ROOT / "data/test.parquet"
    p.add_argument("--test-data", type=Path,
                   default=default_test if default_test.exists() else ROOT / "lytang___llm-aggre_fact",
                   help="apply only: test.parquet, the test .arrow or the HF cache directory")
    args = p.parse_args()
    if args.phase == "fit":
        fit(args.runs, args.models)
    else:
        apply(args.runs, args.models, args.test_data)


if __name__ == "__main__":
    main()
