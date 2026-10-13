"""Build report.json for one run folder from its saved files (no inference).

Everything is recomputed from answers.jsonl, latency.jsonl, throughput.jsonl, batches.jsonl and
run.json, so the decision rule, threshold or temperature can change at any time.
"""
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from . import rules
from .metrics import metrics

# Length buckets (upper edges, the last one open). Fixed so reports of different models line up.
BUCKETS = {
    "claim_chars": [50, 100, 200, 400],
    "doc_chars": [1000, 2000, 4000, 8000, 16000],
    "prompt_tokens": [512, 1024, 2048, 4096, 8192],
}


def read_jsonl(path, skip_header=False):
    path = Path(path)
    if not path.exists():
        return []
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    if skip_header and lines and "identity" in json.loads(lines[0]):
        lines = lines[1:]
    return [json.loads(l) for l in lines]


def bucket_label(value, edges):
    if value is None:
        return "unknown"
    low = 0
    for edge in edges:
        if value < edge:
            return f"{low}-{edge}"
        low = edge
    return f">={low}"


def prompt_tokens(rec):
    """Tokens of the longest single prompt of a row (or the request total if the model reports only that)."""
    tokens = rec.get("prompt_tokens")
    if isinstance(tokens, dict):
        values = [v for v in tokens.values() if v]
        return max(values) if values else None
    return tokens


def stats(values):
    if not values:
        return None
    a = np.asarray(values, dtype=float)
    return {"n": int(a.size), "mean": round(float(a.mean()), 5), "p50": round(float(np.percentile(a, 50)), 5),
            "p90": round(float(np.percentile(a, 90)), 5), "p95": round(float(np.percentile(a, 95)), 5),
            "p99": round(float(np.percentile(a, 99)), 5), "min": round(float(a.min()), 5), "max": round(float(a.max()), 5)}


def binary(labels, scores, threshold):
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    pred = scores > threshold
    out = {"n": int(labels.size), "accuracy": float((pred == labels).mean()) if labels.size else None}
    pos, neg = labels == 1, labels == 0
    if pos.any() and neg.any():
        tpr, tnr = float(pred[pos].mean()), float((~pred[neg]).mean())
        out.update(balanced_accuracy=(tpr + tnr) / 2, supported_recall=tpr, unsupported_recall=tnr,
                   roc_auc=float(roc_auc_score(labels, scores)))
    return out


def choose_decision(run, decision=None):
    if decision is not None:
        return decision
    if run.get("default_decision"):
        return run["default_decision"]
    return run["questions"].get("decision")


def build_run_report(run_dir, decision=None, threshold=None, temperature=None, write=True):
    run_dir = Path(run_dir)
    run = json.loads((run_dir / "run.json").read_text())
    recs = read_jsonl(run_dir / "answers.jsonl", skip_header=True)
    data = run["data"]
    expected = min(data["limit"], data["rows_total"]) if data.get("limit") else data["rows_total"]
    if run.get("model_kind") == "baseline":
        decision = None   # a baseline ignores the questions: only its own rule applies (a threshold override still does)
    if temperature is not None and recs and any(a.get("logits") is None for a in recs[0]["answers"].values()):
        temperature = "n/a"   # the model stores no logits: its answers cannot be re-tempered
    rule = choose_decision(run, decision)
    if rule is not None and threshold is not None:
        rule = {**rule, "threshold": threshold}
    report = {"run": run_dir.name, "model": run["identity"]["model"], "model_name": run.get("model_name"),
              "model_size": run.get("model_size"), "model_kind": run.get("model_kind"),
              "dataset": data["dataset"], "split": data["split"], "questions_hash": run["identity"]["questions_hash"],
              "params_hash": run["identity"].get("params_hash"), "options": run["identity"].get("options"),
              "status": run.get("status"), "decision": rule, "temperature_override": temperature,
              "coverage": {"rows_expected": expected, "rows_scored": len(recs),
                           "complete": len(recs) >= expected and run.get("status") == "complete",
                           "failures_logged": len(read_jsonl(run_dir / "failures.jsonl"))}}

    if temperature == "n/a":
        temperature = None

    # ---- accuracy under the decision rule
    if recs and rule is not None:
        scored = []
        for r in recs:
            try:
                scored.append((r, rules.score(r["answers"], rule, temperature)))
            except (KeyError, ValueError):
                continue
        rows = [{"id": r["id"], "label": r["label"], "dataset": r["source"]} for r, _ in scored]
        scores = [s for _, s in scored]
        if rows:
            m = metrics(rows, scores, rule["threshold"])
            report["accuracy"] = {k: m[k] for k in ("macro_balanced_accuracy", "balanced_accuracy", "accuracy",
                                                     "roc_auc", "supported_recall", "unsupported_recall", "n")}
            report["accuracy"]["macro_roc_auc"] = _macro_auc(rows, scores)
            report["by_source"] = m["by_source"]
            report["by_length"] = {}
            for feature, edges in BUCKETS.items():
                groups = {}
                for (r, s) in scored:
                    value = prompt_tokens(r) if feature == "prompt_tokens" else r.get(feature)
                    groups.setdefault(bucket_label(value, edges), ([], []))
                    groups[bucket_label(value, edges)][0].append(r["label"])
                    groups[bucket_label(value, edges)][1].append(s)
                report["by_length"][feature] = {k: binary(l, s, rule["threshold"]) for k, (l, s) in _ordered(groups, edges)}

    # ---- every question on its own (threshold-free AUC for yes/no questions)
    if recs:
        per_q = {}
        labels = [r["label"] for r in recs]
        for qid, ans in recs[0]["answers"].items():
            entry = {"type": ans["type"]}
            if ans["type"] == "noul":
                values = [rules.take(r["answers"][qid], "yes", temperature) for r in recs if qid in r["answers"]]
                entry["mean_p_yes"] = round(float(np.mean(values)), 4)
                if len(set(labels)) == 2 and len(values) == len(labels):
                    entry["auc_yes_vs_supported"] = round(float(roc_auc_score(labels, values)), 4)
            else:
                keys = ans["keys"]
                entry["mean_probs"] = {k: round(float(np.mean([r["answers"][qid]["probs"][k] for r in recs])), 4) for k in keys}
            per_q[qid] = entry
        report["per_question"] = per_q

    # ---- latency (one row at a time, cold cache)
    lat = read_jsonl(run_dir / "latency.jsonl")
    ok = [r for r in lat if r.get("ok")]
    if lat:
        report["latency"] = {"overall_seconds": stats([r["seconds"] for r in ok]), "failed": len(lat) - len(ok),
                             "by_source": {k: stats(v) for k, v in _group(ok, lambda r: r["source"]).items()},
                             "by_doc_chars": {k: stats(v) for k, v in _ordered_group(ok, "doc_chars").items()},
                             "by_prompt_tokens": {k: stats(v) for k, v in _ordered_group(ok, "prompt_tokens").items()}}

    # ---- throughput
    sweep = read_jsonl(run_dir / "throughput.jsonl")
    batches = read_jsonl(run_dir / "batches.jsonl")
    nq = len(run["questions"]["questions"])
    seconds = sum(b["seconds"] for b in batches)
    rows_ok = sum(b["ok"] for b in batches)
    report["throughput"] = {"batch_size_sweep": sweep,
                            "main": {"rows": rows_ok, "seconds": round(seconds, 2),
                                     "rows_per_s": round(rows_ok / seconds, 3) if seconds else None,
                                     "questions_per_s": round(rows_ok * nq / seconds, 3) if seconds else None,
                                     "prompt_tokens_per_s": round(sum(b["prompt_tokens"] for b in batches) / seconds, 1) if seconds else None,
                                     "batch_seconds": stats([b["seconds"] for b in batches]),
                                     "batch_size": run["identity"]["options"].get("batch")}}

    # ---- resources and start-up (summed over sessions: a resumed run has several)
    sessions = run.get("sessions", [])
    main_energy = sum((s.get("telemetry") or {}).get("main", {}).get("energy_j", 0) or 0 for s in sessions)
    main_rows = sum(s.get("main_rows_scored", 0) or 0 for s in sessions)
    peaks = [ph.get("gpu_mem_peak_mb") for s in sessions for ph in (s.get("telemetry") or {}).values() if ph.get("gpu_mem_peak_mb")]
    rss = [ph.get("rss_peak_mb") for s in sessions for ph in (s.get("telemetry") or {}).values() if ph.get("rss_peak_mb")]
    main_phases = [s["telemetry"]["main"] for s in sessions if (s.get("telemetry") or {}).get("main")]
    report["resources"] = {
        "load_seconds": [s.get("load_seconds") for s in sessions],
        "first_call_seconds": [s.get("first_call_seconds") for s in sessions],
        "gpu_mem_peak_mb": max(peaks) if peaks else None,
        "rss_peak_mb": max(rss) if rss else None,
        "main_energy_j": round(main_energy, 1),
        "energy_per_1k_rows_j": round(main_energy / main_rows * 1000, 1) if main_rows and main_energy else None,
        "main_power_mean_w": _mean([p.get("power_mean_w") for p in main_phases]),
        "main_power_max_w": max([p["power_max_w"] for p in main_phases if p.get("power_max_w")], default=None),
        "main_gpu_util_mean": _mean([p.get("gpu_util_mean") for p in main_phases]),
        "main_temp_max_c": max([p["temp_max_c"] for p in main_phases if p.get("temp_max_c")], default=None),
        "main_sm_clock_min_mhz": min([p["sm_clock_min_mhz"] for p in main_phases if p.get("sm_clock_min_mhz")], default=None),
        "gpu": run.get("environment", {}).get("gpu"),
    }
    if write and decision is None and threshold is None and temperature is None:
        (run_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _macro_auc(rows, scores):
    groups = {}
    for row, s in zip(rows, scores):
        groups.setdefault(row["dataset"], ([], []))
        groups[row["dataset"]][0].append(row["label"])
        groups[row["dataset"]][1].append(s)
    aucs = [roc_auc_score(l, s) for l, s in groups.values() if len(set(l)) == 2]
    return float(np.mean(aucs)) if aucs else None


def _mean(values):
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 1) if values else None


def _group(recs, key):
    out = {}
    for r in recs:
        out.setdefault(key(r), []).append(r["seconds"])
    return out


def _ordered(groups, edges):
    order = [bucket_label(e - 1, edges) for e in edges] + [bucket_label(math.inf, edges), "unknown"]
    return [(k, groups[k]) for k in order if k in groups]


def _ordered_group(recs, feature):
    edges = BUCKETS[feature]
    groups = {}
    for r in recs:
        value = prompt_tokens(r) if feature == "prompt_tokens" else r.get(feature)
        groups.setdefault(bucket_label(value, edges), []).append(r["seconds"])
    return dict(_ordered(groups, edges))
