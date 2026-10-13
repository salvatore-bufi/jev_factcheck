#!/usr/bin/env python3
"""Rebuild reports and comparison tables from saved results (no inference).

    python report.py                                             # LLM-AggreFact test, every question set
    python report.py --threshold 0.5                             # same answers, another threshold
    python report.py --rule support_simple                       # judge by one yes/no question only
    python report.py --decision my_rule.json                     # any rule (see core/rules.py)
    python report.py --temperature 2                             # re-temper models that store logits
    python report.py --calibrate-on dev                          # per-model threshold fitted on the dev split

It refreshes every run's report.json (default rule) and writes, per question set,
results/<dataset>/<split>/summary_<hash>.md with the accuracy table (table.md format: best in bold,
second underlined), per-question AUC, latency, throughput, resources and accuracy by document length.
With a rule, threshold or temperature override the summary is written as summary_<hash>_custom.md
and the per-run report.json files are left as they are.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core import RESULTS, questions as qs, rules  # noqa: E402
from core.metrics import tune_threshold  # noqa: E402
from core.report import BUCKETS, bucket_label, build_run_report, read_jsonl  # noqa: E402

# LLM-AggreFact source names -> the column names of table.md, in its order.
SHORT = {"AggreFact-CNN": "CNN", "AggreFact-XSum": "XSum", "TofuEval-MediaS": "MediaS", "TofuEval-MeetB": "MeetB",
         "Wice": "WiCE", "Reveal": "REVEAL", "ClaimVerify": "ClaimVerify", "FactCheck-GPT": "FactCheck",
         "ExpertQA": "ExpertQA", "Lfqa": "LFQA", "RAGTruth": "RAGTruth"}
ORDER = list(SHORT.values())


def runs_in(split_dir, qhash=None):
    out = []
    for run_json in sorted(Path(split_dir).glob("*/run.json")):
        run = json.loads(run_json.read_text())
        if qhash is None or run["identity"]["questions_hash"] == qhash:
            out.append(run_json.parent)
    return out


def calibrated_threshold(test_dir, dataset, dev_split, rule):
    """Fit the rule's threshold on the same model + question set's dev run (complete only)."""
    dev_dir = RESULTS / dataset / dev_split / test_dir.name
    if not (dev_dir / "run.json").exists():
        return None, f"no {dev_split} run"
    run = json.loads((dev_dir / "run.json").read_text())
    if run.get("status") != "complete":
        return None, f"{dev_split} run incomplete"
    recs = read_jsonl(dev_dir / "answers.jsonl", skip_header=True)
    rows = [{"id": r["id"], "label": r["label"], "dataset": r["source"]} for r in recs]
    scores = [rules.score(r["answers"], rule) for r in recs]
    threshold, _ = tune_threshold(rows, scores)
    return float(threshold), None


def pct(x):
    return None if x is None else round(100 * x, 2)


def highlight(rows, columns):
    """Bold the best and underline the second best value of each numeric column."""
    for col in columns:
        values = sorted({r[col] for r in rows if isinstance(r.get(col), (int, float))}, reverse=True)
        for r in rows:
            v = r.get(col)
            if isinstance(v, (int, float)):
                text = f"{v:.2f}"
                if values and v == values[0]:
                    text = f"**{text}**"
                elif len(values) > 1 and v == values[1]:
                    text = f"<u>{text}</u>"
                r[col] = text
    return rows


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|---|" + "---:|" * (len(header) - 1)]
    for r in rows:
        lines.append("| " + " | ".join("" if r.get(h) is None else str(r.get(h)) for h in header) + " |")
    return "\n".join(lines)


def fmt(x, digits=3):
    if x is None:
        return None
    return f"{x:.{digits}f}" if isinstance(x, float) else x


def label(rep, variants):
    name = rep["model_name"] or rep["model"]
    if variants.get(rep["model"], 0) > 1:      # several option sets / subsets of the same model: tell them apart
        name += f" [{rep.get('params_hash')}]"
        suffix = rep["run"].split(f"_{rep['questions_hash']}", 1)[-1]   # e.g. -limit32 or -<tag>
        name += f" {suffix.lstrip('-')}" if suffix else ""
    return name


def summary(dataset, split, qhash, reports, entry, note):
    variants = {}
    for rep in reports:
        variants[rep["model"]] = variants.get(rep["model"], 0) + 1
    sources = sorted({s for rep in reports for s in rep.get("by_source", {})})
    cols = [SHORT.get(s, s) for s in sources]
    cols = [c for c in ORDER if c in cols] + sorted(c for c in cols if c not in ORDER)
    inverse = {SHORT.get(s, s): s for s in sources}
    acc_rows = []
    for rep in reports:
        acc = rep.get("accuracy") or {}
        name = label(rep, variants)
        if rep.get("model_kind") == "baseline":
            name += " (baseline, ignores the questions)"
        if not rep["coverage"]["complete"]:
            name += f" (incomplete: {rep['coverage']['rows_scored']}/{rep['coverage']['rows_expected']})"
        row = {"Model": name, "Size": rep.get("model_size"), "Threshold": fmt(rep["decision"]["threshold"], 2) if rep.get("decision") else None,
               "Average": pct(acc.get("macro_balanced_accuracy")), "Macro AUC": pct(acc.get("macro_roc_auc"))}
        for c in cols:
            row[c] = pct((rep.get("by_source", {}).get(inverse[c]) or {}).get("balanced_accuracy"))
        acc_rows.append(row)
    acc_rows.sort(key=lambda r: -(r["Average"] or 0))
    acc_rows = highlight(acc_rows, ["Average", "Macro AUC"] + cols)

    q_ids = list(entry["questions"])
    q_rows = [{"Model": label(rep, variants),
               **{q: fmt((rep.get("per_question", {}).get(q) or {}).get("auc_yes_vs_supported")) for q in q_ids}}
              for rep in reports]

    perf_rows = []
    for rep in reports:
        lat = (rep.get("latency") or {}).get("overall_seconds") or {}
        main = (rep.get("throughput") or {}).get("main") or {}
        res = rep.get("resources") or {}
        perf_rows.append({"Model": label(rep, variants),
                          "Latency p50 (s)": fmt(lat.get("p50")), "p95 (s)": fmt(lat.get("p95")), "p99 (s)": fmt(lat.get("p99")),
                          "Rows/s": fmt(main.get("rows_per_s"), 2), "Questions/s": fmt(main.get("questions_per_s"), 2),
                          "Prompt tokens/s": fmt(main.get("prompt_tokens_per_s"), 0),
                          "Load (s)": fmt((res.get("load_seconds") or [None])[0], 1),
                          "First call (s)": fmt((res.get("first_call_seconds") or [None])[0], 2),
                          "GPU mem peak (MB)": res.get("gpu_mem_peak_mb"), "RAM peak (MB)": fmt(res.get("rss_peak_mb"), 0),
                          "Energy / 1k rows (J)": fmt(res.get("energy_per_1k_rows_j"), 0),
                          "Mean power (W)": fmt(res.get("main_power_mean_w"), 0),
                          "Failures": rep["coverage"]["failures_logged"]})

    sweep_sizes = sorted({s["batch_size"] for rep in reports for s in (rep.get("throughput") or {}).get("batch_size_sweep", [])})
    sweep_rows = [{"Model": label(rep, variants),
                   **{f"bs={b}": fmt(next((s["rows_per_s"] for s in rep["throughput"]["batch_size_sweep"] if s["batch_size"] == b), None), 2)
                      for b in sweep_sizes}} for rep in reports]

    edges = BUCKETS["doc_chars"]
    labels = [bucket_label(e - 1, edges) for e in edges] + [bucket_label(float("inf"), edges)]
    len_rows = [{"Model": label(rep, variants),
                 **{f"doc {l}": pct(((rep.get("by_length") or {}).get("doc_chars", {}).get(l) or {}).get("balanced_accuracy")) for l in labels}}
                for rep in reports]
    lat_rows = [{"Model": label(rep, variants),
                 **{f"doc {l}": fmt((((rep.get("latency") or {}).get("by_doc_chars") or {}).get(l) or {}).get("p50")) for l in labels}}
                for rep in reports]

    parts = [f"# {dataset} / {split} — question set `{qhash}` ({entry['name']})", "",
             note, "",
             "## Balanced accuracy (%) per source; Average = unweighted mean over sources", "",
             md_table(["Model", "Size", "Threshold", "Average", "Macro AUC"] + cols, acc_rows), "",
             "## Each question alone: ROC AUC of P(yes) against the label (inverted questions show values below 0.5)", "",
             md_table(["Model"] + q_ids, q_rows), "",
             "## Speed and resources", "",
             "Latency: one row (all questions) at a time, cold cache, on a fixed stratified sample. Rows/s: main pass, batched.", "",
             md_table(list(perf_rows[0]) if perf_rows else ["Model"], perf_rows), ""]
    if sweep_sizes:
        parts += ["## Throughput (rows/s) against batch size", "", md_table(["Model"] + [f"bs={b}" for b in sweep_sizes], sweep_rows), ""]
    parts += ["## Balanced accuracy (%) by document length (characters)", "", md_table(["Model"] + [f"doc {l}" for l in labels], len_rows), "",
              "## Latency p50 (s) by document length (characters)", "", md_table(["Model"] + [f"doc {l}" for l in labels], lat_rows), ""]
    return "\n".join(parts)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="lytang--LLM-AggreFact")
    p.add_argument("--split", default="test")
    p.add_argument("--questions", help="only this question set (JSON); default: every set found")
    p.add_argument("--threshold", type=float, help="override the decision threshold")
    p.add_argument("--decision", help="JSON file with a decision rule (see core/rules.py)")
    p.add_argument("--rule", help="judge by one question: QUESTION or QUESTION:take (e.g. unsupported_detail:no)")
    p.add_argument("--temperature", type=float, help="re-temper answers from their stored logits")
    p.add_argument("--calibrate-on", help="fit each model's threshold on this split (e.g. dev), apply here")
    args = p.parse_args()

    split_dir = RESULTS / args.dataset / args.split
    registry_path = split_dir / "hash_questions.json"
    if not registry_path.exists():
        raise SystemExit(f"No results in {split_dir}")
    registry = json.loads(registry_path.read_text())
    hashes = [qs.questions_hash(qs.load(args.questions)["questions"])] if args.questions else list(registry)

    decision = json.loads(Path(args.decision).read_text()) if args.decision else None
    if args.rule:
        qid, _, how = args.rule.partition(":")
        decision = rules.single_question(qid, how or "yes", args.threshold if args.threshold is not None else 0.5)
    custom = any(x is not None for x in (decision, args.threshold, args.temperature, args.calibrate_on))

    for qhash in hashes:
        dirs = runs_in(split_dir, qhash)
        if not dirs:
            continue
        reports = []
        for run_dir in dirs:
            build_run_report(run_dir)   # refresh report.json with the default rule
            threshold, note = args.threshold, None
            run = json.loads((run_dir / "run.json").read_text())
            rule = decision or run.get("default_decision") or registry[qhash].get("decision") or run["questions"].get("decision")
            if args.calibrate_on and rule:
                threshold, note = calibrated_threshold(run_dir, args.dataset, args.calibrate_on, rule)
                if threshold is None:
                    print(f"{run_dir.name}: no calibration ({note}); using the rule's own threshold")
            rep = build_run_report(run_dir, decision=decision, threshold=threshold, temperature=args.temperature, write=False)
            reports.append(rep)
        note = "Default decision rule of the question set." if not custom else (
            f"Custom report: rule={json.dumps(decision) if decision else 'question set default'}, "
            f"threshold={args.threshold if args.threshold is not None else ('fitted on ' + args.calibrate_on if args.calibrate_on else 'default')}, "
            f"temperature={args.temperature or 'model default'}.")
        text = summary(args.dataset, args.split, qhash, reports, registry[qhash], note)
        out = split_dir / (f"summary_{qhash}_custom.md" if custom else f"summary_{qhash}.md")
        out.write_text(text + "\n")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
