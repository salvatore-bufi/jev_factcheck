#!/usr/bin/env python3
"""Grid search over decision parameters on saved results (no inference).

For every model's run in results/<dataset>/<split>/, recompute the results for every combination of

  * decision rule   the question set's rule (e.g. min of the three checks) and each of its questions alone,
  * temperature     `default` (the probabilities as saved, i.e. the model's own temperature) or a value
                    re-applied to the saved option logits (models without logits only have `default`),
  * threshold       "supported" when the score is strictly above it,

and write a markdown summary (default and best combination per model, then every combination per
model) plus a CSV with everything.

    python grid_search.py
    python grid_search.py --thresholds 0.1,0.2,0.3,0.4,0.5 --temperatures default,0.5,1,2
    python grid_search.py --thresholds 0.05:0.95:0.05 --models jpt-9b rune-26b --rules rule,support_simple

The best combination is chosen on the same split it is measured on, so it is an optimistic
estimate: use it to explore, and confirm a choice on another split.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core import RESULTS, questions as qs  # noqa: E402
from core.metrics import metrics  # noqa: E402
from core.report import read_jsonl  # noqa: E402
from report import ORDER, SHORT  # noqa: E402

METRICS = ("macro_balanced_accuracy", "balanced_accuracy", "accuracy", "roc_auc", "macro_roc_auc",
           "supported_recall", "unsupported_recall")
HEADERS = {"macro_balanced_accuracy": "Average", "balanced_accuracy": "Bal. acc.", "accuracy": "Accuracy",
           "roc_auc": "ROC AUC", "macro_roc_auc": "Macro ROC AUC"}


# ------------------------------------------------------------------ parsing

def parse_values(text, allow_default=False):
    """'0.1,0.2' or 'start:stop:step' (inclusive); 'default' kept as None when allowed."""
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if allow_default and part == "default":
            out.append(None)
        elif part.count(":") == 2:
            start, stop, step = (float(x) for x in part.split(":"))
            n = int(math.floor((stop - start) / step + 1e-9)) + 1
            out += [round(start + i * step, 10) for i in range(n)]
        else:
            out.append(float(part))
    return list(dict.fromkeys(out))


# ------------------------------------------------------------------ loading

class Run:
    """One run folder, loaded once into numpy arrays."""

    def __init__(self, run_dir):
        self.dir = Path(run_dir)
        self.info = json.loads((self.dir / "run.json").read_text())
        recs = read_jsonl(self.dir / "answers.jsonl", skip_header=True)
        if not recs:
            raise ValueError("no answers")
        self.ids = [r["id"] for r in recs]
        self.labels = np.asarray([r["label"] for r in recs], dtype=int)
        self.sources = np.asarray([r["source"] for r in recs])
        self.answers = {}
        for qid in recs[0]["answers"]:
            first = recs[0]["answers"][qid]
            keys = first["keys"]
            entry = {"type": first["type"], "keys": keys,
                     "probs": np.asarray([[r["answers"][qid]["probs"][k] for k in keys] for r in recs], dtype=float)}
            if all(r["answers"][qid].get("logits") for r in recs):
                entry["logits"] = np.asarray([r["answers"][qid]["logits"] for r in recs], dtype=float)
            self.answers[qid] = entry
        self.model = self.info["identity"]["model"]
        self.name = self.info.get("model_name") or self.model
        self.kind = self.info.get("model_kind", "decision")
        self.qhash = self.info["identity"]["questions_hash"]
        self.phash = self.info["identity"].get("params_hash")

    def rules(self):
        """{name: rule}: the set's (or the baseline's own) rule, then each of its inputs alone."""
        base = self.info.get("default_decision") or self.info["questions"].get("decision")
        if base is None:
            return {}
        out = {"rule": base}
        if len(base["inputs"]) > 1:
            for item in base["inputs"]:
                out[item["question"]] = {"combine": "single", "inputs": [item], "threshold": base["threshold"]}
        return out

    def take(self, item, temperature):
        """Vector of the value a rule input takes, for every row; None when the temperature is not applicable."""
        ans = self.answers[item["question"]]
        if temperature is None:
            probs = ans["probs"]
        else:
            if "logits" not in ans:
                return None
            z = ans["logits"] / temperature
            z = z - np.max(np.where(np.isfinite(z), z, -np.inf), axis=1, keepdims=True)
            w = np.where(np.isfinite(z), np.exp(z), 0.0)
            probs = w / w.sum(axis=1, keepdims=True)
        keys, how = ans["keys"], item["take"]
        if how in ("yes", "no"):
            p = probs[:, keys.index("true")]
            return p if how == "yes" else 1.0 - p
        if how.startswith("option:"):
            return probs[:, keys.index(how.split(":", 1)[1])]
        if how.startswith("not_option:"):
            return 1.0 - probs[:, keys.index(how.split(":", 1)[1])]
        if how == "score":
            return probs @ np.asarray([int(k) for k in keys], dtype=float) / (len(keys) - 1)
        raise ValueError(f"unknown take {how!r}")

    def score(self, rule, temperature):
        values = [self.take(item, temperature) for item in rule["inputs"]]
        if any(v is None for v in values):
            return None
        stack = np.vstack(values)
        combine = rule["combine"]
        if combine == "min":
            return stack.min(axis=0)
        if combine == "mean":
            return stack.mean(axis=0)
        if combine == "product":
            return stack.prod(axis=0)
        if combine == "single":
            return stack[0]
        raise ValueError(f"unknown combine {combine!r}")


# ------------------------------------------------------------------ metrics (vectorised over thresholds)

def auc(labels, scores):
    return float(roc_auc_score(labels, scores)) if len(set(labels.tolist())) == 2 else None


def evaluate(run, scores, thresholds):
    """One dict per threshold: overall metrics and per-source balanced accuracy / AUC."""
    th = np.asarray(thresholds, dtype=float)
    labels, sources = run.labels, run.sources
    pred = scores[:, None] > th[None, :]                          # [rows, thresholds]

    def rates(mask):
        pos, neg = mask & (labels == 1), mask & (labels == 0)
        tpr = pred[pos].mean(axis=0) if pos.any() else None
        tnr = (~pred[neg]).mean(axis=0) if neg.any() else None
        acc = (pred[mask] == labels[mask][:, None]).mean(axis=0)
        return tpr, tnr, acc

    tpr, tnr, acc = rates(np.ones_like(labels, dtype=bool))
    per_source = {}
    for source in sorted(set(sources.tolist())):
        mask = sources == source
        s_tpr, s_tnr, _ = rates(mask)
        per_source[source] = {"bacc": None if s_tpr is None or s_tnr is None else (s_tpr + s_tnr) / 2,
                              "auc": auc(labels[mask], scores[mask])}
    overall_auc = auc(labels, scores)
    source_aucs = [v["auc"] for v in per_source.values()]
    macro_auc = float(np.mean(source_aucs)) if all(a is not None for a in source_aucs) else None
    baccs = [v["bacc"] for v in per_source.values()]
    macro_bacc = np.mean(baccs, axis=0) if all(b is not None for b in baccs) else None

    out = []
    for j, t in enumerate(thresholds):
        out.append({
            "threshold": float(t),
            "macro_balanced_accuracy": None if macro_bacc is None else float(macro_bacc[j]),
            "balanced_accuracy": None if tpr is None or tnr is None else float((tpr[j] + tnr[j]) / 2),
            "accuracy": float(acc[j]),
            "roc_auc": overall_auc, "macro_roc_auc": macro_auc,
            "supported_recall": None if tpr is None else float(tpr[j]),
            "unsupported_recall": None if tnr is None else float(tnr[j]),
            "by_source": {s: {"bacc": None if v["bacc"] is None else float(v["bacc"][j]), "auc": v["auc"]}
                          for s, v in per_source.items()},
        })
    return out


def check_against_core(run, scores, threshold, row):
    """The vectorised numbers must equal core.metrics.metrics() on the same scores."""
    rows = [{"label": int(l), "dataset": s} for l, s in zip(run.labels, run.sources)]
    ref = metrics(rows, scores.tolist(), threshold)
    pairs = [(row["macro_balanced_accuracy"], ref["macro_balanced_accuracy"]),
             (row["balanced_accuracy"], ref["balanced_accuracy"]), (row["accuracy"], ref["accuracy"])]
    for ours, theirs in pairs:
        if (ours is None) != (theirs is None) or (ours is not None and abs(ours - theirs) > 1e-9):
            raise AssertionError(f"{run.dir.name}: grid metrics differ from core.metrics ({ours} vs {theirs})")


# ------------------------------------------------------------------ search

def search(run, rule_names, temperatures, thresholds):
    rules = run.rules()
    chosen = {k: v for k, v in rules.items() if rule_names is None or k in rule_names or (k == "rule" and "min" in rule_names)}
    if run.kind == "baseline":
        chosen = {"rule": rules["rule"]}            # a baseline has only its own rule
    results, skipped = [], []
    for rname, rule in chosen.items():
        for t in temperatures:
            scores = run.score(rule, t)
            if scores is None:
                skipped.append((rname, t))
                continue
            for row in evaluate(run, scores, thresholds):
                results.append({"rule": rname, "rule_def": rule, "temperature": t, **row})
    return results, skipped


def default_row(run):
    rules = run.rules()
    if "rule" not in rules:
        return None
    rule = rules["rule"]
    scores = run.score(rule, None)
    row = evaluate(run, scores, [rule["threshold"]])[0]
    check_against_core(run, scores, rule["threshold"], row)
    return {"rule": "rule", "rule_def": rule, "temperature": None, **row}


# ------------------------------------------------------------------ output

def pct(x):
    return None if x is None else round(100 * x, 2)


def rule_label(row):
    if row["rule"] != "rule":
        take = row["rule_def"]["inputs"][0]["take"]
        return f"{row['rule']} ({'P(yes)' if take == 'yes' else '1 - P(yes)' if take == 'no' else take})"
    rd = row["rule_def"]
    if rd["combine"] == "single":
        return f"{rd['inputs'][0]['question']} (own rule)"
    return f"{rd['combine']} of {len(rd['inputs'])}"


def source_columns(runs):
    sources = sorted({s for r in runs for s in r.sources.tolist()})
    cols = [SHORT.get(s, s) for s in sources]
    ordered = [c for c in ORDER if c in cols] + sorted(c for c in cols if c not in ORDER)
    inverse = {SHORT.get(s, s): s for s in sources}
    return ordered, inverse


def table_row(name, setting, row, cols, inverse):
    out = {"Model": name, "Setting": setting, "Rule": rule_label(row),
           "T": "default" if row["temperature"] is None else f"{row['temperature']:g}",
           "Threshold": f"{row['threshold']:.2f}"}
    for key, head in HEADERS.items():
        out[head] = pct(row[key])
    for c in cols:
        out[c] = pct((row["by_source"].get(inverse[c]) or {}).get("bacc"))
    return out


def highlight(rows, numeric):
    for col in numeric:
        values = sorted({r[col] for r in rows if isinstance(r.get(col), (int, float))}, reverse=True)
        for r in rows:
            v = r.get(col)
            if isinstance(v, (int, float)):
                r[col] = f"**{v:.2f}**" if values and v == values[0] else f"{v:.2f}"
    return rows


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" if h in ("Model", "Setting", "Rule") else "---:" for h in header) + "|"]
    for r in rows:
        lines.append("| " + " | ".join("" if r.get(h) is None else str(r.get(h)) for h in header) + " |")
    return "\n".join(lines)


def write_outputs(prefix, args, qhash, entry, per_model, runs, elapsed):
    cols, inverse = source_columns(runs)
    header = ["Model", "Setting", "Rule", "T", "Threshold"] + list(HEADERS.values()) + cols
    numeric = list(HEADERS.values()) + cols

    summary = []
    for run, (rows, skipped, default) in per_model.items():
        if default is not None:
            summary.append(table_row(run.label, "default", default, cols, inverse))
        valid = [r for r in rows if r.get(args.select) is not None]
        if valid:
            best = max(valid, key=lambda r: r[args.select])
            summary.append(table_row(run.label, f"best {HEADERS.get(args.select, args.select)}", best, cols, inverse))
    highlight(summary, numeric)

    n_combos = sum(len(rows) for rows, _, _ in per_model.values())
    parts = [f"# Grid search: {args.dataset} / {args.split}, question set `{qhash}` ({entry.get('name', '')})", "",
             f"Thresholds: {', '.join(f'{t:g}' for t in args.threshold_values)}. "
             f"Temperatures: {', '.join('default' if t is None else f'{t:g}' for t in args.temperature_values)}. "
             f"Rules: the question set's rule and each of its questions alone. "
             f"{n_combos} combinations, {elapsed:.1f} s, no inference.", "",
             "Values in %. Average = unweighted mean of the per-source balanced accuracy (the leaderboard metric); "
             "Bal. acc., Accuracy and ROC AUC are pooled over all rows; Macro ROC AUC is the mean of the per-source AUC. "
             "Source columns: balanced accuracy. `default` = the question set's rule, the model's own temperature and "
             "the set's threshold (it matches report.json). Temperatures other than `default` are recomputed from the "
             "saved logits; models without logits (d1-3B, FactCG) only have `default`.", "",
             f"> The `best` rows are chosen on this same split, by {HEADERS.get(args.select, args.select)}: an optimistic "
             "estimate. Use them to explore; confirm a choice on another split.", "",
             "## Summary (best value of each column in bold)", "", md_table(header, summary), ""]
    for run, (rows, skipped, _) in per_model.items():
        ranked = sorted(rows, key=lambda r: (r.get(args.select) is not None, r.get(args.select) or 0), reverse=True)
        if args.top:
            ranked = ranked[:args.top]
        detail = highlight([table_row(run.label, "", r, cols, inverse) for r in ranked], numeric)
        parts += [f"## {run.label}", "", f"Run `{run.dir.name}`; {len(rows)} combinations"
                  + (f" (showing the top {args.top})" if args.top and len(rows) > args.top else "")
                  + (f"; not applicable (no logits): {len(skipped)} rule/temperature pairs" if skipped else "") + ".", "",
                  md_table([h for h in header if h != "Setting"], detail), ""]
    Path(f"{prefix}.md").write_text("\n".join(parts) + "\n")

    with open(f"{prefix}.csv", "w", newline="") as handle:
        sources = sorted({s for r in runs for s in r.sources.tolist()})
        fields = ["model", "model_name", "run", "rule", "rule_definition", "temperature", "threshold", *METRICS] \
            + [f"bacc_{s}" for s in sources] + [f"auc_{s}" for s in sources]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run, (rows, _, _) in per_model.items():
            for r in rows:
                writer.writerow({"model": run.model, "model_name": run.label, "run": run.dir.name, "rule": r["rule"],
                                 "rule_definition": json.dumps(r["rule_def"]), "temperature": "default" if r["temperature"] is None else r["temperature"],
                                 "threshold": r["threshold"], **{m: r[m] for m in METRICS},
                                 **{f"bacc_{s}": (r["by_source"].get(s) or {}).get("bacc") for s in sources},
                                 **{f"auc_{s}": (r["by_source"].get(s) or {}).get("auc") for s in sources}})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="lytang--LLM-AggreFact")
    p.add_argument("--split", default="test")
    p.add_argument("--questions", help="only this question set (JSON); default: every set found")
    p.add_argument("--models", nargs="+", help="only these models (folder names under models/)")
    p.add_argument("--thresholds", default="0.05:0.95:0.05", help="list (0.1,0.3) or range start:stop:step")
    p.add_argument("--temperatures", default="default,0.5,1,1.5,2", help="list; 'default' = the saved probabilities")
    p.add_argument("--rules", help="comma list: 'rule' (or 'min') = the question set's rule, or question ids; default: all")
    p.add_argument("--select", default="macro_balanced_accuracy", choices=["macro_balanced_accuracy", "balanced_accuracy",
                   "accuracy"], help="metric that picks the best combination per model")
    p.add_argument("--top", type=int, default=0, help="rows per model in the detail tables (0 = all)")
    p.add_argument("--include-incomplete", action="store_true", help="also runs whose status is not complete")
    p.add_argument("--include-limited", action="store_true", help="also -limitN runs")
    p.add_argument("--output", help="output prefix (default results/<dataset>/<split>/grid_<hash>)")
    args = p.parse_args()

    args.threshold_values = parse_values(args.thresholds)
    args.temperature_values = parse_values(args.temperatures, allow_default=True)
    if not args.threshold_values or not args.temperature_values:
        p.error("empty threshold or temperature list")
    if any(t is not None and t <= 0 for t in args.temperature_values):
        p.error("temperatures must be positive")
    rule_names = set(args.rules.split(",")) if args.rules else None

    split_dir = RESULTS / args.dataset / args.split
    registry = json.loads((split_dir / "hash_questions.json").read_text()) if (split_dir / "hash_questions.json").exists() else {}
    wanted_hash = qs.questions_hash(qs.load(args.questions)["questions"]) if args.questions else None

    groups = {}
    for run_json in sorted(split_dir.glob("*/run.json")):
        run_dir = run_json.parent
        info = json.loads(run_json.read_text())
        ident = info["identity"]
        if wanted_hash and ident["questions_hash"] != wanted_hash:
            continue
        if args.models and ident["model"] not in args.models:
            continue
        if ident.get("limit") and not args.include_limited:
            continue
        if info.get("status") != "complete" and not args.include_incomplete:
            print(f"skip {run_dir.name}: status {info.get('status')}")
            continue
        groups.setdefault(ident["questions_hash"], []).append(run_dir)
    if not groups:
        raise SystemExit(f"No matching complete runs in {split_dir}")

    for qhash, dirs in groups.items():
        started = time.perf_counter()
        runs = []
        for d in dirs:
            try:
                runs.append(Run(d))
            except (ValueError, KeyError) as exc:
                print(f"skip {d.name}: {exc}")
        counts = {}
        for r in runs:
            counts[r.model] = counts.get(r.model, 0) + 1
        for r in runs:
            r.label = r.name + (f" [{r.phash}]" if counts[r.model] > 1 else "")
        per_model = {}
        for run in runs:
            rows, skipped = search(run, rule_names, args.temperature_values, args.threshold_values)
            default = default_row(run)
            if rows:
                best = max((r for r in rows if r.get(args.select) is not None), key=lambda r: r[args.select], default=None)
                if best is not None:
                    check_against_core(run, run.score(best["rule_def"], best["temperature"]), best["threshold"], best)
            per_model[run] = (rows, skipped, default)
            print(f"{run.label}: {len(rows)} combinations"
                  + (f", {len(skipped)} rule/temperature pairs not applicable (no logits)" if skipped else ""))
        elapsed = time.perf_counter() - started
        prefix = args.output or str(split_dir / f"grid_{qhash}")
        if args.output and len(groups) > 1:
            prefix = f"{args.output}_{qhash}"
        write_outputs(prefix, args, qhash, registry.get(qhash, {}), per_model, runs, elapsed)
        print(f"wrote {prefix}.md and {prefix}.csv ({elapsed:.1f} s)")


if __name__ == "__main__":
    main()
