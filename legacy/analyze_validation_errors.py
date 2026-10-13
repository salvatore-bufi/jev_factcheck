#!/usr/bin/env python3
"""Reproduce a development-only error audit from explicit cached inputs. No API calls."""
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/validation_error_analysis"
SEED = "jev-validation-errors-20260930:"
COMPONENTS = ("support_simple", "no_unsupported_detail", "no_contradicted_detail")
CLUSTERS = {
    "scope_status": "Scope, qualifiers and event status",
    "paraphrase_synthesis": "Paraphrase and combining evidence",
    "missing_evidence": "Unstated details or incomplete evidence",
    "identity_reference": "Entities, attribution and unresolved references",
    "numeric_precision": "Numbers, units and precision",
    "label_conflict": "Strong apparent label/evidence conflicts",
    "ambiguous": "Broad interpretive claim; unresolved",
}


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def stats(counts):
    positive, negative = counts["TP"] + counts["FN"], counts["TN"] + counts["FP"]
    return {**dict(counts), "n": positive + negative,
            "tpr": counts["TP"] / positive, "fpr": counts["FP"] / negative,
            "balanced_accuracy": (counts["TP"] / positive + counts["TN"] / negative) / 2,
            "accuracy": (counts["TP"] + counts["TN"]) / (positive + negative)}


def main():
    # Explicit paths prevent accidental loading of any held-out evaluation artifacts.
    paths = {"rows": ROOT / "data/selection.jsonl",
             "predictions": ROOT / "results/selection_predictions.jsonl",
             "pack": ROOT / "results/selection_pack.json",
             "freeze": ROOT / "results/frozen.json",
             "annotations": OUT / "review_annotations.jsonl"}
    frozen = json.loads(paths["freeze"].read_text())
    pack = json.loads(paths["pack"].read_text())
    rows, predictions = read_rows(paths["rows"]), read_rows(paths["predictions"])
    pred = {p["id"]: p for p in predictions}
    assert len(rows) == len(pred) == len(predictions) == 2221
    assert {r["id"] for r in rows} == set(pred)
    assert all(r["id"].startswith("dev:") for r in rows)
    assert frozen["candidate"] == "min_support_checks" and frozen["threshold"] == .30
    assert hashlib.sha256(paths["rows"].read_bytes()).hexdigest() == frozen["data_hashes"]["selection.jsonl"]
    assert all(pack["questions"][k] == v for k, v in frozen["pack"]["questions"].items())
    for name in ("bench.py", "metrics.py", "analyze.py", "fetch_data.py"):
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == frozen["code_hashes"][name]

    counts, source_counts, groups = Counter(), defaultdict(Counter), defaultdict(list)
    blockers, gate_changes, high_confidence = Counter(), Counter(), Counter()
    boundary, errors = [], []
    truncated_errors = 0
    counterfactual = {name: defaultdict(Counter) for name in ("without_detail_gate", "decimal_boundary")}
    for row in rows:
        p = pred[row["id"]]
        assert p["model"] == frozen["model"] and p["pack_hash"] == digest(pack)
        document = row["doc"]
        if len(document) > 80000:
            document = document[:60000] + "\n[Middle of long document omitted]\n" + document[-20000:]
        assert p["input_hash"] == digest({"document": document, "claim": row["claim"]})
        answers = p["answers"]
        components = [float(answers["support_simple"]["noul"]),
                      1 - float(answers["unsupported_detail"]["noul"]),
                      1 - float(answers["contradicted_detail"]["noul"])]
        assert components == [p["scores"][name] for name in COMPONENTS]
        score = min(components)
        assert score == p["scores"][frozen["candidate"]]
        accepted = score > .30
        outcome = ("TP" if accepted else "FN") if row["label"] else ("FP" if accepted else "TN")
        counts[outcome] += 1
        source_counts[row["dataset"]][outcome] += 1
        without_gate = min(components[0], components[2]) > .30
        decimal_score = min(Decimal(str(answers["support_simple"]["noul"])),
                            1 - Decimal(str(answers["unsupported_detail"]["noul"])),
                            1 - Decimal(str(answers["contradicted_detail"]["noul"])))
        decimal_accepted = decimal_score > Decimal("0.30")
        for name, decision in (("without_detail_gate", without_gate), ("decimal_boundary", decimal_accepted)):
            kind = ("TP" if decision else "FN") if row["label"] else ("FP" if decision else "TN")
            counterfactual[name][row["dataset"]][kind] += 1
        if without_gate != accepted:
            gate_changes["additional_TP" if row["label"] else "additional_FP"] += 1
        if decimal_accepted != accepted:
            boundary.append({"id": row["id"], "dataset": row["dataset"], "label": row["label"],
                             "float_score": score, "decimal_score": str(decimal_score)})
        if outcome in ("FP", "FN"):
            record = {**row, "score": score, "scores": dict(zip(COMPONENTS, components)), "error_type": outcome}
            errors.append(record)
            groups[(row["dataset"], outcome)].append(record)
            truncated_errors += p["document_truncated"]
            high_confidence["FP_score_ge_0.70"] += outcome == "FP" and score >= .70
            high_confidence["FN_score_le_0.10"] += outcome == "FN" and score <= .10
        if outcome == "FN":
            blockers[" + ".join(k for k, value in zip(COMPONENTS, components) if value <= .30)] += 1

    sample = []
    for key, group in sorted(groups.items()):
        group.sort(key=lambda r: hashlib.sha256((SEED + r["id"]).encode()).hexdigest())
        for row in group[:4]:
            sample.append({**row, "stratum_total_errors": len(group), "review_index": len(sample) + 1})
    assert len(sample) == 88 and len(errors) == 439
    existing = read_rows(OUT / "review_sample.jsonl")
    assert [r["id"] for r in sample] == [r["id"] for r in existing]
    annotations = read_rows(paths["annotations"])
    assert [a["index"] for a in annotations] == list(range(1, 89))
    cluster_counts = defaultdict(Counter)
    for row, note in zip(sample, annotations):
        assert note["cluster"] in CLUSTERS
        cluster_counts[note["cluster"]][row["error_type"]] += 1
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Internal selection sample from official development split; cached predictions only.",
        "inputs": {k: {"path": str(v.relative_to(ROOT)), "sha256": hashlib.sha256(v.read_bytes()).hexdigest()} for k, v in paths.items()},
        "model": frozen["model"], "candidate": frozen["candidate"], "threshold": .30,
        "checks": "Complete unique dev IDs, input hashes, pack hashes, matching frozen questions/model, frozen code hashes, and independent score reconstruction passed.",
        "pooled": stats(counts), "by_source": {s: stats(c) for s, c in sorted(source_counts.items())},
        "macro_balanced_accuracy": mean(stats(c)["balanced_accuracy"] for c in source_counts.values()),
        "fn_blockers": dict(blockers), "high_confidence_errors": dict(high_confidence),
        "truncated_rows": sum(p["document_truncated"] for p in predictions), "truncated_errors": truncated_errors,
        "without_detail_gate_changes": dict(gate_changes), "decimal_boundary_cases": boundary,
        "diagnostic_counterfactuals": {name: {"macro_balanced_accuracy": mean(stats(c)["balanced_accuracy"] for c in cs.values()),
                                               "pooled": stats(sum(cs.values(), Counter()))} for name, cs in counterfactual.items()},
        "review": {"n": 88, "distinct_documents": len({r["group"] for r in sample}),
                   "sampling": "Four errors per source and direction, SHA-256 ordering with fixed prefix.",
                   "hash_prefix": SEED, "FP": 44, "FN": 44,
                   "coverage": "Full reference inspected for 87 cases; abstract and targeted style-related sections for case 24. No independent second annotator.",
                   "counts_are_sample_descriptions_not_population_estimates": True,
                   "clusters": {k: {"name": CLUSTERS[k], **dict(c), "n": sum(c.values())} for k, c in cluster_counts.items()},
                   "assessments": dict(Counter(a["assessment"] for a in annotations))},
        "limitations": ["Selection was already used in model selection and threshold fitting; this is diagnostic development analysis, not fresh validation.",
                        "Manual clusters can overlap conceptually; one dominant category is assigned per case.",
                        "Reasons are evidence-based hypotheses, not model-generated explanations or internal reasoning traces.",
                        "Original gold labels and frozen implementation remain unchanged. No new threshold or question was selected."]}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    (OUT / "error_manifest.jsonl").write_text("".join(json.dumps({k: r[k] for k in ("id", "dataset", "label", "error_type", "score", "scores")}) + "\n" for r in errors))
    appendix = ["# Validation error review: all 88 sampled cases", "",
                "FP = gold unsupported, accepted. FN = gold supported, rejected. Scores are support / no unsupported detail / no contradiction.", "",
                "Interpretations are manual hypotheses. Gold labels are unchanged. Case 24 received targeted rather than full-document review.", ""]
    for row, note in zip(sample, annotations):
        appendix.extend([f"## {row['review_index']}. {row['id']} — {row['dataset']} — {row['error_type']}", "",
                         f"Cluster: **{CLUSTERS[note['cluster']]}**. Assessment: `{note['assessment']}`.", "",
                         f"Claim: {row['claim'].strip()}", "",
                         "Scores: " + " / ".join(f"{row['scores'][k]:.2f}" for k in COMPONENTS) + f"; combined **{row['score']:.2f}**.", "",
                         note["reason"], "", f"[Full validation record]({OUT / 'review_sample.jsonl'}:{row['review_index']})", ""])
    (OUT / "reviewed_cases.md").write_text("\n".join(appendix))
    print(json.dumps({"n": len(rows), "errors": len(errors), "counts": counts,
                      "clusters": summary["review"]["clusters"], "assessments": summary["review"]["assessments"],
                      "macro_balanced_accuracy": summary["macro_balanced_accuracy"],
                      "diagnostic_counterfactuals": summary["diagnostic_counterfactuals"]}, indent=2))


if __name__ == "__main__":
    main()
