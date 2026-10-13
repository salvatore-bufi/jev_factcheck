#!/usr/bin/env python3
"""Offline, tune-only calibration followed by one reused-selection assessment.

Run: python iterate_cached.py
Only the four named development/cache files below are opened as experiment data.
"""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import sklearn
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from bench import MODEL, digest, make_state, read_rows, write_json
from metrics import document_cluster_bootstrap, metrics, tune_threshold


ROOT = Path(__file__).resolve().parent
FEATURES = [
    "support_simple", "support_strict", "no_unsupported_detail",
    "support_binary", "support_ternary", "no_contradicted_detail",
]
CLIP = 0.001
BASELINE_THRESHOLD = 0.30
MODEL_PATH = ROOT / "results/iteration_cached_model.json"
REPORT_PATH = ROOT / "results/iteration_cached.json"


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_development(phase):
    if phase not in ("tune", "selection"):
        raise ValueError("Only development phases are allowed")
    row_path = ROOT / f"data/{phase}.jsonl"
    prediction_path = ROOT / ("results/tune_v1_predictions.jsonl" if phase == "tune"
                              else "results/selection_predictions.jsonl")
    rows = read_rows(row_path)
    predictions = read_rows(prediction_path)
    by_id = {p["id"]: p for p in predictions}
    if (len(by_id) != len(predictions) or len({r["id"] for r in rows}) != len(rows)
            or set(by_id) != {r["id"] for r in rows}):
        raise ValueError("Cache must match development IDs exactly, without duplicates")
    for row in rows:
        pred = by_id[row["id"]]
        if (pred["model"] != MODEL or pred["input_hash"] != digest(make_state(row))
                or row["group"] != digest(" ".join(row["doc"].split()))):
            raise ValueError("Stale input, unexpected model, or invalid document group")
        answers = pred["answers"]
        extracted = [answers["support_simple"]["noul"], answers["support_strict"]["noul"],
                     1 - answers["unsupported_detail"]["noul"],
                     answers["support_binary"]["probabilities"]["supported"],
                     answers["support_ternary"]["probabilities"]["supported"],
                     1 - answers["contradicted_detail"]["noul"]]
        if extracted != [pred["scores"][feature] for feature in FEATURES]:
            raise ValueError("Cached common scores disagree with the recorded answers")
    values = np.asarray([[by_id[r["id"]]["scores"][f] for f in FEATURES] for r in rows])
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Scores must be finite probabilities")
    provenance = {
        "rows": len(rows), "document_groups": len({r["group"] for r in rows}),
        "source_label_counts": {
            f"{source}:{label}": count
            for (source, label), count in sorted(Counter(
                (r["dataset"], r["label"]) for r in rows).items())
        },
        "files_sha256": {str(p.relative_to(ROOT)): file_hash(p)
                         for p in (row_path, prediction_path)},
        "prediction_pack_hashes": sorted({p["pack_hash"] for p in predictions}),
    }
    return rows, values, provenance


def transform(values, name):
    if name == "raw_probability":
        return values
    if name == "clipped_logit":
        clipped = np.clip(values, CLIP, 1 - CLIP)
        return np.log(clipped / (1 - clipped))
    raise ValueError(f"Unknown feature transform: {name}")


def fit(rows, values, c):
    counts = Counter((r["dataset"], r["label"]) for r in rows)
    sources = {r["dataset"] for r in rows}
    if len(counts) != 2 * len(sources):
        raise ValueError("Every training source must contain both labels")
    weights = [len(rows) / (len(counts) * counts[(r["dataset"], r["label"])])
               for r in rows]
    return LogisticRegression(C=c, solver="lbfgs", max_iter=5000, tol=1e-9).fit(
        values, [r["label"] for r in rows], sample_weight=weights)


def serialized_scores(config, values):
    if config["features"] != FEATURES or config["clip"] != CLIP:
        raise ValueError("Unexpected serialized feature order or clipping")
    return expit(transform(values, config["transform"]) @ np.asarray(config["coefficients"])
                 + config["intercept"])


def verify_serialization(model, config, values):
    native = model.predict_proba(transform(values, config["transform"]))[:, 1]
    restored = serialized_scores(config, values)
    np.testing.assert_allclose(restored, native, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(restored > config["threshold"], native > config["threshold"])
    return float(np.max(np.abs(native - restored)))


def baseline_scores(values):
    return values[:, [FEATURES.index(f) for f in
                      ("support_simple", "no_unsupported_detail", "no_contradicted_detail")]].min(axis=1)


def fold_variability(rows, scores, folds, threshold):
    values = [metrics([rows[i] for i in valid], scores[valid], threshold)["macro_balanced_accuracy"]
              for _, valid in folds]
    return {"fold_macro_balanced_accuracy": values, "fold_mean": float(np.mean(values)),
            "fold_standard_deviation": float(np.std(values, ddof=1)),
            "fold_standard_error": float(np.std(values, ddof=1) / np.sqrt(len(values)))}


def choose_candidate(candidates):
    best = max(candidates, key=lambda item: item["macro_balanced_accuracy"])
    cutoff = best["macro_balanced_accuracy"] - best["fold_standard_error"]
    eligible = [c for c in candidates if c["macro_balanced_accuracy"] >= cutoff]
    # Both transforms have six features: prefer lower C, then higher tune OOF BAcc.
    chosen = min(eligible, key=lambda item: (item["C"], -item["macro_balanced_accuracy"]))
    return chosen, {"rule": "within one fold standard error of best pooled OOF score, then lowest C, then highest OOF score",
                    "best_pooled_oof_macro_balanced_accuracy": best["macro_balanced_accuracy"],
                    "best_fold_standard_error": best["fold_standard_error"],
                    "eligibility_cutoff": cutoff, "eligible_candidates": len(eligible),
                    "limitation": "Fold SE is descriptive: folds share training data and threshold is fit on pooled OOF predictions."}


def main():
    tune_rows, tune_values, tune_provenance = load_development("tune")
    groups = np.asarray([r["group"] for r in tune_rows])
    folds = list(GroupKFold(n_splits=5).split(tune_values, groups=groups))
    fold_audit = []
    for train, valid in folds:
        assert not set(groups[train]) & set(groups[valid])
        fold_audit.append({"train_rows": len(train), "validation_rows": len(valid),
                           "train_groups": len(set(groups[train])),
                           "validation_groups": len(set(groups[valid])), "group_overlap": 0})
    candidates = []
    oof_scores = {}
    for name in ("raw_probability", "clipped_logit"):
        values = transform(tune_values, name)
        for c in (0.01, 0.1, 1.0, 10.0):
            scores = np.full(len(tune_rows), np.nan)
            for train, valid in folds:
                model = fit([tune_rows[i] for i in train], values[train], c)
                scores[valid] = model.predict_proba(values[valid])[:, 1]
            assert np.isfinite(scores).all()
            candidate = {"transform": name, "C": c, **tune_threshold(tune_rows, scores)}
            candidate.update(fold_variability(tune_rows, scores, folds, candidate["threshold"]))
            candidates.append(candidate)
            oof_scores[(name, c)] = scores
            print(json.dumps({key: candidate[key] for key in
                              ("transform", "C", "threshold", "macro_balanced_accuracy")}), flush=True)
    chosen, selection_rule = choose_candidate(candidates)
    model = fit(tune_rows, transform(tune_values, chosen["transform"]), chosen["C"])
    config = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "family": "source_class_balanced_logistic_regression", "base_model": MODEL,
        "features": FEATURES, "transform": chosen["transform"], "clip": CLIP,
        "C": chosen["C"], "solver": "lbfgs", "max_iter": 5000, "tol": 1e-9,
        "coefficients": model.coef_[0].tolist(), "intercept": float(model.intercept_[0]),
        "threshold": chosen["threshold"], "positive_rule": "score > threshold",
        "training_phase": "tune", "training_rows": len(tune_rows),
        "training_files_sha256": tune_provenance["files_sha256"],
        "sample_weight": "n_train / (2 * n_sources * count_train(source,label))",
        "hyperparameter_and_threshold_selection": "5-fold document-group OOF on tune only, one-SE preference for lower C",
        "candidate_selection_rule": selection_rule,
        "selection_data_read_before_serialization": False,
        "numpy_version": np.__version__, "sklearn_version": sklearn.__version__,
    }
    # Persist the complete model before the single reused-selection assessment.
    write_json(MODEL_PATH, config)
    config = json.loads(MODEL_PATH.read_text())
    tune_serialization_error = verify_serialization(model, config, tune_values)
    chosen_oof = oof_scores[(chosen["transform"], chosen["C"])]
    tune_baseline = baseline_scores(tune_values)

    selection_rows, selection_values, selection_provenance = load_development("selection")
    if set(groups) & {r["group"] for r in selection_rows}:
        raise ValueError("Tune and selection must have disjoint document groups")
    selection_serialization_error = verify_serialization(model, config, selection_values)
    selection_scores = serialized_scores(config, selection_values)
    selection_baseline = baseline_scores(selection_values)
    selection_primary = metrics(selection_rows, selection_scores, config["threshold"])
    selection_baseline_metrics = metrics(selection_rows, selection_baseline, BASELINE_THRESHOLD)
    baseline_config = {"score": "min(support_simple,no_unsupported_detail,no_contradicted_detail)",
                       "threshold": BASELINE_THRESHOLD, "positive_rule": "score > threshold"}
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Offline cached development only; no network or inference calls",
        "api_calls": 0, "additional_api_cost_usd": 0,
        "test_used": False,
        "limitations": [
            "Selection was already used for earlier model selection and is not fresh validation.",
            "Tune OOF scores select both hyperparameters and threshold; this is not nested CV.",
            "No leaderboard or held-out test superiority is established by this development assessment.",
            "Bootstrap intervals condition on this fitted model and do not include model-search uncertainty.",
        ],
        "model_path": str(MODEL_PATH.relative_to(ROOT)), "model_sha256": file_hash(MODEL_PATH),
        "baseline_config": baseline_config,
        "procedure": {
            "features": FEATURES, "folds": 5, "splitter": "GroupKFold, shuffle=False",
            "groups": "row['group']: hash of whitespace-normalized document",
            "candidates": 8, "threshold_grid": "0.01 through 0.99, step 0.01",
            "candidate_selection_rule": selection_rule,
            "weighting": config["sample_weight"], "fold_audit": fold_audit,
            "selection_assessments": 1,
        },
        "tune": {
            **tune_provenance, "candidates": candidates, "chosen": chosen,
            "baseline_fold_variability": fold_variability(
                tune_rows, tune_baseline, folds, BASELINE_THRESHOLD),
            "baseline": metrics(tune_rows, tune_baseline, BASELINE_THRESHOLD),
            "oof_score_sha256": hashlib.sha256(chosen_oof.astype('<f8').tobytes()).hexdigest(),
        },
        "reused_selection": {
            **selection_provenance, "primary": selection_primary,
            "baseline": selection_baseline_metrics,
            "macro_balanced_accuracy_delta": selection_primary["macro_balanced_accuracy"]
                                               - selection_baseline_metrics["macro_balanced_accuracy"],
            "exceeds_0_80": selection_primary["macro_balanced_accuracy"] > 0.80,
            "document_cluster_bootstrap_95_percent": document_cluster_bootstrap(
                selection_rows, selection_scores, selection_baseline, config["threshold"],
                BASELINE_THRESHOLD, repeats=1000, seed=42),
        },
        "serialization_equivalence": {
            "tune_max_absolute_error": tune_serialization_error,
            "selection_max_absolute_error": selection_serialization_error,
            "predictions_identical": True,
        },
    }
    write_json(REPORT_PATH, report)
    print(json.dumps({"selected": {key: config[key] for key in ("transform", "C", "threshold")},
                      "tune_oof_macro_balanced_accuracy": chosen["macro_balanced_accuracy"],
                      "reused_selection_macro_balanced_accuracy": selection_primary["macro_balanced_accuracy"],
                      "baseline_reused_selection_macro_balanced_accuracy":
                          selection_baseline_metrics["macro_balanced_accuracy"],
                      "report": str(REPORT_PATH.relative_to(ROOT))}), flush=True)


if __name__ == "__main__":
    main()
