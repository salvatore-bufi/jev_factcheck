"""LLM-AggreFact scoring; all reported scores are fractions, not percentages."""

import numpy as np
from sklearn.metrics import roc_auc_score


def _arrays(rows, scores):
    labels = np.asarray([row["label"] for row in rows])
    scores = np.asarray(scores, dtype=float)
    if not len(rows) or scores.shape != labels.shape:
        raise ValueError("Nonempty rows and one score per row are required")
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("Labels must be binary and scores must be finite")
    return labels, scores


def _stats(labels, scores, threshold):
    predictions = scores > threshold
    positive, negative = labels == 1, labels == 0
    tpr = float(predictions[positive].mean()) if positive.any() else None
    tnr = float((~predictions[negative]).mean()) if negative.any() else None
    both_classes = tpr is not None and tnr is not None
    return {
        "n": len(labels),
        "balanced_accuracy": (tpr + tnr) / 2 if both_classes else None,
        "accuracy": float((predictions == labels).mean()),
        "roc_auc": float(roc_auc_score(labels, scores)) if both_classes else None,
        "supported_recall": tpr,
        "unsupported_recall": tnr,
    }


def metrics(rows, scores, threshold=0.5):
    """Macro-average source BAcc; a missing class makes BAcc/AUC undefined."""
    labels, scores = _arrays(rows, scores)
    if not np.isfinite(threshold):
        raise ValueError("Threshold must be finite")
    sources = np.asarray([row["dataset"] for row in rows])
    by_source = {
        source: _stats(labels[sources == source], scores[sources == source], threshold)
        for source in sorted(set(sources))
    }
    balanced = [result["balanced_accuracy"] for result in by_source.values()]
    return {
        **_stats(labels, scores, threshold),
        "macro_balanced_accuracy": float(np.mean(balanced))
        if all(value is not None for value in balanced) else None,
        "by_source": by_source,
    }


def tune_threshold(rows, scores):
    """Optimize dev macro BAcc on .01–.99, then nearest .5, then larger."""
    labels, scores = _arrays(rows, scores)
    sources = np.asarray([row["dataset"] for row in rows])
    grid = np.arange(1, 100) / 100
    source_bacc = []
    for source in sorted(set(sources)):
        positive = (sources == source) & (labels == 1)
        negative = (sources == source) & (labels == 0)
        if not positive.any() or not negative.any():
            raise ValueError("Each source needs both labels for threshold tuning")
        tpr = (scores[positive, None] > grid).mean(axis=0)
        tnr = (scores[negative, None] <= grid).mean(axis=0)
        source_bacc.append((tpr + tnr) / 2)
    macro_bacc = np.mean(source_bacc, axis=0)
    step = max(range(1, 100), key=lambda step: (
        macro_bacc[step - 1], -abs(step - 50), step
    ))
    return {"threshold": step / 100, **metrics(rows, scores, step / 100)}


def document_cluster_bootstrap(rows, scores, baseline_scores, threshold,
                               baseline_threshold=0.5, repeats=1000, seed=42):
    """Paired macro-BAcc percentile CIs, resampling documents within sources.

    Whitespace-normalized identical documents form a cluster. Both models use
    identical draws; replicates missing either class in any source are omitted.
    CI fields are [2.5th, 97.5th percentile], or None when no replicate is valid.
    """
    labels, scores = _arrays(rows, scores)
    _, baseline_scores = _arrays(rows, baseline_scores)
    if not np.isfinite([threshold, baseline_threshold]).all() or repeats < 1:
        raise ValueError("Finite thresholds and positive repeats are required")
    predictions = scores > threshold
    baseline_predictions = baseline_scores > baseline_threshold
    groups = {}
    for index, row in enumerate(rows):
        document = " ".join(row["doc"].split())
        source_groups = groups.setdefault(row["dataset"], {})
        counts = source_groups.setdefault(document, np.zeros(6, dtype=np.int64))
        positive = labels[index] == 1
        counts[0 if positive else 1] += 1
        counts[2 if positive else 3] += predictions[index] == positive
        counts[4 if positive else 5] += baseline_predictions[index] == positive
    contributions = [np.array(list(groups[source].values())) for source in sorted(groups)]
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(repeats):
        total = np.zeros(2)
        for source in contributions:
            counts = source[rng.integers(len(source), size=len(source))].sum(axis=0)
            if not counts[0] or not counts[1]:
                break
            total += (counts[[2, 4]] / counts[0] + counts[[3, 5]] / counts[1]) / 2
        else:
            results.append(total / len(contributions))
    intervals = {"primary": None, "baseline": None, "paired_delta": None}
    if results:
        values = np.asarray(results)
        for key, samples in zip(intervals, (values[:, 0], values[:, 1], values[:, 0] - values[:, 1])):
            intervals[key] = np.percentile(samples, [2.5, 97.5]).tolist()
    return {**intervals, "valid_repeats": len(results), "repeats": repeats}
