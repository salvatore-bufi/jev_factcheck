"""Selection of the shown sentences and the evidence metrics (doc 10 §2). Shared by baselines, training, evaluation.

Where it sits in the workflow:
    scores of the sentences (head or baseline)  ->  [select]  ->  shown sentences  ->  [summarize]  ->  metrics

A PREDICTION is one dict per row, the common format of every method:
    id, source, label           from the row
    p_supported                 the verdict, P(supported)
    scores                      one score per sentence (higher = more likely evidence)
    threshold                   the score above which a sentence is evidence for THIS row
                                (head with document threshold: s0 + delta; fixed threshold: a constant)
    shown                       the sentences shown (select(scores, threshold))
    evidence_sets               gold sets of the row (empty for negatives)
    gold_tok_last               token distance of the last gold sentence from the document start (None if no gold)
Rows are evaluated for evidence only if label = 1 AND the verdict says supported (p_supported > VERDICT_THRESHOLD):
those are the only rows where the system shows sentences.
"""
import math
import statistics

from evidence import config

SENT_BUCKETS = ("0-1", "2-9", "10-19", ">=20")
TOK_BUCKETS = ("<1024", "1024-4096", ">4096")


def select(scores, threshold, max_shown=config.MAX_SHOWN, min_shown=config.MIN_SHOWN):
    """The sentences shown.

    Input:  scores (list of floats, one per sentence), threshold (float), max_shown, min_shown.
    Output: sorted list of indices: the sentences with score > threshold, at most max_shown (the highest),
            at least min_shown (the best one) even if none passes the threshold.
    """
    order = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
    chosen = [k for k in order if scores[k] > threshold][:max_shown]
    return sorted(chosen or order[:min_shown])


def sentence_metrics(shown, evidence_sets):
    """Quality of the shown sentences for one row.

    Input:  shown (indices), evidence_sets (list of alternative gold sets).
    Output: dict precision, recall, f1 (against the UNION of the gold sets), full_set (every sentence of at least
            one gold set shown), exact (shown == one gold set), count_error (shown - gold, in sentences).
    Example: shown {1, 4}, gold sets {1, 4} and {7}: precision 1, recall 2/3, f1 0.8, full_set yes, exact yes.
    """
    gold = set().union(*map(set, evidence_sets)) if evidence_sets else set()
    shown = set(shown)
    tp = len(shown & gold)
    p = tp / len(shown) if shown else 0.0
    r = tp / len(gold) if gold else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "full_set": any(set(e) <= shown for e in evidence_sets),
            "exact": any(set(e) == shown for e in evidence_sets),
            "count_error": len(shown) - len(gold)}


def auprc(scores_per_row, gold_per_row):
    """Area under the precision-recall curve over all sentences of all rows (does not depend on the threshold).

    Input:  scores_per_row (list of score lists), gold_per_row (list of sets of gold indices).
    Output: float, or None if there is no gold sentence.
    """
    from sklearn.metrics import average_precision_score
    y = [int(k in g) for s, g in zip(scores_per_row, gold_per_row) for k in range(len(s))]
    x = [v for s in scores_per_row for v in s]
    return float(average_precision_score(y, x)) if any(y) else None


def sent_bucket(evidence_sets):
    """Distance bucket, in sentences, between the first and the last gold sentence (union of the sets)."""
    gold = sorted(set().union(*map(set, evidence_sets)))
    d = gold[-1] - gold[0]
    return "0-1" if d <= 1 else "2-9" if d <= 9 else "10-19" if d <= 19 else ">=20"


def tok_bucket(gold_tok_last):
    """Bucket of the token position of the last gold sentence from the document start."""
    if gold_tok_last < 1024:
        return "<1024"
    return "1024-4096" if gold_tok_last <= 4096 else ">4096"


def gold_tok_last(entry):
    """Token distance of the end of the last gold sentence from the document start (feature index entry or similar).

    Input:  dict with gold (0/1 list), sent_tok_spans, doc_tok_start.  Output: int, or None without gold.
    """
    gold = [k for k, g in enumerate(entry["gold"]) if g]
    if not gold:
        return None
    return entry["sent_tok_spans"][gold[-1]][1] - entry["doc_tok_start"]


def _mean(values):
    values = [v for v in values if v is not None]
    return round(statistics.mean(values), 4) if values else None


def summarize(preds):
    """All the metrics of a list of predictions.

    Input:  preds (list of prediction dicts, see the file header).
    Output: dict with
        evaluated               positives judged supported (where evidence is measured)
        missed_by_verdict       positives judged NOT supported (no evidence shown)
        precision, recall, f1, full_set, exact, count_error_mean, shown_mean, auprc
        f1_by_sent_distance     F1 per bucket of distance between gold sentences
        f1_by_tok_distance      F1 per bucket of token position of the last gold sentence
        recall_uncertain        recall on positives with P just above the verdict threshold (up to +0.15)
        negatives_lit           share of negatives (label 0) with at least one sentence above the threshold
        score                   checkpoint criterion: mean of f1 and full_set
    """
    thr = config.VERDICT_THRESHOLD
    pos = [p for p in preds if p["label"] == 1 and p["evidence_sets"]]
    ev = [p for p in pos if p["p_supported"] > thr]
    neg = [p for p in preds if p["label"] == 0]
    per = [(p, sentence_metrics(p["shown"], p["evidence_sets"])) for p in ev]
    out = {"evaluated": len(ev), "missed_by_verdict": len(pos) - len(ev)}
    for k in ("precision", "recall", "f1", "full_set", "exact"):
        out[k] = _mean([float(m[k]) for _, m in per])
    out["count_error_mean"] = _mean([m["count_error"] for _, m in per])
    out["shown_mean"] = _mean([len(p["shown"]) for p in ev])
    out["auprc"] = auprc([p["scores"] for p in ev], [set().union(*map(set, p["evidence_sets"])) for p in ev]) \
        if ev else None
    out["f1_by_sent_distance"] = {b: _mean([m["f1"] for p, m in per if sent_bucket(p["evidence_sets"]) == b])
                                  for b in SENT_BUCKETS}
    out["f1_by_tok_distance"] = {b: _mean([m["f1"] for p, m in per if p.get("gold_tok_last") is not None
                                           and tok_bucket(p["gold_tok_last"]) == b]) for b in TOK_BUCKETS}
    out["f1_far_10plus"] = _mean([m["f1"] for p, m in per if sent_bucket(p["evidence_sets"]) in ("10-19", ">=20")])
    out["f1_beyond_1024_tok"] = _mean([m["f1"] for p, m in per if p.get("gold_tok_last") is not None
                                       and p["gold_tok_last"] >= 1024])
    out["recall_uncertain"] = _mean([m["recall"] for p, m in per if p["p_supported"] <= thr + 0.15])
    out["negatives_lit"] = _mean([float(any(s > p["threshold"] for s in p["scores"])) for p in neg]) if neg else None
    out["score"] = _mean([out["f1"], out["full_set"]]) if out["f1"] is not None else None
    out["by_source"] = {}
    for src in sorted({p["source"] for p in ev}):
        sub = [m for p, m in per if p["source"] == src]
        out["by_source"][src] = {"n": len(sub), "f1": _mean([m["f1"] for m in sub]),
                                 "full_set": _mean([float(m["full_set"]) for m in sub])}
    return out


def logit(p):
    """log(p / (1 - p)): turns a probability threshold into a score threshold."""
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def best_fixed_threshold(preds_fn, candidates):
    """Pick the threshold that maximises F1 (used for baselines and the reference loss).

    Input:  preds_fn (function threshold -> list of predictions), candidates (list of thresholds).
    Output: (best threshold, its summary).
    """
    best = None
    for t in candidates:
        m = summarize(preds_fn(t))
        if m["f1"] is not None and (best is None or m["f1"] > best[1]["f1"]):
            best = (t, m)
    return best
