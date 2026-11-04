"""Statistics of a built source, written next to its rows (<source>.stats.json)."""
from collections import Counter

import numpy as np

DISTANCE_BINS = [(0, 0, "0"), (1, 1, "1"), (2, 9, "2-9"), (10, 19, "10-19"), (20, 10 ** 9, ">=20")]


def _quantiles(values):
    if not values:
        return {}
    v = np.asarray(values)
    return {"p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)), "max": int(v.max())}


def compute(rows, skips):
    """rows: built rows; skips: Counter of skip reasons. -> dict."""
    positives = [r for r in rows if r["label"] == 1 and r["evidence"]]
    spread = [r["evidence"][-1] - r["evidence"][0] for r in positives]
    gold = sum(r["meta"].get("gold_sentences", 0) for r in rows)
    one = sum(r["meta"].get("gold_one_to_one", 0) for r in rows)
    overlap = Counter(str(r["meta"].get("aggrefact_overlap")) for r in rows if "aggrefact_overlap" in r["meta"])
    return {
        "rows": len(rows),
        "skipped": dict(Counter(skips).most_common()),
        "by_label": dict(Counter(r["label"] for r in rows)),
        "by_orig_split": dict(Counter(r["orig_split"] for r in rows)),
        "by_evidence_term": dict(Counter(r["evidence_term"] for r in rows)),
        "by_gold_alignment": dict(Counter(r["meta"].get("gold_alignment", "exact") for r in rows)),
        "documents": len({r["doc_id"] for r in rows}),
        "sentences_per_document": _quantiles([len(r["sentences"]) for r in rows]),
        "characters_per_document": _quantiles([len(r["document"]) for r in rows]),
        "prompt_tokens": _quantiles([r["meta"]["prompt_tokens"] for r in rows if "prompt_tokens" in r["meta"]]),
        "evidence_per_positive": {
            "mean": float(np.mean([len(r["evidence"]) for r in positives])) if positives else 0.0,
            "share_with_2_or_more": float(np.mean([len(r["evidence"]) >= 2 for r in positives])) if positives else 0.0,
        },
        "evidence_spread_in_sentences": {name: sum(lo <= d <= hi for d in spread) for lo, hi, name in DISTANCE_BINS},
        "gold_sentences": gold,
        "gold_one_to_one_share": one / gold if gold else None,
        "aggrefact_overlap": dict(overlap) if overlap else None,
    }
