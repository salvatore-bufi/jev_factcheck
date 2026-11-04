"""STEP 9 — Evaluate a trained head on a set: evidence metrics, verdict metrics, sufficiency / necessity (doc 10).

Where it sits in the workflow:
    run folder (evidence.train_head) + features of a set (evidence.extract)  ->  [evidence.eval_evidence]
    ->  <run>/eval/<set>/metrics.json + predictions.jsonl

Run:
    python -m evidence.eval_evidence --run data/evidence/runs/knowline-4b/h2c \
        --features data/evidence/features/knowline-4b/dev_evid_select
    # test sets: ONCE, with the configuration frozen (doc 12 §6); sufficiency needs the GPU and the rows' text
    python -m evidence.eval_evidence --run <run or release> --features .../test_evid_id \
        --rows dataset/evidence/splits/test_evid_id.jsonl --sufficiency 300

What is measured:
    evidence      evidence.metrics.summarize: precision / recall / F1, full set, exact, AUPRC, F1 by distance (in
                  sentences and in tokens), negatives with a sentence above the threshold, recall of uncertain positives
    verdict       BAcc, AUC, ECE of P(supported) over all rows (must equal the verdict without the head: the model
                  is frozen; it is reported to check the pipeline)
    sufficiency   share of rows where the document reduced to the SHOWN sentences still gets "supported"
    necessity     share of rows where the document WITHOUT the shown sentences gets "not supported" (reported only:
                  it can fail with correct evidence when the evidence is repeated in the document)
"""
import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evidence import config  # noqa: E402
from evidence.data import io  # noqa: E402
from evidence.features import FeatureStore, same_features  # noqa: E402
from evidence.head import load_run  # noqa: E402
from evidence.metrics import summarize  # noqa: E402
from evidence.train_head import load_rows, predictions, scores_of  # noqa: E402


def verdict_metrics(labels, probs, thr=config.VERDICT_THRESHOLD, bins=10):
    """Balanced accuracy, AUC and expected calibration error of the verdict.

    Input:  labels (0/1 list), probs (P(supported) list), thr (decision threshold), bins (ECE bins).
    Output: dict bacc, auc, ece, n.
    """
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    pred = [int(p > thr) for p in probs]
    out = {"n": len(labels), "bacc": round(float(balanced_accuracy_score(labels, pred)), 4)}
    out["auc"] = round(float(roc_auc_score(labels, probs)), 4) if len(set(labels)) == 2 else None
    ece = 0.0
    for b in range(bins):
        idx = [i for i, p in enumerate(probs) if b / bins <= p < (b + 1) / bins or (b == bins - 1 and p == 1.0)]
        if idx:
            conf = sum(probs[i] for i in idx) / len(idx)
            acc = sum(labels[i] for i in idx) / len(idx)
            ece += len(idx) / len(probs) * abs(conf - acc)
    out["ece"] = round(ece, 4)
    return out


def sufficiency(preds, rows, preamble, limit):
    """Sufficiency and necessity with the model (2 forwards per row, GPU).

    Input:  preds (predictions of evaluated rows), rows (dict id -> dataset row), preamble, limit (rows, 0 = all).
    Output: dict sufficiency, necessity, n.
    """
    from evidence.extract import rebuild
    from evidence.model import forward_sentences, load_model, p_supported
    from evidence.render import render_row
    model = load_model()
    thr = config.VERDICT_THRESHOLD
    todo = [p for p in preds if p["label"] == 1 and p["evidence_sets"] and p["p_supported"] > thr]
    todo = todo[:limit] if limit else todo
    suff, nec = [], []
    for p in todo:
        row = rows[p["id"]]
        n = len(row["sentences"])
        only = rebuild(row, p["shown"])
        p_only = p_supported(forward_sentences(model, render_row(only, preamble), features=False)["answer_logits"])
        suff.append(p_only > thr)
        rest = [k for k in range(n) if k not in set(p["shown"])]
        if rest:
            without = rebuild(row, rest)
            p_wo = p_supported(forward_sentences(model, render_row(without, preamble), features=False)["answer_logits"])
            nec.append(p_wo <= thr)
    return {"n": len(suff), "sufficiency": round(sum(suff) / max(len(suff), 1), 4),
            "necessity": round(sum(nec) / max(len(nec), 1), 4)}


def main(argv=None):
    """Command line entry point (see the file header)."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True, help="run folder (or release folder) with the head")
    p.add_argument("--features", type=Path, required=True, help="feature folder of the set to evaluate")
    p.add_argument("--rows", type=Path, help="rows file of the set (needed for --sufficiency)")
    p.add_argument("--sufficiency", type=int, default=-1, help="rows for sufficiency/necessity (0 = all, -1 = skip)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    opts = p.parse_args(argv)

    head, cfg, rule = load_run(opts.run, opts.device)
    store = FeatureStore(opts.features)
    if not same_features(cfg["feature_meta"], store.meta):
        raise SystemExit("the features were extracted with different settings than the head's training features")
    rows = load_rows(store, cfg)
    preds = predictions(rows, scores_of(head, rows, opts.device), rule)
    metrics = {"evidence": summarize(preds),
               "verdict": verdict_metrics([r["label"] for r in rows], [pr["p_supported"] for pr in preds]),
               "rule": rule, "run": str(opts.run), "features": str(opts.features)}
    if opts.sufficiency >= 0:
        if not opts.rows:
            raise SystemExit("--sufficiency needs --rows")
        by_id = {r["id"]: r for r in io.read_jsonl(opts.rows)}
        metrics["sufficiency"] = sufficiency(preds, by_id, store.meta["preamble"], opts.sufficiency)

    out = Path(opts.run) / "eval" / opts.features.name
    io.write_json(out / "metrics.json", metrics)
    io.write_jsonl(out / "predictions.jsonl", preds)
    e = metrics["evidence"]
    print(f"{opts.features.name}: evaluated {e['evaluated']}  f1 {e['f1']}  full_set {e['full_set']}  "
          f"auprc {e['auprc']}  f1>=10 {e['f1_far_10plus']}  negatives_lit {e['negatives_lit']}")
    print(f"verdict: {metrics['verdict']}")
    if "sufficiency" in metrics:
        print(f"sufficiency: {metrics['sufficiency']}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
