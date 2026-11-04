"""STEP 4 — Does the verdict still work with the preamble? Which preamble text? (doc 05 §4-5)

Where it sits in the workflow:
    splits  ->  [evidence.check_preamble]  ->  dataset/evidence/preamble_choice.json  ->  read by evidence.config
    (run it BEFORE the full extraction: the features depend on the preamble)

What it does: for each candidate ("none", "P-A", "P-B" in evidence/config.py) it runs the verdict forward on the same
rows and measures BAcc, AUC and ECE. The repo has no LoRA of the binary fine-tuning, so the "verdict" is KnowLine-4B
as it is, and the conditions are B-1D (no preamble) and B-SAND-A / B-SAND-B (doc 05 §4).

Choice: the preamble whose AUC is closest to "none"; within 0.2 AUC points, the shorter text. It is accepted if BAcc
and AUC stay within 0.3 points of "none" and ECE within 0.01; otherwise the file says so ("acceptable": false):
then the verdict would need retraining with the preamble (not available here) or the preamble is dropped (H4).

Run (GPU, ~3 forwards per row):
    python -m evidence.check_preamble                         # dev_evid_select + dev_evid_confirm + dev_evid_neg
    python -m evidence.check_preamble --rows <file.jsonl> --limit 500
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evidence import config  # noqa: E402
from evidence.data import io  # noqa: E402
from evidence.eval_evidence import verdict_metrics  # noqa: E402


def main(argv=None):
    """Command line entry point (see the file header)."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_rows = [config.SPLITS / f"{n}.jsonl" for n in ("dev_evid_select", "dev_evid_confirm", "dev_evid_neg")]
    p.add_argument("--rows", type=Path, nargs="+", default=default_rows)
    p.add_argument("--limit", type=int, default=0, help="at most N rows in total (0 = all)")
    opts = p.parse_args(argv)

    from evidence.model import forward_sentences, load_model, p_supported
    from evidence.render import render_row
    rows = [r for path in opts.rows for r in io.read_jsonl(path)]
    rows = rows[:opts.limit] if opts.limit else rows
    model = load_model()
    labels = [r["label"] for r in rows]
    results = {}
    for name, template in config.PREAMBLES.items():
        probs = [p_supported(forward_sentences(model, render_row(r, template), features=False)["answer_logits"])
                 for r in rows]
        results[name] = verdict_metrics(labels, probs)
        print(f"{name:5s} {results[name]}")

    base = results["none"]
    candidates = [n for n in config.PREAMBLES if n != "none"]
    gap = {n: abs(results[n]["auc"] - base["auc"]) for n in candidates}
    best_gap = min(gap.values())
    close = [n for n in candidates if gap[n] - best_gap <= 0.002]        # within 0.2 AUC points
    chosen = min(close, key=lambda n: len(config.PREAMBLES[n]))
    r = results[chosen]
    acceptable = (abs(r["bacc"] - base["bacc"]) <= 0.003 and abs(r["auc"] - base["auc"]) <= 0.003
                  and abs(r["ece"] - base["ece"]) <= 0.01)
    io.write_json(config.PREAMBLE_CHOICE, {"chosen": chosen, "template": config.PREAMBLES[chosen],
                                           "acceptable": acceptable, "results": results,
                                           "rows": [str(x) for x in opts.rows], "n": len(rows)})
    print(f"chosen: {chosen} ({config.PREAMBLES[chosen]!r}), within tolerance of 'none': {acceptable}")
    print(f"-> {config.PREAMBLE_CHOICE}")


if __name__ == "__main__":
    main()
