"""Run with python test_calibrate_transfer.py. Needs the local dev split; the test cache check needs results/runs/jpt-9b."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import json
import random
import shutil
from tempfile import TemporaryDirectory

import calibrate_transfer as ct
from dev_split import load_dev_rows
from metrics import tune_threshold

ROOT = Path(__file__).resolve().parent


def fake_dev_cache(runs, rows, data_hash):
    """JPT-style probability cache whose support answer is noisy but label-correlated."""
    rng = random.Random(0)
    run_dir = runs / "jpt-9b-dev"
    run_dir.mkdir(parents=True)
    lines = [json.dumps({"identity": {"split": "dev", "data_sha256": data_hash, "limit": 0}})]
    answers = {}
    for row in rows:
        support = min(max(rng.gauss(0.65 if row["label"] else 0.35, 0.2), 0.0), 1.0)
        answers[row["id"]] = {"support_simple": support, "unsupported_detail": 0.0, "contradicted_detail": 0.0}
        lines.append(json.dumps({"id": row["id"], "answers": answers[row["id"]], "prompt_tokens": 1}))
    (run_dir / "answers.jsonl").write_text("\n".join(lines) + "\n")
    return [answers[r["id"]]["support_simple"] for r in rows]


def check():
    rows, data_hash = load_dev_rows()
    with TemporaryDirectory() as tmp:
        runs = Path(tmp)
        try:   # apply without a fitted threshold must not run
            ct.apply(runs, ["jpt-9b"], ROOT / "lytang___llm-aggre_fact")
            raise AssertionError("apply ran without a threshold file")
        except SystemExit:
            pass

        scores = fake_dev_cache(runs, rows, data_hash)
        ct.fit(runs, ["jpt-9b"])
        fitted = json.loads((runs / "jpt-9b-dev/threshold.json").read_text())
        assert fitted["threshold"] == tune_threshold(rows, scores)["threshold"]

        # A changed development cache after the fit is refused.
        cache = runs / "jpt-9b-dev/answers.jsonl"
        original = cache.read_text()
        cache.write_text(original + "\n")
        try:
            ct.fit(runs, ["jpt-9b"])
            raise AssertionError("refit on a changed cache was not refused")
        except SystemExit:
            pass
        cache.write_text(original)

        real = ROOT / "results/runs/jpt-9b"
        if not (real / "answers.jsonl").exists():
            print("skipped apply check: no JPT-9B test cache")
            return
        (runs / "jpt-9b").mkdir()
        shutil.copy(real / "answers.jsonl", runs / "jpt-9b/answers.jsonl")
        with redirect_stdout(io.StringIO()):   # fake threshold: its test score means nothing, so don't print it
            ct.apply(runs, ["jpt-9b"], ROOT / "lytang___llm-aggre_fact")
        applied = json.loads((runs / "jpt-9b/report_dev_threshold.json").read_text())
        original_report = json.loads((real / "report.json").read_text())
        # At 0.30 the script must reproduce the benchmark's own report exactly.
        assert applied["test_at_frozen_030"]["macro_balanced_accuracy"] == original_report["macro_balanced_accuracy"]
        assert (runs / "dev_threshold_summary.md").exists()


if __name__ == "__main__":
    check()
    print("calibrate_transfer checks passed")
