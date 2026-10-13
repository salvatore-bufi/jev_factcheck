"""Offline checks of the benchmark core (no GPU, no model): datasets, question hash registry, decision
rules, answers, and a full runner + report pass with a fake model.

    python tests/test_core.py          # or: python -m pytest tests
"""
import json
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import core  # noqa: E402
from core import adapter as ad, datasets, questions as qs, rules  # noqa: E402

QUESTIONS = json.loads((ROOT / "config/questions/claim_support.json").read_text())

FAKE_ADAPTER = '''
from core.adapter import BaseAdapter, make_answer, option_keys

class Adapter(BaseAdapter):
    NAME = "Fake"
    SIZE = "0B"
    DEFAULTS = {"batch": 3, "temperature": 1.0}

    def load(self):
        self.calls = 0

    def score(self, states, questions):
        out = []
        for s in states:
            if "FAIL" in s["claim"]:
                out.append("synthetic failure")
                continue
            good = "good" in s["claim"]
            answers = {}
            for qid, q in questions.items():
                yes = (2.0 if good else -2.0) * (1 if qid == "support_simple" else -1)
                answers[qid] = make_answer(q["type"], option_keys(q), logits=[0.0, yes], temperature=1.0)
            out.append((answers, {"prompt_tokens": {qid: len(s["document"]) for qid in questions}}))
        return out
'''


def synthetic_rows(n=12):
    rows = []
    for i in range(n):
        label = i % 2
        rows.append({"doc": "d" * (500 + 900 * i), "claim": ("good " if label else "bad ") + f"claim {i}",
                     "label": label, "dataset": "A" if i < n // 2 else "B"})
    rows[-1]["claim"] = "FAIL bad claim"
    return rows


def test_datasets_local_and_subset():
    with TemporaryDirectory() as tmp:
        folder = Path(tmp) / "toy"
        folder.mkdir()
        (folder / "test.jsonl").write_text("\n".join(json.dumps(r) for r in synthetic_rows()) + "\n")
        (folder / "extra.json").write_text(json.dumps(synthetic_rows(4)))
        rows, info = datasets.load("toy", "test", root=tmp)
        assert len(rows) == 12 and rows[0]["id"] == "test:0" and info["split"] == "test" and info["sha256"]
        rows, info = datasets.load("toy", "extra.json", root=tmp)
        assert len(rows) == 4 and info["split"] == "extra"
        sub, _ = datasets.load("toy", "test", limit=5, root=tmp)
        again, _ = datasets.load("toy", "test", limit=5, root=tmp)
        assert [r["id"] for r in sub] == [r["id"] for r in again] and len(sub) == 5
        try:
            datasets.load("toy", "missing", root=tmp)
            raise AssertionError("missing split not reported")
        except SystemExit as exc:
            assert "not found" in str(exc)


def test_question_hash_and_registry():
    h = qs.questions_hash(QUESTIONS["questions"])
    assert len(h) == 8 and h == qs.questions_hash(json.loads(json.dumps(QUESTIONS["questions"])))
    changed = json.loads(json.dumps(QUESTIONS["questions"]))
    changed["support_simple"]["instructions"] += " "
    assert qs.questions_hash(changed) != h
    with TemporaryDirectory() as tmp:
        assert qs.register(tmp, QUESTIONS) == h
        first = (Path(tmp) / "hash_questions.json").read_text()
        qs.register(tmp, QUESTIONS)                                   # same hash: untouched
        assert (Path(tmp) / "hash_questions.json").read_text() == first
        qs.register(tmp, {"name": "other", "questions": changed})     # new hash: appended
        assert set(json.loads((Path(tmp) / "hash_questions.json").read_text())) == {h, qs.questions_hash(changed)}


def test_rules_and_answers():
    a = ad.make_answer("noul", ["false", "true"], logits=[0.0, math.log(3)], temperature=1.0)
    assert abs(a["probs"]["true"] - 0.75) < 1e-12
    assert abs(rules.take(a, "no") - 0.25) < 1e-12
    assert rules.take(a, "yes", temperature=2.0) < 0.75               # re-tempered from the stored logits
    c = ad.make_answer("choice", ["x", "y", "z"], probs={"x": 0.2, "y": 0.5, "z": 0.3})
    s = ad.make_answer("score", ["0", "1", "2"], probs=[0.0, 0.5, 0.5])
    assert rules.take(c, "option:y") == 0.5 and abs(rules.take(s, "score") - 0.75) < 1e-12
    answers = {"support_simple": ad.make_answer("noul", ["true", "false"], probs={"true": 0.9, "false": 0.1}),
               "unsupported_detail": ad.make_answer("noul", ["false", "true"], probs={"true": 0.2, "false": 0.8}),
               "contradicted_detail": ad.make_answer("noul", ["false", "true"], probs={"true": 0.6, "false": 0.4})}
    assert abs(rules.score(answers, QUESTIONS["decision"]) - 0.4) < 1e-12
    qs.validate(QUESTIONS["questions"])
    try:
        qs.validate({"q": {"type": "score", "instructions": "x", "criteria": ["only one"]}})
        raise AssertionError("bad score accepted")
    except SystemExit:
        pass


def test_runner_and_report_end_to_end():
    from core import report as rep, runner
    with TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "models/fake").mkdir(parents=True)
        (tmp / "models/fake/adapter.py").write_text(FAKE_ADAPTER)
        (tmp / "dataset/toy").mkdir(parents=True)
        (tmp / "dataset/toy/test.jsonl").write_text("\n".join(json.dumps(r) for r in synthetic_rows()) + "\n")
        saved = core.MODELS, core.RESULTS, core.CONFIG, datasets.DATASETS
        core.MODELS, core.RESULTS, core.CONFIG, datasets.DATASETS = tmp / "models", tmp / "results", tmp / "config", tmp / "dataset"
        try:
            argv = ["--model", "fake", "--dataset", "toy", "--split", "test",
                    "--questions", str(ROOT / "config/questions/claim_support.json"), "--latency-sample", "5",
                    "--throughput-sizes", "1,4", "--throughput-sample", "6", "--allow-busy-gpu", "--telemetry-interval", "0.05"]
            runner.main(argv)
            h = qs.questions_hash(QUESTIONS["questions"])
            split_dir = tmp / "results/toy/test"
            models_registry = json.loads((split_dir / "hash_fake.json").read_text())
            (phash, entry), = models_registry.items()
            assert entry["options"] == {"batch": 3, "temperature": 1.0}
            run_dir = split_dir / f"fake_{phash}_{h}"
            assert (split_dir / "hash_questions.json").exists()
            for f in ("run.json", "answers.jsonl", "batches.jsonl", "latency.jsonl", "throughput.jsonl",
                      "telemetry.jsonl", "failures.jsonl", "report.json"):
                assert (run_dir / f).exists(), f
            run = json.loads((run_dir / "run.json").read_text())
            assert run["status"] == "incomplete"                     # the FAIL row is never given a score
            report = json.loads((run_dir / "report.json").read_text())
            assert report["coverage"]["rows_scored"] == 11 and report["coverage"]["failures_logged"] >= 1
            assert report["accuracy"]["macro_balanced_accuracy"] == 1.0
            assert report["per_question"]["support_simple"]["auc_yes_vs_supported"] == 1.0
            assert report["latency"]["overall_seconds"]["n"] + report["latency"]["failed"] == 5
            assert {s["batch_size"] for s in report["throughput"]["batch_size_sweep"]} == {1, 4}
            assert "doc_chars" in report["by_length"]
            # resume: a second run scores only the missing row (which fails again) and keeps the rest
            runner.main(argv)
            assert len(rep.read_jsonl(run_dir / "answers.jsonl", skip_header=True)) == 11
            # report-time changes need no inference
            strict = rep.build_run_report(run_dir, threshold=0.99, write=False)
            assert strict["accuracy"]["supported_recall"] < 1.0
            one = rep.build_run_report(run_dir, decision=rules.single_question("support_simple"), write=False)
            assert one["accuracy"]["macro_balanced_accuracy"] == 1.0
            # other options -> another params hash, another folder; the registry gains an entry
            runner.main(argv + ["--opt", "temperature=2.0"])
            assert len(json.loads((split_dir / "hash_fake.json").read_text())) == 2
            assert len([d for d in split_dir.iterdir() if d.is_dir() and d.name.startswith("fake_")]) == 2
            # a model config file is read, and the command line overrides it
            (tmp / "fake.json").write_text(json.dumps({"_comment": "x", "temperature": 3.0, "batch": 5}))
            runner.main(argv + ["--model-config", str(tmp / "fake.json"), "--opt", "batch=4"])
            options = [e["options"] for e in json.loads((split_dir / "hash_fake.json").read_text()).values()]
            assert {"batch": 4, "temperature": 3.0} in options
            # same folder but different data is refused instead of mixed into the cache
            with (tmp / "dataset/toy/test.jsonl").open("a") as f:
                f.write(json.dumps({"doc": "x", "claim": "good extra", "label": 1, "dataset": "A"}) + "\n")
            try:
                runner.main(argv)
                raise AssertionError("changed data was mixed into the cache")
            except SystemExit as exc:
                assert "different configuration" in str(exc)
        finally:
            core.MODELS, core.RESULTS, core.CONFIG, datasets.DATASETS = saved


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("core checks passed")
