"""Offline checks of the focused 10-TDM study (no GPU, no weights): the backbone prompt, the answer conversion of the
Flide adapter, the question and rule files, and analysis/make_runs.py + analysis/focused_10.py end to end on synthetic
runs laid out as the harness writes them.

    python tests/test_focused.py          # or: python -m pytest tests

The end-to-end check reads the LLM-AggreFact test and dev-jev4520 splits from dataset/ for the row ids and documents;
it is skipped when they are not there.
"""
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import rules  # noqa: E402
from core.adapter import load_adapter_class  # noqa: E402

PACK = json.loads((ROOT / "config/questions/claim_support.json").read_text())["questions"]
AGGREFACT = ROOT / "dataset/lytang--LLM-AggreFact"


class CharTokenizer:
    """One token per character: enough for the backbone's single-letter checks and prompt rendering."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


def test_backbone_prompt_has_both_orders_and_no_answer():
    cls = load_adapter_class("llm-backbone")
    chat = cls({"chat": True})
    chat.tok = CharTokenizer()
    assert [chat._letter_id(x) for x in "AB"] == [ord("A"), ord("B")]
    adapter = cls({"chat": False})
    state = {"document": "The sky is blue.", "claim": "The sky is green."}
    yes_first = adapter._render(state, PACK["support_simple"], True)
    no_first = adapter._render(state, PACK["support_simple"], False)
    assert "A. yes\nB. no" in yes_first and "A. no\nB. yes" in no_first
    assert yes_first.endswith("\nAnswer:") and PACK["support_simple"]["instructions"] in yes_first
    for text in (yes_first, no_first):            # the prompt never names the right answer
        assert "supported" not in text.lower() and "label" not in text.lower()


def test_onejev_answer_uses_slot_logits_in_key_order():
    cls = load_adapter_class("onejev-9b")
    adapter = cls()
    q = {"type": "noul", "instructions": "x"}
    p_yes = 1 / (1 + math.exp(-(2.0 - 0.5)))
    a = adapter.answer(q, {"type": "noul", "noul": p_yes}, {"slot_logits": [[2.0, 0.5]], "temperature": 1.0})
    assert a["keys"] == ["false", "true"] and a["logits"] == [0.5, 2.0]
    assert abs(rules.softmax(a["logits"])[1] - a["probs"]["true"]) < 1e-9
    choice = {"type": "choice", "instructions": "x", "criteria": {"a": "", "b": "", "c": ""}}
    a = adapter.answer(choice, {"probabilities": {"a": 0.7, "b": 0.2, "c": 0.1}}, {"slot_logits": [[1.0, 0.0, -1.0]]})
    assert a["logits"] == [1.0, 0.0, -1.0]
    a = adapter.answer(q, {"noul": 0.6}, {"slot_logits": [[1.0, 0.0], [0.0, 1.0]]})   # several orders: probs only
    assert a["logits"] is None and abs(a["probs"]["true"] - 0.6) < 1e-12


def test_question_and_rule_files():
    qs = {p.stem: json.loads(p.read_text()) for p in (ROOT / "config/questions").glob("*.json")}
    assert qs["q1_only"]["questions"]["support_simple"] == PACK["support_simple"]
    assert set(qs["q4_only"]["questions"]) == {"support_every_part"}
    assert set(qs["q5_only"]["questions"]) == {"any_unsupported"}
    assert list(qs["q6_three_way"]["questions"]["support_3way"]["criteria"]) == ["supported", "contradicted", "not_stated"]
    texts = {qs[f"p{i:02d}"]["questions"]["support_simple"]["instructions"] for i in range(1, 11)}
    assert len(texts) == 10 and PACK["support_simple"]["instructions"] not in texts
    yes, no = {"type": "noul", "probs": {"true": 0.8, "false": 0.2}}, {"type": "noul", "probs": {"true": 0.1, "false": 0.9}}
    answers = {"support_simple": yes, "unsupported_detail": no, "contradicted_detail": no}
    got = {n: rules.score(answers, json.loads((ROOT / f"config/rules/{n}.json").read_text())) for n in ("P1", "P2", "P3", "P6")}
    assert abs(got["P1"] - 0.8) < 1e-9 and abs(got["P2"] - 0.8) < 1e-9 and abs(got["P3"] - 0.9) < 1e-9
    assert abs(got["P6"] - (0.8 + 0.9 + 0.9) / 3) < 1e-9


def synthetic_results(root, rng):
    """Harness-like result folders for 3 TDMs (one with every variant) and one backbone, on a slice of the real ids."""
    sys.path.insert(0, str(ROOT))
    from core import datasets

    test, _ = datasets.load("lytang--LLM-AggreFact", "test")
    dev, _ = datasets.load("lytang--LLM-AggreFact", "dev-jev4520")
    test = [r for r in test if int(r["id"].split(":")[1]) % 15 == 0]
    dev = [r for r in dev if int(r["id"].split(":")[1]) % 3 == 0]
    sub = test[::5]
    qf1 = ["support_simple", "unsupported_detail", "contradicted_detail"]

    def put(model, split, rows, qs, skill, options=None, max_doc_chars=80000, tag="", backbone=False):
        folder = root / "results/lytang--LLM-AggreFact" / split / f"{model}_{rng.getrandbits(32):08x}_x{tag}"
        folder.mkdir(parents=True)
        identity = {"model": model, "options": options or {}, "limit": 0, "max_doc_chars": max_doc_chars}
        with (folder / "answers.jsonl").open("w") as f:
            f.write(json.dumps({"identity": identity}) + "\n")
            for r in rows:
                answers = {}
                for q in qs:
                    t = r["label"] if q in ("support_simple", "support_every_part", "support_3way") else 1 - r["label"]
                    z = (2 * t - 1) * skill + rng.gauss(0, 1.5)
                    p = 1 / (1 + math.exp(-z))
                    if q == "support_3way":
                        answers[q] = {"type": "choice", "keys": ["supported", "contradicted", "not_stated"],
                                      "probs": {"supported": p, "contradicted": (1 - p) / 2, "not_stated": (1 - p) / 2}}
                        continue
                    answers[q] = {"type": "noul", "keys": ["false", "true"], "probs": {"true": p, "false": 1 - p}}
                    if backbone:
                        a, b = z + 0.7 + rng.gauss(0, 0.5), z - 0.3 + rng.gauss(0, 0.5)
                        answers[q]["orders"] = {"yes_first": [a / 2, -a / 2], "no_first": [-b / 2, b / 2]}
                f.write(json.dumps({"id": r["id"], "source": r["dataset"], "label": r["label"], "answers": answers}) + "\n")

    for model, skill in (("elephant-1", 1.6), ("d3-flash", 2.0), ("knowline-4b", 1.8)):
        put(model, "test", test, qf1, skill)
        put(model, "dev-jev4520", dev, qf1, skill)
    for tag, qs in (("QF2", ["support_simple"]), ("QF3", ["support_every_part"]), ("QF4", ["any_unsupported"]),
                    ("QF5", ["support_3way"])):
        put("elephant-1", "test4k", sub, qs, 1.6, {"v": tag})
        put("elephant-1", "dev-jev4520", dev, qs, 1.6, {"v": tag})
    put("elephant-1", "test4k", sub, qf1, 1.6, max_doc_chars=32000, tag="-trunc32k")
    for split, rows in (("test", test), ("dev-jev4520", dev)):
        put("llm-backbone", split, rows, qf1, 0.8, {"repo": "Qwen/Qwen3.5-9B"}, backbone=True)
        put("llm-backbone", split, rows, qf1, 0.9, {"repo": "Qwen/Qwen3.5-4B"}, backbone=True)


def test_analysis_end_to_end():
    if not (AGGREFACT / "test.arrow").exists() or not (AGGREFACT / "dev-jev4520.jsonl").exists():
        print("skip test_analysis_end_to_end: LLM-AggreFact is not in dataset/")
        return
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name in ("core", "dataset", "config", "analysis", "Documentation"):
            os.symlink(ROOT / name, root / name)
        synthetic_results(root, random.Random(0))
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        runs = root / "runs.json"
        out = subprocess.run([sys.executable, str(ROOT / "analysis/make_runs.py"), "--out", str(runs)], cwd=root,
                             env=env, capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        cfg = json.loads(runs.read_text())
        assert set(cfg["systems"]) == {"A1", "A3", "A11", "BB9", "BB4"}
        assert set(cfg["variants"]["A1"]) == {"QF2", "QF3", "QF4", "QF5", "trunc32k"}
        assert cfg["groups"]["G9"] == {"backbone": "BB9", "tdms": ["A1", "A3"]}
        out = subprocess.run([sys.executable, str(ROOT / "analysis/focused_10.py"), "--runs", str(runs), "--out",
                              str(root / "tables"), "--boot", "50"], cwd=root, env=env, capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        tables = {p.name: p.read_text() for p in (root / "tables").iterdir()}
        assert set(tables) == {"table_main.md", "table_attribution.md", "table_protocol.md", "table_calibration.md"}
        main = {line.split("|")[2].strip(): line.split("|")[3].strip() for line in tables["table_main.md"].splitlines()[2:]}
        assert main["BB9 R1"] == main["BB9 R2"]          # a monotone rescaling keeps BAcc at a fitted threshold
        assert "AF-4k P4" in tables["table_protocol.md"] and "Δ_recipe" in tables["table_attribution.md"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("focused-study checks passed")
