"""STEP 11b — Build a release folder: everything the server needs, frozen with hashes (doc 12 §3-6).

Where it sits in the workflow:
    best run (evidence.train_head)  ->  [evidence.release]  ->  releases/<name>/  ->  evidence.serve / eval on test

Run:
    python -m evidence.release --run data/evidence/runs/knowline-4b/<id> --name knowline-4b-fc-ev-v1
    python -m evidence.release --verify releases/knowline-4b-fc-ev-v1        # check the hashes against this machine

Writes releases/<name>/:
    head.safetensors, config.json, selection.json   the head, its config (incl. feature settings), the selection rule
    decision.json      every setting of the release in one place (model, question, preamble, threshold, layers,
                       selection, splitter, token limit)
    sentences/         a copy of evidence/sentences.py and of NLTK's English punkt_tab parameters: the splitter is
                       part of the model (doc 12 §4)
    frozen.json        sha256 of the head, the splitter, punkt_tab, render.py, model.py, config.py, question, splits
                       and mix manifests. evidence.serve checks them at start-up (verify())
    report.md          the run's dev metrics (+ test and latency when they exist in the run folder)
    MODEL_CARD.md      what the model does and does not do
The decision model's weights are not copied: they are KnowLine-4B-Gen5.5 at the revision in decision.json.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evidence import config  # noqa: E402
from evidence.data import io  # noqa: E402

CODE = ("sentences.py", "render.py", "model.py", "config.py")


def punkt_dir():
    """Folder of NLTK's English punkt_tab parameters on this machine."""
    import nltk
    return Path(nltk.data.find("tokenizers/punkt_tab/english"))


def hashes():
    """sha256 of the code and data that define the behaviour of the release on THIS machine.

    Output: dict name -> sha256 (code files of evidence/, every punkt_tab file, the mix and splits manifests).
    """
    here = Path(__file__).resolve().parent
    out = {f"evidence/{f}": io.sha256(here / f) for f in CODE}
    for f in sorted(punkt_dir().iterdir()):
        out[f"punkt_tab/english/{f.name}"] = io.sha256(f)
    return out


def verify(folder):
    """Stop if this machine's splitter, punkt_tab or code differ from the release (frozen.json).

    Input:  folder (release folder).  Output: None; raises SystemExit listing the differences.
    """
    frozen = json.loads((Path(folder) / "frozen.json").read_text())["code"]
    now = hashes()
    diff = [k for k in frozen if now.get(k) != frozen[k]]
    if diff:
        raise SystemExit(f"{folder}: different from the release: {diff} (sentence splitting or prompts would differ)")


def model_card(name, decision):
    """Text of MODEL_CARD.md."""
    return f"""# {name}

Fact checking of a claim against a document, with **evidence at sentence level**, in ONE forward of the decision
model {decision['model_repo']} (revision {decision['model_revision'][:7]}). Nothing is generated.

- Input: a document and a claim, in English. Output: P(supported) and, only when the claim is supported
  (P > {decision['threshold']}), 1 to {decision['selection']['max_shown']} sentences of the document that support it,
  with their character positions in the document as sent.
- When the claim is not supported nothing is highlighted: the model does not say what is missing or contradicted.
- The prompt contains one line with the claim before the document: `{decision['preamble']}`.
- Documents up to {decision['max_prompt_tokens']} prompt tokens; longer ones are refused, never truncated.
- Sentences come from the splitter shipped in `sentences/` (NLTK punkt_tab, English, plus fixed rules): the evidence
  head was trained on exactly these sentences.
- The decision model is frozen: the verdict is the same as the model without the evidence head.
- Licences: KnowLine-4B weights Apache-2.0; training data of the head: HoVer (CC BY-SA), SciFact, WiCE (their
  licences).
"""


def build(run_dir, name):
    """Create releases/<name>/ from a run folder (see the file header). Output: the folder Path."""
    run_dir = Path(run_dir)
    out = config.RELEASES / name
    if out.exists():
        raise SystemExit(f"{out} exists: releases are never overwritten; choose another name")
    out.mkdir(parents=True)
    for f in ("head.safetensors", "config.json", "selection.json"):
        shutil.copy2(run_dir / f, out / f)
    cfg = json.loads((run_dir / "config.json").read_text())
    rule = json.loads((run_dir / "selection.json").read_text())
    (out / "sentences").mkdir()
    shutil.copy2(Path(__file__).resolve().parent / "sentences.py", out / "sentences" / "sentences.py")
    shutil.copytree(punkt_dir(), out / "sentences" / "punkt_tab" / "english")
    import nltk
    decision = {
        "model": name, "model_repo": config.MODEL_REPO, "model_revision": config.MODEL_REVISION,
        "question_id": config.QUESTION_ID, "question": config.QUESTION, "take": "yes",
        "preamble": cfg["feature_meta"]["preamble"], "threshold": config.VERDICT_THRESHOLD,
        "evidence": {"mix_layers": cfg["layers_read"], "attention_layers": config.FULL_ATTENTION_LAYERS,
                     "features": cfg["parts"] + (["att"] if cfg["head"].get("use_att") else []),
                     "head_config": cfg["id"]},
        "selection": dict(rule, max_shown=config.MAX_SHOWN, min_shown=config.MIN_SHOWN),
        "segmenter": {"file": "sentences/sentences.py", "nltk": nltk.__version__},
        "max_prompt_tokens": config.MAX_TOKENS_SERVE, "source_run": str(run_dir),
    }
    io.write_json(out / "decision.json", decision)
    manifests = {p.name: io.sha256(p) for p in (config.SPLITS / "manifest.json",
                                                 config.MIXES / "ev_head.manifest.json") if p.is_file()}
    io.write_json(out / "frozen.json", {"head": io.sha256(out / "head.safetensors"), "code": hashes(),
                                        "manifests": manifests, "git_commit": io.git_commit()})
    lines = [f"# Report — {name}", "", f"Source run: `{run_dir}`", ""]
    for f in sorted(run_dir.glob("eval/*/metrics.json")) + [run_dir / "dev_metrics.json", run_dir / "latency.json"]:
        if f.is_file():
            lines += [f"## {f.parent.name if f.name == 'metrics.json' else f.stem}", "", "```json",
                      f.read_text().strip(), "```", ""]
    (out / "report.md").write_text("\n".join(lines))
    (out / "MODEL_CARD.md").write_text(model_card(name, decision))
    return out


def main(argv=None):
    """Command line entry point (see the file header)."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, help="run folder to release")
    p.add_argument("--name", help="release name, e.g. knowline-4b-fc-ev-v1")
    p.add_argument("--verify", type=Path, help="check an existing release against this machine")
    opts = p.parse_args(argv)
    if opts.verify:
        verify(opts.verify)
        print(f"{opts.verify}: hashes match")
    elif opts.run and opts.name:
        print(f"-> {build(opts.run, opts.name)}")
    else:
        p.error("give --run and --name, or --verify")


if __name__ == "__main__":
    main()
