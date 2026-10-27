"""Write analysis/focused_runs.json: which result folder holds which (system, question file, split) of the focused study.

    python analysis/make_runs.py [--out analysis/focused_runs.json]

Folders are recognised by the identity header and the first answer of their answers.jsonl, never by name, and runs
made with --limit are ignored. Two complete folders for the same cell stop the script: delete or move the stale one.
"""
import argparse
import json
from pathlib import Path

ROOT = Path("results/lytang--LLM-AggreFact")
TDMS = {  # adapter folder -> (plan ID, Hugging Face repo, group)
    "elephant-1": ("A1", "baselabs/elephant-1", "G9"),
    "startlux-9b-instant": ("A2", "startlux-models/StartLux-Decision-v1.1-9B-Instant", "G9"),
    "d3-flash": ("A3", "vllm-sr/d3-flash", "G9"),
    "onejev-9b": ("A4", "OmniJev/OneJev1.6-9B", "G9"),
    "nimble-9b": ("A5", "bespokelabs/Bespoke-Nimble-9B-v3", "G9"),
    "clef-flash": ("A7", "Cloudflare/clef-flash", "G9"),
    "knowline-4b": ("A11", "PelaAI/KnowLine-4B-Gen5.5", "G4"),
    "startlux-4b-instant": ("A12", "startlux-models/StartLux-Decision-v1.1-4B-Instant", "G4"),
    "pplx-decider-flash": ("A13", "perplexity-ai/pplx-decider-flash-v1.2", "G4"),
    "jeffan-4b": ("A14", "tattrongvu/Jeffan-4B-2010", "G4"),
}
BACKBONES = {"Qwen/Qwen3.5-9B": ("BB9", "G9"), "Qwen/Qwen3.5-4B": ("BB4", "G4")}
CHECKERS = {"factcg": "factcg_support", "minicheck-7b": "minicheck_support"}


def question_file(identity, first):
    keys = set(first["answers"])
    if identity.get("max_doc_chars") == 32000:
        return "trunc32k" if "unsupported_detail" in keys else None
    for name, key in (("QF1", "unsupported_detail"), ("QF3", "support_every_part"), ("QF4", "any_unsupported"),
                      ("QF5", "support_3way")):
        if key in keys:
            return name
    if keys == {"support_simple"}:
        return "QF2"
    return "QF1" if keys & set(CHECKERS.values()) else None


ap = argparse.ArgumentParser()
ap.add_argument("--out", default="analysis/focused_runs.json")
out_path = Path(ap.parse_args().out)

cells = {}
for answers in sorted(ROOT.glob("*/*/answers.jsonl")):
    with answers.open() as f:
        identity = json.loads(f.readline())["identity"]
        line = f.readline()
    if not line or identity.get("limit"):
        continue
    model, split = identity["model"], answers.parent.parent.name
    if model in TDMS:
        system = TDMS[model][0]
    elif model == "llm-backbone":
        system = BACKBONES.get(identity["options"].get("repo"), (None,))[0]
    elif model in CHECKERS:
        system = model
    else:
        continue
    qf = question_file(identity, json.loads(line))
    if system is None or qf is None or split not in ("test", "dev-jev4520", "test4k"):
        continue
    cell = (system, qf, {"dev-jev4520": "dev"}.get(split, split))
    if cell in cells:
        raise SystemExit(f"two folders for {cell}: {cells[cell]} and {answers.parent}")
    cells[cell] = str(answers.parent)

cfg = {"systems": {}, "groups": {}, "variants": {}, "test4k_ids_from": None}
for model, (sid, repo, group) in TDMS.items():
    if (sid, "QF1", "test") not in cells or (sid, "QF1", "dev") not in cells:
        print(f"missing QF1 test/dev for {sid} ({model}): left out")
        continue
    cfg["systems"][sid] = {"kind": "tdm", "id": sid, "repo": repo,
                           "test": cells[sid, "QF1", "test"], "dev": cells[sid, "QF1", "dev"]}
    cfg["groups"].setdefault(group, {"backbone": None, "tdms": []})["tdms"].append(sid)
    for qf in ("QF2", "QF3", "QF4", "QF5", "trunc32k"):
        if (sid, qf, "test4k") in cells:
            v = {"test": cells[sid, qf, "test4k"]}
            if (sid, qf, "dev") in cells:
                v["dev"] = cells[sid, qf, "dev"]
            cfg["variants"].setdefault(sid, {})[qf] = v
            cfg["test4k_ids_from"] = cfg["test4k_ids_from"] or v["test"]
for repo, (sid, group) in BACKBONES.items():
    if (sid, "QF1", "test") in cells and (sid, "QF1", "dev") in cells and group in cfg["groups"]:
        cfg["systems"][sid] = {"kind": "backbone", "id": sid, "repo": repo,
                               "test": cells[sid, "QF1", "test"], "dev": cells[sid, "QF1", "dev"]}
        cfg["groups"][group]["backbone"] = sid
cfg["groups"] = {g: v for g, v in cfg["groups"].items() if v["backbone"]}
for name, question in CHECKERS.items():
    if (name, "QF1", "test") in cells and (name, "QF1", "dev") in cells:
        cfg["systems"][name] = {"kind": "checker", "id": name, "test": cells[name, "QF1", "test"],
                                "dev": cells[name, "QF1", "dev"],
                                "rule": {"combine": "single", "inputs": [{"question": question, "take": "yes"}]}}
out_path.write_text(json.dumps(cfg, indent=2) + "\n")
print(f"{out_path}: {len(cfg['systems'])} systems, groups {sorted(cfg['groups'])}, "
      f"variants for {len(cfg['variants'])} TDMs")
