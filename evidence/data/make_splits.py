"""STEP 1a — Split the evidence rows into frozen train / dev / test sets, by document.

Where it sits in the workflow:
    evidence.data.build  ->  [evidence.data.make_splits]  ->  evidence.data.make_mix  ->  evidence.extract  -> ...

Run:
    python -m evidence.data.make_splits
Reads:   dataset/evidence/{scifact,hover,wice}.jsonl  (rows built by evidence.data.build)
Writes:  dataset/evidence/splits/<set>.jsonl          the rows of each set, unchanged
         dataset/evidence/splits/<set>.stats.json     counts, labels, gold sentences, distance between gold sentences
         dataset/evidence/splits/manifest.json        input hashes, output hashes, dropped rows with reason, options

Rules (plan: Documentation/fine_tuning_factcheck_evidence/testa_per_frase/04_split_e_mix.md; decisions: CLAUDE_README.md):
    SciFact  train + validation pooled, then 80 / 10 / 10 by doc_id (the original split is by claim, not by
             document: 295 train rows share their abstract with validation)
    HoVer    original train -> 90 / 10 train / dev by doc_id, dev with "exact" rows only;
             original dev, "exact" rows -> test; "db_split" rows outside train are dropped
    WiCE     original train without aggrefact_overlap -> train; original dev + test -> test (out of domain)
Then a document found in more than one stage stays only in the latest (test > dev > train): no document is shared.

Sets written:
    train              every training row (all labels and evidence terms: the mix decides what the loss sees)
    dev_evid_select    half of the dev positives (SciFact, HoVer), by doc_id: every choice is made here
    dev_evid_confirm   the other half: confirms the choices
    dev_evid_neg       the dev negatives (SciFact, HoVer)
    test_evid_id       SciFact and HoVer test, all labels
    test_evid_ood      WiCE dev + test, all labels
"""
import argparse
import hashlib
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:                # lets the file run as a script too, not only with python -m
    sys.path.insert(0, str(ROOT))

from evidence import config  # noqa: E402
from evidence.data import io  # noqa: E402

SOURCES = ("scifact", "hover", "wice")
SETS = ("train", "dev_evid_select", "dev_evid_confirm", "dev_evid_neg", "test_evid_id", "test_evid_ood")
STAGE = {"train": 0, "dev": 1, "test": 2}    # order of the stages: a shared document stays in the latest one
DISTANCE_BUCKETS = ("0-1", "2-9", "10-19", ">=20")


def unit(seed, salt, doc_id):
    """A pseudo-random number in [0, 1) that depends only on (seed, salt, doc_id).

    Input:  seed (int), salt (str, e.g. "hover": makes the draws of different sources independent), doc_id (str).
    Output: float in [0, 1).
    The same document always gets the same number, whatever the order of the rows or how many rows are added:
    "unit(...) < 0.1 -> dev" is therefore a stable 10% split by document.
    """
    digest = hashlib.sha256(f"{seed}:{salt}:{doc_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2 ** 64


def assign_scifact(rows, seed, fractions, dropped):
    """SciFact: ignore the original split, pool everything, then train / dev / test by document.

    Input:  rows (list of row dicts), seed, fractions = (train share, dev share), dropped (Counter, unused here).
    Output: list of (stage, row) with stage in "train" / "dev" / "test".
    """
    f_train, f_dev = fractions
    out = []
    for r in rows:
        u = unit(seed, "scifact", r["doc_id"])
        stage = "train" if u < f_train else "dev" if u < f_train + f_dev else "test"
        out.append((stage, r))
    return out


def assign_hover(rows, seed, dev_fraction, dropped):
    """HoVer: test = original dev ("exact" rows only); dev = a share of the original train's documents ("exact" only).

    Input:  rows, seed, dev_fraction (share of the original train's documents that go to dev), dropped (Counter).
    Output: list of (stage, row). Rows not kept are counted in `dropped` with their reason.
    "db_split" rows took a missing page from the HoVer db: their gold sentence is off by one in ~2% of cases, so they
    may train (with a lower weight, decided by the mix) but never measure.
    """
    out = []
    for r in rows:
        exact = r["meta"]["gold_alignment"] == "exact"
        if r["orig_split"] == "dev":
            stage = "test"
        else:
            stage = "dev" if unit(seed, "hover", r["doc_id"]) < dev_fraction else "train"
        if stage != "train" and not exact:
            dropped[f"{stage}_db_split"] += 1
            continue
        out.append((stage, r))
    return out


def assign_wice(rows, seed, unused, dropped):
    """WiCE: train = original train outside LLM-AggreFact; original dev + test = out-of-domain test.

    Input:  rows, seed (unused), unused, dropped (Counter).
    Output: list of (stage, row).
    meta.aggrefact_overlap is "dev"/"test" when the page is also in LLM-AggreFact (our final benchmark), False when it
    is not, None when unknown. Only False may train: in doubt, the row is dropped.
    """
    out = []
    for r in rows:
        if r["orig_split"] in ("dev", "test"):
            out.append(("test", r))
        elif r["meta"].get("aggrefact_overlap") is False:
            out.append(("train", r))
        else:
            dropped["train_aggrefact_overlap"] += 1
    return out


ASSIGN = {"scifact": assign_scifact, "hover": assign_hover, "wice": assign_wice}


def remove_shared_documents(assigned, dropped):
    """Make sure no document is in two stages: a shared document stays only in the latest stage (test > dev > train).

    Input:  assigned (list of (stage, row) of all sources), dropped (dict source -> Counter).
    Output: the kept (stage, row) pairs. Removed rows are counted as "<stage>_document_in_later_split".
    """
    latest = defaultdict(int)                            # doc_id -> highest stage number it appears in
    for stage, r in assigned:
        latest[r["doc_id"]] = max(latest[r["doc_id"]], STAGE[stage])
    kept = []
    for stage, r in assigned:
        if STAGE[stage] < latest[r["doc_id"]]:
            dropped[r["source"]][f"{stage}_document_in_later_split"] += 1
        else:
            kept.append((stage, r))
    return kept


def make_sets(assigned, seed):
    """Turn the stages into the named sets.

    Input:  assigned (list of (stage, row)), seed.
    Output: dict set name -> list of rows (sorted by id, so the files are stable).
    Dev positives are halved by document into select / confirm; dev negatives go to dev_evid_neg;
    test rows go to test_evid_ood (WiCE) or test_evid_id (SciFact, HoVer).
    """
    sets = {name: [] for name in SETS}
    for stage, r in assigned:
        if stage == "train":
            name = "train"
        elif stage == "dev":
            if r["label"] == 0:
                name = "dev_evid_neg"
            elif unit(seed, "dev_half", r["doc_id"]) < 0.5:
                name = "dev_evid_select"
            else:
                name = "dev_evid_confirm"
        else:
            name = "test_evid_ood" if r["source"] == "wice" else "test_evid_id"
        sets[name].append(r)
    for rows in sets.values():
        rows.sort(key=lambda r: r["id"])
    return sets


def distance_bucket(evidence):
    """Bucket of the distance, in sentences, between the first and the last gold sentence.

    Input:  evidence (sorted list of gold sentence indices, not empty).
    Output: one of "0-1", "2-9", "10-19", ">=20".
    """
    d = evidence[-1] - evidence[0]
    if d <= 1:
        return "0-1"
    if d <= 9:
        return "2-9"
    if d <= 19:
        return "10-19"
    return ">=20"


def set_stats(rows):
    """Summary of one set, saved next to it.

    Input:  rows (list of row dicts).
    Output: dict with counts by source / label / evidence term, gold sentences per positive, distance buckets,
            prompt length (meta.prompt_tokens, estimated by the build with KnowLine's tokenizer).
    """
    pos = [r for r in rows if r["evidence"]]
    toks = [r["meta"]["prompt_tokens"] for r in rows if "prompt_tokens" in r["meta"]]
    buckets = Counter(distance_bucket(r["evidence"]) for r in pos)
    return {
        "rows": len(rows),
        "documents": len({r["doc_id"] for r in rows}),
        "by_source": dict(Counter(r["source"] for r in rows)),
        "by_label": dict(Counter(r["label"] for r in rows)),
        "by_evidence_term": dict(Counter(r["evidence_term"] for r in rows)),
        "gold_per_positive_mean": round(statistics.mean(len(r["evidence"]) for r in pos), 2) if pos else None,
        "positives_with_2plus_gold": sum(len(r["evidence"]) >= 2 for r in pos),
        "gold_distance_sentences": {b: buckets[b] for b in DISTANCE_BUCKETS},
        "prompt_tokens_median": statistics.median(toks) if toks else None,
        "prompt_tokens_max": max(toks) if toks else None,
    }


def check(sets):
    """Stop with an error if the sets break the rules.

    Input:  sets (dict name -> rows).
    Output: None; raises SystemExit if a document sits in two stages, an id repeats, or a WiCE row that overlaps
            LLM-AggreFact is in train.
    """
    stages = defaultdict(set)
    for name, rows in sets.items():
        for r in rows:
            stages[r["doc_id"]].add(name.split("_")[0])           # "train" / "dev" / "test"
    shared = [d for d, s in stages.items() if len(s) > 1]
    ids = [r["id"] for rows in sets.values() for r in rows]
    leaked = [r["id"] for r in sets["train"]
              if r["source"] == "wice" and r["meta"].get("aggrefact_overlap") is not False]
    errors = []
    if shared:
        errors.append(f"{len(shared)} documents in more than one stage (e.g. {shared[:3]})")
    if len(ids) != len(set(ids)):
        errors.append("duplicate ids across sets")
    if leaked:
        errors.append(f"{len(leaked)} WiCE rows overlapping LLM-AggreFact in train")
    if errors:
        raise SystemExit("\n".join(errors))


def main(argv=None):
    """Command line entry point: read the sources, split, check, write sets + stats + manifest."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=config.DATASETS, help="folder with <source>.jsonl")
    p.add_argument("--out", type=Path, default=config.SPLITS)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--scifact-fractions", type=float, nargs=2, default=(0.8, 0.1), metavar=("TRAIN", "DEV"))
    p.add_argument("--hover-dev-fraction", type=float, default=0.10)
    opts = p.parse_args(argv)
    params = {"scifact": opts.scifact_fractions, "hover": opts.hover_dev_fraction, "wice": None}

    # 1. every row of every source gets a stage
    assigned, dropped, inputs = [], defaultdict(Counter), {}
    for src in SOURCES:
        path = opts.data / f"{src}.jsonl"
        assigned += ASSIGN[src](io.read_jsonl(path), opts.seed, params[src], dropped[src])
        inputs[src] = {"file": str(path), "sha256": io.sha256(path)}
    # 2. no shared documents, 3. named sets, 4. checks
    assigned = remove_shared_documents(assigned, dropped)
    sets = make_sets(assigned, opts.seed)
    check(sets)

    # 5. write
    outputs = {}
    for name, rows in sets.items():
        path = opts.out / f"{name}.jsonl"
        io.write_jsonl(path, rows)
        st = set_stats(rows)
        io.write_json(opts.out / f"{name}.stats.json", st)
        outputs[name] = {"file": path.name, "rows": len(rows), "sha256": io.sha256(path)}
        print(f"{name:17s} {len(rows):6d} rows  {st['by_source']}  labels {st['by_label']}")
    build_manifest = opts.data / "manifest.json"
    io.write_json(opts.out / "manifest.json", {
        "inputs": inputs,
        "build_manifest_sha256": io.sha256(build_manifest) if build_manifest.is_file() else None,
        "outputs": outputs,
        "dropped": {s: dict(c) for s, c in dropped.items()},
        "options": {"seed": opts.seed, "scifact_fractions": list(opts.scifact_fractions),
                    "hover_dev_fraction": opts.hover_dev_fraction},
        "git_commit": io.git_commit(),
    })
    print("dropped:", {s: dict(c) for s, c in dropped.items()})


if __name__ == "__main__":
    main()
