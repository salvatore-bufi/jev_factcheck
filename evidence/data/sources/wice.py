"""WiCE (tasksource/wice): Wikipedia claims against the web page they cite, with supporting-sentence sets.

All three releases (train, dev, test) are used; our own split comes later. Document = the page's lines (`evidence`)
joined by line breaks, so titles and list lines stay separate. supported -> label 1 with the alternative
`supporting_sentences` sets; not_supported -> label 0; partially_supported -> label 0 with no evidence loss
(`--wice-partial negative`, default) or dropped (`--wice-partial drop`).

WiCE is also an LLM-AggreFact source. AggreFact keeps WiCE's whole claims while this release splits them into
sub-claims, so the overlap is found by document (the cited page): meta.aggrefact_overlap is "dev" or "test" when the
page is among AggreFact's Wice rows of that split, false otherwise, null when AggreFact is not available locally.
(Measured: WiCE dev is AggreFact dev, WiCE test is AggreFact test, 2 train rows share a page with the test.)
"""
import hashlib
import json
import re
import sys

from evidence.data.schema import DocBuilder, Skip, assemble
from evidence.data.sources import hub_files

NAME = "wice"
REPO = "tasksource/wice"
SPLITS = ("train", "dev", "test")
AGGREFACT = "lytang--LLM-AggreFact"


def doc_key(text):
    """The start of a document reduced to lowercase letters and digits: the same page in WiCE and AggreFact."""
    return re.sub(r"[^a-z0-9]", "", text.replace("(meta data)", "").lower())[:400]


def aggrefact_docs(prov):
    """{doc_key: "dev" | "test"} of AggreFact's Wice rows; None if AggreFact is not available locally."""
    from core import datasets
    docs = {}
    for split in ("dev", "test"):
        try:
            rows, info = datasets.load(AGGREFACT, split)
        except SystemExit as err:
            print(f"wice: AggreFact {split} not available, aggrefact_overlap left empty ({err})", file=sys.stderr)
            return None
        prov.setdefault("aggrefact", {})[split] = info["sha256"]
        for r in rows:
            if r["dataset"].lower() == "wice":
                docs.setdefault(doc_key(r["doc"]), split)
    return docs


def load_raw(opts, prov):
    overlap = aggrefact_docs(prov)
    for split in SPLITS:
        files, revision = hub_files(REPO, f"{split}.jsonl")
        prov.setdefault("files", []).extend(str(f) for f in files)
        prov["revision"] = revision
        with open(files[0]) as handle:
            for i, line in enumerate(handle):
                row = json.loads(line)
                row["orig_split"], row["index"] = split, i
                key = doc_key("\n".join(row["evidence"]))
                row["aggrefact_overlap"] = None if overlap is None else overlap.get(key, False)
                yield row


def convert(raw, opts):
    label_name = raw["label"]
    if label_name == "partially_supported" and getattr(opts, "wice_partial", "negative") == "drop":
        return Skip("partially_supported_dropped")
    if label_name not in ("supported", "not_supported", "partially_supported"):
        return Skip(f"label_{label_name}")
    doc = DocBuilder()
    spans = []
    for i, line in enumerate(raw["evidence"]):
        if i:
            doc.add("\n")
        spans.append(doc.add(re.sub(r"\s*\n\s*", " ", line)))
    document = doc.text()
    if not document.strip():
        return Skip("empty_document")
    gold_sets = []
    if label_name == "supported":
        for s in raw["supporting_sentences"]:
            gold = [spans[i] for i in s if 0 <= i < len(spans) and spans[i][0] < spans[i][1]]
            if gold:
                gold_sets.append(gold)
    meta = raw.get("meta") or {}
    return assemble(
        id=f"wice:{raw['orig_split']}:{meta.get('id', raw['index'])}",
        source=NAME, orig_split=raw["orig_split"], claim=raw["claim"].strip(), document=document,
        label=1 if label_name == "supported" else 0,
        gold_sets=gold_sets,
        evidence_quality="gold" if gold_sets else "none",
        evidence_term="none" if label_name == "partially_supported" else None,
        doc_id="wice:" + hashlib.sha1(document.encode()).hexdigest()[:16],
        meta={"orig_label": label_name, "wice_id": meta.get("id"), "claim_title": meta.get("claim_title"),
              "aggrefact_overlap": raw.get("aggrefact_overlap")},
    )
