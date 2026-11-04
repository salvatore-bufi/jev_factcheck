"""SciFact (allenai/scifact_entailment): biomedical claims against one paper abstract, with rationale sentences.

Document = title, a line break, then the abstract's sentences joined by spaces. SUPPORT -> label 1 with the
rationale as evidence; CONTRADICT -> label 0, rationale kept as `contra`; NEI -> label 0.
"""
import pyarrow.parquet as pq

from evidence.data.schema import DocBuilder, Skip, assemble
from evidence.data.sources import hub_files

NAME = "scifact"
REPO = "allenai/scifact_entailment"
SPLITS = ("train", "validation")
LABELS = {"SUPPORT": 1, "CONTRADICT": 0, "NEI": 0}


def load_raw(opts, prov):
    for split in SPLITS:
        files, revision = hub_files(REPO, f"default/{split}/*.parquet")
        prov.setdefault("files", []).extend(str(f) for f in files)
        prov["revision"] = revision
        for path in files:
            for row in pq.read_table(path).to_pylist():
                row["orig_split"] = split
                yield row


def convert(raw, opts):
    verdict = raw["verdict"]
    if verdict not in LABELS:
        return Skip(f"verdict_{verdict}")
    sentences = [s.strip() for s in raw["abstract"]]
    if not any(sentences):
        return Skip("empty_abstract")
    doc = DocBuilder()
    doc.add((raw.get("title") or "").strip())
    doc.add("\n")
    spans = []
    for i, s in enumerate(sentences):
        if i:
            doc.add(" ")
        spans.append(doc.add(s))
    rationale = [spans[i] for i in raw["evidence"] if 0 <= i < len(spans) and spans[i][0] < spans[i][1]]
    label = LABELS[verdict]
    return assemble(
        id=f"scifact:{raw['orig_split']}:{raw['claim_id']}:{raw['abstract_id']}",
        source=NAME, orig_split=raw["orig_split"], claim=raw["claim"].strip(), document=doc.text(), label=label,
        gold_sets=[rationale] if verdict == "SUPPORT" and rationale else [],
        contra=rationale if verdict == "CONTRADICT" else (),
        evidence_quality="gold" if verdict == "SUPPORT" and rationale else "none",
        doc_id=f"scifact:{raw['abstract_id']}",
        meta={"orig_label": verdict, "claim_id": raw["claim_id"], "abstract_id": raw["abstract_id"]},
    )
