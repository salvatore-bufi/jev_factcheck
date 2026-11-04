"""The row format shared by every evidence source, how a row is assembled, and its validation.

One row is one claim on one document:

    id, source, orig_split        unique id, source name, split in the original release
    claim, document, label        label 1 = the document supports the claim (also what benchmark.py reads)
    sentences                     [start, end) character spans of the document's sentences, from evidence.sentences
    evidence, evidence_sets       indices into `sentences`; `evidence` is the union of the alternative sets;
                                  empty when label = 0
    contra                        sentences that contradict the claim, when the source says (analysis only)
    evidence_quality              gold | construct | ablation | none
    evidence_term                 the evidence loss the row receives: supervised (label 1 with evidence),
                                  mil_neg (label 0), mil_pos (label 1 without evidence), none (no evidence loss)
    doc_id                        rows with the same document share it (splits are made by doc_id)
    meta                          source-specific details, including how the gold sentences were aligned

Weights and quotas are not here (they belong to the training mix), nor token positions (they depend on the model).
"""
from dataclasses import dataclass

from evidence import sentences as sent

QUALITIES = ("gold", "construct", "ablation", "none")
TERMS = ("supervised", "mil_neg", "mil_pos", "none")


@dataclass
class Skip:
    """A raw record that does not become a row; `reason` is counted in the build statistics."""
    reason: str


class DocBuilder:
    """Concatenates text pieces into a document and remembers where each piece landed."""

    def __init__(self):
        self.parts, self.length = [], 0

    def add(self, text):
        """Append text; -> its (start, end) span, trimmed of surrounding whitespace."""
        base = self.length
        self.parts.append(text)
        self.length += len(text)
        start, end = 0, len(text)
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        return base + start, base + end

    def text(self):
        return "".join(self.parts)


def default_term(label, evidence):
    if label == 0:
        return "mil_neg"
    return "supervised" if evidence else "mil_pos"


def assemble(*, id, source, orig_split, claim, document, label, gold_sets=(), contra=(), evidence_quality,
             doc_id, meta=None, evidence_term=None, spans=None):
    """Build a row from character-level gold spans.

    gold_sets: list of alternative evidence sets, each a list of (start, end) character spans in `document`
               (the source's own sentences); they are mapped onto our sentences with evidence.sentences.align.
    contra: character spans of contradicting sentences.
    spans: sent.split(document), when the caller already computed it.
    """
    spans = spans if spans is not None else sent.split(document)
    meta = dict(meta or {})
    sets, gold_total, one_to_one = [], 0, 0
    for gold in gold_sets:
        matched = sent.align(spans, gold)
        gold_total += len(matched)
        one_to_one += sum(len(m) == 1 for m in matched)
        sets.append(sorted({k for m in matched for k in m}))
    sets = [s for s in sets if s]
    contra_idx = sorted({k for m in sent.align(spans, list(contra)) for k in m}) if contra else []
    evidence = sorted({k for s in sets for k in s})
    meta["gold_sentences"] = gold_total
    meta["gold_one_to_one"] = one_to_one
    return {
        "id": id, "source": source, "orig_split": orig_split,
        "claim": claim, "document": document, "label": int(label),
        "sentences": [list(s) for s in spans],
        "evidence": evidence, "evidence_sets": sets, "contra": contra_idx,
        "evidence_quality": evidence_quality,
        "evidence_term": evidence_term or default_term(int(label), evidence),
        "doc_id": doc_id, "meta": meta,
    }


def validate(row):
    """Raise ValueError if the row breaks the format."""
    doc, spans = row["document"], row["sentences"]
    if row["label"] not in (0, 1):
        raise ValueError(f"{row['id']}: label {row['label']!r}")
    if not row["claim"].strip() or not doc.strip():
        raise ValueError(f"{row['id']}: empty claim or document")
    if not spans:
        raise ValueError(f"{row['id']}: no sentences")
    prev = 0
    for s, e in spans:
        if not (prev <= s < e <= len(doc)):
            raise ValueError(f"{row['id']}: bad sentence span {(s, e)}")
        if doc[prev:s].strip():
            raise ValueError(f"{row['id']}: text outside any sentence at {prev}:{s}")
        prev = e
    if doc[prev:].strip():
        raise ValueError(f"{row['id']}: text outside any sentence after {prev}")
    n = len(spans)
    for name in ("evidence", "contra"):
        if any(not 0 <= k < n for k in row[name]) or row[name] != sorted(set(row[name])):
            raise ValueError(f"{row['id']}: bad {name} indices")
    if sorted({k for s in row["evidence_sets"] for k in s}) != row["evidence"]:
        raise ValueError(f"{row['id']}: evidence is not the union of evidence_sets")
    if row["label"] == 0 and row["evidence"]:
        raise ValueError(f"{row['id']}: label 0 with evidence")
    if row["evidence_quality"] not in QUALITIES or row["evidence_term"] not in TERMS:
        raise ValueError(f"{row['id']}: bad evidence_quality or evidence_term")
    term, label, ev = row["evidence_term"], row["label"], row["evidence"]
    if (term == "supervised" and not (label == 1 and ev)) or (term == "mil_neg" and label != 0) \
            or (term == "mil_pos" and (label != 1 or ev)):
        raise ValueError(f"{row['id']}: evidence_term {term} does not fit label {label} and evidence {ev}")
    if not row["doc_id"]:
        raise ValueError(f"{row['id']}: empty doc_id")
