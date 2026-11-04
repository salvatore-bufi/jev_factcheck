"""Offline checks of the evidence data builders (no download, no GPU): sentence splitting and alignment, the row
format, and each source's conversion on hand-written raw records.

    python tests/test_evidence_data.py          # or: python -m pytest tests

Needs NLTK's punkt_tab (see evidence/sentences.py).
"""
from pathlib import Path
import re
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evidence import sentences as sent  # noqa: E402
from evidence.data import schema  # noqa: E402
from evidence.data.sources import hover, scifact, wice  # noqa: E402

OPTS = SimpleNamespace(wice_partial="negative", hover_db_fallback=False, limit=0)


def _covered(text, spans):
    joined = "".join(text[s:e] for s, e in spans)
    return re.sub(r"\s", "", joined) == re.sub(r"\s", "", text)


def _ordered(spans):
    return all(s < e for s, e in spans) and all(spans[i][1] <= spans[i + 1][0] for i in range(len(spans) - 1))


def _evidence_text(row):
    return [row["document"][slice(*row["sentences"][k])] for k in row["evidence"]]


def test_split_cases():
    cases = {
        "title and paragraphs": "Kristian Zahrtmann\nKristian Zahrtmann was a Danish painter. He was born in 1843.\n\n"
                                "Ossian Elgström\nOssian Elgström was a Swedish illustrator. He studied with Zahrtmann.",
        "list": "Ingredients\n- two cups of flour for the base\n- one egg. Beat it well before use.\n1. numbered step here",
        "abbreviations": "Dr. Smith met Mr. Brown at 10 a.m. on Jan. 5 in Washington. They talked about the U.S. budget.",
        "wrapped line": "This sentence is wrapped over two lines,\nand it continues here. Then another one starts.",
        "long": "Start here. " + " ".join(["alpha"] * 70) + "; " + " ".join(["beta"] * 70) + ".",
        "short": "Yes. No. The rest of the text is here.",
    }
    for name, text in cases.items():
        spans = sent.split(text)
        assert _covered(text, spans) and _ordered(spans), name
        assert all(len(text[s:e].split()) <= sent.MAX_WORDS for s, e in spans), name
    spans = sent.split(cases["abbreviations"])
    assert len(spans) == 2, [cases["abbreviations"][s:e] for s, e in spans]
    text = 'It first appeared in "Swamp Thing" Vol. 2 No. 25 (June 1984). See Fig. 3 for the details here.'
    assert len(sent.split(text)) == 2, [text[s:e] for s, e in sent.split(text)]
    spans = sent.split(cases["wrapped line"])
    assert cases["wrapped line"][slice(*spans[0])].endswith("continues here.")
    assert sent.split("") == []


def test_align():
    text = "First sentence is here. Second sentence follows it. Third one closes."
    spans = sent.split(text)
    second = (text.index("Second"), text.index("Third") - 1)
    assert sent.align(spans, [second]) == [[1]]
    both = (text.index("First"), text.index("Third") - 1)           # a gold span covering two of our sentences
    assert sent.align(spans, [both]) == [[0, 1]]
    part = (text.index("Third"), text.index("Third") + 5)           # a small piece of a sentence: largest overlap
    assert sent.align(spans, [part]) == [[2]]


def _valid_row(**kw):
    doc = "Title line\nAlpha beta gamma delta. Epsilon zeta eta theta."
    gold = (doc.index("Epsilon"), len(doc))
    row = schema.assemble(id="t:1", source="t", orig_split="train", claim="A claim.", document=doc, label=1,
                          gold_sets=[[gold]], evidence_quality="gold", doc_id="t:doc")
    row.update(kw)
    return row


def test_validate():
    schema.validate(_valid_row())
    bad = [
        _valid_row(evidence=[9], evidence_sets=[[9]]),             # index out of range
        _valid_row(label=0),                                       # label 0 with evidence
        _valid_row(sentences=[[0, 30], [20, 50]]),                 # overlapping spans
        _valid_row(sentences=[[11, 34]]),                          # text left outside the sentences
        _valid_row(evidence_term="mil_neg"),                       # term does not fit the label
    ]
    for row in bad:
        try:
            schema.validate(row)
        except ValueError:
            continue
        raise AssertionError(f"accepted a bad row: {row}")


def test_docbuilder():
    doc = schema.DocBuilder()
    doc.add("Title")
    doc.add("\n")
    span = doc.add("  Body text.  ")
    assert doc.text()[slice(*span)] == "Body text."


def test_scifact():
    base = {"claim_id": 7, "abstract_id": 42, "title": "A study", "orig_split": "train",
            "abstract": ["Background sentence one is here.", "The drug lowers blood pressure in adults.",
                         "Other results were not significant."]}
    row = scifact.convert(dict(base, claim="The drug lowers blood pressure.", verdict="SUPPORT", evidence=[1]), OPTS)
    schema.validate(row)
    assert row["label"] == 1 and row["evidence_term"] == "supervised"
    assert _evidence_text(row) == ["The drug lowers blood pressure in adults."]
    row = scifact.convert(dict(base, claim="The drug raises blood pressure.", verdict="CONTRADICT", evidence=[1]), OPTS)
    schema.validate(row)
    assert row["label"] == 0 and row["evidence"] == [] and row["evidence_term"] == "mil_neg" and len(row["contra"]) == 1
    row = scifact.convert(dict(base, claim="Unrelated.", verdict="NEI", evidence=[]), OPTS)
    assert row["label"] == 0 and row["evidence_quality"] == "none"


def test_wice():
    base = {"claim": "Jane Doe was born in Rome.", "orig_split": "test", "index": 3, "aggrefact_overlap": "test",
            "meta": {"id": "test1-0", "claim_title": "Jane Doe"},
            "evidence": ["(meta data) TITLE: Jane Doe biography", "# Jane Doe", "Jane Doe was born in Rome in 1950.",
                         "She moved to Paris as a child.", "Born:", "Rome, Italy"]}
    row = wice.convert(dict(base, label="supported", supporting_sentences=[[2], [4, 5]]), OPTS)
    schema.validate(row)
    assert row["label"] == 1 and len(row["evidence_sets"]) == 2 and row["meta"]["aggrefact_overlap"] == "test"
    assert any("born in Rome in 1950" in t for t in _evidence_text(row))
    row = wice.convert(dict(base, label="partially_supported", supporting_sentences=[[2]]), OPTS)
    schema.validate(row)
    assert row["label"] == 0 and row["evidence_term"] == "none" and row["evidence"] == []
    assert isinstance(wice.convert(dict(base, label="partially_supported", supporting_sentences=[]),
                                   SimpleNamespace(wice_partial="drop")), schema.Skip)
    row = wice.convert(dict(base, label="not_supported", supporting_sentences=[[]]), OPTS)
    assert row["label"] == 0 and row["evidence_term"] == "mil_neg"


def _hover_raw(label, facts, db_pages=None):
    context = {
        "Kristian Zahrtmann": ["Kristian Zahrtmann was a Danish painter.", " He taught at an art school in Copenhagen."],
        "Distractor Page": ["This page talks about something else entirely.", " It has two sentences."],
        "Ossian Elgström": ["Ossian Elgström was a Swedish illustrator and writer.", " He studied with Zahrtmann."],
    }
    claim = {"uid": "u1", "claim": "An illustrator studied with a Danish painter.", "label": label, "num_hops": 2,
             "hpqa_id": "h1", "supporting_facts": facts, "orig_split": "train"}
    return {"hover": claim, "context": context, "db_pages": db_pages or {}}


def test_hover():
    facts = [["Kristian Zahrtmann", 0], ["Ossian Elgström", 1]]
    row = hover.convert(_hover_raw("SUPPORTED", facts), OPTS)
    schema.validate(row)
    texts = _evidence_text(row)
    assert row["label"] == 1 and any("Danish painter" in t for t in texts) and any("studied with" in t for t in texts)
    assert row["meta"]["gold_alignment"] == "exact" and row["doc_id"] == "hover:h1"
    row = hover.convert(_hover_raw("NOT_SUPPORTED", facts), OPTS)
    assert row["label"] == 0 and row["evidence"] == [] and row["evidence_term"] == "mil_neg"
    missing = facts + [["Copenhagen Academy", 1]]
    assert hover.convert(_hover_raw("SUPPORTED", missing), OPTS).reason == "page_missing"
    page = {"Copenhagen Academy": "Copenhagen Academy is a school of fine arts. Zahrtmann taught there for years."}
    row = hover.convert(_hover_raw("SUPPORTED", missing, page), SimpleNamespace(hover_db_fallback=True))
    schema.validate(row)
    assert row["meta"]["gold_alignment"] == "db_split"
    assert any("taught there for years" in t for t in _evidence_text(row))
    assert hover.convert(_hover_raw("SUPPORTED", [["Kristian Zahrtmann", 5]]), OPTS).reason == "bad_sent_id"


def test_length_filter():
    from evidence.data.build import build_source

    class WordLength:                                  # stands in for PromptLength: one token per word
        def prompt_tokens(self, row):
            doc = len(row["document"].split())
            return doc, doc + 2 * len(row["claim"].split()) + 10

    short = "Short document with one sentence only."
    long_doc = " ".join(["Another sentence of five words."] * 40)
    source = SimpleNamespace(
        NAME="fake",
        load_raw=lambda opts, prov: iter([("a", short), ("b", long_doc)]),
        convert=lambda raw, opts: schema.assemble(id=raw[0], source="fake", orig_split="train", claim="A claim here.",
                                                  document=raw[1], label=0, evidence_quality="none", doc_id=raw[0]))
    rows, skips, _ = build_source(source, SimpleNamespace(limit=0, max_tokens=50), WordLength())
    assert [r["id"] for r in rows] == ["a"] and skips == {"too_long": 1}
    assert rows[0]["meta"]["doc_tokens"] == 6 and rows[0]["meta"]["prompt_tokens"] == 6 + 6 + 10


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
