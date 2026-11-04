"""Checks of the rendering: from sentences to prompt tokens (CPU; needs KnowLine's tokenizer in the HF cache).

    python tests/test_evidence_render.py          # or: python -m pytest tests

What is checked (doc 06 §7):
    - document_in_prompt gives exactly what the authors' render_value writes (indentation, line breaks);
    - without preamble the prompt is byte-identical to the benchmark's (adapter: state {"document", "claim"});
    - with the preamble, the prompt contains it once per copy and ends like the benchmark's;
    - the tokens of every sentence decode to the sentence, on hard documents: quotes, backslashes, line breaks, tabs,
      "<", emoji, non-Latin text, a document without line breaks;
    - a request read twice takes its sentences from the second copy.
Skipped (with a message) if the KnowLine snapshot is not in the Hugging Face cache.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evidence import config  # noqa: E402
from evidence import sentences as sent  # noqa: E402

DOCS = [
    'Rose "Byrne" is an\nAustralian actress. She starred in films.',
    "A path C:\\temp\\x was used.\tTabs\tare here. Less < than and > more.\n\nNew paragraph with emoji 🎬 here.",
    "Москва — столица России. Το Παρίσι είναι πρωτεύουσα. 東京は日本の首都です。 Done here.",
    "One line only, no breaks at all. Second sentence on the same line. Third.",
    "Title\nFirst real sentence of the page is here. Another one follows it.\r\nWindows line end above.",
]


def available():
    try:
        from evidence.render import frontend
        frontend()
        return True
    except SystemExit as err:
        print("skipped:", err)
        return False


def test_document_in_prompt_matches_render_value():
    if not available():
        return
    from evidence.render import document_in_prompt, frontend
    ks = frontend().ks
    for doc in DOCS:
        text, origin = document_in_prompt(doc)
        expected = ks.render_value(doc, 1) if "\n" in doc else doc
        assert text == expected, (text, expected)
        assert len(origin) == len(text) and all(0 <= o < len(doc) for o in origin)


def test_prompt_identical_without_preamble():
    if not available():
        return
    from evidence.render import frontend, render
    fe = frontend()
    doc, claim = DOCS[0], "Rose Byrne is an actress."
    request = {"state": {"document": doc, "claim": claim}, "questions": {config.QUESTION_ID: config.QUESTION}}
    benchmark = fe.kl.render_request(request, fe.tok, fe.layout)["prompts"][config.QUESTION_ID][0][0]
    ours = render(doc, claim, sent.split(doc), None)["prompt"]
    assert ours == benchmark
    with_pre = render(doc, claim, sent.split(doc), "Claim under review: {claim}")
    assert with_pre["prompt"].count("Claim under review: " + claim) == (2 if with_pre["twice"] else 1)
    assert with_pre["prompt"].endswith(benchmark[-200:])


def test_sentence_tokens_decode_to_sentences():
    if not available():
        return
    from evidence.render import check_row
    for i, doc in enumerate(DOCS):
        row = {"document": doc, "claim": "A claim.", "sentences": sent.split(doc)}
        for preamble in (None, "Claim under review: {claim}"):
            assert check_row(row, preamble) == [], (i, preamble)


def test_second_copy():
    if not available():
        return
    from evidence.render import render
    doc = DOCS[0]
    ex = render(doc, "Rose Byrne is an actress.", sent.split(doc), "Claim under review: {claim}")
    assert ex["twice"]
    first_token = ex["sent_tok_spans"][0][0]
    # the first sentence's tokens come after the "Read the same input again" line
    from evidence.render import frontend
    before = frontend().tok.decode(ex["input_ids"][:first_token])
    assert "Read the same input again before answering:" in before


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
