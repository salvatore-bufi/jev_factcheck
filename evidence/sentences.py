"""Sentence splitting and gold-sentence alignment.

`split` is the only sentence splitter of the evidence pipeline: the training data, the rendering and the server
must cut documents identically, or the evidence head reads the wrong positions. Changing it invalidates every
built dataset and every trained head (its sha256 is recorded in the build manifest).

Rules (Documentation/fine_tuning_factcheck_evidence/02_dati_fonti.md §2):
1. a line break is a hard boundary after a blank line, before a list item, and after a line that does not end in
   punctuation (a title); other line breaks (also after a line ending in a comma or a dash) join into one block;
2. inside a block, NLTK's Punkt (punkt_tab, English) gives the sentences;
3. sentences over 60 words are cut at "; " or ": " nearest the middle, then at ", ", then at a space;
4. sentences under 3 words are merged into the next sentence of the same paragraph (or the previous one).

Punkt needs the `punkt_tab` data: `python -c "import nltk; nltk.download('punkt_tab')"`. NLTK refuses a data
directory that other users can write to (`chmod -R go-w ~/nltk_data`), or point NLTK_DATA to a private copy.
"""
import bisect
import re
from functools import lru_cache

MAX_WORDS = 60
MIN_WORDS = 3
MAX_SENTENCES = 512

_LIST_ITEM = re.compile(r"\s*(?:[-*•]\s|\d+[.)]\s)")
_FINAL_PUNCT = tuple('.!?;:"\')]”’,-–(')   # a line ending otherwise is a title: the block ends there
# Abbreviations missing from punkt_tab's English model ("Vol. 2 No. 25" was cut twice); words that often end a
# sentence ("etc.", "est.") are left out. "no" is in: "No. 1" is common in encyclopedic text, and the cost is that a
# sentence ending in "no." followed by a common sentence starter may be joined to the next one.
_EXTRA_ABBREVIATIONS = {"vol", "vols", "no", "fig", "figs", "eq", "eqs", "approx", "ca", "cf", "mt", "sgt", "capt",
                        "gov", "dept", "pp", "al", "e.g", "i.e"}
_WORD = re.compile(r"\S+")


@lru_cache(maxsize=1)
def _punkt():
    try:
        from nltk.tokenize import PunktTokenizer
        tokenizer = PunktTokenizer("english")
        tokenizer._params.abbrev_types.update(_EXTRA_ABBREVIATIONS)
        return tokenizer
    except LookupError as err:
        raise SystemExit("NLTK punkt_tab is not available: run python -c \"import nltk; nltk.download('punkt_tab')\" "
                         "and make the data directory private (chmod -R go-w ~/nltk_data), "
                         "or set NLTK_DATA to a private copy") from err


def _words(text, start, end):
    return len(_WORD.findall(text, start, end))


def _trim(text, start, end):
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _paragraphs(text):
    """-> list of paragraphs, each a list of (start, end) blocks for Punkt. Paragraphs are split by blank lines."""
    paragraphs, blocks, block = [], [], None
    lines, a = [], 0
    for line in text.split("\n"):
        lines.append((a, a + len(line)))
        a += len(line) + 1
    for a, b in lines:
        line = text[a:b]
        if not line.strip():                                   # blank line: closes block and paragraph
            if block:
                blocks.append(block)
            if blocks:
                paragraphs.append(blocks)
            blocks, block = [], None
            continue
        if block is not None and _LIST_ITEM.match(line):       # a list item starts a new block
            blocks.append(block)
            block = None
        block = (a, b) if block is None else (block[0], b)
        if not line.rstrip().endswith(_FINAL_PUNCT):           # a title (or the last line of an item) ends the block
            blocks.append(block)
            block = None
    if block:
        blocks.append(block)
    if blocks:
        paragraphs.append(blocks)
    return paragraphs


def _cut_long(text, start, end):
    """Cut a span until every piece has at most MAX_WORDS words."""
    if _words(text, start, end) <= MAX_WORDS:
        return [(start, end)]
    middle = (start + end) // 2
    for pattern in (r"[;:]\s", r",\s", r"\s"):
        cuts = [m.end() for m in re.finditer(pattern, text[start:end])]
        cuts = [start + c for c in cuts if 0 < c < end - start]
        if cuts:
            cut = min(cuts, key=lambda c: abs(c - middle))
            left, right = _trim(text, start, cut), _trim(text, cut, end)
            if left[0] < left[1] and right[0] < right[1]:
                return _cut_long(text, *left) + _cut_long(text, *right)
    return [(start, end)]


def _merge_short(text, spans):
    """Merge spans under MIN_WORDS words into the next span (the previous one for the last span)."""
    spans = list(spans)
    i = 0
    while len(spans) > 1 and i < len(spans):
        if _words(text, *spans[i]) >= MIN_WORDS:
            i += 1
            continue
        if i + 1 < len(spans):
            spans[i:i + 2] = [(spans[i][0], spans[i + 1][1])]
        else:
            spans[i - 1:i + 1] = [(spans[i - 1][0], spans[i][1])]
            i -= 1
    return spans


def split(text):
    """-> list of (start, end) character spans, in order, not overlapping, covering every non-space character."""
    punkt = _punkt()
    spans = []
    for blocks in _paragraphs(text):
        para = []
        for a, b in blocks:
            for s, e in punkt.span_tokenize(text[a:b]):
                s, e = _trim(text, a + s, a + e)
                if s < e:
                    para.extend(_cut_long(text, s, e))
        spans.extend(_merge_short(text, para))
    return spans


def align(spans, gold):
    """Map gold character spans onto `spans` (the output of `split`).

    A sentence k matches gold sentence g when the overlap covers at least half of k or half of g; when nothing
    reaches half, the sentence with the largest overlap is taken. -> list, per gold span, of the matched indices.
    """
    starts = [s for s, _ in spans]
    out = []
    for g0, g1 in gold:
        lo = max(bisect.bisect_right(starts, g0) - 1, 0)
        hits, best = [], (0, None)
        for k in range(lo, len(spans)):
            s, e = spans[k]
            if s >= g1:
                break
            overlap = min(e, g1) - max(s, g0)
            if overlap <= 0:
                continue
            if overlap * 2 >= e - s or overlap * 2 >= g1 - g0:
                hits.append(k)
            if overlap > best[0]:
                best = (overlap, k)
        out.append(hits or ([best[1]] if best[1] is not None else []))
    return out
