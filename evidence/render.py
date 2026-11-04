"""STEP 2 — Build KnowLine-4B's prompt for a row and find which prompt tokens belong to each sentence.

Where it sits in the workflow:
    rows (splits / mix)  ->  [evidence.render]  ->  evidence.model.forward_sentences  ->  features / head / server

Why this file matters most: the evidence head reads, for every sentence, the hidden vector of the sentence's tokens.
If a sentence's token positions are wrong by even one token, the head reads the wrong vectors. Everything here is
checked (tests/test_evidence_render.py): the decoded tokens of every sentence give back the sentence.

How KnowLine renders a request (front end of the authors, imported from the model snapshot, NOT rewritten here):
    the state {"document": ..., "claim": ...} becomes ONE user turn, as indented text:
        document:
          <line 1 of the document>          <- every line indented by 2 spaces (when the document has line breaks)
          <line 2 ...>
        claim: <claim>
    then a user turn with the instruction, "Question: ...", "Options: A. Yes: yes / B. No: no",
    and the assistant turn "Answer:". The verdict is read at the LAST token (answer position).
    Requests of at most 3,000 tokens are written TWICE ("Read the same input again before answering:"):
    we take the sentences from the SECOND copy (it has already seen the whole document and the claim).

The preamble (doc 05): one line "Claim under review: <claim>" written BEFORE the state, in the same user turn.
It is passed to the authors' renderer as a ready-made user message, so their code is used unchanged; with no
preamble the prompt is byte-identical to the benchmark's (models/knowline-4b/adapter.py).

Main function:
    render(document, claim, sentences, preamble)  ->  dict with input_ids, answer_pos, sent_tok_spans, ...
"""
import bisect
import sys
import unicodedata
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evidence import config  # noqa: E402


# ------------------------------------------------------------------ the authors' front end
def snapshot_dir():
    """Folder of the KnowLine snapshot in the Hugging Face cache.

    Input:  nothing (repo and revision come from evidence/config.py).
    Output: Path of the snapshot; raises SystemExit if it is missing (run `hf download PelaAI/KnowLine-4B-Gen5.5`).
    """
    from core.datasets import hub_cache
    path = hub_cache() / f"models--{config.MODEL_REPO.replace('/', '--')}" / "snapshots" / config.MODEL_REVISION
    if not (path / "knowline_layout.py").is_file():
        raise SystemExit(f"{config.MODEL_REPO} not in the HF cache ({path}); run `hf download {config.MODEL_REPO}`")
    return path


@lru_cache(maxsize=1)
def frontend():
    """Load (once) the tokenizer and the authors' rendering code.

    Input:  nothing.
    Output: SimpleNamespace with
        tok        the tokenizer (transformers AutoTokenizer of the snapshot)
        ks         module knowline_server (render_value, ...)
        kl         module knowline_layout (render_request, Layout, labels_and_ids)
        layout     the Layout used by the benchmark (read twice, 3,000 tokens)
        label_ids  token ids of the labels "A" and "B" (A = Yes, B = No)
    """
    from transformers import AutoTokenizer
    folder = snapshot_dir()
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
    import knowline_layout as kl
    import knowline_server as ks
    tok = AutoTokenizer.from_pretrained(str(folder))
    layout = kl.Layout(read_twice=config.READ_TWICE, twice_max_tokens=config.TWICE_MAX_TOKENS) \
        if config.READ_TWICE else kl.Layout()
    labels, ids = kl.labels_and_ids(tok)
    return SimpleNamespace(tok=tok, ks=ks, kl=kl, layout=layout, folder=folder, label_ids=ids[:2])


# ------------------------------------------------------------------ document text as it appears in the prompt
def document_in_prompt(document):
    """The document as KnowLine writes it in the prompt, and where every prompt character comes from.

    Input:  document (str, the original text).
    Output: (text, origin)
        text    the document as rendered by knowline_server.render_value: unchanged if it has no line break,
                otherwise every line indented by two spaces (lines split like str.splitlines)
        origin  list, one int per character of `text`: index of the original character it comes from
                (the two indentation spaces point to the first character of their line, a line break to the
                original line-break character)
    """
    if "\n" not in document:
        return document, list(range(len(document)))
    out, origin, pos = [], [], 0
    pieces = document.splitlines(keepends=True)        # same line breaks as render_value's splitlines()
    for i, piece in enumerate(pieces):
        line = piece.splitlines()[0] if piece.splitlines() else ""
        if i:                                           # the "\n" joining two lines
            out.append("\n")
            origin.append(pos - 1)
        out.append("  ")                                # indentation of the line
        origin += [pos, pos]
        out.append(line)
        origin += range(pos, pos + len(line))
        pos += len(piece)
    return "".join(out), origin


def state_text(document, claim):
    """The state turn without preamble: what render_value gives for {"document", "claim"}.

    Input:  document, claim (str).
    Output: (text of the user turn, the part before the document text, e.g. "document:\\n" or "document: ").
    """
    ks = frontend().ks
    text = ks.render_value({"document": document, "claim": claim})
    head = "document:\n" if "\n" in document else "document: "
    return text, head


# ------------------------------------------------------------------ tokens -> sentences
def token_char_spans(offsets, doc_start, doc_len, origin):
    """For every prompt token that overlaps the document: the original characters it covers.

    Input:  offsets    (start, end) of every token in the prompt string (tokenizer return_offsets_mapping)
            doc_start  index in the prompt string where the document text starts
            doc_len    length of the document text in the prompt
            origin     output of document_in_prompt (prompt-document char -> original char)
    Output: (token indices in the prompt, list of (start, end) in ORIGINAL document characters), same order.
    """
    doc_end = doc_start + doc_len
    idx, spans = [], []
    for k, (a, b) in enumerate(offsets):
        a, b = max(a, doc_start), min(b, doc_end)       # the part of the token inside the document
        if a >= b:
            continue
        idx.append(k)
        spans.append((origin[a - doc_start], origin[b - 1 - doc_start] + 1))
    return idx, spans


def sentence_token_spans(tok_char, sentences, text):
    """Assign every document token to a sentence; give the first and last token of each sentence.

    Input:  tok_char   list of (start, end) original characters of the document tokens (output of token_char_spans)
            sentences  list of (start, end) sentence spans of the original document (evidence.sentences.split)
            text       the original document
    Output: list of (first, last) positions INTO tok_char (not prompt indices), one pair per sentence.
    Rule: a token belongs to the sentence that contains its last non-space character (so ".\"" glued to the next
    word goes to the sentence it closes). Tokens of only spaces belong to no sentence.
    Raises ValueError if a sentence gets no token (very rare: a one-character sentence fused into a token).
    """
    starts = [s for s, _ in sentences]
    first, last = [None] * len(sentences), [None] * len(sentences)
    for i, (a, b) in enumerate(tok_char):
        c = b - 1
        while c >= a and text[c].isspace():
            c -= 1
        if c < a:
            continue                                    # only spaces
        k = bisect.bisect_right(starts, c) - 1          # the sentence whose start is the last one <= c
        if k < 0 or c >= sentences[k][1]:
            continue                                    # character between two sentences
        if first[k] is None:
            first[k] = i
        last[k] = i
    missing = [k for k, f in enumerate(first) if f is None]
    if missing:
        raise ValueError(f"sentences without tokens of their own: {missing}")
    return list(zip(first, last))


# ------------------------------------------------------------------ the main function
def render(document, claim, sentences, preamble=None):
    """Build the prompt of one row and locate its sentences.

    Input:  document   original document text
            claim      claim text
            sentences  list of [start, end) character spans of the document's sentences (the row's "sentences")
            preamble   template with "{claim}" (e.g. "Claim under review: {claim}"), or None for no preamble
    Output: dict
            prompt          the prompt string (ends with "Answer:")
            input_ids       list of token ids (what the model reads)
            answer_pos      index of the last token: where the verdict is read
            sent_tok_spans  list of [first, last] prompt-token indices of each sentence (inclusive), in the copy
                            of the document read last (the second one when the request is read twice)
            n_tok           number of tokens of each sentence
            doc_tok_start   index of the first token of that copy of the document (for "distance in tokens")
            twice           True if the request was written twice
            label_ids       token ids of "A" (Yes) and "B" (No)
    """
    fe = frontend()
    body, head = state_text(document, claim)
    if preamble:
        state = [{"role": "user", "content": preamble.format(claim=claim) + "\n\n" + body}]
    else:
        state = {"document": document, "claim": claim}     # exactly what the benchmark sends
    request = {"state": state, "questions": {config.QUESTION_ID: config.QUESTION}}
    r = fe.kl.render_request(request, fe.tok, fe.layout)
    prompt, keys = r["prompts"][config.QUESTION_ID][0]
    if keys[:2] != ["true", "false"]:
        raise ValueError(f"unexpected answer keys {keys}")

    # where is the (last copy of the) document in the prompt string
    doc_text, origin = document_in_prompt(document)
    anchor = prompt.rfind(head + doc_text)
    if anchor < 0:
        raise ValueError("document not found in the prompt")
    doc_start = anchor + len(head)

    # tokens with their character offsets; must be the very ids the model reads
    enc = fe.tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    if ids != fe.tok.encode(prompt, add_special_tokens=False):
        raise ValueError("tokenization with offsets differs from plain encode")

    idx, tok_char = token_char_spans(enc["offset_mapping"], doc_start, len(doc_text), origin)
    rel = sentence_token_spans(tok_char, [tuple(s) for s in sentences], document)
    spans = [[idx[a], idx[b]] for a, b in rel]
    return {
        "prompt": prompt, "input_ids": ids, "answer_pos": len(ids) - 1,
        "sent_tok_spans": spans, "n_tok": [b - a + 1 for a, b in spans],
        "doc_tok_start": idx[0], "twice": r["twice"], "label_ids": list(fe.label_ids),
    }


def render_row(row, preamble=None):
    """render() for a dataset row (dict with document, claim, sentences)."""
    return render(row["document"], row["claim"], row["sentences"], preamble)


def _squash(text):
    """Text without spaces, in Unicode NFKC form (decoding may normalise accents and diacritics)."""
    return "".join(unicodedata.normalize("NFKC", text).split())


def check_row(row, preamble=None):
    """Self-check of one row: the decoded tokens of every sentence must give back the sentence (spaces aside).

    Input:  row (dataset row), preamble.
    Output: list of indices of the sentences that FAIL (empty list = all good).
    Tolerance: a token that straddles two sentences (e.g. '!"' glued to the next word) belongs to one of them only,
    so the decoded text may contain a few characters of a neighbour, or miss up to 3 characters at a boundary.
    """
    fe = frontend()
    ex = render_row(row, preamble)
    bad = []
    for k, ((a, b), (s, e)) in enumerate(zip(ex["sent_tok_spans"], row["sentences"])):
        decoded = _squash(fe.tok.decode(ex["input_ids"][a:b + 1]))
        sentence = _squash(row["document"][s:e])
        inner = sentence[1:-3] if len(sentence) > 6 else sentence     # the sentence without its boundary chars
        if inner not in decoded or len(decoded) > len(sentence) + 6:
            bad.append(k)
    return bad


if __name__ == "__main__":
    # quick check from the command line:  python -m evidence.render [n rows per set]
    from evidence.data import io
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    for name in ("dev_evid_select", "dev_evid_neg", "test_evid_ood"):
        rows = io.read_jsonl(config.SPLITS / f"{name}.jsonl")[:n]
        fails = sum(bool(check_row(r, config.preamble_template())) for r in rows)
        print(f"{name}: {len(rows) - fails}/{len(rows)} rows with every sentence located correctly")
