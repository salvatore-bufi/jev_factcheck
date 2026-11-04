"""HoVer claims on HotpotQA distractor contexts: multi-hop evidence spread over several Wikipedia pages.

HoVer gives the claims (2-4 hops), SUPPORTED / NOT_SUPPORTED labels and the gold facts as (page title, sentence
number), numbered as in HotpotQA. HotpotQA distractor (joined on `hpqa_id`) gives, for that question, its pages
already split into sentences plus ~8 distractor pages. Document = the context pages and the supporting pages, in a
random order fixed by the claim uid; each page is its title, a line break, its text; pages are separated by a
blank line.

Supporting pages missing from the HotpotQA context (3-4 hop claims): the row is skipped (`page_missing`), or with
--hover-db-fallback the page text comes from HoVer's wiki_wo_links.db and is split with our splitter; sentence i
is then our i-th sentence of that page (meta.gold_alignment = "db_split").
"""
import json
import random
import sqlite3
import unicodedata
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from evidence import sentences as sent
from evidence.data.schema import DocBuilder, Skip, assemble
from evidence.data.sources import hub_files

NAME = "hover"
ROOT = Path(__file__).resolve().parents[3]
HOVER = ROOT / "dataset" / "HoVer" / "data"
HOVER_SPLITS = {"train": "hover_train_release_v1.1.json", "dev": "hover_dev_release_v1.1.json"}
HOTPOT = "hotpotqa/hotpot_qa"
LABELS = {"SUPPORTED": 1, "NOT_SUPPORTED": 0}


def nfc(text):
    return unicodedata.normalize("NFC", text)


def hotpot_contexts(ids, prov):
    """-> {hotpot id: {title (NFC): [sentences]}} for the given ids, from the cached distractor parquet files."""
    files, revision = hub_files(HOTPOT, "distractor/*.parquet")
    prov["hotpot"] = {"revision": revision, "files": [str(f) for f in files]}
    wanted = pa.array(sorted(ids))
    out = {}
    for path in files:
        table = pq.read_table(path, columns=["id", "context"])
        table = table.filter(pc.is_in(table["id"], value_set=wanted))
        for row in table.to_pylist():
            ctx = row["context"]
            out[row["id"]] = {nfc(t): list(s) for t, s in zip(ctx["title"], ctx["sentences"])}
    return out


class WikiDB:
    """HoVer's wiki_wo_links.db (DrQA layout: documents(id, text)); titles are looked up as given, NFC and NFD."""

    def __init__(self, path=HOVER / "wiki_wo_links.db"):
        self.conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)

    def text(self, title):
        for key in (title, nfc(title), unicodedata.normalize("NFD", title)):
            row = self.conn.execute("SELECT text FROM documents WHERE id = ?", (key,)).fetchone()
            if row:
                return row[0]
        return None


def load_raw(opts, prov):
    claims = []
    for split, name in HOVER_SPLITS.items():
        path = HOVER / name
        prov.setdefault("files", []).append(str(path))
        rows = json.loads(path.read_text())
        claims += [dict(r, orig_split=split) for r in rows]
    if getattr(opts, "limit", 0):
        claims = claims[:opts.limit]
    contexts = hotpot_contexts({c["hpqa_id"] for c in claims}, prov)
    db = WikiDB() if getattr(opts, "hover_db_fallback", False) else None
    for c in claims:
        ctx = contexts.get(c["hpqa_id"])
        db_pages = {}
        if ctx is not None and db is not None:
            for title, _ in c["supporting_facts"]:
                if nfc(title) not in ctx and nfc(title) not in db_pages:
                    db_pages[nfc(title)] = db.text(title)
        yield {"hover": c, "context": ctx, "db_pages": db_pages}


def convert(raw, opts):
    claim, ctx = raw["hover"], raw["context"]
    if claim.get("label") not in LABELS:
        return Skip("unlabelled")
    if ctx is None:
        return Skip("hotpot_missing")
    facts = [(nfc(t), int(i)) for t, i in claim["supporting_facts"]]
    support_titles = list(dict.fromkeys(t for t, _ in facts))
    missing = [t for t in support_titles if t not in ctx]
    db_pages = raw.get("db_pages") or {}
    if missing and not all(db_pages.get(t) for t in missing):
        return Skip("page_missing")

    titles = list(dict.fromkeys(list(ctx) + missing))
    random.Random(claim["uid"]).shuffle(titles)
    doc, sentence_spans, page_ranges = DocBuilder(), {}, {}
    for j, title in enumerate(titles):
        if j:
            doc.add("\n\n")
        doc.add(title)
        doc.add("\n")
        if title in ctx:
            for i, s in enumerate(ctx[title]):
                sentence_spans[(title, i)] = doc.add(s.replace("\n", " "))
        else:
            page_ranges[title] = doc.add(db_pages[title].replace("\n", " "))
    document = doc.text()
    spans = sent.split(document)

    gold = []
    for title, i in facts:
        if title in page_ranges:                     # page split by us: sentence i of that page
            a, b = page_ranges[title]
            inside = [s for s in spans if s[1] > a and s[0] < b]     # the title line may be merged into the first
            if i >= len(inside):
                return Skip("db_index_out_of_range")
            gold.append(tuple(inside[i]))
        else:
            span = sentence_spans.get((title, i))
            if span is None or span[0] >= span[1]:
                return Skip("bad_sent_id")
            gold.append(span)

    label = LABELS[claim["label"]]
    return assemble(
        id=f"hover:{claim['orig_split']}:{claim['uid']}",
        source=NAME, orig_split=claim["orig_split"], claim=claim["claim"].strip(), document=document, label=label,
        gold_sets=[gold] if label == 1 else [],
        evidence_quality="gold" if label == 1 else "none",
        doc_id=f"hover:{claim['hpqa_id']}",
        spans=spans,
        meta={"orig_label": claim["label"], "num_hops": claim.get("num_hops"), "hpqa_id": claim["hpqa_id"],
              "supporting_facts": claim["supporting_facts"], "pages": titles,
              "gold_alignment": "db_split" if page_ranges else "exact"},
    )
