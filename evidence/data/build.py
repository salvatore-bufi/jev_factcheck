"""Build the evidence rows of one or more sources into dataset/evidence/<source>.jsonl.

    python -m evidence.data.build --source scifact wice hover
    python -m evidence.data.build --source hover --limit 200 --hover-db-fallback
    python -m evidence.data.build --max-tokens 16384          # stricter length filter (default 32,768; 0 = off)

Rows whose prompt would not fit --max-tokens tokens of --length-model (KnowLine-4B by default; see
evidence/data/length.py) are dropped as too_long. Every row is validated (evidence/data/schema.py) before anything is written. Next to the rows go
<source>.stats.json and the folder's manifest.json (files read, revisions, sha256 of the output, options, git
commit, sha256 of evidence/sentences.py). The build fails when a source's gold sentences map one-to-one onto our
sentences less often than --min-alignment.
"""
import argparse
import itertools
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evidence import sentences  # noqa: E402
from evidence.data import io, schema, stats  # noqa: E402
from evidence.data.length import PromptLength  # noqa: E402
from evidence.data.sources import registry  # noqa: E402


def build_source(module, opts, length=None):
    """-> (rows, skip reasons, provenance). With `length` (a PromptLength), rows whose prompt would exceed
    opts.max_tokens are skipped as too_long, and every kept row records meta.doc_tokens and meta.prompt_tokens."""
    prov, rows, skips, ids = {}, [], Counter(), {}
    raw_rows = module.load_raw(opts, prov)
    if opts.limit:
        raw_rows = itertools.islice(raw_rows, opts.limit)
    for raw in raw_rows:
        row = module.convert(raw, opts)
        if isinstance(row, schema.Skip):
            skips[row.reason] += 1
            continue
        if len(row["sentences"]) > sentences.MAX_SENTENCES:
            skips["too_many_sentences"] += 1
            continue
        if length is not None:
            row["meta"]["doc_tokens"], row["meta"]["prompt_tokens"] = length.prompt_tokens(row)
            if row["meta"]["prompt_tokens"] > opts.max_tokens:
                skips["too_long"] += 1
                continue
        schema.validate(row)
        if row["id"] in ids:
            if ids[row["id"]] != row:                     # same id, different content: the id scheme is wrong
                raise SystemExit(f"{module.NAME}: duplicate id {row['id']} with different content")
            skips["duplicate_record"] += 1                # the release repeats a record (SciFact validation does)
            continue
        ids[row["id"]] = row
        rows.append(row)
    return rows, skips, prov


def main(argv=None):
    sources = registry()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", nargs="+", choices=sorted(sources), default=sorted(sources))
    p.add_argument("--out", type=Path, default=io.OUT)
    p.add_argument("--limit", type=int, default=0, help="raw records per source (quick checks); 0 = all")
    p.add_argument("--wice-partial", choices=["negative", "drop"], default="negative",
                   help="partially supported WiCE claims: label 0 without evidence loss, or dropped")
    p.add_argument("--hover-db-fallback", action="store_true",
                   help="take HoVer pages missing from the HotpotQA context from wiki_wo_links.db")
    p.add_argument("--min-alignment", type=float, default=0.9,
                   help="minimum share of gold sentences mapped one-to-one onto our sentences")
    p.add_argument("--max-tokens", type=int, default=32768,
                   help="drop rows whose prompt would exceed this many tokens of --length-model (0 = no filter)")
    p.add_argument("--length-model", default="knowline-4b", help="model whose tokenizer measures the prompt")
    opts = p.parse_args(argv)

    length = PromptLength(opts.length_model) if opts.max_tokens else None
    failed = []
    for name in opts.source:
        rows, skips, prov = build_source(sources[name], opts, length)
        if length is not None:
            prov["length_filter"] = dict(length.describe, max_tokens=opts.max_tokens)
        st = stats.compute(rows, skips)
        path = opts.out / f"{name}.jsonl"
        io.write_jsonl(path, rows)
        io.write_json(opts.out / f"{name}.stats.json", st)
        io.update_manifest(opts.out, name, {
            "file": path.name, "rows": len(rows), "sha256": io.sha256(path), "provenance": prov,
            "options": {k: v for k, v in vars(opts).items() if k not in ("source", "out")},
            "git_commit": io.git_commit(), "sentences_py_sha256": io.sha256(sentences.__file__),
        })
        share = st["gold_one_to_one_share"]
        print(f"{name}: {len(rows)} rows -> {path}  labels {st['by_label']}  terms {st['by_evidence_term']}  "
              f"skipped {st['skipped']}  one-to-one {share if share is None else round(share, 3)}")
        if share is not None and share < opts.min_alignment:
            failed.append(f"{name}: one-to-one alignment {share:.3f} < {opts.min_alignment}")
    if failed:
        raise SystemExit("\n".join(failed))


if __name__ == "__main__":
    main()
