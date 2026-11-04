"""STEP 5 — Feature extraction (phase 1a): one forward per row, the sentence vectors saved to disk.

Where it sits in the workflow:
    splits / mix  ->  [evidence.extract]  ->  data/evidence/features/knowline-4b/<set>/  ->  evidence.train_head

Why: the decision model is frozen, so the vectors it gives for each sentence never change. We compute them ONCE
and the head then trains on them in minutes, without the model in memory.

Run (GPU):
    python -m evidence.extract --rows dataset/evidence/mixes/ev_head.jsonl dataset/evidence/splits/dev_evid_select.jsonl
    python -m evidence.extract --rows ... --limit 300          # a quick sample
Reads:   the rows files given (each file = one set, named after the file)
Writes:  data/evidence/features/knowline-4b/<set>/
             shard-00000.safetensors ...   SHARD_ROWS rows per file. Tensors (concatenated over the rows' sentences):
                 h_end.<layer>   [N, D] bf16      last token of each sentence, one tensor per layer (19, 23, 27, 31)
                 h_mean.<layer>  [N, D] bf16      mean of each sentence's tokens
                 att             [N, 128] fp16    attention of the answer position on each sentence
                 n_tok           [N] int16        tokens of each sentence
                 q.<layer>       [R, D] fp32      answer-position vector, one row per ROW (not per sentence)
                 answer_logits   [R, 2] fp32      log-probabilities of Yes / No
             index.jsonl       one line per row: id, shard, row number in the shard, first sentence offset, n,
                               gold (0/1 per sentence), evidence_sets, label, source, evidence_term, row_weight, omega,
                               token positions of the sentences (for "distance in tokens"), P(supported)
             meta.json         model, revision, preamble, layers, hashes of the code that defines the features;
                               training refuses features made with different settings
             skipped.json      rows not extracted, with the reason
Resumes: complete shards are not recomputed.

Long rows (doc 06 §4): a training row (from a mix) longer than --max-tokens is SHORTENED (positives lose non-gold
sentences, starting from the farthest from the evidence; negatives lose their last sentences); dev / test rows
are never cut: they are skipped and counted.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evidence import config  # noqa: E402
from evidence.data import io  # noqa: E402

SHARD_ROWS = 250          # ~1 GB per shard at ~50 sentences per row: a shard is built in RAM before being written


# ------------------------------------------------------------------ shortening long training rows
def rebuild(row, keep):
    """A copy of the row whose document keeps only some sentences.

    Input:  row (dataset row), keep (sorted list of sentence indices to keep).
    Output: new row: document = kept sentences joined ("\\n" where the original had a line break after the sentence,
            else a space), sentences = their new spans, evidence / evidence_sets remapped, meta.shortened = True.
    The kept spans are used as they are (not re-split): they are our sentences already.
    """
    doc, spans = row["document"], row["sentences"]
    new_index = {old: new for new, old in enumerate(keep)}
    parts, new_spans, pos = [], [], 0
    for j, k in enumerate(keep):
        s, e = spans[k]
        if j:
            prev = keep[j - 1]                                   # the previous kept sentence
            gap = doc[spans[prev][1]:spans[prev + 1][0]]         # original text between it and its next sentence
            parts.append("\n" if "\n" in gap else " ")
            pos += 1
        parts.append(doc[s:e])
        new_spans.append([pos, pos + e - s])
        pos += e - s
    out = dict(row)
    out["document"] = "".join(parts)
    out["sentences"] = new_spans
    out["evidence"] = [new_index[k] for k in row["evidence"] if k in new_index]
    out["evidence_sets"] = [[new_index[k] for k in es if k in new_index] for es in row["evidence_sets"]]
    out["contra"] = [new_index[k] for k in row.get("contra", []) if k in new_index]
    out["meta"] = dict(row["meta"], shortened=True, original_sentences=len(spans))
    return out


def shorten(row, ex, max_tokens, preamble):
    """Shorten a training row until its prompt fits max_tokens.

    Input:  row, ex (its rendering, too long), max_tokens, preamble.
    Output: (shortened row, its rendering).
    Positives: drop non-gold sentences, farthest from any gold sentence first (labels stay valid).
    Negatives: drop sentences from the end.
    """
    from evidence.render import render_row
    n = len(row["sentences"])
    gold = set(row["evidence"])
    if gold:
        order = sorted((k for k in range(n) if k not in gold), key=lambda k: -min(abs(k - g) for g in gold))
    else:
        order = list(range(n - 1, 0, -1))
    keep = set(range(n))
    while len(ex["input_ids"]) > max_tokens:
        excess = len(ex["input_ids"]) - max_tokens
        n_tok = dict(zip(sorted(keep), ex["n_tok"]))
        removed = 0
        while order and removed < excess * 1.05 + 8:      # remove about as many tokens as needed, a bit more
            k = order.pop(0)
            keep.discard(k)
            removed += n_tok.get(k, 0)
        if not order and removed == 0:
            raise ValueError("cannot shorten the row enough")
        row_short = rebuild(row, sorted(keep))
        ex = render_row(row_short, preamble)
    return row_short, ex


# ------------------------------------------------------------------ one row
def index_entry(row, ex, out, shard, shard_row, offset):
    """The index line of one extracted row (see the file header)."""
    from evidence.model import p_supported
    gold = [0] * len(row["sentences"])
    for k in row["evidence"]:
        gold[k] = 1
    return {
        "id": row["id"], "source": row["source"], "label": row["label"],
        "shard": shard, "row": shard_row, "offset": offset, "n": len(row["sentences"]),
        "gold": gold, "evidence_sets": row["evidence_sets"],
        "evidence_term": row["evidence_term"],
        "row_weight": row.get("row_weight", 1.0), "omega": row.get("omega", 1.0),
        "doc_id": row["doc_id"], "sentences": row["sentences"],
        "sent_tok_spans": ex["sent_tok_spans"], "doc_tok_start": ex["doc_tok_start"],
        "prompt_tokens": len(ex["input_ids"]), "twice": ex["twice"],
        "p_supported": p_supported(out["answer_logits"]),
        "shortened": bool(row["meta"].get("shortened")),
    }


def save_shard(path, outs):
    """Concatenate the features of the rows of one shard and write them (safetensors).

    Input:  path, outs (list of forward_sentences outputs, in row order).
    Output: None (file written atomically: .tmp then rename, so an interrupted run never leaves a broken shard).
    """
    from safetensors.torch import save_file
    tensors = {}
    for j, layer in enumerate(config.SENTENCE_LAYERS):
        tensors[f"h_end.{layer}"] = torch.cat([o["h_end"][:, j] for o in outs]).contiguous()
        tensors[f"h_mean.{layer}"] = torch.cat([o["h_mean"][:, j] for o in outs]).contiguous()
        tensors[f"q.{layer}"] = torch.stack([o["q"][j] for o in outs]).contiguous()
    tensors["att"] = torch.cat([o["att"] for o in outs]).contiguous()
    tensors["n_tok"] = torch.cat([o["n_tok"] for o in outs]).contiguous()
    tensors["answer_logits"] = torch.stack([o["answer_logits"] for o in outs]).contiguous()
    tmp = path.with_suffix(".tmp")
    save_file(tensors, str(tmp))
    tmp.rename(path)


def feature_meta(preamble):
    """Settings that define the features: two feature folders can be mixed only if these are equal."""
    here = Path(__file__).resolve().parent
    return {
        "model": config.MODEL_NAME, "repo": config.MODEL_REPO, "revision": config.MODEL_REVISION,
        "preamble": preamble, "sentence_layers": config.SENTENCE_LAYERS,
        "attention_layers": config.FULL_ATTENTION_LAYERS, "question": config.QUESTION,
        "read_twice": config.READ_TWICE,
        "sha256": {f: io.sha256(here / f) for f in ("sentences.py", "render.py", "model.py")},
    }


# ------------------------------------------------------------------ one set
def extract_set(model, rows_path, out_dir, preamble, max_tokens, limit):
    """Extract the features of one rows file into out_dir (resuming complete shards).

    Input:  model, rows_path (jsonl), out_dir, preamble (template or None), max_tokens, limit (0 = all rows).
    Output: None. Prints progress; writes shards, index.jsonl, meta.json, skipped.json.
    """
    from evidence.model import forward_sentences
    from evidence.render import render_row
    rows = io.read_jsonl(rows_path)
    if limit:
        rows = rows[:limit]
    is_training = bool(rows) and "row_weight" in rows[0]              # mix rows may be shortened
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = feature_meta(preamble)
    meta_path = out_dir / "meta.json"
    if meta_path.is_file() and json.loads(meta_path.read_text())["sha256"] != meta["sha256"]:
        print(f"  WARNING {out_dir}: existing shards were made with different code; delete the folder to redo them")
    io.write_json(meta_path, dict(meta, rows_file=str(rows_path), rows_sha256=io.sha256(rows_path)))

    index_path, skipped_path = out_dir / "index.jsonl", out_dir / "skipped.json"
    index = {e["id"]: e for e in io.read_jsonl(index_path)} if index_path.is_file() else {}
    skipped = json.loads(skipped_path.read_text()) if skipped_path.is_file() else {}
    t0, done = time.time(), 0
    for shard in range((len(rows) + SHARD_ROWS - 1) // SHARD_ROWS):
        path = out_dir / f"shard-{shard:05d}.safetensors"
        if path.is_file():
            continue                                                   # resume
        outs, entries, offset = [], [], 0
        for row in rows[shard * SHARD_ROWS:(shard + 1) * SHARD_ROWS]:
            try:
                ex = render_row(row, preamble)
                if len(ex["input_ids"]) > max_tokens:
                    if not is_training:
                        skipped[row["id"]] = "too_long"
                        continue
                    row, ex = shorten(row, ex, max_tokens, preamble)
                out = forward_sentences(model, ex)
            except ValueError as err:                                  # e.g. a sentence without tokens
                skipped[row["id"]] = f"render: {err}"
                continue
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                skipped[row["id"]] = "out_of_memory"
                continue
            entries.append(index_entry(row, ex, out, shard, len(outs), offset))
            outs.append(out)
            offset += entries[-1]["n"]
        if outs:
            save_shard(path, outs)
        for e in entries:
            index[e["id"]] = e
        io.write_jsonl(index_path, sorted(index.values(), key=lambda e: (e["shard"], e["row"])))
        io.write_json(skipped_path, skipped)
        done += len(entries)
        rate = done / max(time.time() - t0, 1e-6)
        print(f"  {out_dir.name}: shard {shard} written ({len(index)} rows, {len(skipped)} skipped, {rate:.1f} rows/s)")
    report(out_dir)


def report(out_dir):
    """Print what a feature folder contains, including the 'evidence beyond 1,024 tokens' share (doc 04 §3)."""
    index = io.read_jsonl(out_dir / "index.jsonl") if (out_dir / "index.jsonl").is_file() else []
    pos = [e for e in index if any(e["gold"])]
    far = sum(max(e["sent_tok_spans"][k][1] for k, g in enumerate(e["gold"]) if g) - e["doc_tok_start"] > 1024
              for e in pos)
    print(f"{out_dir.name}: {len(index)} rows, {len(pos)} with evidence; "
          f"evidence beyond 1,024 tokens: {far / max(len(pos), 1):.3f} of them; "
          f"shortened: {sum(e['shortened'] for e in index)}")


def main(argv=None):
    """Command line entry point."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rows", type=Path, nargs="+", required=True, help="rows files (jsonl); one set per file")
    p.add_argument("--out", type=Path, default=config.FEATURES / config.MODEL_NAME)
    p.add_argument("--preamble", default=None, help="none | P-A | P-B (default: the chosen one)")
    p.add_argument("--max-tokens", type=int, default=config.MAX_TOKENS_EXTRACT)
    p.add_argument("--limit", type=int, default=0, help="first N rows of each file (quick runs)")
    p.add_argument("--suffix", default="", help="appended to the set name (e.g. _nopre for experiment H4)")
    opts = p.parse_args(argv)

    from evidence.model import load_model
    preamble = config.preamble_template(opts.preamble)
    model = load_model()
    for rows_path in opts.rows:
        name = rows_path.stem + opts.suffix + (f"_limit{opts.limit}" if opts.limit else "")
        print(f"{rows_path} -> {opts.out / name}  (preamble: {preamble!r})")
        extract_set(model, rows_path, opts.out / name, preamble, opts.max_tokens, opts.limit)


if __name__ == "__main__":
    main()
