"""Entry point: fact-check every claim in a dataset's claims/ against its documents/.

input/<dataset>/ is shared at the repo root across all codebases. Each run is
namespaced at the repo root by dataset and by a deterministic hash of its
hyperparameters (see run_identity.py): results go under
../outputs/<dataset>/<run_id>/. Re-running an identical configuration lands
back on the same folder.


FactCG does not retrieve: for each claim, EVERY document in
documents/ is compared directly against the claim (chunked internally if
long), and the document with the highest support_prob is reported as the
best_source. There is no vector index to build/load, so there is no --load
flag and no data_store/.

FactCG is a locally-loaded
transformers sequence-classification model (yaxili96/FactCG-DeBERTa-v3-Large,
see description_claude.md and factcg_client.py) -- a deterministic binary
classifier, so there is no --temperature and no --retries on unparseable
output: a forward pass always yields a valid probability.

Usage:
    python main.py
    python main.py --gpu 1   # restrict this process to physical GPU 1

One JSON file per claim plus a summary.json are written to the resolved
outputs directory, alongside a run_params.json manifest of the full config.
"""
import os
import sys

# Must happen before `factcg_client` (-> transformers -> torch) is imported below:
# CUDA_VISIBLE_DEVICES only takes effect if set before CUDA/torch initializes.
if "--gpu" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[sys.argv.index("--gpu") + 1]

import argparse
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

import config
import run_identity
from chunking import chunk_document
from factcg_client import FactCGModel


def parse_args():
    parser = argparse.ArgumentParser(description="FactCG (FactCG-DeBERTa-v3-Large) fact-checking module")
    parser.add_argument("--dataset", default=config.DATASET_NAME,
                        help="name of the input/<dataset>/{claims,documents} folder to use")
    parser.add_argument("--claims-dir", type=Path, default=None,
                        help="override the claims directory (default: input/<dataset>/claims)")
    parser.add_argument("--documents-dir", type=Path, default=None,
                        help="override the documents directory (default: input/<dataset>/documents)")
    parser.add_argument("--outputs-dir", type=Path, default=None,
                        help="override the outputs directory (default: auto, hashed on the full run config)")
    parser.add_argument("--model", default=config.MODEL_ID, help="HF Hub model id (AutoModelForSequenceClassification)")
    parser.add_argument("--max-length", type=int, default=config.MAX_LENGTH,
                        help="tokenizer max_length (analogous to num_ctx)")
    parser.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE,
                        help="max words per document chunk (approximate token proxy, see chunking.py)")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE,
                        help="document chunks scored per forward pass")
    parser.add_argument("--threshold", type=float, default=config.SUPPORT_THRESHOLD,
                        help="support_prob threshold above which the verdict is 'Supported'")
    parser.add_argument("--device", default=None, help="torch device override (default: cuda if available, else cpu)")
    parser.add_argument("--gpu", default=config.GPU_DEVICE,
                        help="physical GPU index (or comma-separated indices) to restrict this "
                             "process to, via CUDA_VISIBLE_DEVICES -- e.g. --gpu 1. Orthogonal to "
                             "--device: --gpu restricts which physical GPUs are visible at all, "
                             "--device picks whether/how to use CUDA within whatever's visible.")
    parser.add_argument("--resume", action="store_true",
                        help="skip claims that already have a non-error output file in outputs_dir "
                             "(same dataset/model/params -> same outputs_dir, see run_identity.py); "
                             "claims that previously errored are retried")
    return parser.parse_args()


def resolve_run(args):
    """Fill in claims/documents/outputs dirs left as None, deriving them from
    --dataset and the (CLI-resolved) hyperparameters. Returns the run manifest
    to persist alongside the outputs.
    """
    if args.claims_dir is None:
        args.claims_dir = config.REPO_ROOT / "input" / args.dataset / "claims"
    if args.documents_dir is None:
        args.documents_dir = config.REPO_ROOT / "input" / args.dataset / "documents"

    run_id, run_params = run_identity.run_identity(
        config.MODEL_NAME, args.dataset,
        {
            "llm_model": args.model,
            "max_length": args.max_length,
            "chunk_size": args.chunk_size,
            "threshold": args.threshold,
        },
    )
    if args.outputs_dir is None:
        args.outputs_dir = config.REPO_ROOT / "outputs" / args.dataset / run_id

    return run_params


def load_claims(claims_dir: Path):
    """Read every .json file in claims_dir; each holds one claim or a list."""
    claims = []
    files = sorted(claims_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No .json claim files found in {claims_dir}")
    for path in files:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [data]
        for item in data:
            if "claim" not in item or "claim_id" not in item:
                raise ValueError(f"{path.name}: every claim needs 'claim' and 'claim_id' fields")
            claims.append(item)
    seen = set()
    for c in claims:
        if c["claim_id"] in seen:
            print(f"WARNING: duplicate claim_id {c['claim_id']} — its output file will be overwritten")
        seen.add(c["claim_id"])
    return claims


def load_documents(documents_dir: Path):
    """Read every .txt file in documents_dir into {filename: text}."""
    files = sorted(documents_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files found in {documents_dir}")
    return {path.name: path.read_text(encoding="utf-8", errors="replace") for path in files}


def score_document(doc_text: str, claim_text: str, args, model: FactCGModel):
    """Chunk one document and score every chunk against the claim; return the max."""
    chunks = chunk_document(doc_text, args.chunk_size)
    chunk_records = []
    for batch_start in range(0, len(chunks), args.batch_size):
        batch_chunks = chunks[batch_start:batch_start + args.batch_size]
        batch_claims = [claim_text] * len(batch_chunks)
        support_probs, all_probs = model.score_pairs(batch_chunks, batch_claims)
        for offset, (support_prob, probs) in enumerate(zip(support_probs, all_probs)):
            chunk_records.append(
                {
                    "chunk_index": batch_start + offset,
                    "support_prob": round(support_prob, 6),
                    "probs": [round(p, 6) for p in probs],
                }
            )
    best = max(chunk_records, key=lambda c: c["support_prob"])
    return {
        "support_prob": best["support_prob"],
        "best_chunk_index": best["chunk_index"],
        "best_chunk_text": chunks[best["chunk_index"]],
        "chunks": chunk_records,
    }


def check_claim(claim_item: dict, documents: dict, args, model: FactCGModel) -> dict:
    """Compare a claim against its candidate documents and assemble the output record.

    If the claim has a "candidate_documents" field (a list of filenames into
    `documents`, written by scripts/dataset_common.py), only those are scored --
    this is what lets a claim be checked against its own N-document pool instead
    of every document in the dataset. Without it (e.g. example/aggregatefact_*),
    every document in `documents` is scored, same as before.
    """
    claim_text = claim_item["claim"]
    metadata = {k: v for k, v in claim_item.items() if k not in ("claim", "claim_id")}
    started = time.time()

    candidate_names = claim_item.get("candidate_documents")
    scored_documents = (
        {name: documents[name] for name in candidate_names if name in documents}
        if candidate_names else documents
    )

    document_scores = {}
    for filename, doc_text in scored_documents.items():
        document_scores[filename] = score_document(doc_text, claim_text, args, model)

    best_file = max(document_scores, key=lambda f: document_scores[f]["support_prob"])
    best = document_scores[best_file]
    support_prob = best["support_prob"]
    verdict = "Supported" if support_prob > args.threshold else "Refuted"

    record = {
        "claim_id": claim_item["claim_id"],
        "claim": claim_text,
        "claim_metadata": metadata,
        "verdict": verdict,
        "support_prob": support_prob,
        "best_source": {
            "file": best_file,
            "chunk_index": best["best_chunk_index"],
            "text": best["best_chunk_text"],
        },
        "document_scores": [
            {
                "file": filename,
                "support_prob": scores["support_prob"],
                "best_chunk_index": scores["best_chunk_index"],
                "chunks": scores["chunks"],
            }
            for filename, scores in document_scores.items()
        ],
        "run_info": {
            "model": args.model,
            "max_length": args.max_length,
            "chunk_size": args.chunk_size,
            "threshold": args.threshold,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": None,  # filled below
        },
    }
    record["run_info"]["elapsed_seconds"] = round(time.time() - started, 2)
    return record


def summary_entry(record: dict, out_file: Path) -> dict:
    return {
        "claim_id": record.get("claim_id"),
        "claim": record.get("claim"),
        "verdict": record.get("verdict"),
        "support_prob": record.get("support_prob"),
        "best_source": record.get("best_source", {}).get("file") if record.get("best_source") else None,
        "sub_dataset": record.get("claim_metadata", {}).get("sub_dataset"),
        "gold_label": record.get("claim_metadata", {}).get("gold_label"),
        "output_file": out_file.name,
    }


def main():
    args = parse_args()
    run_params = resolve_run(args)
    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    run_identity.write_manifest(args.outputs_dir, run_params, "run_params.json")

    claims = load_claims(args.claims_dir)
    print(f"Loaded {len(claims)} claims from {args.claims_dir}")
    documents = load_documents(args.documents_dir)
    print(f"Loaded {len(documents)} documents from {args.documents_dir}")

    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading {args.model} ...")
    model = FactCGModel(args.model, config.CACHE_DIR, args.max_length, args.device)
    print(f"Model loaded on {model.device}")

    summary = []
    skipped = 0
    for claim_item in tqdm(claims, desc="Fact-checking"):
        out_file = args.outputs_dir / f"claim_{claim_item['claim_id']}.json"
        if args.resume and out_file.exists():
            with open(out_file, encoding="utf-8") as f:
                existing = json.load(f)
            if "error" not in existing:
                summary.append(summary_entry(existing, out_file))
                skipped += 1
                continue
        try:
            record = check_claim(claim_item, documents, args, model)
        except Exception:
            print(f"claim_id {claim_item['claim_id']}: FAILED\n{traceback.format_exc()}")
            record = {
                "claim_id": claim_item["claim_id"],
                "claim": claim_item["claim"],
                "error": traceback.format_exc(limit=3),
            }
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        summary.append(summary_entry(record, out_file))

    if args.resume and skipped:
        print(f"Resumed: skipped {skipped}/{len(claims)} already-completed claims")

    with open(args.outputs_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Done. {len(summary)} results in {args.outputs_dir} (see summary.json)")


if __name__ == "__main__":
    main()
