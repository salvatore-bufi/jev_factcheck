#!/usr/bin/env python3
"""Evaluate CLM-v0.1-8B (Contrastive-LM/CLM-v0.1-8B) on the pinned LLM-AggreFact test split, with vLLM.

Run from the repository root (or anywhere):

    python clm-8b/benchmark.py --limit 1       # one claim against its document (smoke test)
    python clm-8b/benchmark.py                 # full test split, resumable
    python clm-8b/benchmark.py --offline       # rebuild the report from cached answers

CLM is not a generative model. It is a "System One" model made of two small projection heads
(a 73 MB checkpoint) on top of a frozen Qwen3-8B encoder: the state text (the document and claim,
then the question) and each answer option's text are embedded by Qwen3-8B (last-token pooling),
projected by the state head and the action head, and the answer is the softmax over
`scale * cosine(state, option)`. For a yes/no (noul) question the options are `false: ...` and
`true: ...`, and the answer is P(true).

The reference implementation serves this through `vllm serve Qwen/Qwen3-8B --runner pooling` plus
`clm-serve`. This script does the same in-process with vLLM and reuses the `contrastive-lm`
package's own text layout (clm.schema) and head code (clm.heads), so the inputs are the same.

The experiment is a fixed transfer evaluation, like the JPT-9B, Rune and Winnow runs in the README:
the frozen three-question pack (packs/claim_support.json), the minimum rule and the threshold
> 0.30, with the same 80,000-character document cap. Nothing is tuned for CLM. Texts are NOT
truncated to the server's default 2,048 tokens (that would cut long documents): the encoder runs at
--max-model-len (default 32768), and a longer prompt is reported as a failure. The cache stores the
raw logits (scale * cosine), so --temperature (default 1) and --threshold only rescore it; do not
pick either from test results.

Data: data/test.parquet if present, else the Hugging Face cache directory
./lytang___llm-aggre_fact; override with --data. Models are read from the local Hugging Face cache.
"""
import argparse
from datetime import datetime, timezone
import glob
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))   # shared metrics.py, evaluate.state_for

# No nvcc on some machines: vLLM's FlashInfer sampler would JIT-compile CUDA at start-up. Nothing is
# sampled here (embeddings only). Must be set before importing vLLM.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")   # fork + OpenMP in this process is unsafe

from metrics import metrics  # noqa: E402

ENCODER = "Qwen/Qwen3-8B"     # the encoder the heads were trained against
HEADS_GLOB = str(Path.home() / ".cache/huggingface/hub/models--Contrastive-LM--CLM-v0.1-8B/snapshots/*/CLM_v0.1-8B.pt")
NAME = "CLM-v0.1-8B"
SIZE = "8B"
TEMPERATURE = 1.0             # the CLM client's default
THRESHOLD = 0.30              # frozen Jev rule, applied unchanged
MAX_DOC_CHARS = 80000
PACK = ROOT / "packs/claim_support.json"
INVERT = {"support_simple": False, "unsupported_detail": True, "contradicted_detail": True}
SHORT = {
    "AggreFact-CNN": "CNN", "AggreFact-XSum": "XSum", "TofuEval-MediaS": "MediaS",
    "TofuEval-MeetB": "MeetB", "Wice": "WiCE", "Reveal": "REVEAL", "ClaimVerify": "ClaimVerify",
    "FactCheck-GPT": "FactCheck", "ExpertQA": "ExpertQA", "Lfqa": "LFQA", "RAGTruth": "RAGTruth",
}
COLUMNS = ["CNN", "XSum", "MediaS", "MeetB", "WiCE", "REVEAL", "ClaimVerify", "FactCheck",
           "ExpertQA", "LFQA", "RAGTruth"]


# ---------------------------------------------------------------- data

def resolve_data(path):
    """Accept test.parquet, a Hugging Face `datasets` cache .arrow file, or the cache directory."""
    path = Path(path)
    if path.is_dir():
        found = sorted(path.rglob("llm-aggre_fact-test.arrow"))
        if len(found) != 1:
            raise SystemExit(f"Expected exactly one llm-aggre_fact-test.arrow under {path}, found {len(found)}")
        path = found[0]
    if not path.exists():
        raise SystemExit(f"{path} not found")
    return path


def read_table(path):
    if path.suffix == ".arrow":
        import pyarrow as pa
        with path.open("rb") as handle:
            return pa.ipc.open_stream(handle).read_all()
    import pyarrow.parquet as pq
    return pq.read_table(path)


def load_rows(path, limit, split="test"):
    if split == "dev":   # Jev's verified tune + selection pools (dev_split.py); ids are dev:<row>
        from dev_split import load_dev_rows
        rows = load_dev_rows()[0]
    else:
        rows = read_table(Path(path)).to_pylist()
        for i, row in enumerate(rows):
            row["id"] = f"test:{i}"
            row.pop("contamination_identifier", None)
    if limit:
        # Deterministic pseudo-random subset (hash order) so a smoke test spans several sources.
        order = sorted(range(len(rows)), key=lambda i: hashlib.sha256(rows[i]["id"].encode()).hexdigest())
        rows = [rows[i] for i in sorted(order[:limit])]
    return rows


# ---------------------------------------------------------------- cache

def read_cache(path, identity):
    done = {}
    if not path.exists():
        return done
    with path.open() as handle:
        header = json.loads(handle.readline())
        if header.get("identity") != identity:
            raise SystemExit(f"{path} was produced with a different configuration; use another --output")
        for line in handle:
            if line.strip():
                record = json.loads(line)
                done[record["id"]] = record
    return done


# ---------------------------------------------------------------- scoring rule

def p_yes(logits, temperature):
    """Softmax over the two option-letter logits [false, true] at `temperature`; returns P(true)."""
    a, b = logits
    if not (math.isfinite(a) or math.isfinite(b)):
        raise ValueError("no finite option logit")
    peak = max(a, b)
    wa, wb = math.exp((a - peak) / temperature), math.exp((b - peak) / temperature)
    return wb / (wa + wb)


def decide(logits_by_question, temperature):
    """Frozen rule: min(support, 1 - unsupported detail, 1 - contradicted detail)."""
    parts = []
    for question, logits in logits_by_question.items():
        p = p_yes(logits, temperature)
        parts.append(1 - p if INVERT[question] else p)
    return min(parts)


# ---------------------------------------------------------------- inference

def find_heads(path=None):
    path = path or (sorted(glob.glob(HEADS_GLOB)) or [None])[0]
    if not path or not Path(path).exists():
        raise SystemExit("CLM_v0.1-8B.pt not found in the Hugging Face cache; run `hf download Contrastive-LM/CLM-v0.1-8B`")
    return path


class Scorer:
    """Qwen3-8B embeddings (vLLM, last-token pooling) -> CLM heads -> scale * cosine logits per option."""

    def __init__(self, args):
        import numpy as np
        from clm.heads import HeadPair
        from clm.schema import candidates, state_text
        from vllm import LLM

        self.np, self.state_text = np, state_text
        self.pack = json.loads(PACK.read_text())["questions"]
        self.max_len = args.max_model_len
        # The encoder must start BEFORE any CPU torch work in this process: vLLM forks its engine process,
        # and a fork after OpenMP has run (the heads below) makes the child hang or segfault in libgomp.
        self.llm = LLM(model=ENCODER, runner="pooling", dtype="bfloat16", max_model_len=args.max_model_len,
                       gpu_memory_utilization=args.gpu_util, cpu_offload_gb=args.cpu_offload_gb,
                       enable_prefix_caching=True, max_num_seqs=args.max_num_seqs,
                       max_num_batched_tokens=args.max_batched_tokens, seed=0)
        self.heads = HeadPair("clm", find_heads(args.heads), device="cpu").ensure()   # tiny MLPs: keep the GPU for the encoder
        self.scale = self.heads.scale
        self.tok = self.llm.get_tokenizer()
        # The option texts are identical for every row: embed and project them once.
        self.options = {}
        for qid, q in self.pack.items():
            keys, texts = candidates(q)
            assert keys == ["false", "true"], keys
            self.options[qid] = self.heads.project_actions(self.embed(texts)).cpu().numpy()   # [2, proj]

    def embed(self, texts):
        """[n, 4096] L2-normalised float32 embeddings, as the reference Embedder returns them."""
        np = self.np
        out = self.llm.embed(texts, use_tqdm=False)
        vecs = np.asarray([o.outputs.embedding for o in out], dtype=np.float32)
        return vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)

    def score(self, states):
        """states -> per row ({qid: [logit false, logit true]}, prompt tokens), or an error string."""
        np = self.np
        texts, owners, results = [], [], [None] * len(states)
        for i, state in enumerate(states):
            per_q = {qid: self.state_text(state, q.get("instructions")) for qid, q in self.pack.items()}
            n = max(len(self.tok.encode(t)) for t in per_q.values())
            if n > self.max_len:
                results[i] = f"prompt of {n} tokens exceeds --max-model-len {self.max_len}"
                continue
            for qid, text in per_q.items():
                texts.append(text)
                owners.append((i, qid, n))
        rows = {}
        if texts:
            zs = self.heads.project_states(self.embed(texts)).cpu().numpy()
            for (i, qid, n), z in zip(owners, zs):
                logits = (self.scale * (self.options[qid] @ z)).tolist()   # [false, true]
                rows.setdefault(i, {})[qid] = (logits, n)
        for i, per_q in rows.items():
            results[i] = ({q: v for q, (v, _) in per_q.items()}, max(n for _, n in per_q.values()))
        return results


def score_rows(rows, scorer, args, cache_path, identity, done):
    from evaluate import state_for
    todo = [r for r in rows if r["id"] not in done]
    new_file = not cache_path.exists()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    started, failures = time.time(), 0
    with cache_path.open("a") as out:
        if new_file:
            out.write(json.dumps({"identity": identity}) + "\n")
        for start in range(0, len(todo), args.window):
            window = todo[start:start + args.window]
            results = scorer.score([state_for(r, args.max_doc_chars) for r in window])
            for row, result in zip(window, results):
                if isinstance(result, str):   # never invent a score for a failed row
                    failures += 1
                    print(f"  {row['id']}: {result}", flush=True)
                    continue
                logits, tokens = result
                record = {"id": row["id"], "logits": logits, "prompt_tokens": tokens}
                done[row["id"]] = record
                out.write(json.dumps(record) + "\n")
            out.flush()
            finished = min(start + args.window, len(todo))
            print(f"{finished}/{len(todo)} new rows ({finished / max(time.time() - started, 1e-9):.1f} rows/s, "
                  f"{failures} failed)", flush=True)


# ---------------------------------------------------------------- report

def build_report(rows, done, args, identity, complete):
    scored = [r for r in rows if r["id"] in done]
    scores = [decide(done[r["id"]]["logits"], args.temperature) for r in scored]
    result = metrics(scored, scores, args.threshold)
    by = {SHORT.get(k, k): v for k, v in result["by_source"].items()}
    avg = result["macro_balanced_accuracy"]
    report = {
        "model": "CLM-v0.1-8B", "temperature": args.temperature, "threshold": args.threshold,
        "score_rule": "min(support, 1 - unsupported_detail, 1 - contradicted_detail) > threshold",
        "pack": "packs/claim_support.json", "max_doc_chars": args.max_doc_chars,
        "rows_total": len(rows), "rows_scored": len(scored), "complete": complete,
        "setting": "fixed transfer of the frozen Jev checker; no tuning for this model",
        "identity": identity, "reported_at": datetime.now(timezone.utc).isoformat(),
        "macro_balanced_accuracy": avg,
        "pooled": {k: result[k] for k in ["n", "balanced_accuracy", "accuracy", "roc_auc",
                                          "supported_recall", "unsupported_recall"]},
        "by_source": by,
        "prompt_tokens_max_per_row_sum": sum(done[r["id"]]["prompt_tokens"] for r in scored),
    }
    cells = [f"{by[c]['balanced_accuracy'] * 100:.2f}" if c in by and by[c]["balanced_accuracy"] is not None else "n/a"
             for c in COLUMNS]
    avg_cell = f"{avg * 100:.2f}" if avg is not None else "n/a"
    row_md = f"| {NAME} (ours; fixed Jev rule) | {SIZE} | {avg_cell} | " + " | ".join(cells) + " |"
    return report, row_md


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_data = ROOT / "data/test.parquet"
    if not default_data.exists():
        default_data = ROOT / "lytang___llm-aggre_fact"
    p.add_argument("--data", default=str(default_data),
                   help="test.parquet, a HF datasets cache llm-aggre_fact-test.arrow, or the cache directory")
    p.add_argument("--output", default=str(ROOT / "results/runs/clm-8b"))
    p.add_argument("--limit", type=int, default=0, help="score a deterministic subset (0 = all rows)")
    p.add_argument("--temperature", type=float, default=TEMPERATURE,
                   help="temperature of the option softmax (default 1, the CLM client's); rescoring the cache is "
                        "free, but do not pick a value from test results")
    p.add_argument("--threshold", type=float, default=THRESHOLD,
                   help="decision threshold (default: the frozen 0.30); rescoring the cache is free, but do not "
                        "pick a value from test results")
    p.add_argument("--max-doc-chars", type=int, default=MAX_DOC_CHARS, help="75%% head + 25%% tail; 0 = no cap")
    p.add_argument("--heads", default=None, help="path to CLM_v0.1-8B.pt (default: the Hugging Face cache)")
    p.add_argument("--max-model-len", type=int, default=32768, help="longest text (tokens) the encoder accepts")
    p.add_argument("--cpu-offload-gb", type=float, default=0, help="GB of the 16 GB encoder kept in system RAM")
    p.add_argument("--max-num-seqs", type=int, default=32, help="concurrent sequences")
    p.add_argument("--max-batched-tokens", type=int, default=8192, help="prefill tokens per step (activation memory)")
    p.add_argument("--gpu-util", type=float, default=0.92, help="fraction of GPU memory vLLM may use")
    p.add_argument("--window", type=int, default=128, help="rows embedded together (3 state texts each)")
    p.add_argument("--gpu", default=None, help="physical GPU id(s), sets CUDA_VISIBLE_DEVICES")
    p.add_argument("--offline", action="store_true", help="only rebuild the report from cached answers")
    args = p.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    data = resolve_data(args.data)
    out_dir = Path(args.output)
    if args.limit:   # a smoke test must never share a cache or report with a full run
        out_dir = out_dir.with_name(out_dir.name + f"-limit{args.limit}")
    heads_path = find_heads(args.heads)
    identity = {
        "model": "CLM-v0.1-8B", "encoder": ENCODER, "max_doc_chars": args.max_doc_chars,
        "heads_sha256": hashlib.sha256(Path(heads_path).read_bytes()).hexdigest(),
        "pack_sha256": hashlib.sha256(PACK.read_bytes()).hexdigest(), "limit": args.limit,
        "data_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
    }
    rows = load_rows(data, args.limit)
    cache_path = out_dir / "answers.jsonl"
    done = read_cache(cache_path, identity)

    if not args.offline and len(done) < len(rows):
        scorer = Scorer(args)
        print(f"{len(rows) - len(done)} of {len(rows)} rows to score", flush=True)
        score_rows(rows, scorer, args, cache_path, identity, done)

    if not done:
        raise SystemExit("No cached answers; run without --offline")
    complete = all(r["id"] in done for r in rows)
    report, row_md = build_report(rows, done, args, identity, complete)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    header = "| Model | Size | Average | " + " | ".join(COLUMNS) + " |"
    status = "" if complete else (f"\n\n**Incomplete: {report['rows_scored']}/{report['rows_total']} rows scored; "
                                  "not a benchmark score.**")
    if args.limit:
        status += "\n\n**Subset run (--limit): not a benchmark score.**"
    (out_dir / "table_row.md").write_text(header + "\n|---|---:|---:|" + "---:|" * len(COLUMNS) + "\n" + row_md + status + "\n")
    print(header)
    print(row_md + status)
    print(f"Report: {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
