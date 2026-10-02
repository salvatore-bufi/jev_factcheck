#!/usr/bin/env python3
"""Evaluate JPT-9B (kirp/jpt-9b) on the pinned LLM-AggreFact test split, in-process with vLLM.

Run from the repository root (or anywhere):

    python jpt-9b/benchmark.py --limit 1       # one claim against its document (smoke test)
    python jpt-9b/benchmark.py                 # full test split, resumable
    python jpt-9b/benchmark.py --offline       # rebuild the report from cached answers

JPT-9B implements the typed-decision (System One) interface: for each question it
reads the probability of every option label from ONE forward pass. This script
reproduces that recipe with vLLM instead of a server. Prompts come from the
`llm2jev` package (the model card's reference implementation: chat template,
thinking off, "Answer:" prefill, labels A/B, softmax at the card's fixed
temperature 1.087), so the answers match what `llm2jev` would return.

The experiment is a fixed transfer evaluation, like the CLM run described in the
README: the frozen three-question pack (packs/claim_support.json), the minimum
rule and the threshold > 0.30, with the same 80,000-character document cap. Nothing
is tuned for JPT-9B. Raw answers are cached, so changing --threshold only rescores
the cache; do not choose a threshold from test results.

Data: data/test.parquet if present, else the Hugging Face cache directory
./lytang___llm-aggre_fact; override with --data. The model is read from the local
Hugging Face cache (~/.cache/huggingface/hub).
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))   # shared metrics.py, evaluate.state_for

# No nvcc here: vLLM's FlashInfer top-k/top-p sampler JIT-compiles CUDA and would crash at
# start-up. This script never samples (one greedy token, label logprobs only), so use the
# native sampler. Must be set before vLLM is imported.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from metrics import metrics  # noqa: E402

MODEL = "kirp/jpt-9b"
NAME = "JPT-9B"
SIZE = "9B"
TEMPERATURE = 1.087           # fixed by the model card; fit once on a held-out split
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


def load_rows(path, limit):
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

def decide(answers):
    """Frozen rule: min(support, 1 - unsupported detail, 1 - contradicted detail)."""
    parts = [(1 - v) if INVERT[k] else v for k, v in answers.items()]
    return min(parts)


# ---------------------------------------------------------------- inference

class Scorer:
    """One vLLM engine; one forward pass per question; label probabilities as llm2jev computes them."""

    def __init__(self, args):
        from llm2jev.prompt import find_labels, render
        from llm2jev.scoring import answer, softmax
        from transformers import AutoProcessor, AutoTokenizer
        from vllm import LLM

        self.render, self.answer, self.softmax = render, answer, softmax
        try:
            self.processor = AutoProcessor.from_pretrained(MODEL)
            self.processor.apply_chat_template
        except (OSError, ValueError, AttributeError, ImportError):
            self.processor = AutoTokenizer.from_pretrained(MODEL)
        self.tok = getattr(self.processor, "tokenizer", self.processor)
        # Same probe llm2jev runs at start-up: labels must be single tokens after "Answer:".
        _, probe, _ = render(self.processor, "x", {"q": {"type": "noul"}}, ["A", "B"], "chat")
        self.labels, self.ids = find_labels(self.tok, probe["q"][0])
        self.pack = json.loads(PACK.read_text())["questions"]
        self.max_len = args.max_model_len
        self.llm = LLM(model=MODEL, dtype="bfloat16", max_model_len=args.max_model_len,
                       gpu_memory_utilization=args.gpu_util, enable_prefix_caching=True,
                       limit_mm_per_prompt={"image": 0, "video": 0},   # text only: skip the vision tower budget
                       max_logprobs=20, seed=0, max_num_seqs=args.max_num_seqs,
                       max_num_batched_tokens=args.max_batched_tokens)

    def prompts(self, state):
        """-> {question id: prompt text}; the document prefix is identical across questions (prefix cache)."""
        _, rendered, _ = self.render(self.processor, state, self.pack, self.labels, "chat")
        return {qid: text for qid, (text, _) in rendered.items()}

    def score(self, states):
        """states: list of System One states -> list of ({qid: P(yes)}, prompt tokens) or an error string."""
        from vllm import SamplingParams
        params = SamplingParams(max_tokens=1, temperature=0.0, logprobs=len(self.ids[:2]),
                                logprob_token_ids=self.ids[:2])
        texts, owners, results = [], [], [None] * len(states)
        for i, state in enumerate(states):
            per_q = self.prompts(state)
            n = max(len(self.tok.encode(t, add_special_tokens=False)) for t in per_q.values())
            if n + 1 > self.max_len:
                results[i] = f"prompt of {n} tokens exceeds --max-model-len {self.max_len}"
                continue
            for qid, text in per_q.items():
                texts.append(text)
                owners.append((i, qid, n))
        answers = {}
        outputs = self.llm.generate(texts, params, use_tqdm=False) if texts else []
        for (i, qid, n), out in zip(owners, outputs):
            top = out.outputs[0].logprobs[0]
            row = [top[t].logprob if t in top else float("-inf") for t in self.ids[:2]]
            probs = self.softmax(row, TEMPERATURE)
            # option order for noul is ["true", "false"] -> label A = yes
            answers.setdefault(i, {})[qid] = (self.answer({"type": "noul"}, ["true", "false"], probs)["noul"], n)
        for i, per_q in answers.items():
            results[i] = ({q: p for q, (p, _) in per_q.items()}, max(n for _, n in per_q.values()))
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
                answers, tokens = result
                record = {"id": row["id"], "answers": answers, "prompt_tokens": tokens}
                done[row["id"]] = record
                out.write(json.dumps(record) + "\n")
            out.flush()
            finished = min(start + args.window, len(todo))
            print(f"{finished}/{len(todo)} new rows ({finished / max(time.time() - started, 1e-9):.1f} rows/s, "
                  f"{failures} failed)", flush=True)


# ---------------------------------------------------------------- report

def build_report(rows, done, args, identity, complete):
    scored = [r for r in rows if r["id"] in done]
    scores = [decide(done[r["id"]]["answers"]) for r in scored]
    result = metrics(scored, scores, args.threshold)
    by = {SHORT.get(k, k): v for k, v in result["by_source"].items()}
    avg = result["macro_balanced_accuracy"]
    report = {
        "model": MODEL, "temperature": TEMPERATURE, "threshold": args.threshold,
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
    p.add_argument("--output", default=str(ROOT / "results/runs/jpt-9b"))
    p.add_argument("--limit", type=int, default=0, help="score a deterministic subset (0 = all rows)")
    p.add_argument("--threshold", type=float, default=THRESHOLD,
                   help="decision threshold (default: the frozen 0.30); rescoring the cache is free, but do not "
                        "pick a value from test results")
    p.add_argument("--max-doc-chars", type=int, default=MAX_DOC_CHARS, help="75%% head + 25%% tail; 0 = no cap")
    p.add_argument("--max-model-len", type=int, default=32768, help="longest prompt (tokens) the engine accepts")
    p.add_argument("--max-num-seqs", type=int, default=32,
                   help="concurrent sequences; Qwen3.5's DeltaNet layers need one state block each, "
                        "and a 24 GB GPU holds few after the 17 GB of weights")
    p.add_argument("--max-batched-tokens", type=int, default=8192, help="prefill tokens per step (activation memory)")
    p.add_argument("--gpu-util", type=float, default=0.92, help="fraction of GPU memory vLLM may use")
    p.add_argument("--window", type=int, default=64, help="rows submitted to vLLM together (3 prompts each)")
    p.add_argument("--gpu", default=None, help="physical GPU id(s), sets CUDA_VISIBLE_DEVICES")
    p.add_argument("--offline", action="store_true", help="only rebuild the report from cached answers")
    args = p.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")   # weights come from the local cache

    data = resolve_data(args.data)
    out_dir = Path(args.output)
    if args.limit:   # a smoke test must never share a cache or report with a full run
        out_dir = out_dir.with_name(out_dir.name + f"-limit{args.limit}")
    identity = {
        "model": MODEL, "temperature": TEMPERATURE, "max_doc_chars": args.max_doc_chars,
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
