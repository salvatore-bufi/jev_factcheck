"""STEP 6 — Simple methods the head must beat (doc 07), measured with the same metrics.

Where it sits in the workflow:
    features of a dev set (evidence.extract) + its rows  ->  [evidence.baselines]  ->  data/evidence/runs/baselines.json

Methods (each gives one score per sentence):
    rand  E-RAND  random scores: the floor
    lex   E-LEX   BM25 of the sentence against the claim (lower-case words, no stopwords, within the document):
                  the main reference. If the head does not beat it clearly, a word match would do almost as well
    att   E-ATT   attention of the answer position on the sentence (features "att"): the best 8 (layer, head,
                  length normalisation) combinations, chosen on half A of the set, averaged as normalised ranks
    abl   E-ABL   greedy ablation with the model: the smallest group of sentences that, alone, makes KnowLine say
                  "supported" (hundreds of forwards per row: reference only; needs the GPU)
(E-EMB of the plan needs sentence-transformers, which is not installed: left out.)

Protocol: the set is cut in two halves by document (A, B). Everything is chosen on A (E-ATT combinations; for every
method the better of "top k" with k = mean number of gold sentences, or a threshold on the score normalised to
[0, 1] within the row); the numbers reported are measured on B.

Run:
    python -m evidence.baselines --features data/evidence/features/knowline-4b/dev_evid_select \
        --rows dataset/evidence/splits/dev_evid_select.jsonl --method rand lex att
    python -m evidence.baselines ... --method abl --per-source 100          # GPU
"""
import argparse
import json
import math
import random
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evidence import config  # noqa: E402
from evidence.data import io  # noqa: E402
from evidence.data.make_splits import unit  # noqa: E402
from evidence.features import FeatureStore  # noqa: E402
from evidence.metrics import gold_tok_last, select, summarize  # noqa: E402

STOPWORDS = set("""a an the of in on at to for from by with and or but not no is are was were be been being has have
had do does did it its this that these those as which who whom whose what when where why how than then so such he she
they them his her their him we us our you your i me my also into over under about after before between during""".split())
WORD = re.compile(r"[a-z0-9]+")


# ------------------------------------------------------------------ scores per method
def words(text):
    """Lower-case words of a text without stopwords (for BM25)."""
    return [w for w in WORD.findall(text.lower()) if w not in STOPWORDS]


def bm25(query, docs, k1=1.5, b=0.75):
    """BM25 score of every "doc" (here: every sentence of the document) for the query (the claim).

    Input:  query (list of words), docs (list of lists of words), k1 and b (the usual BM25 constants).
    Output: list of floats, one per doc. IDF is computed over the sentences of this document only.
    """
    n = len(docs)
    avg = sum(len(d) for d in docs) / max(n, 1) or 1.0
    df = Counter(w for d in docs for w in set(d))
    out = []
    for d in docs:
        tf = Counter(d)
        score = 0.0
        for w in set(query):
            if tf[w]:
                idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
                score += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(d) / avg))
        out.append(score)
    return out


def lex_scores(row):
    """E-LEX: BM25 of each sentence against the claim. Input: dataset row. Output: list of floats."""
    sents = [words(row["document"][s:e]) for s, e in row["sentences"]]
    return bm25(words(row["claim"]), sents)


def rand_scores(row, seed=config.SEED):
    """E-RAND: random scores, reproducible per row. Output: list of floats."""
    rng = random.Random(f"{seed}:{row['id']}")
    return [rng.random() for _ in row["sentences"]]


def att_columns(feat, beta):
    """All 128 attention features of a row, divided by n_tok^beta. Output: list of n lists of 128 floats."""
    att = feat["att"].float()
    n_tok = feat["n_tok"].float().clamp_min(1)
    return (att / n_tok[:, None] ** beta).tolist()


def ranks(values):
    """Normalised ranks in [0, 1] (1 = highest) of a list of values."""
    order = sorted(range(len(values)), key=lambda k: values[k])
    out = [0.0] * len(values)
    for r, k in enumerate(order):
        out[k] = r / max(len(values) - 1, 1)
    return out


def choose_att_combos(feats_a, top=8):
    """E-ATT step 2-3 on half A: the `top` best (column, beta) combinations by AUPRC.

    Input:  feats_a (feature rows of half A with evidence), top.
    Output: list of (column index 0..127, beta).
    """
    from evidence.metrics import auprc
    golds = [set(k for k, g in enumerate(f["gold"].tolist()) if g) for f in feats_a]
    results = []
    for beta in (0.0, 0.5, 1.0):
        cols = [att_columns(f, beta) for f in feats_a]
        for c in range(len(cols[0][0])):
            results.append((auprc([[row[c] for row in fc] for fc in cols], golds) or 0.0, c, beta))
    results.sort(reverse=True)
    return [(c, beta) for _, c, beta in results[:top]]


def att_scores(feat, combos):
    """E-ATT score of each sentence: mean over the chosen combinations of the normalised rank of the sentence."""
    per_combo = []
    for c, beta in combos:
        cols = att_columns(feat, beta)
        per_combo.append(ranks([row[c] for row in cols]))
    return [statistics.mean(v) for v in zip(*per_combo)]


# ------------------------------------------------------------------ E-ABL (with the model)
def ablation_group(model, row, preamble, max_k=6):
    """E-ABL: greedy search of the smallest group of sentences that alone makes the model say "supported" (doc 03 §2).

    Input:  model (evidence.model.load_model), row (dataset row), preamble, max_k (largest group tried).
    Output: sorted list of sentence indices, or None if no group of max_k sentences is enough.
    One forward per candidate; reduced documents are short, so each forward is cheap.
    """
    from evidence.extract import rebuild
    from evidence.model import forward_sentences, p_supported
    from evidence.render import render_row
    thr = config.VERDICT_THRESHOLD

    def p_of(keep):
        short = rebuild(row, sorted(keep))
        return p_supported(forward_sentences(model, render_row(short, preamble), features=False)["answer_logits"])

    n = len(row["sentences"])
    chosen = []
    while len(chosen) < max_k:
        tried = {k: p_of(chosen + [k]) for k in range(n) if k not in chosen}
        best = max(tried, key=tried.get)
        chosen.append(best)
        if tried[best] > thr:
            break
    else:
        return None
    for k in list(chosen):                                  # clean-up: drop sentences that are not needed
        rest = [j for j in chosen if j != k]
        if rest and p_of(rest) > thr:
            chosen = rest
    return sorted(chosen)


# ------------------------------------------------------------------ evaluation of one method
def make_preds(items, scores, rule):
    """Predictions (metrics format) of a method with a selection rule.

    Input:  items (list of (feature index entry, dataset row)), scores (list of score lists), rule:
            {"kind": "topk", "k": int} or {"kind": "threshold", "t": float on min-max normalised scores}.
    Output: list of prediction dicts.
    """
    preds = []
    for (e, row), s in zip(items, scores):
        lo, hi = min(s), max(s)
        norm = [(v - lo) / (hi - lo) if hi > lo else 0.0 for v in s]
        if rule["kind"] == "topk":
            order = sorted(range(len(s)), key=lambda k: -s[k])
            shown, thr = sorted(order[:rule["k"]]), sorted(norm, reverse=True)[min(rule["k"], len(s)) - 1] - 1e-9
        else:
            shown, thr = select(norm, rule["t"]), rule["t"]
        preds.append({"id": e["id"], "source": e["source"], "label": e["label"], "p_supported": e["p_supported"],
                      "scores": norm, "threshold": thr, "shown": shown, "evidence_sets": e["evidence_sets"],
                      "gold_tok_last": gold_tok_last(e)})
    return preds


def evaluate_method(items_a, scores_a, items_b, scores_b):
    """Choose the rule on A, measure on B.

    Output: (rule, summary on B).
    """
    pos_a = [e for e, _ in items_a if any(e["gold"])]
    k = max(1, round(statistics.mean(sum(e["gold"]) for e in pos_a))) if pos_a else 2
    rules = [{"kind": "topk", "k": k}] + [{"kind": "threshold", "t": t / 20} for t in range(1, 20)]
    best = max(rules, key=lambda r: summarize(make_preds(items_a, scores_a, r))["f1"] or 0.0)
    return best, summarize(make_preds(items_b, scores_b, best))


def main(argv=None):
    """Command line entry point (see the file header)."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", type=Path, required=True, help="feature folder of the set (for att, p_supported)")
    p.add_argument("--rows", type=Path, required=True, help="rows file of the same set (for the text)")
    p.add_argument("--method", nargs="+", default=["rand", "lex", "att"], choices=["rand", "lex", "att", "abl"])
    p.add_argument("--per-source", type=int, default=100, help="E-ABL: rows per source (it is slow)")
    p.add_argument("--out", type=Path, default=config.RUNS / "baselines.json")
    opts = p.parse_args(argv)

    store = FeatureStore(opts.features)
    rows = {r["id"]: r for r in io.read_jsonl(opts.rows)}
    items = [(e, rows[e["id"]]) for e in store.index if e["id"] in rows]
    half = lambda e: "A" if unit(config.SEED, "baseline_half", e["doc_id"]) < 0.5 else "B"
    a_pos = [i for i, (e, _) in enumerate(items) if half(e) == "A"]
    b_pos = [i for i, (e, _) in enumerate(items) if half(e) == "B"]
    results = json.loads(opts.out.read_text()) if opts.out.is_file() else {}   # other sets / methods kept
    key = opts.features.name

    for method in opts.method:
        if method == "abl":
            from evidence.model import load_model
            model = load_model()
            preamble = store.meta["preamble"]
            per_src, scores, chosen_items = Counter(), [], []
            for e, row in items:
                if not any(e["gold"]) or e["p_supported"] <= config.VERDICT_THRESHOLD \
                        or per_src[e["source"]] >= opts.per_source:
                    continue
                per_src[e["source"]] += 1
                group = ablation_group(model, row, preamble) or []
                scores.append([1.0 if k in group else 0.0 for k in range(e["n"])])
                chosen_items.append((e, row))
            summary = summarize(make_preds(chosen_items, scores, {"kind": "threshold", "t": 0.5}))
            results.setdefault(key, {})["E-ABL"] = {"rule": "own group", "rows": len(chosen_items), **summary}
        else:
            if method == "rand":
                scores = [rand_scores(row) for _, row in items]
            elif method == "lex":
                scores = [lex_scores(row) for _, row in items]
            else:
                feats = [store.get(i, layers=[], parts=()) for i in range(len(items))]
                combos = choose_att_combos([feats[i] for i in a_pos if any(feats[i]["gold"].tolist())])
                scores = [att_scores(f, combos) for f in feats]
            rule, summary = evaluate_method([items[i] for i in a_pos], [scores[i] for i in a_pos],
                                            [items[i] for i in b_pos], [scores[i] for i in b_pos])
            name = {"rand": "E-RAND", "lex": "E-LEX", "att": "E-ATT"}[method]
            results.setdefault(key, {})[name] = {"rule": rule, **summary}
        r = results[key][{"rand": "E-RAND", "lex": "E-LEX", "att": "E-ATT", "abl": "E-ABL"}[method]]
        print(f"{method}: f1 {r['f1']} full_set {r['full_set']} auprc {r['auprc']} f1>=10 {r['f1_far_10plus']} "
              f"f1>1024tok {r['f1_beyond_1024_tok']} (evaluated {r['evaluated']})")
    io.write_json(opts.out, results)
    print(f"-> {opts.out}")


if __name__ == "__main__":
    main()
