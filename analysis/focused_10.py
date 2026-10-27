"""Focused 10-TDM study: every table from the saved answers, no inference.

    python analysis/focused_10.py --runs analysis/focused_runs.json --out results/focused_10 [--boot 10000]

The runs file (guide, Step 8) maps each system to its result folders. Thresholds: T-AF (threshold at a dev-score quantile) is fitted on the dev run of the
same model, question set and options; T-oracle on the test rows themselves (upper bound only); T-0.5 is fixed.
Backbone answers are re-read as R0, R1 or R2 from the two option orders stored by models/llm-backbone.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, ".")
from core import datasets, rules  # noqa: E402

QUANTILES = np.linspace(0.0005, 0.9995, 1999)
RAG = "RAGTruth"
POSITIVE = {"support_simple": True, "unsupported_detail": False, "contradicted_detail": False,
            "support_every_part": True, "any_unsupported": False}
SELECTION = "Documentation/paper_idea/selection/selection_0501b8c.csv"
RULES = {n: json.loads(Path(f"config/rules/{n}.json").read_text()) for n in ("P1", "P2", "P3", "P5", "P6")}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def read_run(folder):
    out = {}
    with open(Path(folder) / "answers.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if "identity" not in r:
                out[r["id"]] = r
    return out


def doc_keys(split):
    rows, _ = datasets.load("lytang--LLM-AggreFact", split)
    return {r["id"]: hashlib.md5(r["doc"].encode()).hexdigest() for r in rows}


# ---- backbone readouts ---------------------------------------------------------------------------------------------

def backbone_p(answer, mode, p0=0.5, temp=1.0):
    """P(yes) under R0 (first order), R1 (both orders, prior p0 divided out) or R2 (R1 at temperature temp)."""
    yf = answer["orders"]["yes_first"]
    p1 = sigmoid(yf[0] - yf[1])
    if mode == "R0":
        return float(p1)
    nf = answer["orders"]["no_first"]
    z = logit(0.5 * (p1 + sigmoid(nf[1] - nf[0]))) - logit(p0)
    return float(sigmoid(z / (temp if mode == "R2" else 1.0)))


def fit_backbone(dev_run):
    """Per question, on dev: label prior p0 = mean P(yes) (labels unused) and the temperature of least log loss."""
    calib = {}
    for q, positive in POSITIVE.items():
        rows = [r for r in dev_run.values() if q in r["answers"]]
        if not rows:
            continue
        raw = np.array([backbone_p(r["answers"][q], "R1") for r in rows])
        p0 = float(raw.mean())
        y = np.array([r["label"] if positive else 1 - r["label"] for r in rows])
        z = logit(raw) - logit(p0)

        def loss(t):
            p = np.clip(sigmoid(z / t), 1e-12, 1 - 1e-12)
            return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        calib[q] = (p0, float(min(np.arange(0.25, 4.01, 0.05), key=loss)))
    return calib


def scores(run, rule, readout=None, calib=None):
    """id -> score of the rule; a backbone's answers are re-read under its readout first."""
    out = {}
    for rid, r in run.items():
        answers = r["answers"]
        if readout:
            answers = {}
            for q, a in r["answers"].items():
                p = backbone_p(a, readout, *(calib or {}).get(q, (0.5, 1.0)))
                answers[q] = {**a, "probs": {"true": p, "false": 1 - p}}
        try:
            out[rid] = float(rules.score(answers, rule))
        except KeyError:
            continue
    return out


# ---- metrics -------------------------------------------------------------------------------------------------------

class Split:
    """Labels, scores and sources of a fixed list of rows (kept in the given order), grouped by source once."""

    def __init__(self, run, sc, ids):
        self.ids = [i for i in ids if i in sc and i in run]
        self.src = np.array([run[i]["source"] for i in self.ids])
        self.y = np.array([int(run[i]["label"]) for i in self.ids])
        self.s = np.array([sc[i] for i in self.ids])
        self.groups = {k: np.where(self.src == k)[0] for k in sorted(set(self.src))}

    def bacc(self, thr, idx=None, drop_rag=False):
        per = {}
        for k, g in self.groups.items():
            if drop_rag and k == RAG:
                continue
            g = g if idx is None else idx[k]
            y, p = self.y[g], self.s[g] > thr
            if (y == 1).any() and (y == 0).any():
                per[k] = 0.5 * (p[y == 1].mean() + (~p[y == 0]).mean())
        return float(np.mean(list(per.values()))), per

    def candidates(self):
        """Thresholds at the score quantiles: a monotone re-scaling (prior, temperature) moves them with the scores."""
        return np.unique(np.quantile(self.s, QUANTILES))

    def fit(self):
        return float(max(self.candidates(), key=lambda t: self.bacc(t)[0]))

    def auc(self):
        return float(np.mean([roc_auc_score(self.y[g], self.s[g]) for g in self.groups.values()
                              if len(set(self.y[g])) == 2]))

    def admission(self, recall=0.90):
        """Mean over sources of the share of unsupported rows accepted when 90% of supported rows are accepted."""
        out = []
        for g in self.groups.values():
            y, s = self.y[g], self.s[g]
            if (y == 1).any() and (y == 0).any():
                out.append(float((s[y == 0] >= np.quantile(s[y == 1], 1 - recall)).mean()))
        return float(np.mean(out))

    def ece(self, bins=10):
        idx = np.clip((self.s * bins).astype(int), 0, bins - 1)
        return float(sum(abs(self.s[idx == b].mean() - self.y[idx == b].mean()) * (idx == b).mean()
                         for b in range(bins) if (idx == b).any()))

    def brier(self):
        return float(np.mean((self.s - self.y) ** 2))


def draws(split, keys, n, seed=2026):
    """Document-level bootstrap resamples (documents drawn with replacement inside each source). The same seed gives
    the same resamples to every system, so differences between systems are paired."""
    rng = np.random.default_rng(seed)
    docs = {}
    for k, g in split.groups.items():
        d = {}
        for j in g:
            d.setdefault(keys[split.ids[j]], []).append(j)
        docs[k] = [np.array(v) for v in d.values()]
    for _ in range(n):
        yield {k: np.concatenate([lst[p] for p in rng.integers(0, len(lst), len(lst))]) for k, lst in docs.items()}


def ci(values):
    lo, hi = np.percentile(values, [2.5, 97.5])
    return f"[{100 * lo:.1f}, {100 * hi:.1f}]"


def pct(x):
    return f"{100 * x:.2f}"


def table(path, header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    Path(path).write_text("\n".join(lines) + "\n")


# ---- main ----------------------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", default="results/focused_10")
    ap.add_argument("--boot", type=int, default=10000)
    a = ap.parse_args()
    cfg = json.loads(Path(a.runs).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    keys = {**doc_keys("test"), **doc_keys("dev-jev4520")}

    # systems under the primary protocol (P1 for TDMs and backbones; a checker's own rule)
    systems = {}
    for label, c in cfg["systems"].items():
        test, dev = read_run(c["test"]), read_run(c["dev"])
        rule = RULES[c.get("rule", "P1")] if isinstance(c.get("rule", "P1"), str) else c["rule"]
        modes = ("R0", "R1", "R2") if c["kind"] == "backbone" else (None,)
        calib = fit_backbone(dev) if c["kind"] == "backbone" else None
        for mode in modes:
            name = f"{label} {mode}" if mode else label
            systems[name] = {**c, "label": label, "mode": mode, "test": test, "dev": dev, "rule": rule,
                             "sc_test": scores(test, rule, mode, calib), "sc_dev": scores(dev, rule, mode, calib),
                             "calib": calib}
    common = sorted(set.intersection(*(set(s["sc_test"]) for s in systems.values())),
                    key=lambda i: int(i.split(":")[1]))
    for s in systems.values():
        s["S"] = Split(s["test"], s["sc_test"], common)
        s["D"] = Split(s["dev"], s["sc_dev"], sorted(s["sc_dev"]))
        s["thr"] = s["D"].fit()
    ref = next(iter(systems.values()))["S"]
    def boots():
        return draws(ref, keys, a.boot)

    # E0/E1/E5/E6: main table on the rows every system answered
    rows = []
    for name, s in systems.items():
        S, t = s["S"], s["thr"]
        m, per = S.bacc(t)
        b = [S.bacc(t, idx)[0] for idx in boots()]
        rows.append([s.get("id", ""), name, pct(m), ci(b), pct(S.bacc(t, drop_rag=True)[0]), pct(S.bacc(0.5)[0]),
                     pct(S.bacc(S.fit())[0]), f"{S.auc():.3f}", pct(S.admission()), f"{S.ece():.3f}",
                     f"{S.brier():.3f}", f"{t:.3f}", len(s["sc_test"])] + [pct(per.get(k, float("nan"))) for k in S.groups])
    table(out / "table_main.md", ["ID", "System", "Macro BAcc T-AF", "95% CI", "no RAGTruth", "T-0.5", "T-oracle",
                                   "Macro AUC", "Admission@90", "ECE", "Brier", "T-AF", "rows answered"]
          + list(ref.groups), rows)

    # E2: attribution per backbone group
    rows = []
    for gname, g in cfg["groups"].items():
        bb = {m: systems[f"{g['backbone']} {m}"] for m in ("R0", "R1", "R2")}
        base_t = bb["R2"]["thr"]
        base_b = np.array([bb["R2"]["S"].bacc(base_t, idx)[0] for idx in boots()])
        r0, r1 = bb["R0"]["S"].bacc(bb["R0"]["thr"])[0], bb["R1"]["S"].bacc(bb["R1"]["thr"])[0]
        rows.append([gname, f"{g['backbone']} R0 / R1 / R2", f"{pct(r0)} / {pct(r1)} / {pct(bb['R2']['S'].bacc(base_t)[0])}",
                     f"Δ_readout {pct(r1 - r0)}; ECE R1 {bb['R1']['S'].ece():.3f} → R2 {bb['R2']['S'].ece():.3f}; "
                     f"T-0.5 R1 {pct(bb['R1']['S'].bacc(0.5)[0])} → R2 {pct(bb['R2']['S'].bacc(0.5)[0])}", ""])
        deltas = []
        for label in g["tdms"]:
            s = systems[label]
            d = s["S"].bacc(s["thr"])[0] - bb["R2"]["S"].bacc(base_t)[0]
            db = np.array([s["S"].bacc(s["thr"], idx)[0] for idx in boots()]) - base_b
            deltas.append(d)
            rows.append([gname, label, pct(s["S"].bacc(s["thr"])[0]), f"Δ_recipe {pct(d)} {ci(db)}", ""])
        deltas = np.array(deltas)
        if len(deltas) >= 2:
            sd = deltas.std(ddof=1)
            rows.append([gname, "summary", "", f"mean Δ {pct(deltas.mean())}; SD {pct(sd)}; "
                         f"max − min {pct(deltas.max() - deltas.min())}", f"H1 spread ≥ mean: {sd >= deltas.mean()}"])
    table(out / "table_attribution.md", ["Group", "System", "Macro BAcc (T-AF)", "Δ and 95% CI", "Note"], rows)

    # E4: protocol sensitivity of the TDMs
    tdms = [n for n, s in systems.items() if s["kind"] == "tdm"]
    proto = {n: {} for n in tdms}
    sub = [i for i in common if i in set(read_run(cfg["test4k_ids_from"]))] if cfg.get("test4k_ids_from") else []
    for n in tdms:
        s = systems[n]
        for rname in ("P1", "P2", "P3", "P6"):
            st, sd = scores(s["test"], RULES[rname]), scores(s["dev"], RULES[rname])
            S, D = Split(s["test"], st, common), Split(s["dev"], sd, sorted(sd))
            proto[n][f"{rname} T-0.5"] = S.bacc(0.5)[0]
            proto[n][f"{rname} T-AF"] = S.bacc(D.fit())[0]
            proto[n][f"{rname} T-oracle"] = S.bacc(S.fit())[0]
        v = cfg.get("variants", {}).get(n, {})
        if sub and v:
            proto[n]["AF-4k P1 C2"] = Split(s["test"], s["sc_test"], sub).bacc(s["thr"])[0]
            for key, rname in (("QF2", "P1"), ("QF5", "P5"), ("trunc32k", "P1")):
                if key in v:
                    vt = read_run(v[key]["test"])
                    vd = read_run(v[key]["dev"]) if "dev" in v[key] else s["dev"]
                    st, sd = scores(vt, RULES[rname]), scores(vd, RULES[rname])
                    proto[n][f"AF-4k {key}"] = Split(vt, st, sub).bacc(Split(vd, sd, sorted(sd)).fit())[0]
            if "QF3" in v and "QF4" in v:
                t3, t4 = read_run(v["QF3"]["test"]), read_run(v["QF4"]["test"])
                d3, d4 = read_run(v["QF3"]["dev"]), read_run(v["QF4"]["dev"])

                def p4(a3, a4):
                    return {i: 0.5 * (a3[i]["answers"]["support_every_part"]["probs"]["true"]
                                      + 1 - a4[i]["answers"]["any_unsupported"]["probs"]["true"])
                            for i in a3 if i in a4}
                st, sd = p4(t3, t4), p4(d3, d4)
                proto[n]["AF-4k P4"] = Split(t3, st, sub).bacc(Split(d3, sd, sorted(sd)).fit())[0]
                gap = [abs(t3[i]["answers"]["support_every_part"]["probs"]["true"]
                           - (1 - t4[i]["answers"]["any_unsupported"]["probs"]["true"])) for i in sub if i in t3 and i in t4]
                proto[n]["polarity gap"] = float(np.mean(gap))
    cols = sorted({k for d in proto.values() for k in d if k != "polarity gap"})
    rows = []
    for n in tdms:
        vals = [proto[n][c] for c in cols if c in proto[n] and "oracle" not in c and not c.startswith("AF-4k")]
        rows.append([n] + [pct(proto[n][c]) if c in proto[n] else "" for c in cols]
                    + [pct(max(vals) - min(vals)) if vals else "", f"{proto[n].get('polarity gap', float('nan')):.3f}"])
    ranks = {c: [proto[n].get(c, np.nan) for n in tdms] for c in cols}
    tau = [f"{c}: {kendalltau(ranks['P1 T-AF'], ranks[c], nan_policy='omit')[0]:.2f}" for c in cols if c != "P1 T-AF"]
    table(out / "table_protocol.md", ["System"] + cols + ["range over full-test P1-P3, P6 (T-0.5, T-AF)", "polarity gap"], rows)
    (out / "table_protocol.md").write_text((out / "table_protocol.md").read_text()
                                            + "\nKendall τ against P1 T-AF: " + "; ".join(tau) + "\n")

    # E6: threshold transfer across sources (P1): fit on one source's dev rows, apply to another source's test rows
    rows = []
    for n in tdms:
        s = systems[n]
        S, D = s["S"], s["D"]
        cand = D.candidates()
        per = [D.bacc(t)[1] for t in cand]
        fit = {k: float(cand[int(np.argmax([p.get(k, 0) for p in per]))]) for k in D.groups}
        losses = [S.bacc(fit[j])[1][j] - S.bacc(fit[i])[1][j] for i in fit for j in fit if i != j and j in S.groups]
        rows.append([n, pct(float(np.mean(losses))), pct(float(np.max(losses))), f"{S.ece():.3f}", f"{S.brier():.3f}"])
    table(out / "table_calibration.md", ["System", "mean transfer loss", "worst transfer loss", "ECE", "Brier"], rows)

    # E9: Decision Index ranking against AggreFact (descriptive, n = number of TDMs)
    board = {r["model_url"].split("huggingface.co/")[-1]: r for r in csv.DictReader(open(SELECTION))}
    di = [(systems[n]["S"].bacc(systems[n]["thr"])[0], float(board[r]["full_score"]), float(board[r]["public_score"]),
           float(board[r]["grounding_pct"])) for n in tdms if (r := systems[n].get("repo")) in board]
    if len(di) >= 4:
        af = [d[0] for d in di]
        rows = [[q, f"{spearmanr(af, [d[i] for d in di])[0]:.2f}", len(di)] for i, q in ((1, "Full"), (2, "Public"), (3, "Grounding pct"))]
        table(out / "table_decision_index.md", ["Decision Index quantity", "Spearman ρ with AF macro BAcc", "n"], rows)
    print(f"tables in {out}: rows answered by every system = {len(common)}")


if __name__ == "__main__":
    main()
