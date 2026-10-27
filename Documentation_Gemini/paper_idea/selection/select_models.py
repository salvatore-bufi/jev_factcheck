"""Model selection for the research plan, from one pinned snapshot of the Decision Index.

Source (the only source of leaderboard numbers used here):
  Hugging Face dataset multimodalart/decision-index-leaderboard, commit 0501b8cae77a94b3c56dc925609d88c6210835f7
  ("Decision Index 0.4.1", generated 2026-10-26T15:30:58Z), licence CC BY 4.0.
  leaderboard.parquet: one row per entrant (full_score, same_skills_eq, new_domains_eq, public_score, ...).
  benchmarks.parquet: one row per (entrant, public benchmark) with score, requests, answered.

    python Documentation/paper_idea/selection/select_models.py

writes, next to this file:
  snapshot/leaderboard.parquet, snapshot/benchmarks.parquet, snapshot/SHA256SUMS   (the pinned copy)
  selection_0501b8c.csv   one row per eligible entrant: metrics, flags, selected tier and the rule that selected it
  selection_0501b8c.md    the selected models and the motivation numbers, as tables

No local benchmark result is used. The manual inputs are the constants below, each with its reason.
"""
import hashlib
import math
import io
import urllib.request
from pathlib import Path

import pandas as pd

REPO = "multimodalart/decision-index-leaderboard"
COMMIT = "0501b8cae77a94b3c56dc925609d88c6210835f7"
FILES = ("leaderboard.parquet", "benchmarks.parquet")
HERE = Path(__file__).resolve().parent
SNAP = HERE / "snapshot"
TAG = COMMIT[:7]

MAX_PARAMS_B = 9.7           # nominal 9B: the Qwen3.5 "9B" class serves 9.4-9.7B parameters with its vision tower
MIN_COVERAGE = 0.99          # answered / requests over the public suite
STRONG_SHARE = 0.25          # S4: top quarter of its backbone group by combined rank (rounded up, at least 2, ties kept)
TIE_BAND = 0.9               # the board's own tie band on the Full score
GROUNDING = ["RAGTruth", "HoVer", "ANLI", "ContractNLI", "NLI4CT"]
CORE_GROUPS = {"G9": "Qwen/Qwen3.5-9B", "G9B": "Qwen/Qwen3.5-9B-Base", "G4": "Qwen/Qwen3.5-4B"}

# Excluded although eligible on the numbers (C8 of the plan), with the reason.
EXCLUDED = {
    "Bespoke Nimble 9B v2": "superseded by Bespoke Nimble 9B v3",
    "Sieve 9B": "superseded by Sieve 9B v2",
    "Sieve 2B": "superseded by Sieve 2B v3",
    "RSI-Jev v5.0 VL 3B": "superseded by RSI-Jev v6.1",
    "vLLM-SR Decision 2.0 Lux 9B": "superseded by vLLM-SR Decision 3.0 (d3)",
    "vLLM-SR Decision 2.0 Nox 4B": "superseded by vLLM-SR Decision 3.0 (d3)",
    "vLLM-SR Decision 2.0 Sol 2B": "superseded by vLLM-SR Decision 3.0 (d3)",
    "vLLM-SR Decision 2.0 Eos 0.8B": "superseded by vLLM-SR Decision 3.0 (d3)",
    "vLLM-SR Decision 2.0 Kai 0.6B": "superseded by vLLM-SR Decision 3.0 (d3)",
    "Saracura PT-BR 4B": "language-specific (Brazilian Portuguese)",
}

# Backbone family for non-Qwen entrants whose board base_model is an intermediate checkpoint (resolved from its card).
FAMILY_OF_BASE = {
    "StandardThinking/StandardOne-8B": "Ministral-3-8B",       # card: mistralai/Ministral-3-8B-Instruct-2512-BF16
    "mistralai/Ministral-3-8B-Base-2512": "Ministral-3-8B",
    "google/gemma-4-E4B": "Gemma-4-E4B",
    "google/gemma-4-E4B-it": "Gemma-4-E4B",
    "LiquidAI/LFM2.5-VL-3B": "LFM2.5-2.6B",                    # card: built on LiquidAI/LFM2.5-2.6B-Base
    "LiquidAI/LFM2.5-2.6B-Base": "LFM2.5-2.6B",
}

# Encoder backbone families (technique "encoder" or GLi*), from the board base_model.
ENCODER_FAMILY = (("deberta", "DeBERTa-v3"), ("modernbert", "ModernBERT"), ("ettin", "Ettin"))

# Recipe families with several sizes and one runtime (S3b); matched on the board's model name.
LADDERS = {
    "StartLux v1.1": r"^StartLux-Decision v1\.1 ",
    "d3": r"^vLLM-SR d3-",
    "OneJev 1.6": r"^OneJev1\.6-",
    "JPT": r"^JPT-",
    "Kev": r"^Kev ",
}

# S5, published LLM-AggreFact leaderboard entries <= 9B that are on the board
# (llm-aggrefact.github.io, read 2026-10-26; a public external source, not a local run).
AGGREFACT_PUBLISHED = {"Cloudflare clef-flash": 76.6, "Nace.AI Drex v1.5": 76.5}

# Controls chosen from the literature, not from board strength.
CONTROLS = {
    "CLM-v0.1-8B Qwen3-8B heads": "contrastive readout (cosine between state and option embeddings): readout control",
    "Laya": "the open TDM most used in prior evaluations (Arize, Red Hat, PremiseCheck): encoder floor",
}

# Rows shown only as motivation: training-free readouts and reference systems above the size limit.
MOTIVATION = ["Ryotide Qwen v0.3.1", "Jobe Qwen3.5-4B", "openvons Qwen3-4B", "SemIf Qwen3.5-4B Base",
              "KnowLine 4B Gen5.5", "decisio 0.4.0 (Gemma 4 12B)", "Winnow-12B", "FlexDM · Gemma-4-31B",
              "AnyJev L0 · Gemma-4-31B", "Jeffan-31b-1710", "Fastino GLiDE 1.2", "Jev 1.14",
              "baselabs elephant-1 (9B)"]


def snapshot():
    """Download the pinned files once, then always read the local copy; record and check sha256."""
    SNAP.mkdir(exist_ok=True)
    sums = {}
    for name in FILES:
        path = SNAP / name
        if not path.exists():
            url = f"https://huggingface.co/datasets/{REPO}/resolve/{COMMIT}/{name}"
            path.write_bytes(urllib.request.urlopen(url, timeout=120).read())
        sums[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    record = SNAP / "SHA256SUMS"
    lines = "".join(f"{sums[n]}  {n}\n" for n in FILES)
    if record.exists() and record.read_text() != lines:
        raise SystemExit(f"snapshot changed: {record} does not match the files")
    record.write_text(lines)
    (SNAP / "SOURCE.txt").write_text(
        f"Copy of https://huggingface.co/datasets/{REPO} at commit {COMMIT}\n"
        "(Decision Index 0.4.1, generated 2026-10-26T15:30:58Z). Licence: CC BY 4.0, (c) the Decision Index maintainers.\n")
    return {n: pd.read_parquet(io.BytesIO((SNAP / n).read_bytes())) for n in FILES}, sums


def family(row):
    base = row["base_model"] if isinstance(row["base_model"], str) else ""
    if base in CORE_GROUPS.values():
        return base
    if base in FAMILY_OF_BASE:
        return FAMILY_OF_BASE[base]
    if row["technique"] != "autoregressive":
        low = base.lower()
        for key, name in ENCODER_FAMILY:
            if key in low:
                return f"encoder:{name}"
        return "encoder:other"
    return FAMILY_OF_BASE.get(base, base or "unknown")


def rank_sum(frame):
    return frame.full_score.rank(ascending=False) + frame.public_score.rank(ascending=False)


def main():
    data, sums = snapshot()
    board, bench = data["leaderboard.parquet"], data["benchmarks.parquet"]

    cov = bench.groupby("engine")[["answered", "requests"]].sum()
    board = board.join((cov.answered / cov.requests).rename("coverage"), on="engine")

    pool = board[(board.params_b <= MAX_PARAMS_B) & (board.kind != "inference technique") & (~board.is_jev)
                 & board.model_url.fillna("").str.startswith("https://huggingface.co/")
                 & (board.coverage >= MIN_COVERAGE)].copy()
    pool["excluded"] = pool.model.map(EXCLUDED)
    pool = pool[pool.excluded.isna()].drop(columns="excluded")
    pool["family"] = pool.apply(family, axis=1)

    # S4: combined rank R = rank(Full) + rank(Public) inside the backbone group; strong = top quarter of the group
    grp = pool.groupby("family")
    pool["group_n"] = grp.full_score.transform("size")
    pool["rank_full_in_group"] = grp.full_score.rank(ascending=False, method="min")
    pool["rank_public_in_group"] = grp.public_score.rank(ascending=False, method="min")
    pool["combined_rank"] = pool.rank_full_in_group + pool.rank_public_in_group
    pool["combined_pos"] = pool.groupby("family").combined_rank.rank(method="min")
    k = pool.group_n.map(lambda n: max(2, math.ceil(STRONG_SHARE * n)))
    pool["strong"] = pool.combined_pos <= k

    # M4: mean percentile rank, inside the eligible pool, on the five grounding-type public benchmarks
    g = bench[bench.benchmark.isin(GROUNDING) & bench.engine.isin(pool.engine)].pivot(
        index="engine", columns="benchmark", values="score")
    pct = g.rank(pct=True)
    pool = pool.join(g.add_prefix("raw_"), on="engine").join((100 * pct.mean(axis=1)).rename("grounding_pct"), on="engine")

    pool["tier"], pool["rule"] = "", ""

    def pick(mask, tier, rule):
        idx = pool.index[mask & (pool.tier == "")]
        pool.loc[idx, "tier"] = tier
        pool.loc[idx, "rule"] = rule

    # Core groups: strong members, then one member per training kind not yet covered, then fill to >= 2
    for gid, base in CORE_GROUPS.items():
        in_group = pool.family == base
        pick(in_group & pool.strong, f"A:{gid}", "S4 top quarter of the group by rank(Full) + rank(Public)")
        members = pool[in_group].sort_values(["combined_rank", "full_score"], ascending=[True, False])
        for kind, rows in members.groupby("kind", sort=False):
            if not (pool.loc[rows.index, "tier"] != "").any():
                pick(pool.index == rows.index[0], f"A:{gid}", f"S3 training-kind coverage ({kind}): best rank-sum")
        while (pool.tier == f"A:{gid}").sum() < 2:
            nxt = members[pool.loc[members.index, "tier"] == ""].index[0]
            pick(pool.index == nxt, f"A:{gid}", "S3 group size >= 2: best rank-sum")

    # S5: published LLM-AggreFact entries
    pick(pool.model.isin(AGGREFACT_PUBLISHED), "A:other", "S5 published LLM-AggreFact entry to reproduce")

    # Non-Qwen decoder families (any readout, encoder-style included) with >= 2 entrants on the board: all members
    base_txt = pool.base_model.fillna("").str.lower()
    nq = pool[~base_txt.str.contains("qwen") & ~pool.family.str.startswith("encoder") & (pool.family != "unknown")]
    multi = nq.family.value_counts()
    for fam in multi[multi >= 2].index:
        pick(pool.family == fam, f"N:{fam}", "S3e non-Qwen family with >= 2 entrants: all members")

    # Encoders: the best rank-sum entrant of each declared encoder backbone family
    for _, name in ENCODER_FAMILY:
        rows = pool[pool.family == f"encoder:{name}"]
        if len(rows):
            best = rows.assign(rs=rank_sum(rows)).sort_values("rs").index[0]
            pick(pool.index == best, "N:encoders", f"S3d encoder family {name}: best rank-sum")

    # Ladders anchored to a selected core model: every other size of the same recipe
    for name, pattern in LADDERS.items():
        fam = pool.model.str.contains(pattern, regex=True)
        if (fam & pool.tier.str.startswith("A:")).any() and fam.sum() >= 3:
            pick(fam, f"B:{name}", "S3b size ladder of a selected recipe")

    # Non-Qwen singletons with Full >= 30: level B
    single = multi[multi == 1].index
    pick(pool.family.isin(single) & (pool.full_score >= 30), "B:non-Qwen singleton", "S3e non-Qwen singleton, Full >= 30")

    # Controls from the literature
    for model, why in CONTROLS.items():
        hit = board.model == model
        if hit.any() and not (pool.model == model).any():
            extra = board[hit].assign(family="control", strong=False, tier="C", rule=f"control: {why}")
            pool = pd.concat([pool, extra])
        else:
            pick(pool.model == model, "C", f"control: {why}")

    pool["aggrefact_published"] = pool.model.map(AGGREFACT_PUBLISHED)
    cols = ["tier", "rule", "model", "model_url", "params_b", "family", "base_model", "kind", "technique",
            "rank", "tied", "full_score", "same_skills_eq", "new_domains_eq", "public_score", "public_rank",
            "group_n", "rank_full_in_group", "rank_public_in_group", "combined_rank", "combined_pos", "strong",
            "grounding_pct"] + [f"raw_{b}" for b in GROUNDING] + \
           ["coverage", "aggrefact_published", "revision"]
    out = pool[cols].sort_values(["tier", "full_score"], ascending=[True, False])
    out.to_csv(HERE / f"selection_{TAG}.csv", index=False)

    md = [f"# Model selection from Decision Index 0.4.1 (snapshot {TAG})\n",
          f"Source: `{REPO}` at commit `{COMMIT}` (CC BY 4.0). sha256: "
          + ", ".join(f"`{n}` {s[:12]}…" for n, s in sums.items()) + ".\n",
          f"Eligible pool: {len(pool[pool.tier != 'C'])} entrants (params ≤ {MAX_PARAMS_B}B, open weights on Hugging Face, "
          f"public coverage ≥ {MIN_COVERAGE}, not a training-free technique, {len(EXCLUDED)} declared exclusions). "
          f"Strong = top quarter of its backbone group by rank(Full) + rank(Public) (at least 2, ties kept). "
          f"Grounding % = mean percentile rank in the pool "
          f"on {', '.join(GROUNDING)}. Full-score differences below {TIE_BAND} share a rank on the board.\n",
          "| Tier | Model | Params (B) | Base (board) | Kind | Full (rank) | Public (rank) | In-group position | Strong | Grounding % | Rule |",
          "|---|---|---:|---|---|---|---|---|---|---:|---|"]
    for _, r in out[out.tier != ""].iterrows():
        md.append(f"| {r.tier} | {r.model} | {r.params_b:.2f} | {r.base_model} | {r.kind} | {r.full_score:.2f} ({int(r['rank'])}) | "
                  f"{r.public_score:.2f} ({int(r.public_rank)}) | "
                  f"{'' if pd.isna(r.combined_pos) else f'{int(r.combined_pos)} of {int(r.group_n)}'} | {'yes' if r.strong else ''} | "
                  f"{'' if pd.isna(r.grounding_pct) else f'{r.grounding_pct:.0f}'} | {r.rule} |")
    mot = board[board.model.isin(MOTIVATION)].join(board.set_index("engine").coverage, on="engine", rsuffix="_")
    md += ["\n## Motivation rows (training-free readouts and references)\n",
           "| Model | Params (B) | Kind | Base | Full (rank) | Public (rank) |", "|---|---:|---|---|---|---|"]
    for _, r in mot.sort_values("params_b", na_position="last").iterrows():
        p = "" if pd.isna(r.params_b) else f"{r.params_b:.2f}"
        md.append(f"| {r.model} | {p} | {r.kind if isinstance(r.kind, str) else 'hosted API'} | "
                  f"{r.base_model if isinstance(r.base_model, str) else ''} | {r.full_score:.2f} ({int(r['rank'])}) | "
                  f"{r.public_score:.2f} ({int(r.public_rank)}) |")
    (HERE / f"selection_{TAG}.md").write_text("\n".join(md) + "\n")
    print(f"{len(out)} rows written; selected: {(out.tier != '').sum()}")
    print(out[out.tier != ""].groupby("tier").model.apply(list).to_string())


if __name__ == "__main__":
    main()
