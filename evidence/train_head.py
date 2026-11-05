"""STEP 8 — Train the evidence head on saved features (phase 1a: the decision model is not in memory, it cannot change).

Where it sits in the workflow:
    evidence.extract (features of the mix and of the dev sets)  ->  [evidence.train_head]  ->  data/evidence/runs/...
    ->  evidence.eval_evidence / evidence.release

Run:
    python -m evidence.train_head --config evidence/configs/h0.json \
        --train data/evidence/features/knowline-4b/ev_head \
        --dev data/evidence/features/knowline-4b/dev_evid_select \
        --confirm data/evidence/features/knowline-4b/dev_evid_confirm \
        --dev-neg data/evidence/features/knowline-4b/dev_evid_neg \
        --eval-every 1 --log-steps 20 --train-eval 2000      # (these three are the defaults)
    python -m evidence.train_head --config evidence/configs/h0.json --train <features> --overfit 64
        # sanity check before any real run: the head must learn 64 rows by heart (F1 close to 1)

The config (evidence/configs/<id>.json) says everything about one experiment:
    layers_read  which layers to read from the features ([31] or [19, 23, 27, 31])
    parts        "end" and/or "mean"
    head         options of EvidenceHead (evidence/head.py)
    loss         {"name": "ref", alpha, gamma, margin}  or  {"name": "zlpr", mu_neg, mu_pos, rho}  (+ pair_weight)
    train        epochs, batch, lr, weight_decay, patience, seed, terms (which evidence_term rows are used)

Writes data/evidence/runs/knowline-4b/<config id>[_<tag>]/:
    config.json        the config + the feature folders + their meta
    head.safetensors   the weights of the best epoch
    selection.json     the selection rule chosen on dev (doc 09 §2): delta (document threshold) or tau (fixed)
    log.jsonl          one line per evaluation (every --eval-every epochs): train / dev loss, F1, full set
    steps.jsonl        the training loss every --log-steps optimizer steps (live: tail -f)
    tb/                the same curves for TensorBoard, if installed (live: tensorboard --logdir data/evidence/runs)
    dev_metrics.json   metrics on dev_evid_select, dev_evid_confirm and dev_evid_neg with the chosen rule
and appends one line to data/evidence/runs/registry.md (doc 11).
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evidence import config  # noqa: E402
from evidence.data import io  # noqa: E402
from evidence.features import FeatureStore, same_features  # noqa: E402
from evidence.head import build_head  # noqa: E402
from evidence.losses import TERMS, pair_loss, ref_loss, zlpr_loss  # noqa: E402
from evidence.metrics import gold_tok_last, logit, select, summarize  # noqa: E402
from evidence.model import p_supported  # noqa: E402


# ------------------------------------------------------------------ batches
def collate(rows, device):
    """Pad a list of rows (dicts from FeatureStore.get) into batch tensors.

    Input:  rows, device.
    Output: dict of tensors: q [B, L, D], h_end [B, N, L, D], h_mean (same or None), att [B, N, 128], n_tok [B, N],
            mask [B, N], gold [B, N], omega [B, N], term [B] (long), weight [B]. N = most sentences in the batch.
    """
    n = max(r["n"] for r in rows)
    B = len(rows)
    L, D = rows[0]["q"].shape
    has_mean = rows[0]["h_mean"] is not None
    h_end = torch.zeros(B, n, L, D, dtype=torch.bfloat16)
    h_mean = torch.zeros(B, n, L, D, dtype=torch.bfloat16) if has_mean else None
    att = torch.zeros(B, n, rows[0]["att"].shape[1])
    n_tok, mask = torch.ones(B, n), torch.zeros(B, n)
    gold, omega = torch.zeros(B, n), torch.ones(B, n)
    for i, r in enumerate(rows):
        k = r["n"]
        h_end[i, :k] = r["h_end"]
        if has_mean:
            h_mean[i, :k] = r["h_mean"]
        att[i, :k] = r["att"].float()
        n_tok[i, :k] = r["n_tok"].float()
        gold[i, :k] = r["gold"]
        mask[i, :k] = 1
        omega[i, :k] = r["omega"]
    batch = {"q": torch.stack([r["q"] for r in rows]), "h_end": h_end, "h_mean": h_mean, "att": att,
             "n_tok": n_tok, "mask": mask, "gold": gold, "omega": omega,
             "term": torch.tensor([TERMS[r["evidence_term"]] for r in rows]),
             "weight": torch.tensor([float(r["row_weight"]) for r in rows])}
    return {k: (v.to(device) if v is not None else None) for k, v in batch.items()}


def length_batches(lengths, size, seed):
    """Group rows of similar length (less padding), in random order.

    Input:  lengths (sentences per row), size (rows per batch), seed.
    Output: list of lists of row positions.
    """
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    groups = [order[i:i + size] for i in range(0, len(order), size)]
    random.Random(seed).shuffle(groups)
    return groups


def run_head(head, batch):
    """Forward of the head on a collated batch. Output: (s [B, N], s0 [B], S or None)."""
    return head(batch["q"], batch["h_end"], batch["h_mean"], batch["att"], batch["n_tok"], batch["mask"])


def batch_loss(head, batch, loss_cfg):
    """Loss of one batch: per-row loss x row weight, summed, divided by the weights of the rows that have a loss.

    Input:  head, batch (from collate), loss_cfg (the config's "loss").
    Output: scalar tensor.
    """
    s, s0, S = run_head(head, batch)
    if loss_cfg["name"] == "ref":
        per_row = ref_loss(s, batch["gold"], batch["mask"], batch["omega"], alpha=loss_cfg.get("alpha", 10.0),
                           gamma=loss_cfg.get("gamma", 0.5), margin=loss_cfg.get("margin", 1.0))
        active = (batch["term"] == TERMS["supervised"]).float()        # the reference loss sees positives only
        per_row = per_row * active
    else:
        per_row = zlpr_loss(s, s0, batch["gold"], batch["mask"], batch["omega"], batch["term"],
                            mu_neg=loss_cfg.get("mu_neg", 0.3), mu_pos=loss_cfg.get("mu_pos", 0.3),
                            rho=loss_cfg.get("rho", 0.1))
        active = (batch["term"] != TERMS["none"]).float()
    w = batch["weight"] * active
    loss = (per_row * w).sum() / w.sum().clamp_min(1e-6)
    if S is not None and loss_cfg.get("pair_weight"):
        loss = loss + loss_cfg["pair_weight"] * pair_loss(S, batch["gold"], batch["mask"])
    return loss


@torch.no_grad()
def mean_loss(head, rows, loss_cfg, device, size=32):
    """Average loss on a list of rows, in eval mode (no dropout, no gradients): the "error" on train or dev.

    Input:  head, rows (dicts from FeatureStore.get), loss_cfg, device, size (rows per forward).
    Output: float, the mean of the batch losses weighted by the number of rows (None if rows is empty).
    """
    if not rows:
        return None
    head.eval()
    total = 0.0
    for i in range(0, len(rows), size):
        chunk = rows[i:i + size]
        total += batch_loss(head, collate(chunk, device), loss_cfg).item() * len(chunk)
    return total / len(rows)


# ------------------------------------------------------------------ logging
class RunLogger:
    """Writes the training curves while the run goes on, so they can be watched live.

    Two files in the run folder, one JSON per line (readable with `tail -f`):
        steps.jsonl  every `every` optimizer steps: {"step", "epoch", "loss", "lr"}
        log.jsonl    every evaluation (every --eval-every epochs): train / dev loss and metrics
    plus, if the `tensorboard` package is installed, the same numbers in <run>/tb
    (watch them with `tensorboard --logdir data/evidence/runs`).
    With run_dir=None (overfit check) it only prints.
    """

    def __init__(self, run_dir=None, every=20):
        """Input: run_dir (folder of the run, or None), every (log the step loss every N steps)."""
        self.every, self.writer, self.steps_path, self.log_path = every, None, None, None
        self._acc = []                                  # step losses since the last step line
        if run_dir is None:
            return
        self.steps_path, self.log_path = run_dir / "steps.jsonl", run_dir / "log.jsonl"
        self.steps_path.write_text("")                  # a new run with the same id starts from empty files
        self.log_path.write_text("")
        try:
            import shutil
            from torch.utils.tensorboard import SummaryWriter
            shutil.rmtree(run_dir / "tb", ignore_errors=True)   # old curves of the same run id would mix
            self.writer = SummaryWriter(str(run_dir / "tb"))
        except ImportError:
            print("tensorboard not installed: curves only in steps.jsonl / log.jsonl (pip install tensorboard)")

    def step(self, step, epoch, loss, lr):
        """Record the loss of one optimizer step; write the mean of the last `every` steps."""
        self._acc.append(loss)
        if len(self._acc) < self.every:
            return
        entry = {"step": step, "epoch": epoch, "loss": round(sum(self._acc) / len(self._acc), 5), "lr": lr}
        self._acc = []
        if self.steps_path:
            with open(self.steps_path, "a") as handle:
                handle.write(json.dumps(entry) + "\n")
        if self.writer:
            self.writer.add_scalar("loss/train_step", entry["loss"], step)
            self.writer.add_scalar("lr", lr, step)
            self.writer.flush()

    def epoch(self, entry, step):
        """Record one evaluation (dict with train_* and dev_* numbers): print, log.jsonl, TensorBoard."""
        print(json.dumps(entry))
        if self.log_path:
            with open(self.log_path, "a") as handle:
                handle.write(json.dumps(entry) + "\n")
        if self.writer:
            for split in ("train", "dev"):              # same chart for train and dev -> easy to compare
                for name in ("loss", "f1", "full_set", "score"):
                    value = entry.get(f"{split}_{name}")
                    if value is not None:
                        self.writer.add_scalar(f"{name}/{split}", value, entry["epoch"])
            self.writer.add_scalar("loss/train_running", entry["train_loss_running"], entry["epoch"])
            self.writer.flush()

    def close(self):
        """Close the TensorBoard writer (if any)."""
        if self.writer:
            self.writer.close()


# ------------------------------------------------------------------ predictions
@torch.no_grad()
def scores_of(head, rows, device, size=32):
    """Head scores of a list of rows.

    Input:  head, rows (dicts from FeatureStore.get), device, size (rows per forward).
    Output: list of (scores list, s0 float), one per row.
    """
    head.eval()
    out = []
    for i in range(0, len(rows), size):
        chunk = rows[i:i + size]
        s, s0, _ = run_head(head, collate(chunk, device))
        for j, r in enumerate(chunk):
            out.append((s[j, :r["n"]].float().cpu().tolist(), float(s0[j])))
    return out


def predictions(rows, scored, rule):
    """Turn head scores into prediction dicts (format of evidence/metrics.py).

    Input:  rows, scored (output of scores_of), rule {"kind": "doc", "delta": d} or {"kind": "fixed", "tau": t}.
    Output: list of predictions.
    """
    preds = []
    for r, (s, s0) in zip(rows, scored):
        thr = s0 + rule["delta"] if rule["kind"] == "doc" else logit(rule["tau"])
        preds.append({"id": r["id"], "source": r["source"], "label": r["label"],
                      "p_supported": p_supported(r["answer_logits"]), "scores": s, "s0": s0, "threshold": thr,
                      "shown": select(s, thr), "evidence_sets": r["evidence_sets"], "gold_tok_last": gold_tok_last(r)})
    return preds


def choose_rule(rows, scored, loss_name):
    """Pick the selection rule on dev (doc 09 §2).

    Input:  dev rows, their scores, loss name ("ref" -> fixed threshold tau; "zlpr" -> document threshold + delta).
    Output: (rule dict, its summary on these rows).
    zlpr: delta in [-1, 1] by 0.1; delta != 0 is kept only if it gains at least 1 point of F1.
    ref:  tau in 0.05 ... 0.95, the best F1.
    """
    if loss_name == "ref":
        cands = [{"kind": "fixed", "tau": round(t, 2)} for t in [i / 20 for i in range(1, 20)]]
    else:
        cands = [{"kind": "doc", "delta": round(d / 10, 1)} for d in range(-10, 11)]
    results = [(rule, summarize(predictions(rows, scored, rule))) for rule in cands]
    best_rule, best = max(results, key=lambda rm: rm[1]["f1"] or 0.0)
    if loss_name != "ref":
        zero = next(m for rule, m in results if rule["delta"] == 0.0)
        if (best["f1"] or 0) - (zero["f1"] or 0) < 0.01:
            best_rule, best = {"kind": "doc", "delta": 0.0}, zero
    return best_rule, best


# ------------------------------------------------------------------ training
def load_rows(store, cfg, terms=None, limit=0):
    """Rows of a feature store in RAM, reading only the layers / parts of the config.

    Input:  store (FeatureStore), cfg (experiment config), terms (keep only these evidence terms, or None), limit.
    Output: list of row dicts.
    """
    idx = [i for i, e in enumerate(store.index) if terms is None or e["evidence_term"] in terms]
    if limit:
        idx = idx[:limit]
    return [store.get(i, cfg["layers_read"], tuple(cfg["parts"])) for i in idx]


def train(cfg, train_rows, dev_rows, device, logger=None, eval_every=1, train_eval_rows=None):
    """Train the head; keep the best evaluated epoch on dev (early stopping).

    Input:  cfg (experiment config), train_rows, dev_rows (rows of dev_evid_select; may be the train rows for the
            overfit check), device,
            logger (RunLogger; None -> print only),
            eval_every (evaluate train and dev every N epochs; the last epoch is always evaluated),
            train_eval_rows (rows of the train set on which the train loss / F1 are measured; None -> no train metrics).
    Output: (head with the best weights, best dev score, list of evaluation logs).
    Early stopping: `patience` counts evaluations without improvement (= epochs when eval_every is 1).
    """
    logger = logger or RunLogger()
    t = cfg["train"]
    torch.manual_seed(t.get("seed", config.SEED))
    d_model = train_rows[0]["q"].shape[1]
    head = build_head(cfg, d_model, n_att=train_rows[0]["att"].shape[1]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=t["lr"], weight_decay=t.get("weight_decay", 0.01))
    steps = t["epochs"] * math.ceil(len(train_rows) / t["batch"])
    warm = max(1, int(0.05 * steps))
    # linear warm-up over the first 5% of the steps, then cosine decay
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda k: min(1.0, (k + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(k, steps) / steps)))
    lengths = [r["n"] for r in train_rows]
    best, best_state, bad, history, step = -1.0, None, 0, [], 0
    for epoch in range(t["epochs"]):
        head.train()
        t0, losses = time.time(), []
        for group in length_batches(lengths, t["batch"], t.get("seed", config.SEED) + epoch):
            batch = collate([train_rows[i] for i in group], device)
            loss = batch_loss(head, batch, cfg["loss"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
            step += 1
            logger.step(step, epoch, losses[-1], sched.get_last_lr()[0])
        if (epoch + 1) % eval_every != 0 and epoch != t["epochs"] - 1:
            continue                                    # no evaluation this epoch
        # dev: loss + metrics with the rule chosen on dev
        scored = scores_of(head, dev_rows, device)
        rule, m = choose_rule(dev_rows, scored, cfg["loss"]["name"])
        entry = {"epoch": epoch, "step": step,
                 "train_loss_running": round(sum(losses) / len(losses), 5),   # mean over the epoch, with dropout
                 "dev_loss": round(mean_loss(head, dev_rows, cfg["loss"], device), 5),
                 "dev_f1": m["f1"], "dev_full_set": m["full_set"], "dev_score": m["score"], "rule": rule}
        # train: the same numbers on a fixed sample of training rows, same rule -> compare with dev (overfitting)
        if train_eval_rows:
            m_tr = summarize(predictions(train_eval_rows, scores_of(head, train_eval_rows, device), rule))
            entry.update(train_loss=round(mean_loss(head, train_eval_rows, cfg["loss"], device), 5),
                         train_f1=m_tr["f1"], train_full_set=m_tr["full_set"], train_score=m_tr["score"])
        entry.update(seconds=round(time.time() - t0, 1), layer_weights=head.layer_weights())
        history.append(entry)
        logger.epoch(entry, step)
        score = m["score"] or 0.0
        if score > best:
            best, bad = score, 0
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        else:
            bad += 1
            if bad >= t.get("patience", 2):
                break
    head.load_state_dict(best_state)
    return head, best, history


def append_registry(run_dir, cfg, metrics_select, metrics_neg, train_dir):
    """One line in data/evidence/runs/registry.md (doc 11): the numbers used to keep or drop a change."""
    path = config.RUNS / "registry.md"
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("| run | config | features | F1 | full set | F1 >= 10 sent. | F1 > 1024 tok | negatives lit "
                        "| description | decision |\n|---|---|---|---:|---:|---:|---:|---:|---|---|\n")
    fmt = lambda v: "" if v is None else f"{v:.3f}"
    line = (f"| {run_dir.name} | {cfg['id']} | {Path(train_dir).name} | {fmt(metrics_select['f1'])} | "
            f"{fmt(metrics_select['full_set'])} | {fmt(metrics_select['f1_far_10plus'])} | "
            f"{fmt(metrics_select['f1_beyond_1024_tok'])} | {fmt(metrics_neg['negatives_lit'] if metrics_neg else None)} "
            f"| {cfg.get('description', '')} | |\n")
    with open(path, "a") as handle:
        handle.write(line)


def main(argv=None):
    """Command line entry point (see the file header)."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--train", type=Path, required=True, help="feature folder of the training mix")
    p.add_argument("--dev", type=Path, help="feature folder of dev_evid_select")
    p.add_argument("--confirm", type=Path, help="feature folder of dev_evid_confirm")
    p.add_argument("--dev-neg", type=Path, help="feature folder of dev_evid_neg")
    p.add_argument("--tag", default="", help="suffix of the run folder (e.g. seed14)")
    p.add_argument("--seed", type=int, help="overrides train.seed")
    p.add_argument("--overfit", type=int, default=0, help="train and evaluate on the first N supervised rows")
    p.add_argument("--limit", type=int, default=0, help="first N training rows (quick runs)")
    p.add_argument("--eval-every", type=int, default=1, help="evaluate train and dev loss / metrics every N epochs")
    p.add_argument("--log-steps", type=int, default=20, help="write the mean step loss every N steps (steps.jsonl)")
    p.add_argument("--train-eval", type=int, default=2000, help="training rows sampled for the train loss / F1")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    opts = p.parse_args(argv)

    cfg = json.loads(opts.config.read_text())
    if opts.seed is not None:
        cfg["train"]["seed"] = opts.seed
    train_store = FeatureStore(opts.train)
    terms = set(cfg["train"].get("terms", ["supervised", "mil_neg"]))

    if opts.overfit:
        rows = load_rows(train_store, cfg, terms={"supervised"}, limit=opts.overfit)
        cfg["train"].update(epochs=max(cfg["train"]["epochs"], 30), patience=1000, batch=min(16, len(rows)))
        head, best, _ = train(cfg, rows, rows, opts.device, eval_every=opts.eval_every)
        print(f"overfit on {len(rows)} rows: best train score (mean of F1 and full set) {best:.3f} (want ~1)")
        return

    if not opts.dev:
        raise SystemExit("--dev is required (feature folder of dev_evid_select)")
    for folder in (opts.dev, opts.confirm, opts.dev_neg):
        if folder and not same_features(train_store.meta, FeatureStore(folder).meta):
            raise SystemExit(f"{folder} was extracted with different settings than {opts.train}")
    train_rows = load_rows(train_store, cfg, terms=terms, limit=opts.limit)
    dev_rows = load_rows(FeatureStore(opts.dev), cfg)
    print(f"train rows {len(train_rows)} (terms {sorted(terms)}), dev rows {len(dev_rows)}")

    run_dir = config.RUNS / config.MODEL_NAME / (cfg["id"] + (f"_{opts.tag}" if opts.tag else ""))
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(run_dir, every=opts.log_steps)
    # fixed random sample of the training rows for the train loss / F1 (the whole mix would be slow to score)
    sample = random.Random(cfg["train"].get("seed", config.SEED)).sample(
        train_rows, min(opts.train_eval, len(train_rows)))
    print(f"curves -> {run_dir}/log.jsonl (every {opts.eval_every} epochs), {run_dir}/steps.jsonl "
          f"(every {opts.log_steps} steps); train metrics on {len(sample)} rows")

    head, best, _ = train(cfg, train_rows, dev_rows, opts.device, logger, opts.eval_every, sample)
    logger.close()

    # selection rule on dev_evid_select, then metrics on every dev set with that rule
    rule, m_select = choose_rule(dev_rows, scores_of(head, dev_rows, opts.device), cfg["loss"]["name"])
    metrics = {"dev_evid_select": m_select}
    m_neg = None
    for name, folder in (("dev_evid_confirm", opts.confirm), ("dev_evid_neg", opts.dev_neg)):
        if folder:
            rows = load_rows(FeatureStore(folder), cfg)
            metrics[name] = summarize(predictions(rows, scores_of(head, rows, opts.device), rule))
            if name == "dev_evid_neg":
                m_neg = metrics[name]

    from safetensors.torch import save_file
    save_file({k: v.contiguous().cpu() for k, v in head.state_dict().items()}, str(run_dir / "head.safetensors"))
    io.write_json(run_dir / "config.json", dict(cfg, features={"train": str(opts.train), "dev": str(opts.dev)},
                                                feature_meta=train_store.meta, d_model=train_rows[0]["q"].shape[1]))
    io.write_json(run_dir / "selection.json", rule)
    io.write_json(run_dir / "dev_metrics.json", metrics)
    append_registry(run_dir, cfg, m_select, m_neg, opts.train)
    print(f"run -> {run_dir}\nrule {rule}\n"
          + "\n".join(f"{k}: f1 {v['f1']} full_set {v['full_set']} negatives_lit {v['negatives_lit']}"
                      for k, v in metrics.items()))


if __name__ == "__main__":
    main()
