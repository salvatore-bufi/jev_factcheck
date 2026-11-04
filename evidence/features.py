"""Read the saved features (written by evidence.extract) as one dict per row, for training and evaluation.

Where it sits in the workflow:
    evidence.extract  ->  data/evidence/features/<model>/<set>/  ->  [evidence.features]  ->  train_head / eval / baselines

The shards are opened lazily (safetensors memory-maps the files): only the rows asked for are read from disk, and
only the layers asked for. The full features (~100 GB for the whole mix) never need to fit in RAM.

Use:
    store = FeatureStore(config.FEATURES / "knowline-4b" / "ev_head")
    row = store.get(0, layers=[31], parts=("end",))     # dict: q [1, D], h_end [n, 1, D], h_mean None, att, ...
    len(store), store.index[0]["id"]
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evidence.data import io  # noqa: E402


class FeatureStore:
    """The features of one set (one folder written by evidence.extract)."""

    def __init__(self, folder, ids=None):
        """Input:  folder (Path of the set), ids (optional set of row ids to keep, e.g. only some rows).
        Builds: self.index (list of index entries), self.meta (meta.json), lazily opened shard handles."""
        self.folder = Path(folder)
        self.meta = json.loads((self.folder / "meta.json").read_text())
        self.index = io.read_jsonl(self.folder / "index.jsonl")
        if ids is not None:
            self.index = [e for e in self.index if e["id"] in ids]
        self._shards = {}

    def __len__(self):
        return len(self.index)

    def _shard(self, k):
        """The open safetensors file of shard k (opened once, then kept)."""
        from safetensors import safe_open
        if k not in self._shards:
            self._shards[k] = safe_open(str(self.folder / f"shard-{k:05d}.safetensors"), framework="pt")
        return self._shards[k]

    def get(self, i, layers, parts=("end", "mean")):
        """The features of row i.

        Input:  i       position in self.index
                layers  list of layer numbers to read (subset of meta["sentence_layers"], e.g. [31] or [19,23,27,31])
                parts   which sentence vectors: "end" (last token) and / or "mean" (token mean)
        Output: dict
                q       [L, D] float32           answer-position vector in those layers
                h_end   [n, L, D] bf16 or None
                h_mean  [n, L, D] bf16 or None
                att     [n, 128] fp16
                n_tok   [n] int16
                answer_logits [2] float32
                plus the index fields: gold [n] (tensor 0/1), evidence_sets, evidence_term, row_weight, omega, ...
        """
        e = self.index[i]
        f = self._shard(e["shard"])
        a, b = e["offset"], e["offset"] + e["n"]

        def sentences(name):
            return torch.stack([f.get_slice(f"{name}.{layer}")[a:b] for layer in layers], 1)

        out = dict(e)
        # layers=[] reads no hidden vector at all (e.g. the attention baseline)
        out["q"] = torch.stack([f.get_slice(f"q.{layer}")[e["row"]] for layer in layers]) if layers else None
        out["h_end"] = sentences("h_end") if "end" in parts and layers else None
        out["h_mean"] = sentences("h_mean") if "mean" in parts and layers else None
        out["att"] = f.get_slice("att")[a:b]
        out["n_tok"] = f.get_slice("n_tok")[a:b]
        out["answer_logits"] = f.get_slice("answer_logits")[e["row"]]
        out["gold"] = torch.tensor(e["gold"], dtype=torch.float)
        return out

    def load_all(self, layers, parts=("end", "mean")):
        """All rows in RAM (only for small sets such as dev, or few layers). Output: list of dicts from get()."""
        return [self.get(i, layers, parts) for i in range(len(self))]


def same_features(meta_a, meta_b):
    """True if two feature folders were made with the same model, preamble, layers and code (can be mixed)."""
    keys = ("model", "revision", "preamble", "sentence_layers", "attention_layers", "question", "sha256")
    return all(meta_a.get(k) == meta_b.get(k) for k in keys)
