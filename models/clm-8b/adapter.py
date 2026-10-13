"""CLM-v0.1-8B (Contrastive-LM/CLM-v0.1-8B): Qwen3-8B embeddings (vLLM pooling) + the CLM projection heads.

CLM does not generate: the state text (the state fields as prose, a blank line, then the question)
and each option text (`false: ...` / `true: ...` for a noul, the option's own text for a choice or
score) are embedded by Qwen3-8B (last-token pooling), projected by the state head and the action
head, and the answer is softmax(scale * cosine). Text layout and heads come from the contrastive-lm
package itself (clm.schema, clm.heads). Option texts do not depend on the row, so they are embedded
once per question. Texts are not truncated to the reference server's default 2,048 tokens.
"""
import glob
import json
from pathlib import Path

from core.adapter import BaseAdapter, make_answer
from core.datasets import hub_cache

ENCODER = "Qwen/Qwen3-8B"


def find_heads(path=None):
    path = path or (sorted(glob.glob(str(hub_cache() / "models--Contrastive-LM--CLM-v0.1-8B/snapshots/*/CLM_v0.1-8B.pt"))) or [None])[0]
    if not path or not Path(path).exists():
        raise SystemExit(f"CLM_v0.1-8B.pt not found under {hub_cache()}; run `hf download Contrastive-LM/CLM-v0.1-8B` "
                         "(or pass --opt clm-8b:heads=PATH)")
    return path


def check_encoder():
    """The run is offline, so Qwen/Qwen3-8B must already be complete in the cache: fail with a clear message."""
    for snap in sorted(glob.glob(str(hub_cache() / "models--Qwen--Qwen3-8B/snapshots/*"))):
        index = Path(snap) / "model.safetensors.index.json"
        if index.exists():
            shards = set(json.loads(index.read_text())["weight_map"].values())
            missing = sorted(f for f in shards if not (Path(snap) / f).exists())
            if not missing and (Path(snap) / "config.json").exists() and (Path(snap) / "tokenizer.json").exists():
                return
            raise SystemExit(f"{snap} is incomplete (missing {missing or 'config/tokenizer'}); run `hf download Qwen/Qwen3-8B`.")
    raise SystemExit(f"The encoder Qwen/Qwen3-8B (16 GB) is not in the Hugging Face cache {hub_cache()}; CLM needs it in "
                     "addition to its heads. Run `hf download Qwen/Qwen3-8B` or copy models--Qwen--Qwen3-8B into that cache.")


class Adapter(BaseAdapter):
    NAME = "CLM-v0.1-8B"
    SIZE = "8B"
    DEFAULTS = {
        "encoder_revision": "b968826d9c46dd6066d109eabc6255188de91218",   # the encoder revision CLM was trained on
        "heads": "",                 # path to CLM_v0.1-8B.pt (default: the Hugging Face cache)
        "temperature": 1.0,          # the CLM client's default
        "max_model_len": 32768,
        "cpu_offload_gb": 0.0,
        "gpu_util": 0.92,
        "max_num_seqs": 32,
        "max_batched_tokens": 8192,
        "batch": 128,
    }

    def load(self):
        import numpy as np
        from clm.heads import HeadPair
        from clm.schema import candidates, state_text
        from vllm import LLM

        check_encoder()
        self.np, self.state_text, self.candidates = np, state_text, candidates
        self.heads_path = find_heads(self.opt["heads"] or None)
        # vLLM's engine must start BEFORE any CPU torch work here (the heads): a fork after OpenMP has run
        # makes the engine hang or segfault in libgomp.
        rev = self.opt["encoder_revision"]
        self.llm = LLM(model=ENCODER, revision=rev, tokenizer_revision=rev, runner="pooling", dtype="bfloat16",
                       max_model_len=self.opt["max_model_len"], gpu_memory_utilization=self.opt["gpu_util"],
                       cpu_offload_gb=self.opt["cpu_offload_gb"], enable_prefix_caching=True,
                       max_num_seqs=self.opt["max_num_seqs"], max_num_batched_tokens=self.opt["max_batched_tokens"], seed=0)
        self.heads = HeadPair("clm", self.heads_path, device="cpu").ensure()   # tiny MLPs: keep the GPU for the encoder
        self.scale = self.heads.scale
        self.tok = self.llm.get_tokenizer()
        self.option_cache = {}

    def describe(self):
        import hashlib
        return {"encoder": ENCODER, "encoder_revision": self.opt["encoder_revision"], "heads": self.heads_path,
                "heads_sha256": hashlib.sha256(Path(self.heads_path).read_bytes()).hexdigest(),
                "scale": self.scale, "temperature": self.opt["temperature"]}

    def embed(self, texts):
        """[n, 4096] L2-normalised float32 embeddings, as the reference Embedder returns them."""
        np = self.np
        out = self.llm.embed(texts, use_tqdm=False)
        vecs = np.asarray([o.outputs.embedding for o in out], dtype=np.float32)
        return vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)

    def options(self, qid, question):
        key = (qid, json.dumps(question, sort_keys=True))
        if key not in self.option_cache:
            keys, texts = self.candidates(question)
            self.option_cache[key] = (keys, self.heads.project_actions(self.embed(texts)).cpu().numpy())
        return self.option_cache[key]

    def reset_cache(self):
        try:
            self.llm.reset_prefix_cache()
        except Exception:
            pass

    def score(self, states, questions):
        texts, owners, results = [], [], [None] * len(states)
        for i, state in enumerate(states):
            per_q = {qid: self.state_text(state, q.get("instructions")) for qid, q in questions.items()}
            lengths = {qid: len(self.tok.encode(t)) for qid, t in per_q.items()}
            if max(lengths.values()) > self.opt["max_model_len"]:
                results[i] = f"text of {max(lengths.values())} tokens exceeds max_model_len {self.opt['max_model_len']}"
                continue
            for qid, t in per_q.items():
                texts.append(t)
                owners.append((i, qid, lengths))
        if not texts:
            return results
        zs = self.heads.project_states(self.embed(texts)).cpu().numpy()
        rows = {}
        for (i, qid, lengths), z in zip(owners, zs):
            keys, opts = self.options(qid, questions[qid])
            logits = (self.scale * (opts @ z)).tolist()
            rows.setdefault(i, ({}, lengths))[0][qid] = make_answer(
                questions[qid]["type"], keys, logits=logits, temperature=self.opt["temperature"])
        for i, (answers, lengths) in rows.items():
            results[i] = (answers, {"prompt_tokens": lengths})
        return results
