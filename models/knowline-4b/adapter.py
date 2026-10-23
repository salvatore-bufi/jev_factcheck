"""KnowLine-4B-Gen5.5 (PelaAI/KnowLine-4B-Gen5.5) through the model repository's own front end, knowline_server.py.

A Qwen3.5-4B decision model. The front end (knowline_server.py with knowline_layout.py, shipped in the model repo and
imported from the snapshot) renders one prompt per question: the state as one user turn (indented text, not JSON), then
a user turn with the instruction, "Question: ...", "Options: A. Yes: ... B. No: ...", and the assistant turn opening with
"Answer:". Gen5.5 reads twice: a text-only request of at most twice_max_tokens tokens (read once) is written a second
time after "Read the same input again before answering:". Each question is one prefill; the answer is a softmax over
the label tokens. The readout temperature (T = 1.50) is folded into the weights, so the front end uses temperature 1.

Backends, as in the repo's knowline_engine.py:
  hf      transformers in process, bf16 (no server; the authors report it matches their runs on 81 of 82 answers);
  sglang  an SGLang server with the flags of serve_knowline.sh (FP8 at load), started by the adapter: the setting of
          the authors' Decision Index runs. Needs sglang==0.5.21 in this environment (or sglang_python).

The adapter runs the steps of knowline_server.KnowLine.run with its own functions (render_request, warm, score) and keeps
the label log-probabilities, which run() turns into probabilities and drops, so answers can be re-tempered afterwards.
"""
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from core.adapter import BaseAdapter, make_answer, option_keys
from core.datasets import hub_cache

REPO = "PelaAI/KnowLine-4B-Gen5.5"
SGLANG_FLAGS = ["--served-model-name", "m", "--tp", "1", "--quantization", "fp8", "--mamba-radix-cache-strategy",
                "extra_buffer", "--enable-fp32-lm-head"]        # serve_knowline.sh / knowline_engine.py


def snapshot(revision):
    path = hub_cache() / f"models--{REPO.replace('/', '--')}" / "snapshots" / revision
    need = ["config.json", "model.safetensors.index.json", "tokenizer.json", "knowline_server.py", "knowline_layout.py"]
    if not all((path / f).exists() for f in need):
        raise SystemExit(f"{REPO} (revision {revision}) is not complete in the Hugging Face cache {hub_cache()}; "
                         f"run `hf download {REPO}`")
    return path


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Adapter(BaseAdapter):
    NAME = "KnowLine-4B-Gen5.5"
    SIZE = "4B"
    DEFAULTS = {
        "revision": "779944d2cc8ffa19cef386aee83c83bcb967ade2",
        "backend": "hf",               # hf (transformers, bf16) or sglang (FP8 at load: the authors' runs)
        "read_twice": True,            # Gen5.5's layout; False gives the Gen5 prompts
        "twice_max_tokens": 3000,      # longest request (tokens, read once) that is read twice
        "workers": 16,                 # threads scoring the questions of a row (the front end's default)
        "mem": 0.72,                   # sglang: --mem-fraction-static (sizes the KV cache only)
        "sglang_python": "",           # sglang: an interpreter with SGLang, if not this one
        "startup_timeout": 1800.0,
        "batch": 32,
    }

    def load(self):
        from concurrent.futures import ThreadPoolExecutor
        import torch
        from transformers import AutoTokenizer

        self.torch, self.server = torch, None
        self.folder = snapshot(self.opt["revision"])
        sys.path.insert(0, str(self.folder))
        import knowline_server as ks
        from knowline_layout import Layout, render_request

        self.ks, self.render_request = ks, render_request
        tok = AutoTokenizer.from_pretrained(str(self.folder))
        if self.opt["backend"] == "hf":
            if not torch.cuda.is_available():
                raise SystemExit(f"{self.NAME} needs a GPU")
            backend = ks.HF(str(self.folder), device="cuda:0")
        elif self.opt["backend"] == "sglang":
            backend = ks.SGLang(self.start_sglang())
        else:
            raise SystemExit(f"{self.NAME}: backend must be hf or sglang, not {self.opt['backend']!r}")
        self.layout = Layout(read_twice=True, twice_max_tokens=self.opt["twice_max_tokens"]) if self.opt["read_twice"] else None
        self.engine = ks.KnowLine(tok, backend, 1.0, self.opt["workers"], {}, self.layout)
        self.pool = ThreadPoolExecutor(self.opt["workers"])

    def start_sglang(self):
        import requests

        port = free_port()
        cmd = [self.opt["sglang_python"] or sys.executable, "-m", "sglang.launch_server", "--model-path", str(self.folder),
               *SGLANG_FLAGS, "--mem-fraction-static", str(self.opt["mem"]), "--port", str(port)]
        self.log = open(Path(__file__).resolve().parent / "sglang.log", "w")
        self.server = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
        url, deadline = f"http://127.0.0.1:{port}", time.time() + self.opt["startup_timeout"]
        while True:
            if self.server.poll() is not None:
                raise SystemExit(f"SGLang exited with code {self.server.returncode}; see {self.log.name}")
            try:
                if requests.get(f"{url}/health", timeout=3).status_code == 200:
                    return url
            except requests.RequestException:
                pass
            if time.time() > deadline:
                raise SystemExit(f"SGLang did not become healthy in {self.opt['startup_timeout']:.0f} s")
            time.sleep(5)

    def describe(self):
        import transformers

        return {"repo": REPO, "revision": self.opt["revision"], "front_end": "knowline_server.py + knowline_layout.py",
                "backend": self.opt["backend"], "dtype": "bfloat16" if self.opt["backend"] == "hf" else "fp8 (SGLang)",
                "temperature": 1.0, "folded_temperature": 1.5, "layout": None if self.layout is None else vars(self.layout),
                "labels": self.engine.labels[:2], "transformers": transformers.__version__,
                "sglang": " ".join(SGLANG_FLAGS + ["--mem-fraction-static", str(self.opt["mem"])])
                if self.opt["backend"] == "sglang" else None}

    def close(self):
        if getattr(self, "pool", None):
            self.pool.shutdown(wait=False)
        if getattr(self, "server", None) is not None and self.server.poll() is None:
            os.killpg(self.server.pid, signal.SIGTERM)
            try:
                self.server.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(self.server.pid, signal.SIGKILL)

    def run(self, state, questions):
        """knowline_server.KnowLine.run, returning the label log-probabilities instead of the answers."""
        e, ks = self.engine, self.ks
        if self.layout is None:
            prefix, prompts, media = ks.render(e.processor, state, questions, e.labels)
            twice = False
        else:
            r = self.render_request({"state": state, "questions": questions}, e.processor, self.layout, labels=e.labels)
            prefix, media, twice = r["prefix"], r["media"], r["twice"]
            prompts = {qid: r["prompts"][qid][0] for qid in questions}
        items = list(prompts.items())
        if len(items) > 1:
            e.backend.warm(prefix, media)
        results = list(self.pool.map(lambda it: e.backend.score(it[1][0], media, e.ids[:len(it[1][1])]), items))
        return {qid: (keys, row, n) for (qid, (_, keys)), (row, n) in zip(items, results)}, twice

    def score(self, states, questions):
        results = []
        for state in states:
            try:
                scored, twice = self.run(state, questions)
            except Exception as exc:             # e.g. out of memory on a very long document, or a backend error
                results.append(f"{type(exc).__name__}: {str(exc)[:200]}")
                self.torch.cuda.empty_cache()
                continue
            answers = {}
            for qid, q in questions.items():
                keys, row, _ = scored[qid]       # the front end's keys: noul "true", "false" (Yes is label A)
                by_key = dict(zip(keys, row))
                ans = make_answer(q["type"], option_keys(q), logits=[by_key[k] for k in option_keys(q)], temperature=1.0)
                ans["read_twice"] = twice
                answers[qid] = ans
            results.append((answers, {"prompt_tokens": {qid: scored[qid][2] for qid in questions}}))
        return results
