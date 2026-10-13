"""Winnow-12B NVFP4 (EldanRing/Winnow-12B, gguf/Winnow-12B-NVFP4.gguf) on the authors' llama.cpp-based server.

The GGUF runs only on Winnow's own server (`/v1/systemone`); the server renders the prompts and reads
the option logits itself. This adapter starts the server (serve.sh) on the benchmark GPU, sends one
request per row with all its questions (the server reads the state once and answers the questions as
parallel branches), asks for diagnostics so the raw option logits are saved too, and stops the server
at the end. Latency therefore includes the local HTTP round trip. Build the server once with setup.sh.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from core.adapter import BaseAdapter, make_answer, option_keys

HERE = Path(__file__).resolve().parent
SERVED_NAME = "Winnow-12B"


class Adapter(BaseAdapter):
    NAME = "Winnow-12B-NVFP4"
    SIZE = "12B"
    DEFAULTS = {
        "port": 8091,
        "context": "32k",            # holds the longest benchmark documents (80,000 characters is about 20k tokens)
        "workers": 4,                # concurrent requests (the server runs 4 native branches)
        "timeout": 300.0,            # seconds per request
        "startup_timeout": 900.0,
        "url": "",                   # use an already running server instead of starting one
        "batch": 64,
    }

    def load(self):
        import requests

        self.requests = requests
        self.process = None
        self.cold = False
        self.local = threading.local()
        if self.opt["url"]:
            self.url = self.opt["url"].rstrip("/")
        else:
            if not (HERE / "src/.build/bin/winnow-server").exists():
                raise SystemExit("The Winnow server is not built: run models/winnow-12b/setup.sh once")
            self.url = f"http://127.0.0.1:{self.opt['port']}"
            gpu = (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0]
            env = dict(os.environ, GPU=gpu, PORT=str(self.opt["port"]), CONTEXT=str(self.opt["context"]), PYTHON=sys.executable)
            self.log = open(HERE / "server.log", "w")
            self.process = subprocess.Popen(["bash", str(HERE / "serve.sh")], env=env, stdout=self.log,
                                            stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.time() + self.opt["startup_timeout"]
        while True:
            try:
                if requests.get(self.url + "/health", timeout=3).status_code == 200:
                    break
            except requests.RequestException:
                pass
            if self.process is not None and self.process.poll() is not None:
                raise SystemExit(f"The Winnow server exited during start-up; see {HERE / 'server.log'}")
            if time.time() > deadline:
                raise SystemExit(f"The Winnow server did not become ready at {self.url}")
            time.sleep(2)
        names = [m.get("name") for m in requests.get(self.url + "/v1/models", timeout=10).json().get("models", [])]
        if SERVED_NAME not in names:
            raise SystemExit(f"Server at {self.url} serves {names}, expected {SERVED_NAME!r}")
        self.pool = ThreadPoolExecutor(self.opt["workers"])

    def describe(self):
        return {"repo": "EldanRing/Winnow-12B", "file": "gguf/Winnow-12B-NVFP4.gguf", "runtime": "winnow-inference (llama.cpp)",
                "url": self.url, "context": self.opt["context"], "temperature": 1.0}

    def reset_cache(self):
        self.cold = True   # the next request does not reuse the server's cached prefix

    def close(self):
        if getattr(self, "pool", None):
            self.pool.shutdown(wait=False)
        if getattr(self, "process", None) and self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)

    def ask(self, state, questions, reuse):
        session = getattr(self.local, "session", None)
        if session is None:
            session = self.local.session = self.requests.Session()
        body = {"state": state, "questions": questions, "winnow": {"diagnostics": True, "reuse_prefix": reuse}}
        try:
            r = session.post(self.url + "/v1/systemone", json=body, timeout=self.opt["timeout"])
        except self.requests.RequestException as exc:
            return f"request failed: {exc}"
        if r.status_code != 200:
            return f"HTTP {r.status_code}: {r.text[:200]}"
        out = r.json()
        answers = {}
        try:
            for qid, q in questions.items():
                a = out["answers"][qid]
                keys = option_keys(q)
                probs = {"true": a["noul"], "false": 1 - a["noul"]} if q["type"] == "noul" else a["probabilities"]
                logits = (a.get("winnow") or {}).get("logits")
                answers[qid] = make_answer(q["type"], keys, probs=probs, logits=logits, temperature=1.0)
        except (KeyError, TypeError, ValueError) as exc:
            return f"unexpected response ({exc}): {json.dumps(out)[:200]}"
        return answers, {"prompt_tokens": out.get("usage", {}).get("input_tokens")}

    def score(self, states, questions):
        reuse = not self.cold
        self.cold = False
        if len(states) == 1:
            return [self.ask(states[0], questions, reuse)]
        return list(self.pool.map(lambda s: self.ask(s, questions, reuse), states))
