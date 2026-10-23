"""StartLux-Decision-v1.1-27B-AutoThink (startlux-models/StartLux-Decision-v1.1-27B-AutoThink) on the authors' server.

An AutoThink model, read causally: one read-out per question, and when its top probability is below the gate the
model writes a thought (sampled: temperature 0.6, top-p 0.95, top-k 20; up to think_budget tokens, with a loop
detector), reads the letters again and answers (1 - mix) * first + mix * after. The weights repository has no
inference code, and the gate, the thought and the mixture are the package's, so the benchmark runs the authors'
package, startlux-decide, from this folder's own environment (.venv, built by setup.sh). `startlux-decide serve
--launch-vllm` starts one vLLM server on the benchmark GPU and the /v1/systemone server in front of it. The package
renders the prompts ("Evidence: / Question: / Options: A) yes B) no", chat template with thinking off) and reads the
option-letter logits at the type's temperature (decision_config.json). The adapter sends one request per row with
all its questions and `return_details`, and stops the servers at the end. Latency therefore includes the local HTTP
round trip.

Logits: the server reports probabilities at the type's temperature T. An answer read in one pass stores T * log(p) of
its read-out (the letter logits up to a constant), so it can be re-tempered afterwards; an answer mixed with a
read-out after a thought has no logits. Every answer also stores its first read-out and its thought (used, tokens,
how it ended).
"""
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from core.adapter import BaseAdapter, make_answer, option_keys
from core.datasets import hub_cache

HERE = Path(__file__).resolve().parent
REPO = "startlux-models/StartLux-Decision-v1.1-27B-AutoThink"
SERVE = HERE / ".venv/bin/startlux-decide"
PACKAGE_COMMIT = "a99977d9332decc9fb27a2488b4a83b533b6cd5f"   # pinned in setup.sh


def snapshot(revision):
    """The model folder in the Hugging Face cache (the pinned revision, or refs/main when revision is empty)."""
    root = hub_cache() / f"models--{REPO.replace('/', '--')}"
    if not revision and (root / "refs/main").exists():
        revision = (root / "refs/main").read_text().strip()
    path = root / "snapshots" / (revision or "-")
    if not (path / "decision_config.json").exists() or not (path / "model.safetensors.index.json").exists():
        raise SystemExit(f"{REPO} (revision {revision or 'main'}) is not in the Hugging Face cache {hub_cache()}; "
                         f"run `hf download {REPO}`")
    return path, revision


class Adapter(BaseAdapter):
    NAME = "StartLux-27B-AutoThink"
    SIZE = "27B"
    DEFAULTS = {
        "revision": "dd62d5cd84a483d75bdd548880bbf08bdd3f67a4",
        "port": 8092,                  # the decision server
        "vllm_port": 8101,             # the vLLM server behind it
        "max_model_len": 32768,        # holds the longest benchmark documents (80,000 characters is about 20k tokens)
        "gpu_util": 0.88,              # the package's default
        "quantization": "",            # "fp8": weights quantized at load (54 GB in bf16: use it on a 48 GB GPU)
        "workers": 8,                  # concurrent requests
        "timeout": 900.0,              # seconds per request
        "startup_timeout": 1800.0,
        "url": "",                     # use an already running decision server instead of starting one
        "batch": 64,
        "think_mode": "auto",          # auto: think when the first read-out's top probability is below gate; never; always
        "gate": 0.6,                   # the package default (the card's published numbers use 0.9)
        "mix": 0.75,                   # final = (1 - mix) * first read-out + mix * read-out after the thought
        "think_budget": 4096,          # most thought tokens
    }
    THINKS = True
    CONTROLS = ("think_mode", "gate", "mix", "think_budget")   # request controls taken from the options

    def load(self):
        import requests

        self.requests = requests
        self.local = threading.local()
        self.process = None
        self.folder, self.revision = snapshot(self.opt["revision"])
        self.temperature = json.loads((self.folder / "decision_config.json").read_text())["temperature_by_type"]
        if self.opt["url"]:
            self.url = self.opt["url"].rstrip("/")
        else:
            if not SERVE.exists():
                raise SystemExit(f"The StartLux environment is not set up: run {HERE / 'setup.sh'} once")
            self.url = f"http://127.0.0.1:{self.opt['port']}"
            gpu = (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0]
            cmd = [str(SERVE), "serve", "--model", str(self.folder), "--name", self.NAME,
                   "--port", str(self.opt["port"]), "--launch-vllm", "--gpus", gpu,
                   "--vllm-base-port", str(self.opt["vllm_port"]), "--vllm-log-dir", str(HERE / "vllm_logs"),
                   "--max-model-len", str(self.opt["max_model_len"]),
                   "--gpu-memory-utilization", str(self.opt["gpu_util"]), "--no-images",
                   "--startup-timeout", str(self.opt["startup_timeout"])]
            if self.opt["quantization"]:
                cmd += ["--quantization", self.opt["quantization"]]
            self.log = open(HERE / "server.log", "w")
            self.process = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.time() + self.opt["startup_timeout"] + 60
        while True:
            try:
                r = requests.get(self.url + "/health", timeout=3)
                if r.status_code == 200:
                    self.health = r.json()
                    break
            except requests.RequestException:
                pass
            if self.process is not None and self.process.poll() is not None:
                raise SystemExit(f"The StartLux server exited during start-up; see {HERE / 'server.log'} and "
                                 f"{HERE / 'vllm_logs'}")
            if time.time() > deadline:
                raise SystemExit(f"The StartLux server did not become ready at {self.url}")
            time.sleep(5)
        kind = "auto_think" if self.THINKS else "instant"
        if self.health.get("kind") != kind:
            raise SystemExit(f"Server at {self.url} serves a {self.health.get('kind')!r} model, expected {kind!r}")
        self.controls = {k: self.opt[k] for k in self.CONTROLS}
        self.pool = ThreadPoolExecutor(self.opt["workers"])

    def describe(self):
        return {"repo": REPO, "revision": self.revision, "runtime": f"startlux-decide @ {PACKAGE_COMMIT[:7]}, vLLM",
                "url": self.url, "temperature_by_type": self.temperature, "controls": self.controls,
                "server": {k: self.health.get(k) for k in ("model", "kind", "prompt_attention", "engine", "defaults",
                                                           "context_length")}}

    def close(self):
        if getattr(self, "pool", None):
            self.pool.shutdown(wait=False)
        if getattr(self, "process", None) and self.process.poll() is None:
            self.process.send_signal(signal.SIGTERM)     # the server stops its vLLM server on the way out
            try:
                self.process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)

    def answer(self, q, a):
        """One /v1/systemone answer (with details) as the benchmark stores it."""
        keys = option_keys(q)
        probs = {"true": a["noul"], "false": 1 - a["noul"]} if q["type"] == "noul" else a["probabilities"]
        details = a["details"]
        first = details["first_pass"]["probabilities"]
        thought = details.get("thought") or {}
        if thought.get("used"):
            out = make_answer(q["type"], keys, probs=probs)
        else:
            t = self.temperature.get(q["type"], 1.0)
            out = make_answer(q["type"], keys, probs=probs, temperature=t,
                              logits=[t * math.log(first[k]) if first[k] > 0 else float("-inf") for k in keys])
        out["first_pass"] = {k: first[k] for k in keys}
        out["thought"] = {"used": bool(thought.get("used")), "tokens": thought.get("tokens", 0),
                          "ended": thought.get("ended")}
        return out

    def ask(self, state, questions):
        session = getattr(self.local, "session", None)
        if session is None:
            session = self.local.session = self.requests.Session()
        body = {"state": state, "questions": questions, "return_details": True, **self.controls}
        try:
            r = session.post(self.url + "/v1/systemone", json=body, timeout=self.opt["timeout"])
        except self.requests.RequestException as exc:
            return f"request failed: {exc}"
        if r.status_code != 200:
            return f"HTTP {r.status_code}: {r.text[:200]}"
        out = r.json()
        try:
            answers = {qid: self.answer(q, out["answers"][qid]) for qid, q in questions.items()}
        except (KeyError, TypeError, ValueError) as exc:
            return f"unexpected response ({exc}): {json.dumps(out)[:200]}"
        return answers, {"prompt_tokens": out.get("usage", {}).get("input_tokens")}

    def score(self, states, questions):
        if len(states) == 1:
            return [self.ask(states[0], questions)]
        return list(self.pool.map(lambda s: self.ask(s, questions), states))
