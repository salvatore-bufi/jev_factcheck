"""OneJev1.6-9B (OmniJev/OneJev1.6-9B) on Flide, the serving engine its model card names.

A full fine-tune of Qwen3.5-9B. The checkpoint declares its format in decision_config.json ({"format_id":
"onejev-slot-v2", "package": "flide"}): it is not run by the qev engine of models/elephant-1. Flide's OneJev family
renders the prompt (a system turn and the state block as a shared prefix; each question as a suffix with "Question:",
"Options:" A. yes / B. no with their descriptions, "Answer with one letter: A, B."), forks the prefix into every
question in one forward pass, and reads the answer slots from the output head at the last token of each question. The
checkpoint has no readout.json and no attention override, so attention is causal and the readout is the LM-head slots
(`slot_lm`); it has no calibration.json, so the temperature is 1. Debiasing is Flide's default (debias=1: one order).

The adapter starts `flide serve` from this folder's own environment (.venv, built by setup.sh) on the benchmark GPU,
sends one /v1/systemone request per row with all its questions and `?debug=1`, and stores the raw slot logits from the
response's diagnostics. Latency therefore includes the local HTTP round trip.
"""
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from core.adapter import BaseAdapter, make_answer, option_keys
from core.datasets import hub_cache

HERE = Path(__file__).resolve().parent
REPO = "OmniJev/OneJev1.6-9B"
FLIDE_COMMIT = "08b87d033ef9bf76fc01b0a930517b67127b8e85"   # pinned in setup.sh


def snapshot(revision):
    """The model folder in the Hugging Face cache (the pinned revision, or refs/main when revision is empty)."""
    root = hub_cache() / f"models--{REPO.replace('/', '--')}"
    if not revision and (root / "refs/main").exists():
        revision = (root / "refs/main").read_text().strip()
    path = root / "snapshots" / (revision or "-")
    if not (path / "decision_config.json").exists() or not (path / "model.safetensors").exists():
        raise SystemExit(f"{REPO} (revision {revision or 'main'}) is not in the Hugging Face cache {hub_cache()}; "
                         f"run `hf download {REPO}`")
    return path, revision


def serve_command():
    """flide of this environment, else of this folder's own .venv (setup.sh)."""
    for path in (Path(sys.executable).parent / "flide", HERE / ".venv/bin/flide"):
        if path.exists():
            return path
    return None


class Adapter(BaseAdapter):
    NAME = "OneJev1.6-9B"
    SIZE = "9B"
    DEFAULTS = {
        "revision": "259e5470b0abfd795a0bbb4598d6e319fa224a38",
        "port": 8094,
        "dtype": "bfloat16",
        "max_branch_tokens": 32768,    # state plus one question; longer requests are refused (logged), never cut
        "max_request_tokens": 65536,   # state plus all questions
        "quantize": "",                # "fp8": Flide's FP8 at load (Ada, Hopper, Blackwell)
        "workers": 8,                  # concurrent requests
        "timeout": 900.0,              # seconds per request
        "startup_timeout": 1800.0,
        "url": "",                     # use an already running Flide server instead of starting one
        "batch": 64,
    }

    def load(self):
        import requests

        self.requests = requests
        self.local = threading.local()
        self.process = None
        self.folder, self.revision = snapshot(self.opt["revision"])
        self.format = json.loads((self.folder / "decision_config.json").read_text())
        if self.opt["url"]:
            self.url = self.opt["url"].rstrip("/")
        else:
            serve = serve_command()
            if serve is None:
                raise SystemExit(f"flide is not installed: run {HERE / 'setup.sh'} once")
            self.url = f"http://127.0.0.1:{self.opt['port']}"
            gpu = (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0]
            cmd = [str(serve), "serve", "--model", str(self.folder), "--name", self.NAME, "--host", "127.0.0.1",
                   "--port", str(self.opt["port"]), "--dtype", self.opt["dtype"],
                   "--max-branch-tokens", str(self.opt["max_branch_tokens"]),
                   "--max-request-tokens", str(self.opt["max_request_tokens"])]
            if self.opt["quantize"]:
                cmd += ["--quantize", self.opt["quantize"]]
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "HF_HUB_OFFLINE": "1"}
            self.log = open(HERE / "server.log", "w")
            self.process = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True,
                                            env=env)
        deadline = time.time() + self.opt["startup_timeout"]
        while True:
            try:
                r = requests.get(self.url + "/health", timeout=3)
                if r.status_code == 200:
                    self.health = r.json()
                    break
            except requests.RequestException:
                pass
            if self.process is not None and self.process.poll() is not None:
                raise SystemExit(f"The Flide server exited during start-up; see {HERE / 'server.log'}")
            if time.time() > deadline:
                raise SystemExit(f"The Flide server did not become ready at {self.url}")
            time.sleep(5)
        if "onejev" not in str(self.health.get("format", "")).lower():
            raise SystemExit(f"Server at {self.url} serves format {self.health.get('format')!r}, expected OneJev")
        self.pool = ThreadPoolExecutor(self.opt["workers"])

    def describe(self):
        return {"repo": REPO, "revision": self.revision, "runtime": f"flide @ {FLIDE_COMMIT[:7]}", "url": self.url,
                "decision_config": self.format, "server": self.health}

    def close(self):
        if getattr(self, "pool", None):
            self.pool.shutdown(wait=False)
        if getattr(self, "process", None) and self.process.poll() is None:
            self.process.send_signal(signal.SIGTERM)
            try:
                self.process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)

    def answer(self, q, a, diag):
        """One answer as the benchmark stores it. Slot logits come in the rendered option order: yes, no for a yes/no
        question, the criteria order for a choice (debias=1 renders one order only)."""
        keys = option_keys(q)
        probs = {"true": a["noul"], "false": 1 - a["noul"]} if q["type"] == "noul" else a["probabilities"]
        logits, t = None, None
        if diag and len(diag.get("slot_logits", [])) == 1:
            slots, t = diag["slot_logits"][0], diag.get("temperature", 1.0)
            logits = [slots[1], slots[0]] if q["type"] == "noul" else list(slots[:len(keys)])
        return make_answer(q["type"], keys, probs=probs, logits=logits, temperature=t)

    def ask(self, state, questions):
        session = getattr(self.local, "session", None)
        if session is None:
            session = self.local.session = self.requests.Session()
        body = {"model": "jev-latest", "state": state, "questions": questions}
        try:
            r = session.post(self.url + "/v1/systemone?debug=1", json=body, timeout=self.opt["timeout"])
        except self.requests.RequestException as exc:
            return f"request failed: {exc}"
        if r.status_code != 200:
            return f"HTTP {r.status_code}: {r.text[:200]}"
        out = r.json()
        meta = next((v for k, v in out.items() if k not in ("model", "answers", "usage") and isinstance(v, dict)), {})
        diag = meta.get("diagnostics", {})
        try:
            answers = {qid: self.answer(q, out["answers"][qid], diag.get(qid)) for qid, q in questions.items()}
        except (KeyError, TypeError, ValueError) as exc:
            return f"unexpected response ({exc}): {json.dumps(out)[:200]}"
        return answers, {"prompt_tokens": out.get("usage", {}).get("input_tokens")}

    def score(self, states, questions):
        if len(states) == 1:
            return [self.ask(states[0], questions)]
        return list(self.pool.map(lambda s: self.ask(s, questions), states))
