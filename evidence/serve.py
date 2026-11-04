"""STEP 11 — The server: for each request ONE forward that returns the verdict and the evidence sentences (doc 12).

Where it sits in the workflow:
    release folder (evidence.release) or run folder (evidence.train_head)  ->  [evidence.serve]  ->  HTTP clients

Per request {document, claim}:
    1. split the document into sentences (evidence/sentences.py: the same splitter as in training)
    2. build the prompt with the preamble and find the tokens of each sentence (evidence/render.py)
    3. ONE forward with the hooks: P(supported) + the sentence features (evidence/model.py)
    4. if P(supported) <= threshold: answer without evidence
    5. head -> s[k], s0 -> sentences above the threshold (1 to 5) -> their character positions in the ORIGINAL document
It is the same code as training and evaluation: what was measured is what runs.

Answer:
    {"p_supported": 0.93, "supported": true,
     "evidence": [{"sentence": 0, "start": 0, "end": 61, "score": 0.91, "text": "..."}, ...]}
    start / end are character positions in the document as the client sent it.

Run:
    python -m evidence.serve --run releases/knowline-4b-fc-ev-v1 --port 8090
    curl -s localhost:8090/v1/evidence -d '{"document": "...", "claim": "..."}'
    python -m evidence.serve --run <folder> --demo           # one example request, no server
"""
import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evidence import config  # noqa: E402


class Service:
    """Model + head + settings, loaded once; answer() serves one request."""

    def __init__(self, folder, device="cuda"):
        """Input: folder (run folder or release folder), device.
        Loads the decision model, the head, the selection rule and the preamble the head was trained with.
        For a release folder, the hashes in frozen.json are checked first (same splitter, same code)."""
        from evidence.head import load_run
        from evidence.model import load_model
        folder = Path(folder)
        if (folder / "frozen.json").is_file():
            from evidence.release import verify
            verify(folder)
        self.head, self.cfg, self.rule = load_run(folder, device)
        self.preamble = self.cfg["feature_meta"]["preamble"]
        self.layer_pos = [config.SENTENCE_LAYERS.index(layer) for layer in self.cfg["layers_read"]]
        self.model = load_model(device)
        self.lock = threading.Lock()                 # one forward at a time on the GPU

    @torch.no_grad()
    def answer(self, document, claim, spans=None, timings=None):
        """Verdict and evidence for one (document, claim).

        Input:  document, claim (str); spans (optional, the sentence spans if already computed);
                timings (optional list: the time of the head + selection is appended, for evidence.latency).
        Output: dict p_supported, supported, evidence (list of {sentence, start, end, score, text}), or
                {"error": "document_too_long" | "empty_document", ...}.
        """
        import time
        from evidence.metrics import select
        from evidence.model import forward_sentences, p_supported
        from evidence.render import render
        from evidence.sentences import split
        if not document.strip():
            return {"error": "empty_document"}
        spans = spans if spans is not None else split(document)
        ex = render(document, claim, spans, self.preamble)
        if len(ex["input_ids"]) > config.MAX_TOKENS_SERVE:
            return {"error": "document_too_long", "prompt_tokens": len(ex["input_ids"]),
                    "max_tokens": config.MAX_TOKENS_SERVE}
        with self.lock:
            feats = forward_sentences(self.model, ex, to_cpu=False)
        p = p_supported(feats["answer_logits"])
        if p <= config.VERDICT_THRESHOLD:
            return {"p_supported": p, "supported": False, "evidence": []}
        t0 = time.perf_counter()
        idx = self.layer_pos
        s, s0, _ = self.head(feats["q"][None, idx], feats["h_end"][None, :, idx],
                             feats["h_mean"][None, :, idx] if "mean" in self.cfg["parts"] else None,
                             feats["att"][None], feats["n_tok"][None].float(),
                             torch.ones(1, len(spans), device=feats["q"].device))
        s, s0 = s[0].float().cpu(), float(s0[0])
        from evidence.metrics import logit
        thr = s0 + self.rule["delta"] if self.rule["kind"] == "doc" else logit(self.rule["tau"])
        shown = select(s.tolist(), thr)
        if timings is not None:
            timings.append(time.perf_counter() - t0)
        return {"p_supported": p, "supported": True,
                "evidence": [{"sentence": k, "start": spans[k][0], "end": spans[k][1],
                              "score": round(float(torch.sigmoid(s[k] - thr)), 4),
                              "text": document[spans[k][0]:spans[k][1]]} for k in shown]}


def make_handler(service):
    """HTTP handler class bound to a Service: POST /v1/evidence, GET /health."""
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send(200, {"ok": True}) if self.path == "/health" else self._send(404, {"error": "not_found"})

        def do_POST(self):
            if self.path != "/v1/evidence":
                return self._send(404, {"error": "not_found"})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                out = service.answer(str(req["document"]), str(req["claim"]))
                self._send(400 if "error" in out else 200, out)
            except (KeyError, ValueError, TypeError) as err:
                self._send(400, {"error": "bad_request", "detail": str(err)})
    return Handler


def main(argv=None):
    """Command line entry point (see the file header)."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True, help="release folder or run folder")
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--demo", action="store_true", help="answer one example and exit")
    opts = p.parse_args(argv)
    service = Service(opts.run)
    if opts.demo:
        doc = ("Saint Paul School of Theology is a seminary located in Kansas City, Missouri. It was founded in 1958. "
               "The school offers Master of Divinity degrees. Many students come from the Midwest. "
               "Nancy R. Howell is Professor of Theology at Saint Paul School of Theology.")
        print(json.dumps(service.answer(doc, "Nancy R. Howell teaches at a seminary in Kansas City."), indent=2))
        return
    print(f"serving on http://127.0.0.1:{opts.port}/v1/evidence")
    ThreadingHTTPServer(("127.0.0.1", opts.port), make_handler(service)).serve_forever()


if __name__ == "__main__":
    main()
