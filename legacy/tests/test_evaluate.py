"""Offline checks of model swapping, caching, arithmetic and API failures."""
from contextlib import redirect_stdout
from copy import deepcopy
from decimal import Decimal
import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import requests
import evaluate as ev


def raises(kind, fn):
    try:
        fn()
    except kind:
        return
    raise AssertionError(f"Expected {kind.__name__}")


def arithmetic():
    pack = json.loads((ev.ROOT / "packs/claim_support.json").read_text())
    ev.validate_pack(pack)
    answers = {"support_simple": {"noul": .9}, "unsupported_detail": {"noul": .7},
               "contradicted_detail": {"noul": .1}}
    score = ev.extract(answers, pack)["min_support_checks"]
    assert score == Decimal("0.30") and not score > Decimal("0.30")
    for bad in [float("nan"), -1, 2, float("inf")]:
        invalid = deepcopy(answers)
        invalid["support_simple"]["noul"] = bad
        raises(ValueError, lambda: ev.extract(invalid, pack))
    binary = json.loads((ev.ROOT / "packs/binary_support.json").read_text())
    ev.validate_pack(binary)
    assert ev.extract({"support": {"probabilities": {"supported": .8}}}, binary)["support"] == Decimal(".8")
    bad = deepcopy(pack)
    bad["extractions"]["support_simple"]["answer_key"] = "absent"
    raises(KeyError, lambda: ev.validate_pack(bad))
    rubric = {"questions": {"q": {"type": "score", "instructions": "How supported?", "criteria": ["No", "Partial", "Full"]}},
              "extractions": {"normalized": {"answer_key": "q", "field": "score", "divisor": 2}}}
    ev.validate_pack(rubric)
    assert ev.extract({"q": {"score": 1.5}}, rubric)["normalized"] == Decimal(".75")
    row = {"doc": "abcdMIDDLEwxyz", "claim": "synthetic"}
    assert ev.state_for(row, 8)["document"] == "abcdMI\n[Middle of long document omitted]\nyz"
    assert ev.state_for(row, 0)["document"] == row["doc"]


def transport():
    body = {"model": "jev-1.13.0", "answers": {"q": {"type": "noul", "noul": .8}},
            "usage": {"input_tokens": 12, "output_tokens": 3}}
    response = lambda status: Mock(status_code=status, headers={"Retry-After": "0"}, json=Mock(return_value=body))
    with patch.object(ev, "load_key", return_value="synthetic-key"), patch.object(ev.time, "sleep"), \
            patch.object(ev.requests, "Session") as sessions:
        session = sessions.return_value
        session.request.side_effect = [response(429), response(529), response(200)]
        client = ev.Client(10)
        result = client.ask("jev-1.13.0", {"document": "Synthetic.", "claim": "Synthetic."}, {"q": {"type": "noul"}})
        assert result["attempts"] == 3
        request = session.request.call_args
        assert request.args == ("POST", ev.API + "/systemone")
        assert set(request.kwargs["json"]) == {"model", "state", "questions"}
        assert set(request.kwargs["json"]["state"]) == {"document", "claim"}
        assert request.kwargs["allow_redirects"] is False
        session.request.side_effect = [response(401)]
        raises(RuntimeError, lambda: client.request("GET", "/models"))
        session.request.side_effect = [requests.Timeout("do not expose synthetic secret")] * 5
        try:
            client.request("GET", "/models")
        except RuntimeError as error:
            assert "secret" not in str(error)
        else:
            raise AssertionError("Expected timeout")
    for mutation in [{"model": ""}, {"answers": {}}, {"usage": {"input_tokens": -1, "output_tokens": 0}}]:
        raises(RuntimeError, lambda: ev.validate_response({**body, **mutation}, {"q": {"type": "noul"}}))
    with patch.dict(os.environ, {"TYPESAFE_API_KEY": "must-not-leak", "CLM_API_KEY": "local-key"}, clear=True), \
            patch.object(ev, "load_key", side_effect=AssertionError("Custom endpoint read Jev key")), \
            patch.object(ev.requests, "Session") as sessions:
        sessions.return_value.request.return_value = response(200)
        client = ev.Client(10, "http://127.0.0.1:8700/v1/", timeout=600)
        client.request("GET", "/models")
        request = sessions.return_value.request.call_args
        assert request.args == ("GET", "http://127.0.0.1:8700/v1/models")
        assert request.kwargs["headers"] == {} and request.kwargs["timeout"] == (15, 600)
        ev.Client(10, "http://127.0.0.1:8700/v1", "CLM_API_KEY").request("GET", "/models")
        assert sessions.return_value.request.call_args.kwargs["headers"] == {"Authorization": "Bearer local-key"}
        raises(RuntimeError, lambda: ev.Client(10, "http://localhost:8700/v1", "MISSING_KEY"))
    for bad in ["file:///tmp/api", "https://user:secret@example.org/v1", "https://example.org/v1?key=secret", "https://example.org/v1#key"]:
        raises(ValueError, lambda: ev.api_url(bad))


def cache_and_models():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        data, output = root / "dev.jsonl", root / "runs"
        rows = [{"id": str(i), "document": "A fictional reference.", "claim": "yes" if i % 2 else "no",
                 "label": i % 2, "dataset": f"source-{i//2}"} for i in range(4)]
        data.write_text("".join(json.dumps(r) + "\n" for r in rows))
        parsed = ev.load_data(data, 42)
        assert len(parsed) == 4 and parsed == ev.load_data(data, 42)
        assert all(r["dataset"] == "source-0" for r in parsed[:2])
        import pyarrow as pa
        import pyarrow.parquet as pq
        pq.write_table(pa.Table.from_pylist(rows), root / "dev.parquet")
        assert ev.load_data(root / "dev.parquet", 42) == parsed
        calls = []

        def ask(model, state, questions):
            calls.append((model, state, questions))
            assert set(state) == {"document", "claim"}
            answer_model = "jev-99.0.0" if model == "jev-latest" else model
            answers = {k: {"type": "noul", "noul": .9 if k == "support_simple" else (.1 if state["claim"] == "yes" or k == "contradicted_detail" else .7)} for k in questions}
            return {"model": answer_model, "answers": answers, "usage": {"input_tokens": 100, "output_tokens": 10}, "seconds": .1, "attempts": 1}

        base = ["run", "--model", "jev-latest", "--data", str(data), "--output", str(output), "--workers", "1"]
        with redirect_stdout(io.StringIO()), patch.object(ev, "Client") as client:
            client.return_value.ask.side_effect = ask
            ev.main(base + ["--dry-run"])
            assert not output.exists() and client.call_count == 0
            ev.main(base + ["--limit", "2"])
            assert len(calls) == 2 and calls[0][0] == "jev-latest" and calls[1][0] == "jev-99.0.0"
            ev.main(base + ["--limit", "0"])
            assert len(calls) == 4 and all(c[0] == "jev-99.0.0" for c in calls[1:])
            report = json.loads((output / "comparison.json").read_text())[0]
            assert report["complete"] and report["metrics"]["macro_balanced_accuracy"] == 1
            assert report["estimated_cost_usd"] is None and report["usage"]["input_tokens"] == 400
            ev.main(base + ["--limit", "0", "--offline", "--threshold", "0.95"])
            assert len(calls) == 4
            report = json.loads((output / "comparison.json").read_text())[0]
            assert report["metrics"]["supported_recall"] == 0
            ev.main(base + ["--model", "other-2.0.0", "--limit", "2"])
            assert len(calls) == 6 and len(list(output.glob("*/run.json"))) == 2
            # Alternate score recipes with identical questions reuse raw responses.
            pack = json.loads((ev.ROOT / "packs/claim_support.json").read_text())
            pack["decision"] = {"score": "support_simple", "threshold": .5}
            custom = root / "custom.json"
            custom.write_text(json.dumps(pack))
            ev.main(base + ["--pack", str(custom), "--limit", "0"])
            assert len(calls) == 6
            # A new question isolates the cache, even if its output key is unchanged.
            pack["questions"]["support_simple"]["instructions"] += " Additional generic check."
            custom.write_text(json.dumps(pack))
            ev.main(base + ["--pack", str(custom), "--limit", "2"])
            assert len(calls) == 8 and len(list(output.glob("*/run.json"))) == 3
            # A changed resolved version stops the run and preserves earlier successes.
            def changed(model, state, questions):
                result = ask(model, state, questions)
                if model == "jev-99.0.0":
                    result["model"] = "jev-100.0.0"
                return result
            client.return_value.ask.side_effect = changed
            raises(RuntimeError, lambda: ev.main(base + ["--run-name", "changed-alias", "--limit", "2"]))
            run = next(p for p in output.glob("*/run.json") if json.loads(p.read_text())["identity"]["run_name"] == "changed-alias")
            assert len(ev.read_rows(run.with_name("responses.jsonl"))) == 1
            # Copying another configuration's cache must fail before any new call.
            records = ev.read_rows(run.with_name("responses.jsonl"))
            records[0]["cache_key"] = "wrong-configuration"
            run.with_name("responses.jsonl").write_text(json.dumps(records[0]) + "\n")
            count = len(calls)
            raises(RuntimeError, lambda: ev.main(base + ["--run-name", "changed-alias", "--limit", "2"]))
            assert len(calls) == count
        # A malformed dataset fails before any inference.
        rows[1]["id"] = rows[0]["id"]
        data.write_text("".join(json.dumps(r) + "\n" for r in rows))
        raises(ValueError, lambda: ev.load_data(data, 42))


def custom_endpoint_cache():
    with TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()), patch.object(ev, "Client") as client:
        root = Path(tmp)
        data, output = root / "synthetic.jsonl", root / "runs"
        data.write_text(json.dumps({"doc": "Synthetic.", "claim": "Synthetic.", "label": 1}) + "\n")
        model = "Contrastive-LM/CLM-v0.1-8B"
        client.return_value.ask.return_value = {
            "model": model, "answers": {k: {"type": "noul", "noul": .9 if k == "support_simple" else .1}
                                       for k in ["support_simple", "unsupported_detail", "contradicted_detail"]},
            "usage": {"input_tokens": 30, "output_tokens": 0}, "seconds": .1, "attempts": 1}
        base = ["run", "--model", model, "--data", str(data), "--output", str(output), "--limit", "0"]
        for endpoint in ["http://127.0.0.1:8700/v1", "http://127.0.0.1:8701/v1"]:
            ev.main(base + ["--base-url", endpoint, "--timeout", "600"])
            assert client.call_args.args == (8, endpoint, None, 600)
            report = json.loads((output / "comparison.json").read_text())[0]
            assert report["base_url"] == endpoint and report["complete"]
            assert report["usage"]["output_tokens"] == 0
        assert client.return_value.ask.call_count == 2 and len(list(output.glob("*/run.json"))) == 2
        ev.main(base + ["--base-url", "http://127.0.0.1:8700/v1/", "--offline"])
        assert client.return_value.ask.call_count == 2
        for path in output.glob("*/run.json"):
            assert "base_url" in json.loads(path.read_text())["identity"]


if __name__ == "__main__":
    arithmetic()
    transport()
    cache_and_models()
    custom_endpoint_cache()
    print("System One runner checks passed: synthetic data and mocked HTTP only.")
