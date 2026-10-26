"""Bespoke-Nimble-9B-v3 as a Decision Index engine (github.com/apolinario/decision-index).

    python -m decision_index run --engine nimble.evaluation.decision_index_engine:NimbleEngine \
        --option adapter=<Bespoke-Nimble-9B-v3 snapshot> --rows <suite rows> --out <run dir>

Bespoke-Nimble-9B-v3 is a LoRA adapter on Qwen/Qwen3.5-9B that answers each question by scoring one candidate token
per option. A request's questions share one context. By default the engine runs their common prompt
prefix once and then only each question's short tail on a copy of the cache (shared_prefix=True, the
fastest path); shared_prefix=False scores every full prompt in padded batches instead. Prompts, option
codes and probabilities follow the adapter's saved contract (schema_config.json), which pins the two
prompt files in this package by hash. Nothing is truncated: a request the contract cannot encode is
refused as Unsupported. An empty string state is passed as its JSON form, as empty objects already are.
"""
import copy
import hashlib
import json
from pathlib import Path

import torch
from decision_index.engines import Engine, Unsupported

from nimble.evaluation import extended_schema

HERE = Path(__file__).resolve().parent
MAX_CHOICES = 255


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_contract(contract, tokenizer):
    """Refuse an adapter whose prompt code or candidate tokens differ from this package's."""
    if contract.get("task") != "schema_candidate_classification_v2":
        raise ValueError("Unsupported adapter task")
    for name in ("parallel_schema.py", "extended_schema.py"):
        if contract["prompt_source_sha256"].get(name) != sha(HERE / name):
            raise ValueError(f"Prompt code differs from the adapter's contract: {name}")
    codes = extended_schema.codes_for(MAX_CHOICES, tokenizer)
    schema = {"decision": {"type": "enum", "description": "Select one candidate.", "choices": codes}}
    prepared = extended_schema.prepare_prompts(tokenizer, "Codebook validation.", schema, 32768)
    if codes != contract["candidate_codes"] or prepared.candidate_ids[0] != contract["candidate_token_ids"]:
        raise ValueError("Tokenizer candidate codes differ from the adapter's contract")


def native_question(question):
    """Yes/no questions are read from true/false criteria; map yes/no keys onto them."""
    if question["type"] != "noul":
        return question
    criteria = question.get("criteria") or {}
    if set(criteria) <= {"true", "false"}:
        return question
    if set(criteria) == {"yes", "no"}:
        return dict(question, criteria={"true": criteria["yes"], "false": criteria["no"]})
    raise Unsupported(f"yes/no question with criteria keys {sorted(criteria)} has no true/false reading")


def native_schema(body):
    """The request's state as context text, and each question as a schema field."""
    def text(value):
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    schema = {}
    for name, question in body["questions"].items():
        field = {"description": text(question["instructions"])}
        criteria = question.get("criteria") or {}
        if question["type"] == "noul":
            field.update(type="boolean", choices=[False, True],
                         choice_descriptions={k: text(v) for k, v in criteria.items() if v is not None})
        elif question["type"] == "choice":
            field.update(type="enum", choices=list(criteria),
                         choice_descriptions={k: text(v) for k, v in criteria.items() if v is not None})
        elif question["type"] == "score":
            field.update(type="enum", choices=[str(i) for i in range(len(criteria))],
                         choice_descriptions={str(i): text(v) for i, v in enumerate(criteria)})
        else:
            raise ValueError("Unsupported primitive")
        schema[name] = field
    return text(body["state"]), schema


def collate(rows, pad_id):
    """Left-padded prompts and right-padded candidate token ids."""
    length = max(len(r["input_ids"]) for r in rows)
    width = max(len(r["candidate_ids"]) for r in rows)
    for r in rows:
        n = len(r["candidate_ids"])
        if not 1 <= n <= MAX_CHOICES or len(set(r["candidate_ids"])) != n:
            raise ValueError("Invalid candidate token set")
    return {
        "input_ids": torch.tensor([[pad_id] * (length - len(r["input_ids"])) + list(r["input_ids"]) for r in rows]),
        "attention_mask": torch.tensor([[0] * (length - len(r["input_ids"])) + [1] * len(r["input_ids"]) for r in rows]),
        "candidate_ids": torch.tensor([r["candidate_ids"] + [0] * (width - len(r["candidate_ids"])) for r in rows]),
        "candidate_mask": torch.tensor([[True] * len(r["candidate_ids"]) + [False] * (width - len(r["candidate_ids"]))
                                        for r in rows]),
    }


def candidate_logits(model, inputs):
    logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                   use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
    selected = logits.gather(1, inputs["candidate_ids"])
    return selected.masked_fill(~inputs["candidate_mask"], -torch.inf)


def decision_result(row, logits):
    logits = logits[:len(row["choices"])].double()
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Non-finite candidate logits")
    probs = logits.softmax(-1)
    choices = row["choices"]
    return {"prediction": choices[logits.argmax().item()],
            "probabilities": {extended_schema.choice_key(k): v for k, v in zip(choices, probs.tolist())}}


def chunks(rows, budget):
    """Consecutive groups whose padded size, longest prompt times rows, stays within budget."""
    group, longest = [], 0
    for row in rows:
        size = len(row["input_ids"])
        if group and max(longest, size) * (len(group) + 1) > budget:
            yield group
            group, longest = [], 0
        group.append(row)
        longest = max(longest, size)
    if group:
        yield group


def load_base(model_id, revision):
    from transformers import AutoConfig, Qwen3_5ForConditionalGeneration
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Bespoke-Nimble-9B-v3 needs a CUDA GPU with BF16 support")
    config = AutoConfig.from_pretrained(model_id, revision=revision, trust_remote_code=False)
    if config.model_type != "qwen3_5":
        raise ValueError(f"Unsupported architecture: {config.model_type}")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_id, revision=revision, config=config, dtype=torch.bfloat16, attn_implementation="sdpa",
        trust_remote_code=False).to("cuda")
    model.config.use_cache = False
    return model


class NimbleEngine(Engine):
    name = "nimble"
    latency = "In-process request wall time on one GPU: prompt construction and batched forward passes."

    def __init__(self, adapter, max_tokens=32768, batch_tokens=65536, model=None, revision=None, shared_prefix=True,
                 cache_tokens=131072, **options):
        super().__init__(adapter=adapter, max_tokens=max_tokens, batch_tokens=batch_tokens,
                         shared_prefix=shared_prefix, cache_tokens=cache_tokens, **options)
        from peft import PeftModel
        from transformers import AutoTokenizer
        adapter = Path(adapter)
        self.contract = json.loads((adapter / "schema_config.json").read_text())
        model = model or self.contract["model"]
        revision = revision or self.contract["revision"]
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        validate_contract(self.contract, self.tokenizer)
        self.shared_prefix = shared_prefix if isinstance(shared_prefix, bool) else str(shared_prefix).lower() == "true"
        self.cache_tokens = int(cache_tokens)
        self.model = PeftModel.from_pretrained(load_base(model, revision), adapter).eval()
        self.max_tokens, self.batch_tokens = int(max_tokens), int(batch_tokens)
        self.provenance = {"adapter_sha256": sha(adapter / "adapter_model.safetensors"), "base_model": model,
                           "revision": revision, "contract_task": self.contract.get("task"),
                           "max_tokens": self.max_tokens, "shared_prefix": self.shared_prefix}

    def __call__(self, state, questions):
        if isinstance(state, str) and not state.strip():
            state = json.dumps(state)
        body = {"state": state, "questions": {k: native_question(q) for k, q in questions.items()}}
        context, schema = native_schema(body)
        try:
            prompt = extended_schema.prepare_prompts(self.tokenizer, context, schema, self.max_tokens)
        except ValueError as exc:
            if "limit is" in str(exc):
                raise Unsupported(f"maximum context length: {exc}") from None
            raise
        rows = [dict(input_ids=prompt.full_ids[i], candidate_ids=prompt.candidate_ids[i], choices=prompt.choices[i])
                for i in range(len(prompt.names))]
        if self.shared_prefix and len(rows) > 1 and len(prompt.prefix_ids) >= 256:
            results = self.score_shared(prompt, rows)
        else:
            results = self.score(rows)
        answers = {}
        for name, out in zip(prompt.names, results):
            probabilities = out["probabilities"]
            if questions[name]["type"] == "noul":
                answers[name] = {"type": "noul", "noul": probabilities["true"]}
            else:
                answers[name] = {"type": "choice", "choice": out["prediction"], "probabilities": probabilities}
        return {"model": self.name, "answers": answers, "usage": {"input_tokens": sum(map(len, prompt.full_ids))}}, None

    def score(self, rows):
        results = []
        for group in chunks(rows, self.batch_tokens):
            results.extend(self.score_group(group))
        return results

    def score_group(self, group):
        """Score one batch; on out-of-memory, split it, and refuse only a single prompt that does not fit."""
        try:
            batch = {k: v.to("cuda") for k, v in collate(group, self.tokenizer.pad_token_id).items()}
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                logits = candidate_logits(self.model, batch).cpu()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(group) == 1:
                raise Unsupported(f"out of GPU memory for one {len(group[0]['input_ids'])}-token prompt") from None
            half = len(group) // 2
            return self.score_group(group[:half]) + self.score_group(group[half:])
        return [decision_result(row, logits[i]) for i, row in enumerate(group)]

    def score_shared(self, prompt, rows):
        """Run the questions' shared prefix once, then only each question's short tail on a copy of its cache.

        Every prompt of a request is the same text up to the requested field's name. The prefix runs
        once with the cache on; the cache (attention keys and values, and the linear-attention layers'
        convolution and recurrent state) is copied across a batch of tails, which are right-padded so
        padding never comes between the prefix and a tail. Each tail's answer is read at its own last token.
        """
        prefix, tails = prompt.prefix_ids, prompt.suffix_ids
        per_batch = max(1, self.cache_tokens // len(prefix))
        results = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            try:
                out = self.model(input_ids=torch.tensor([prefix], device="cuda"), use_cache=True, logits_to_keep=1)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                return self.score(rows)
            cache = out.past_key_values
            for start in range(0, len(rows), per_batch):
                group = list(range(start, min(start + per_batch, len(rows))))
                lengths = [len(tails[i]) for i in group]
                width = max(lengths)
                ids = torch.tensor([tails[i] + [self.tokenizer.pad_token_id] * (width - len(tails[i])) for i in group],
                                   device="cuda")
                mask = torch.tensor([[1] * len(prefix) + [1] * n + [0] * (width - n) for n in lengths], device="cuda")
                batch_cache = copy.deepcopy(cache)
                batch_cache.reorder_cache(torch.zeros(len(group), dtype=torch.long, device="cuda"))
                logits = self.model(input_ids=ids, attention_mask=mask, past_key_values=batch_cache, use_cache=True,
                                    logits_to_keep=width).logits
                last = logits[torch.arange(len(group), device="cuda"), torch.tensor(lengths, device="cuda") - 1].float()
                for j, i in enumerate(group):
                    candidates = torch.tensor(rows[i]["candidate_ids"], device="cuda")
                    results.append(decision_result(rows[i], last[j, candidates].cpu()))
                del batch_cache, logits
        return results

    def runtime(self):
        return {"torch": torch.__version__, "gpu": torch.cuda.get_device_name()}

    def synchronize(self):
        torch.cuda.synchronize()
