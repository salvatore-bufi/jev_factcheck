"""JPT-9B (kirp/jpt-9b) in-process with vLLM, through the llm2jev prompt (the model card's reference).

Prompt and readout are llm2jev's: chat template with thinking off, the state, the question with all
its options labelled A, B, ..., an "Answer:" prefill, and a softmax over the option-label logits at
the model card's fixed temperature (1.087). One forward pass per question; the questions of a row
share the document prefix through vLLM's prefix cache. noul options are A = true ("Yes"), B = false.
"""
from core.adapter import BaseAdapter, make_answer

MODEL = "kirp/jpt-9b"


class Adapter(BaseAdapter):
    NAME = "JPT-9B"
    SIZE = "9B"
    DEFAULTS = {
        "revision": "b447cc7ee105c0a76a22f8fde8ecf05074dc8be0",   # weights used for the published runs
        "temperature": 1.087,          # fixed by the model card (fit once on a held-out split)
        "max_model_len": 32768,        # tokens; longer prompts fail (never shortened silently)
        "gpu_util": 0.92,
        "max_num_seqs": 32,            # Qwen3.5's DeltaNet layers need one state block per sequence
        "max_batched_tokens": 8192,
        "batch": 64,                   # rows per vLLM call
    }

    def load(self):
        from llm2jev.prompt import find_labels, render
        from transformers import AutoProcessor, AutoTokenizer
        from vllm import LLM

        rev = self.opt["revision"]
        self.render = render
        try:
            self.processor = AutoProcessor.from_pretrained(MODEL, revision=rev)
            self.processor.apply_chat_template
        except (OSError, ValueError, AttributeError, ImportError):
            self.processor = AutoTokenizer.from_pretrained(MODEL, revision=rev)
        self.tok = getattr(self.processor, "tokenizer", self.processor)
        # llm2jev's start-up probe: up to 64 labels that are single tokens right after "Answer:".
        _, probe, _ = render(self.processor, "x", {"q": {"type": "noul"}}, ["A", "B"], "chat")
        self.labels, self.ids = find_labels(self.tok, probe["q"][0], n=64)
        self.llm = LLM(model=MODEL, revision=rev, tokenizer_revision=rev, dtype="bfloat16",
                       max_model_len=self.opt["max_model_len"], gpu_memory_utilization=self.opt["gpu_util"],
                       enable_prefix_caching=True, limit_mm_per_prompt={"image": 0, "video": 0},
                       max_logprobs=64, seed=0, max_num_seqs=self.opt["max_num_seqs"],
                       max_num_batched_tokens=self.opt["max_batched_tokens"])

    def describe(self):
        return {"repo": MODEL, "revision": self.opt["revision"], "prompt": "llm2jev chat", "temperature": self.opt["temperature"]}

    def reset_cache(self):
        try:
            self.llm.reset_prefix_cache()
        except Exception:
            pass

    def score(self, states, questions):
        from vllm import SamplingParams

        prompts, owners, results = [], [], [None] * len(states)
        for i, state in enumerate(states):
            _, rendered, _ = self.render(self.processor, state, questions, self.labels, "chat")
            lengths = {qid: len(self.tok.encode(text, add_special_tokens=False)) for qid, (text, _) in rendered.items()}
            if max(lengths.values()) + 1 > self.opt["max_model_len"]:
                results[i] = f"prompt of {max(lengths.values())} tokens exceeds max_model_len {self.opt['max_model_len']}"
                continue
            for qid, (text, keys) in rendered.items():
                prompts.append(text)
                owners.append((i, qid, keys, lengths))
        if not prompts:
            return results
        k = max(len(keys) for _, _, keys, _ in owners)
        params = SamplingParams(max_tokens=1, temperature=0.0, logprobs=k, logprob_token_ids=self.ids[:k])
        outputs = self.llm.generate(prompts, params, use_tqdm=False)
        rows = {}
        for (i, qid, keys, lengths), out in zip(owners, outputs):
            top = out.outputs[0].logprobs[0]
            logits = [top[t].logprob if t in top else float("-inf") for t in self.ids[:len(keys)]]
            rows.setdefault(i, ({}, lengths))[0][qid] = make_answer(
                questions[qid]["type"], keys, logits=logits, temperature=self.opt["temperature"])
        for i, (answers, lengths) in rows.items():
            results[i] = (answers, {"prompt_tokens": lengths})
        return results
