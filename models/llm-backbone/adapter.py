"""A plain backbone LLM read as a decision model: both option orders of every yes/no question (R0, R1).

Neutral prompt, the same for every backbone: the state (document, claim), the question, two lettered options and
"Answer with the letter of the correct option only.", in the chat template with thinking off (or a completion prompt
ending in "Answer:" for a pre-trained checkpoint). Read-out: next-token log-probabilities of the letters A and B.
Each yes/no question is asked twice, with "yes" as option A and then as option B; both orders share the document
prefix through vLLM's prefix cache.

Stored per answer:
  logits       [0, logit(p)], p = mean P(yes) of the two orders (readout "r1") or of the first order ("r0");
               the label-prior correction and the temperature (R1, R2) are applied by the analysis script
  orders       {"yes_first": [lp_A, lp_B], "no_first": [lp_A, lp_B]}, the raw letter log-probabilities
  letter_mass  mean probability on the two letters (low values: the model wanted to answer something else)
"""
import math

from core.adapter import BaseAdapter, make_answer

PROMPT = ("Evaluate the state below and answer the question.\n\n"
          "document:\n{document}\n\nclaim:\n{claim}\n\n"
          "Question: {question}\nA. {a}\nB. {b}\n"
          "Answer with the letter of the correct option only.")


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def logit(p, eps=1e-6):
    p = min(max(p, eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


class Adapter(BaseAdapter):
    NAME = "Backbone LLM"
    SIZE = "4B/9B"                    # the size of the configured repo (Qwen3.5-4B or -9B)
    DEFAULTS = {
        "repo": "Qwen/Qwen3.5-9B",
        "revision": "",                # pin the downloaded snapshot (commit sha)
        "chat": True,                  # False: completion prompt for a pre-trained (Base) checkpoint
        "readout": "r1",               # "r1": both orders (R0 is kept in `orders`); "r0": first order only
        "max_model_len": 32768,        # longer prompts fail and are logged, never shortened
        "gpu_util": 0.90,
        "max_num_seqs": 32,
        "max_batched_tokens": 8192,
        "batch": 64,
    }

    def load(self):
        from transformers import AutoTokenizer
        from vllm import LLM

        rev = self.opt["revision"] or None
        self.tok = AutoTokenizer.from_pretrained(self.opt["repo"], revision=rev)
        self.ids = [self._letter_id(letter) for letter in ("A", "B")]
        self.llm = LLM(model=self.opt["repo"], revision=rev, tokenizer_revision=rev, dtype="bfloat16",
                       max_model_len=self.opt["max_model_len"], gpu_memory_utilization=self.opt["gpu_util"],
                       enable_prefix_caching=True, limit_mm_per_prompt={"image": 0, "video": 0},
                       max_logprobs=2, seed=0, max_num_seqs=self.opt["max_num_seqs"],
                       max_num_batched_tokens=self.opt["max_batched_tokens"])

    def _letter_id(self, letter):
        """Token id of the letter as the first answer token (chat) or after "Answer:" (completion)."""
        tail, sep = ("", "") if self.opt["chat"] else ("Answer:", " ")
        base = self.tok.encode(tail, add_special_tokens=False)
        full = self.tok.encode(tail + sep + letter, add_special_tokens=False)
        extra = full[len(base):]
        if full[:len(base)] != base or len(extra) != 1:
            raise SystemExit(f"{letter!r} is not a single token here: {extra}")
        return extra[0]

    def _render(self, state, question, yes_first):
        a, b = ("yes", "no") if yes_first else ("no", "yes")
        text = PROMPT.format(document=state["document"], claim=state["claim"],
                             question=question["instructions"], a=a, b=b)
        if not self.opt["chat"]:
            return text + "\nAnswer:"
        messages = [{"role": "user", "content": text}]
        try:
            return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                enable_thinking=False)
        except TypeError:          # templates without a thinking switch
            return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def describe(self):
        return {"repo": self.opt["repo"], "revision": self.opt["revision"], "prompt": "neutral-v1",
                "readout": self.opt["readout"], "chat": self.opt["chat"]}

    def reset_cache(self):
        try:
            self.llm.reset_prefix_cache()
        except Exception:
            pass

    def score(self, states, questions):
        from vllm import SamplingParams

        for qid, q in questions.items():
            if q["type"] != "noul":
                raise SystemExit(f"llm-backbone reads yes/no questions only; {qid!r} is {q['type']!r}")
        orders = (True, False) if self.opt["readout"] == "r1" else (True,)
        prompts, owners, rows = [], [], [None] * len(states)
        for i, state in enumerate(states):
            rendered = {(qid, o): self._render(state, q, o) for qid, q in questions.items() for o in orders}
            lengths = {k: len(self.tok.encode(t, add_special_tokens=False)) for k, t in rendered.items()}
            if max(lengths.values()) + 1 > self.opt["max_model_len"]:
                rows[i] = f"prompt of {max(lengths.values())} tokens exceeds max_model_len {self.opt['max_model_len']}"
                continue
            rows[i] = {"lp": {}, "tokens": {qid: lengths[(qid, True)] for qid in questions}}
            for key, text in rendered.items():
                prompts.append(text)
                owners.append((i, key))
        if prompts:
            params = SamplingParams(max_tokens=1, temperature=0.0, logprobs=2, logprob_token_ids=self.ids)
            for (i, key), out in zip(owners, self.llm.generate(prompts, params, use_tqdm=False)):
                top = out.outputs[0].logprobs[0]
                rows[i]["lp"][key] = [top[t].logprob if t in top else float("-inf") for t in self.ids]
        results = []
        for row in rows:
            if not isinstance(row, dict):
                results.append(row)
                continue
            answers = {}
            for qid in questions:
                yes_first = row["lp"][(qid, True)]
                if all(math.isinf(x) for x in yes_first):
                    answers = f"no letter probability for {qid}"
                    break
                p = sigmoid(yes_first[0] - yes_first[1])                  # "yes" is A
                record = {"yes_first": yes_first}
                mass = [math.exp(yes_first[0]) + math.exp(yes_first[1])]
                if len(orders) == 2:
                    no_first = row["lp"][(qid, False)]
                    p = 0.5 * (p + sigmoid(no_first[1] - no_first[0]))    # "yes" is B
                    record["no_first"] = no_first
                    mass.append(math.exp(no_first[0]) + math.exp(no_first[1]))
                answer = make_answer("noul", ["false", "true"], logits=[0.0, logit(p)], temperature=1.0)
                answer["orders"] = record
                answer["letter_mass"] = sum(mass) / len(mass)
                answers[qid] = answer
            results.append(answers if isinstance(answers, str) else (answers, {"prompt_tokens": row["tokens"]}))
        return results
