"""Bespoke-MiniCheck-7B (bespokelabs/Bespoke-MiniCheck-7B) in-process with vLLM: a baseline, NOT a decision model.

A grounded fact-checker (InternLM2.5-7B fine-tuned by Bespoke Labs and the LLM-AggreFact authors). It ignores the
question set. The adapter reproduces the authors' scorer, minicheck.inference.LLMCheck.score (github.com/Liyan06/MiniCheck
@b58b9fa, Apache-2.0), step for step:
  - the document is split into sentences (NLTK, newlines kept) and packed into chunks of at most
    max_model_len - 300 tokens (on this benchmark every document is one chunk);
  - the claim is split into sentences (NLTK) and every (chunk, sentence) pair becomes one prompt: the fixed system
    prompt, then "Document: <chunk>\\nClaim: <sentence>", in the model's chat template;
  - one greedy token with the 5 top log-probabilities; P(supported) = the summed probability of the top tokens that
    read "yes" (any case);
  - SentenceFusion: for each claim sentence the best chunk, and the claim's score is the weakest sentence:
    min over sentences of max over chunks. Supported when > 0.5.
The score is stored as one pseudo-question `minicheck_support`, judged with that rule. There are no logits (the score
is a sum over a top-5 list, not a softmax over options), so it cannot be re-tempered.
"""
import nltk
from nltk.tokenize import sent_tokenize

from core.adapter import BaseAdapter, make_answer

MODEL = "bespokelabs/Bespoke-MiniCheck-7B"
SYSTEM_PROMPT = ("Determine whether the provided claim is consistent with the corresponding document. Consistency in "
                 "this context implies that all information presented in the claim is substantiated by the document. "
                 "If not, it should be considered inconsistent. Please assess the claim's consistency with the document "
                 "by responding with either \"Yes\" or \"No\".")
USER_PROMPT = "Document: [DOCUMENT]\nClaim: [CLAIM]"
CLAIM_RESERVE = 300                 # tokens kept free for the claim: chunk size = max_model_len - 300 (upstream)


def sent_tokenize_with_newlines(text):
    """Sentences of every line, with a "\\n" item between lines (upstream)."""
    out = []
    for block in text.split("\n"):
        out.extend(sent_tokenize(block))
        out.append("\n")
    return out[:-1]


class Adapter(BaseAdapter):
    NAME = "Bespoke-MiniCheck-7B"
    SIZE = "7B"
    KIND = "baseline"
    DEFAULT_DECISION = {"combine": "single", "inputs": [{"question": "minicheck_support", "take": "yes"}],
                        "threshold": 0.5}
    DEFAULTS = {
        "revision": "1ed7786bcda3fa1dc35f7c4ed9e3f36b785d33b8",
        "max_model_len": 32768,         # upstream default; chunks are max_model_len - 300 tokens
        "gpu_util": 0.90,               # vLLM's default, as upstream
        "prefix_caching": True,         # upstream option (off by default there): the sentences of a claim share the document
        "eager": True,                  # no CUDA graphs: vLLM 0.28-0.31 capture them calling InternLM2's forward
                                        # without intermediate_tensors, which it requires; one token per prompt, so
                                        # graphs would save almost nothing
        "batch": 64,                    # rows per vLLM call
    }

    def load(self):
        from vllm import LLM, SamplingParams

        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            raise SystemExit("NLTK's punkt_tab is missing: python -c \"import nltk; nltk.download('punkt_tab')\"")
        rev = self.opt["revision"]
        self.llm = LLM(model=MODEL, revision=rev, tokenizer_revision=rev, dtype="bfloat16", trust_remote_code=True,
                       seed=2024, max_model_len=self.opt["max_model_len"], gpu_memory_utilization=self.opt["gpu_util"],
                       enable_prefix_caching=self.opt["prefix_caching"], enforce_eager=self.opt["eager"])
        self.tok = self.llm.get_tokenizer()
        self.tok.padding_side = "left"
        stop = [self.tok.eos_token_id]
        eot = self.tok.convert_tokens_to_ids("<|eot_id|>")      # upstream adds it when the tokenizer has it
        if eot is not None:
            stop.append(eot)
        self.params = SamplingParams(temperature=0, max_tokens=1, stop_token_ids=stop, logprobs=5)
        self.chunk_size = self.opt["max_model_len"] - CLAIM_RESERVE

    def describe(self):
        return {"repo": MODEL, "revision": self.opt["revision"], "kind": "baseline (ignores the questions)",
                "scorer": "minicheck LLMCheck @b58b9fa (SentenceFusion: min over claim sentences of max over chunks)",
                "chunk_tokens": self.chunk_size, "prefix_caching": self.opt["prefix_caching"],
                "rule": "minicheck_support > 0.5"}

    def reset_cache(self):
        try:
            self.llm.reset_prefix_cache()
        except Exception:
            pass

    def chunks(self, doc):
        """Upstream get_all_chunks_per_doc: sentences packed greedily into chunks of at most chunk_size tokens."""
        out, current, count = [], [], 0
        for sentence in sent_tokenize_with_newlines(doc) or [""]:
            n = len(self.tok(sentence, add_special_tokens=False)["input_ids"])
            if count + n > self.chunk_size:
                out.append(" ".join(current))
                current, count = [sentence], n
            else:
                current.append(sentence)
                count += n
        if current:
            out.append(" ".join(current))
        out = [c.replace(" \n ", "\n").strip() for c in out]
        return [c for c in out if c != ""] or [""]

    def prompt(self, chunk, sentence):
        user = USER_PROMPT.replace("[DOCUMENT]", chunk).replace("[CLAIM]", sentence)
        return self.tok.apply_chat_template([{"role": "system", "content": SYSTEM_PROMPT},
                                             {"role": "user", "content": user}],
                                            add_generation_prompt=True, tokenize=False)

    @staticmethod
    def support(output):
        """Upstream get_support_prob: the summed probability of the top-5 tokens that read "yes"."""
        import math

        return sum(math.exp(t.logprob) for t in output.outputs[0].logprobs[0].values()
                   if t.decoded_token.lower() == "yes")

    def score(self, states, questions):
        prompts, owners, results, plans = [], [], [None] * len(states), {}
        for i, state in enumerate(states):
            chunks = self.chunks(state["document"])
            sentences = sent_tokenize(state["claim"]) or [state["claim"]]
            texts = [self.prompt(c, s) for c in chunks for s in sentences]
            lengths = [len(self.tok(t, add_special_tokens=False)["input_ids"]) for t in texts]
            if max(lengths) + 1 > self.opt["max_model_len"]:
                results[i] = f"prompt of {max(lengths)} tokens exceeds max_model_len {self.opt['max_model_len']}"
                continue
            plans[i] = (len(chunks), len(sentences), sum(lengths))
            prompts.extend(texts)
            owners.extend([i] * len(texts))
        outputs = self.llm.generate(prompts, self.params, use_tqdm=False) if prompts else []
        probs = {}
        for i, out in zip(owners, outputs):
            probs.setdefault(i, []).append(self.support(out))
        for i, (n_chunks, n_sentences, tokens) in plans.items():
            matrix = [probs[i][c * n_sentences:(c + 1) * n_sentences] for c in range(n_chunks)]   # chunks x sentences
            per_sentence = [max(row[s] for row in matrix) for s in range(n_sentences)]
            best = min(per_sentence)
            answer = make_answer("noul", ["false", "true"], probs={"true": best, "false": 1 - best})
            answer.update(chunks=n_chunks, sentences=n_sentences, per_sentence=per_sentence)
            results[i] = ({"minicheck_support": answer}, {"prompt_tokens": tokens})
        return results
