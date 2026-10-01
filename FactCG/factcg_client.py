
"""Transformers client for FactCG-DeBERTa-v3-Large.

Reproduces the exact prompt template and inference recipe used by the
original FactCG library's DeBERTa inference path (derenlei/FactCG,
factcg/inference.py:Inferencer, use_hf_ckpt=True -- upstream's own default
and recommended loading path, verified against the actual source: it loads
yaxili96/FactCG-DeBERTa-v3-Large via AutoModelForSequenceClassification,
exactly as documented in description_claude.md):

- Model + tokenizer are loaded once and kept in memory (unlike Ollama-served
  models, this runs in-process), on GPU if available.
- Input is a single formatted string per (document_chunk, claim) pair (not a
  sentence-pair encoding -- consistent with the model's config.json
  reporting type_vocab_size: 0), built from the same INSTRUCTION_TEMPLATE
  upstream uses (factcg/utils.py).
- support_prob is softmax(logits)[:, 1] -- verified against upstream's own
  worked example (an inconsistent claim scores ~0.065, a consistent one
  ~0.784), i.e. index 1 = "Supported", matching LLM-AggreFact's gold_label
  convention directly.
- Deliberate deviation from upstream: inference.py reassigns
  tokenizer.pad_token = tokenizer.eos_token, a workaround needed for T5
  (no native pad token) inherited by a shared code path. Verified directly
  against this tokenizer: DeBERTa already has a correct pad_token ('[PAD]',
  id 0); eos_token here just aliases to '[SEP]' (id 2). Padded positions are
  always masked out via attention_mask regardless of the literal token id,
  so this client keeps the tokenizer's own correct pad_token instead of
  overwriting it.
"""
from typing import List, Tuple

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

INSTRUCTION_TEMPLATE = (
    '{text_a}\n\nChoose your answer: based on the paragraph above can we conclude that "{text_b}"?'
    "\n\nOPTIONS:\n- Yes\n- No\nI think the answer is "
)


class FactCGModel:
    def __init__(self, model_id: str, cache_dir, max_length: int, device: str = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        config = AutoConfig.from_pretrained(
            model_id, num_labels=2, finetuning_task="text-classification", cache_dir=str(cache_dir)
        )
        config.problem_type = "single_label_classification"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, cache_dir=str(cache_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id, config=config, cache_dir=str(cache_dir)
        ).to(self.device)
        self.model.eval()
        self.max_length = max_length

    @torch.no_grad()
    def score_pairs(self, documents: List[str], claims: List[str]) -> Tuple[List[float], List[List[float]]]:
        """Score aligned lists of (document_chunk, claim) pairs in one forward pass.

        Returns (support_probs, all_probs): support_probs[i] = P(class 1 = "Supported")
        for pair i; all_probs[i] = [P(class 0), P(class 1)] for the same pair.
        """
        text_list = [
            INSTRUCTION_TEMPLATE.format(text_a=doc, text_b=claim) for doc, claim in zip(documents, claims)
        ]
        inputs = self.tokenizer(
            text_list, truncation=True, padding="longest", max_length=self.max_length, return_tensors="pt"
        ).to(self.device)
        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu()
        return probs[:, 1].tolist(), probs.tolist()
