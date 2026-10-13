"""Sentence-aware chunking of long documents.

Mirrors the greedy sentence-packing algorithm used by the original FactCG
library (derenlei/FactCG, factcg/inference.py:Inferencer.chunking_src):
split the document into sentences, then greedily pack whole sentences into a
chunk until the next sentence would push the chunk over `chunk_size` words
(counted via nltk.word_tokenize, the same proxy the upstream implementation
uses). Sentences are never split mid-way; a single sentence longer than
`chunk_size` still becomes its own (oversized) chunk.
"""
from typing import List

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")


def chunk_document(text: str, chunk_size: int) -> List[str]:
    """Greedily pack sentences into chunks of at most `chunk_size` words."""
    sentences = sent_tokenize(text) or [""]

    chunks = []
    current_chunk: List[str] = []
    current_word_count = 0
    for sentence in sentences:
        sentence_word_count = len(word_tokenize(sentence))
        if current_word_count + sentence_word_count > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_word_count = sentence_word_count
        else:
            current_chunk.append(sentence)
            current_word_count += sentence_word_count
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return [c.strip() for c in chunks if c.strip()]
