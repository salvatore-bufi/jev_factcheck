"""Default configuration for the fact-checking module.

Every value can be overridden from the command line (see main.py --help).

This module lives in a per-codebase folder (FactCG/) alongside sibling
folders for other fact-checking implementations under test (AICCTU/,
MiniCheck/). Claims and documents are shared across all codebases and
datasets, and live at the repo root under input/<dataset>/. Results are
namespaced at the repo root by dataset and, within it, by a deterministic
hash of the hyperparameters that produced them (see run_identity.py) --
this folder's name (MODEL_NAME) is only the "codebase" component of that
identity, computed at runtime by main.py.

Unlike AICCTU, FactCG does not do retrieval (no embeddings, no FAISS index,
no data_store/): like MiniCheck, it compares a claim directly against every
document's text, so there is nothing to build/persist besides the outputs/
themselves. Unlike both AICCTU and MiniCheck (Ollama-served LLMs), FactCG is
a locally-loaded transformers sequence-classification model
(yaxili96/FactCG-DeBERTa-v3-Large, see description_claude.md) -- a
deterministic binary classifier, not a generative model, so there is no
notion of --temperature or --retries on unparseable output.
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent       # this codebase's folder, .../FactCG
REPO_ROOT = BASE_DIR.parent                       # shared repo root
MODEL_NAME = BASE_DIR.name                        # codebase name, used as a prefix in run_identity

# Dataset selection (see --dataset): input/<dataset>/{claims,documents}
DATASET_NAME = "example"

# Physical GPU index (or comma-separated indices) this process is restricted to via
# CUDA_VISIBLE_DEVICES (see --gpu in main.py). None = don't touch, use whatever's already
# in the environment.
GPU_DEVICE = None

# Model (transformers AutoModelForSequenceClassification, loaded once and
# reused across all claims/documents in a run -- see factcg_client.py).
MODEL_ID = "yaxili96/FactCG-DeBERTa-v3-Large"
CACHE_DIR = BASE_DIR / "cache"   # HF download cache, self-contained inside FactCG/

MAX_LENGTH = 2048   # tokenizer max_length, matches the upstream FactCG Inferencer
CHUNK_SIZE = 550    # max words per document chunk (nltk word_tokenize count), matches upstream chunking_src default
BATCH_SIZE = 8      # chunks scored per forward pass; conservative default for a 16GB GPU at up to MAX_LENGTH tokens/example

# Binary verdict threshold on support_prob (same convention as MiniCheck:
# pred_label = 1 if support_prob > threshold else 0).
SUPPORT_THRESHOLD = 0.5

# Veracity labels for the binary verdict this codebase produces.
LABELS = ["Supported", "Refuted"]
