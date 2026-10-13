"""Shared code of the decision-model benchmark: datasets, questions, runner, profiling and reports."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent      # repository root
DATASETS = ROOT / "dataset"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
CONFIG = ROOT / "config"
