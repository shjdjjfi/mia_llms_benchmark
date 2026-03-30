"""EM-MIAs pipeline implementation."""

from .data import ensure_dataset
from .features import FeatureExtractor
from .model import train_and_evaluate

__all__ = ["ensure_dataset", "FeatureExtractor", "train_and_evaluate"]
