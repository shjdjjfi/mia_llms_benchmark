"""EM-MIAs pipeline implementation."""

from .data import ensure_dataset
from .features import FeatureExtractor

__all__ = ["ensure_dataset", "FeatureExtractor"]
