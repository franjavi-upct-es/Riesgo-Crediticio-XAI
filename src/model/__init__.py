# src/model/__init__.py
"""Model training, evaluation, experiment tracking, and artifact management."""

from src.model.registry import load_model_artifacts
from src.model.train import train_model

__all__ = ["load_model_artifacts", "train_model"]
