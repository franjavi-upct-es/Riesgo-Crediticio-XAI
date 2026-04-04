# src/explain/__init__.py
"""Explainability modules for model interpretation."""

from src.explain.shap_engine import ShapEngine, ShapExplanation

__all__ = ["ShapEngine", "ShapExplanation"]
