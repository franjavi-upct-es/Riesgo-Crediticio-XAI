# src/explain/shap_engine.py
"""SHAP explanation engine.

Encapsulates SHAP computation, output normalization, and explanation
formatting. Handles the various output formats that different versions
of the shap library produce (list of arrays, 3D arrays, etc.).
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import shap
import structlog
import xgboost as xgb

from src.config import settings

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class ShapFactor:
    """A single feature's contribution to a prediction.

    Attributes:
        feature: The feature name (after one-hot encoding).
        shap_value: Signed SHAP value (positive = increases risk).
        input_value: The feature value in the input sample.
        impact: Human-readable direction ("increases" or "reduces").
    """

    feature: str
    shap_value: float
    input_value: float | int | str
    impact: str


@dataclass(frozen=True)
class ShapExplanation:
    """Complete SHAP explanation for a single prediction.

    Attributes:
        base_value: The model's expected output (average prediction).
        factors: List of significant feature contributions.
        all_shap_values: Raw SHAP values for all features (for visualization).
    """

    base_value: float
    factors: list[ShapFactor]
    all_shap_values: list[float] = field(default_factory=list)


class ShapEngine:
    """Manages SHAP explainer lifecycle and produces structured explanations.

    The engine wraps shap.TreeExplainer and normalizes the various output
    formats into a consistent ShapExplanation dataclass.
    """

    def __init__(self, model: xgb.XGBClassifier) -> None:
        """Initialize the SHAP TreeExplainer.

        Args:
            model: A trained XGBoost classifier.
        """
        self._explainer = shap.TreeExplainer(model)
        logger.info("shap_engine_initialized")

    def explain(
        self,
        X: pd.DataFrame,
        significance_threshold: float | None = None,
    ) -> ShapExplanation:
        """Compute SHAP explanation for a single-row input.

        Args:
            X: DataFrame with shape (1, n_features).
            significance_threshold: Minimum absolute SHAP value to include
                a factor in the explanation. Defaults to config value.

        Returns:
            Structured ShapExplanation with base value and significant factors.

        Raises:
            ValueError: If X does not have exactly one row.
        """
        if len(X) != 1:
            raise ValueError(f"Expected a single-row DataFrame, got {len(X)} rows.")

        threshold = significance_threshold or settings.shap.significance_threshold

        # Compute raw SHAP values
        shap_values_raw = self._explainer.shap_values(X)
        shap_row = self._normalize_shap_values(shap_values_raw)
        base_value = self._extract_base_value()

        # Ensure alignment between SHAP values and feature columns
        feature_names = X.columns.tolist()
        input_values = X.values[0]

        if len(shap_row) != len(feature_names):
            logger.warning(
                "shap_feature_mismatch",
                shap_length=len(shap_row),
                feature_count=len(feature_names),
            )
            # Truncate to the shorter length rather than silently failing
            min_len = min(len(shap_row), len(feature_names))
            shap_row = shap_row[:min_len]
            feature_names = feature_names[:min_len]
            input_values = input_values[:min_len]

        # Build factor list for significant contributions
        factors = []
        for feat, sv, iv in zip(feature_names, shap_row, input_values):
            if abs(sv) > threshold:
                factors.append(
                    ShapFactor(
                        feature=feat,
                        shap_value=round(sv, 4),
                        input_value=_to_native(iv),
                        impact="increases" if sv > 0 else "reduces",
                    )
                )

        # Sort by absolute impact (most influential first)
        factors.sort(key=lambda f: abs(f.shap_value), reverse=True)

        return ShapExplanation(
            base_value=round(base_value, 4),
            factors=factors,
            all_shap_values=[round(float(v), 6) for v in shap_row],
        )

    def _normalize_shap_values(self, raw: object) -> np.ndarray:
        """Normalize SHAP output into a 1D array for the positive class.

        The shap library returns different formats depending on version and
        model type: list of 2 arrays, 3D ndarray, or plain 2D ndarray.
        This method handles all known variants.

        Args:
            raw: Raw output from explainer.shap_values().

        Returns:
            1D numpy array of SHAP values for one sample.
        """
        arr: np.ndarray

        if isinstance(raw, list):
            # Binary classifier: [class_0_array, class_1_array]
            arr = np.asarray(raw[1] if len(raw) > 1 else raw[0])
        elif isinstance(raw, np.ndarray):
            if raw.ndim == 3:
                # Shape: (n_classes, n_samples, n_features) or (n_samples, n_features, n_classes)
                arr = raw[1] if raw.shape[0] == 2 else raw[0]
            elif raw.ndim == 2:
                arr = raw
            else:
                arr = raw
        else:
            # Fallback: try to convert to array
            arr = np.asarray(raw)

        # Flatten to 1D for a single sample
        if arr.ndim == 2:
            arr = arr[0]
        elif arr.ndim > 2:
            arr = arr.ravel()

        return arr

    def _extract_base_value(self) -> float:
        """Extract the expected (base) value from the explainer.

        Returns:
            The base value for the positive class.
        """
        ev = self._explainer.expected_value

        if isinstance(ev, (list, tuple, np.ndarray)):
            vals = np.asarray(ev)
            return float(vals[1] if len(vals) > 1 else vals[0])

        return float(ev)


def _to_native(value: object) -> float | int | str:
    """Convert numpy scalars and arrays to native Python types.

    Args:
        value: A value that may be a numpy type.

    Returns:
        A native Python scalar suitable for JSON serialization.
    """
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()  # type: ignore[no-any-return]
    if isinstance(value, np.generic):
        return value.item()  # type: ignore[no-any-return]
    return value  # type: ignore[return-value]
