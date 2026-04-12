# tests/unit/test_shap_engine.py
"""Unit tests for src.explain.shap_engine.

Tests the normalization logic that handles the various output formats
from different SHAP library versions (list of arrays, 3D arrays, 2D arrays).
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from src.explain.shap_engine import ShapEngine, ShapExplanation, _to_native


class TestShapNormalization:
    """Test the SHAP value normalization for different output formats."""

    def _make_engine_with_raw(self, raw_output, expected_value=0.35):
        """Helper: create a ShapEngine with mocked explainer returning raw_output."""
        with patch("src.explain.shap_engine.shap") as mock_shap:
            mock_explainer = MagicMock()
            mock_explainer.shap_values.return_value = raw_output
            mock_explainer.expected_value = expected_value
            mock_shap.TreeExplainer.return_value = mock_explainer

            engine = ShapEngine.__new__(ShapEngine)
            engine._explainer = mock_explainer
            return engine

    def test_normalizes_list_of_two_arrays(self):
        """Binary classifier: shap_values returns [class_0, class_1]."""
        raw = [
            np.array([[0.1, -0.2, 0.05]]),  # class 0
            np.array([[-0.1, 0.2, -0.05]]),  # class 1
        ]
        engine = self._make_engine_with_raw(raw, expected_value=[0.3, 0.35])

        X = pd.DataFrame([[10, 20, 30]], columns=["a", "b", "c"])
        result = engine.explain(X)

        assert isinstance(result, ShapExplanation)
        assert result.base_value == 0.35  # Picks class 1
        assert len(result.all_shap_values) == 3

    def test_normalizes_single_array_in_list(self):
        """Some versions return a single-element list."""
        raw = [np.array([[0.1, -0.2, 0.05]])]
        engine = self._make_engine_with_raw(raw, expected_value=0.4)

        X = pd.DataFrame([[10, 20, 30]], columns=["a", "b", "c"])
        result = engine.explain(X)

        assert len(result.all_shap_values) == 3

    def test_normalizes_2d_ndarray(self):
        """Plain 2D array: shape (1, n_features)."""
        raw = np.array([[0.3, -0.15, 0.08]])
        engine = self._make_engine_with_raw(raw, expected_value=0.4)

        X = pd.DataFrame([[10, 20, 30]], columns=["a", "b", "c"])
        result = engine.explain(X)

        assert abs(result.all_shap_values[0] - 0.3) < 1e-5

    def test_normalizes_3d_ndarray(self):
        """3D array: shape (n_classes, n_samples, n_features)."""
        raw = np.array(
            [
                [[0.1, -0.2, 0.05]],  # class 0
                [[-0.1, 0.2, -0.05]],  # class 1
            ]
        )
        engine = self._make_engine_with_raw(raw, expected_value=[0.3, 0.35])

        X = pd.DataFrame([[10, 20, 30]], columns=["a", "b", "c"])
        result = engine.explain(X)

        # Should pick class 1 values
        assert abs(result.all_shap_values[0] - (-0.1)) < 1e-5

    def test_filters_by_significance_threshold(self):
        """Factors below the threshold should be excluded."""
        raw = np.array([[0.5, 0.0001, -0.3]])
        engine = self._make_engine_with_raw(raw, expected_value=0.4)

        X = pd.DataFrame([[10, 20, 30]], columns=["big_pos", "tiny", "big_neg"])
        result = engine.explain(X, significance_threshold=0.01)

        factor_names = [f.feature for f in result.factors]
        assert "big_pos" in factor_names
        assert "big_neg" in factor_names
        assert "tiny" not in factor_names

    def test_factors_sorted_by_absolute_magnitude(self):
        """Factors should be ordered by |shap_value| descending."""
        raw = np.array([[0.1, -0.5, 0.3]])
        engine = self._make_engine_with_raw(raw, expected_value=0.4)

        X = pd.DataFrame([[10, 20, 30]], columns=["small", "biggest", "medium"])
        result = engine.explain(X, significance_threshold=0.01)

        magnitudes = [abs(f.shap_value) for f in result.factors]
        assert magnitudes == sorted(magnitudes, reverse=True)

    def test_impact_direction_labels(self):
        """Positive SHAP → 'increases', negative → 'reduces'."""
        raw = np.array([[0.5, -0.3]])
        engine = self._make_engine_with_raw(raw, expected_value=0.4)

        X = pd.DataFrame([[10, 20]], columns=["pos", "neg"])
        result = engine.explain(X, significance_threshold=0.01)

        impacts = {f.feature: f.impact for f in result.factors}
        assert impacts["pos"] == "increases"
        assert impacts["neg"] == "reduces"

    def test_rejects_multi_row_input(self):
        """explain() should only accept single-row DataFrames."""
        raw = np.array([[0.1, 0.2]])
        engine = self._make_engine_with_raw(raw, expected_value=0.4)

        X = pd.DataFrame([[10, 20], [30, 40]], columns=["a", "b"])
        with pytest.raises(ValueError, match="single-row"):
            engine.explain(X)


class TestToNative:
    """Test numpy-to-Python type conversion utility."""

    def test_converts_np_int64(self):
        assert _to_native(np.int64(42)) == 42
        assert isinstance(_to_native(np.int64(42)), int)

    def test_converts_np_float64(self):
        assert _to_native(np.float64(3.14)) == pytest.approx(3.14)
        assert isinstance(_to_native(np.float64(3.14)), float)

    def test_converts_np_array(self):
        result = _to_native(np.array([1, 2, 3]))
        assert result == [1, 2, 3]

    def test_passes_through_native_types(self):
        assert _to_native(42) == 42
        assert _to_native("hello") == "hello"
        assert _to_native(3.14) == 3.14
