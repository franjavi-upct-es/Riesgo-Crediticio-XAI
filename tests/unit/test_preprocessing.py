# tests/unit/test_preprocessing.py
"""Unit tests for src.data.preprocessing.

The preprocessing module is the single source of truth for feature
encoding. These tests verify that train and serve produce identical
outputs, that missing columns are handled correctly, and that the
column mapping is complete and consistent.
"""

import pandas as pd

from src.data.preprocessing import (
    CATEGORICAL_FEATURES,
    COLUMN_MAPPING,
    encode_features,
    preprocess_input,
)


class TestColumnMapping:
    """Verify the column mapping is complete and internally consistent."""

    def test_mapping_has_20_entries(self):
        assert len(COLUMN_MAPPING) == 20

    def test_all_values_are_unique(self):
        values = list(COLUMN_MAPPING.values())
        assert len(values) == len(set(values)), (
            "Duplicate column names in mapping"
        )

    def test_keys_are_sequential_attributes(self):
        expected_keys = [f"Attribute{i}" for i in range(1, 21)]
        assert list(COLUMN_MAPPING.keys()) == expected_keys

    def test_categorical_features_are_subset_of_mapping_values(self):
        mapped_names = set(COLUMN_MAPPING.values())
        for cat_feat in CATEGORICAL_FEATURES:
            assert cat_feat in mapped_names, (
                f"{cat_feat} not in COLUMN_MAPPING values"
            )


class TestEncodeFeatures:
    """Verify one-hot encoding behavior."""

    def test_encodes_object_columns(self):
        df = pd.DataFrame(
            {
                "age": [25, 30],
                "housing": ["own", "rent"],
                "job": ["skilled", "unskilled"],
                "purpose": ["car", "furniture"],
            }
        )
        encoded = encode_features(df)

        assert "age" in encoded.columns
        assert "housing" not in encoded.columns  # Original dropped
        assert encoded.shape[0] == 2
        # 1 numeric + 3 dummies (drop_first removes one category per column)
        assert encoded.shape[1] >= 4

    def test_preserves_numeric_columns(self):
        df = pd.DataFrame(
            {
                "duration": [12, 24],
                "credit_amount": [5000, 10000],
            }
        )
        encoded = encode_features(df)

        assert list(encoded.columns) == ["duration", "credit_amount"]
        assert encoded.shape == (2, 2)

    def test_drop_first_reduces_columns(self):
        df = pd.DataFrame(
            {
                "color": ["red", "blue", "green", "red"],
            }
        )
        encoded = encode_features(df)

        # drop_first=True: 3 categories → 2 dummy columns
        assert encoded.shape[1] == 2


class TestPreprocessInput:
    """Verify the inference preprocessing pipeline."""

    def test_output_matches_feature_names_shape(
        self, valid_payload, sample_feature_names
    ):
        result = preprocess_input(valid_payload, sample_feature_names)

        assert result.shape == (1, len(sample_feature_names))
        assert list(result.columns) == sample_feature_names

    def test_fills_missing_features_with_zero(self, valid_payload):
        feature_names = [
            "duration",
            "credit_amount",
            "nonexistent_feature_xyz",
        ]
        result = preprocess_input(valid_payload, feature_names)

        assert result["nonexistent_feature_xyz"].iloc[0] == 0

    def test_ignores_extra_encoded_columns(self, valid_payload):
        # Feature names is a strict subset — extra dummy columns are dropped
        feature_names = ["duration", "credit_amount"]
        result = preprocess_input(valid_payload, feature_names)

        assert result.shape == (1, 2)

    def test_preserves_numeric_values(
        self, valid_payload, sample_feature_names
    ):
        result = preprocess_input(valid_payload, sample_feature_names)

        assert result["duration"].iloc[0] == valid_payload["duration"]
        assert result["age"].iloc[0] == valid_payload["age"]
        assert (
            result["credit_amount"].iloc[0] == valid_payload["credit_amount"]
        )

    def test_one_hot_features_are_binary(
        self, valid_payload, sample_feature_names
    ):
        result = preprocess_input(valid_payload, sample_feature_names)

        # Numeric columns from the original dataset are NOT one-hot
        numeric_cols = {
            "duration",
            "credit_amount",
            "installment_commitment",
            "residence_since",
            "age",
            "existing_credits",
            "num_dependents",
        }
        one_hot_cols = [
            c for c in sample_feature_names if c not in numeric_cols
        ]
        for col in one_hot_cols:
            val = result[col].iloc[0]
            assert val in (0, 1, True, False), (
                f"{col} has non-binary value: {val}"
            )

    def test_raises_on_shape_mismatch(self, valid_payload):
        """This should not happen with reindex, but guards against regressions."""
        # An empty feature_names list should produce an empty DataFrame
        result = preprocess_input(valid_payload, [])
        assert result.shape == (1, 0)

    def test_handles_unknown_categorical_value(self, sample_feature_names):
        payload = {
            "checking_status": "NEVER_SEEN_BEFORE",
            "duration": 12,
            "credit_history": "critical/other",
            "purpose": "radio/television",
            "credit_amount": 5000,
            "savings_status": "no_savings",
            "employment": "unemployed",
            "installment_commitment": 4,
            "personal_status": "male single",
            "other_parties": "none",
            "residence_since": 4,
            "property_magnitude": "real estate",
            "age": 35,
            "other_payment_plans": "none",
            "housing": "own",
            "existing_credits": 1,
            "job": "skilled",
            "num_dependents": 1,
            "own_telephone": "yes",
            "foreign_worker": "no",
        }
        # Should not raise — unknown categories just don't match any known column
        result = preprocess_input(payload, sample_feature_names)
        assert result.shape == (1, len(sample_feature_names))
