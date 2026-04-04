# src/data/preprocessing.py
"""Shared preprocessing logic for training and inference.

This module is the SINGLE source of truth for feature encoding. Both
the training pipeline and the prediction API import from here, which
eliminates train/serve skew by construction.

Key design decision: we use pd.get_dummies with drop_first=True during
training to produce the canonical feature set, then save the resulting
column names as `feature_names`. At inference time, we apply the same
get_dummies call and reindex against the saved feature_names so the
input always matches the model's expected shape.
"""

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

# Mapping from UCI generic attribute names to human-readable names.
# This is the canonical reference — all modules import from here.
COLUMN_MAPPING: dict[str, str] = {
    "Attribute1": "checking_status",
    "Attribute2": "duration",
    "Attribute3": "credit_history",
    "Attribute4": "purpose",
    "Attribute5": "credit_amount",
    "Attribute6": "savings_status",
    "Attribute7": "employment",
    "Attribute8": "installment_commitment",
    "Attribute9": "personal_status",
    "Attribute10": "other_parties",
    "Attribute11": "residence_since",
    "Attribute12": "property_magnitude",
    "Attribute13": "age",
    "Attribute14": "other_payment_plans",
    "Attribute15": "housing",
    "Attribute16": "existing_credits",
    "Attribute17": "job",
    "Attribute18": "num_dependents",
    "Attribute19": "own_telephone",
    "Attribute20": "foreign_worker",
}

# Categorical features that require one-hot encoding.
CATEGORICAL_FEATURES: list[str] = [
    "checking_status",
    "credit_history",
    "purpose",
    "savings_status",
    "employment",
    "personal_status",
    "other_parties",
    "property_magnitude",
    "other_payment_plans",
    "housing",
    "job",
    "own_telephone",
    "foreign_worker",
]


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply one-hot encoding to categorical columns.

    This function is used during training to produce the canonical encoded
    feature set. The resulting column order defines `feature_names`.

    Args:
        df: DataFrame with raw (pre-encoding) feature columns.

    Returns:
        DataFrame with one-hot encoded categorical features.
    """
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    logger.debug("features_encoded", n_raw=df.shape[1], n_encoded=encoded.shape[1])
    return encoded


def preprocess_input(
    input_dict: dict,
    feature_names: list[str],
) -> pd.DataFrame:
    """Transform a single prediction input into a model-ready feature vector.

    Applies the same encoding used during training and aligns the result
    with the model's expected feature order. Missing columns are filled
    with 0 (the absence indicator for one-hot features).

    Args:
        input_dict: Raw input data as a flat dictionary.
        feature_names: The canonical list of feature column names from training.

    Returns:
        DataFrame with shape (1, len(feature_names)) ready for model inference.

    Raises:
        ValueError: If the result shape does not match feature_names.
    """
    input_df = pd.DataFrame([input_dict])
    encoded = encode_features(input_df)

    # Align to the training schema: keep only known columns, fill missing with 0.
    aligned = encoded.reindex(columns=feature_names, fill_value=0)

    if aligned.shape[1] != len(feature_names):
        raise ValueError(
            f"Feature alignment produced {aligned.shape[1]} columns, expected {len(feature_names)}."
        )

    logger.debug(
        "input_preprocess",
        input_column=len(encoded.columns),
        aligned_columns=aligned.shape[1],
    )

    return aligned
