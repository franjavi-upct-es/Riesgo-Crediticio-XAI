# src/data/preprocessing.py
"""Schema-driven preprocessing with sklearn Pipeline.

Replaces the fragile pd.get_dummies approach with a fitted
ColumnTransformer that handles unknown categories gracefully
and carries the feature schema as part of the serialized pipeline.

The pipeline is fitted during training and serialized alongside the
model. At inference time, the same fitted pipeline transforms inputs
identically — eliminating train/serve skew by construction.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import structlog
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.data.schema import DatasetSchema

logger = structlog.get_logger(__name__)

# Legacy mapping preserved for backward compatibility with existing tests
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


def build_preprocessing_pipeline(schema: DatasetSchema) -> ColumnTransformer:
    """Build a sklearn ColumnTransformer from a dataset schema.

    Creates an encoder that handles categorical features with
    OneHotEncoder (unknown categories → all zeros) and passes
    numerical features through with StandardScaler.

    Args:
        schema: The dataset schema defining feature types.

    Returns:
        An unfitted ColumnTransformer ready for .fit().
    """
    categorical_cols = schema.categorical_features
    numerical_cols = schema.numerical_features

    transformers = []

    if numerical_cols:
        num_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )
        transformers.append(("num", num_pipeline, numerical_cols))

    if categorical_cols:
        transformers.append(
            (
                "cat",
                OneHotEncoder(
                    sparse_output=False,
                    handle_unknown="ignore",
                    drop="first",
                ),
                categorical_cols,
            )
        )

    pipeline = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    logger.info(
        "preprocessing_pipeline_built",
        dataset_id=schema.id,
        n_numerical=len(numerical_cols),
        n_categorical=len(categorical_cols),
    )

    return pipeline


def fit_and_save_pipeline(
    pipeline: ColumnTransformer,
    X: pd.DataFrame,
    output_path: Path,
) -> list[str]:
    """Fit the preprocessing pipeline and save it.

    Args:
        pipeline: Unfitted ColumnTransformer.
        X: Training features (raw, before encoding).
        output_path: Path to save the fitted pipeline.

    Returns:
        List of output feature names after transformation.
    """
    pipeline.fit(X)
    feature_names = [re.sub(r"[\[\]<>]", "_", name) for name in pipeline.get_feature_names_out()]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)

    logger.info(
        "pipeline_fitted_and_saved",
        path=str(output_path),
        n_input_features=X.shape[1],
        n_output_features=len(feature_names),
    )

    return feature_names


def load_pipeline(path: Path) -> ColumnTransformer:
    """Load a fitted preprocessing pipeline from disk.

    Args:
        path: Path to the serialized pipeline.

    Returns:
        The fitted ColumnTransformer.

    Raises:
        FileNotFoundError: If the pipeline file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Preprocessing pipeline not found: {path}")

    pipeline = joblib.load(path)
    logger.info("pipeline_loaded", path=str(path))
    return pipeline


def preprocess_with_pipeline(
    input_dict: dict[str, Any],
    pipeline: ColumnTransformer,
    feature_names: list[str],
) -> pd.DataFrame:
    """Transform a single prediction input using the fitted pipeline.

    Args:
        input_dict: Raw input data as a dictionary.
        pipeline: Fitted ColumnTransformer.
        feature_names: Expected output feature names.

    Returns:
        DataFrame with shape (1, n_features) ready for model inference.
    """
    input_df = pd.DataFrame([input_dict])
    transformed = pipeline.transform(input_df)

    result = pd.DataFrame(transformed, columns=feature_names)

    logger.debug(
        "input_preprocessed_via_pipeline",
        input_columns=len(input_dict),
        output_columns=result.shape[1],
    )

    return result


# ---------------------------------------------------------------------------
# Legacy functions (backward compatibility for existing tests/code)
# ---------------------------------------------------------------------------


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Legacy: Apply one-hot encoding to categorical columns.

    Preserved for backward compatibility with existing tests and the
    training pipeline. New code should use build_preprocessing_pipeline().
    """
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    logger.debug("features_encoded", n_raw=df.shape[1], n_encoded=encoded.shape[1])
    return encoded


def preprocess_input(
    input_dict: dict,
    feature_names: list[str],
) -> pd.DataFrame:
    """Legacy: Transform a single prediction input using get_dummies.

    Preserved for backward compatibility. New code should use
    preprocess_with_pipeline().
    """
    input_df = pd.DataFrame([input_dict])
    encoded = encode_features(input_df)
    aligned = encoded.reindex(columns=feature_names, fill_value=0)

    if aligned.shape[1] != len(feature_names):  # pragma: no cover
        raise ValueError(
            f"Feature alignment produced {aligned.shape[1]} columns, expected {len(feature_names)}."
        )

    return pd.DataFrame(aligned)
