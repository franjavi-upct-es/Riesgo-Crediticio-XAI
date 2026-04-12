# src/data/adapter.py
"""Dataset adapter — loads raw data from any supported source.

Provides a unified interface for loading credit risk datasets regardless
of their origin (UCI repository, CSV file, etc.). The adapter uses the
DatasetSchema to know how to find, load, map columns, and extract the
target variable.

Usage::

    >>> from src.data.schema import load_dataset_schema
    >>> from src.data.adapter import load_dataset
    >>> 
    >>> schema = load_dataset_schema("german_credit")
    >>> X, y = load_dataset(schema)
"""

import contextlib
from pathlib import Path

import pandas as pd
import structlog

from src.data.schema import DatasetSchema

logger = structlog.get_logger(__name__)

# Default directory for CSV/Parquet files
DATA_DIR = Path("data/raw")


def load_dataset(
    schema: DatasetSchema,
    data_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load a dataset using its schema definition.

    Dispatches to the appropriate loader based on schema.source.type,
    applies column mapping, extracts and transforms the target variable,
    and returns features + target aligned to the schema.

    Args:
        schema: The dataset schema describing source, columns, and target.
        data_dir: Base directory for CSV/Parquet files. Defaults to data/raw/.

    Returns:
        Tuple of (X, y) where:
            - X: DataFrame with columns matching schema.feature_names.
            - y: Series with binary labels (0/1).

    Raises:
        ValueError: If the source type is not supported.
        RuntimeError: If loading fails.
    """
    source_type = schema.source.type
    logger.info(
        "loading_dataset", dataset_id=schema.id, source_type=source_type
    )

    if source_type == "uci":
        X_raw, y_raw = _load_uci(schema)
    elif source_type == "csv":
        X_raw, y_raw = _load_csv(schema, data_dir or DATA_DIR)
    elif source_type == "parquet":
        X_raw, y_raw = _load_parquet(schema, data_dir or DATA_DIR)
    else:
        raise ValueError(f"Unsupported source type: {source_type}")

    # Select only the features defined in the schema
    available = [f for f in schema.feature_names if f in X_raw.columns]
    missing = [f for f in schema.feature_names if f not in X_raw.columns]

    if missing:
        logger.warning("features_missing_from_source", missing=missing)

    X = X_raw[available].copy()

    logger.info(
        "dataset_loaded",
        dataset_id=schema.id,
        n_samples=len(X),
        n_features=X.shape[1],
        class_distribution=y_raw.value_counts().to_dict(),
    )

    return X, y_raw


def _load_uci(schema: DatasetSchema) -> tuple[pd.DataFrame, pd.Series]:
    """Load from the UCI ML Repository using ucimlrepo."""
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:
        raise RuntimeError(
            "ucimlrepo package is required for UCI datasets. "
            "Install it with: pip install ucimlrepo"
        ) from exc

    dataset_id = schema.source.uci_dataset_id
    if dataset_id is None:
        raise ValueError(f"UCI dataset ID not specified for {schema.id}")

    try:
        dataset = fetch_ucirepo(id=dataset_id)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to fetch UCI dataset (id={dataset_id}). "
            "Check your network connection."
        ) from exc

    X = dataset.data.features.copy()
    y_df = dataset.data.targets.copy()

    # Apply column mapping if defined
    if schema.source.column_mapping:
        X = X.rename(columns=schema.source.column_mapping)

    # Extract and map target
    y = _extract_target(y_df, schema)

    return X, y


def _load_csv(
    schema: DatasetSchema, data_dir: Path
) -> tuple[pd.DataFrame, pd.Series]:
    """Load from a CSV file."""
    filename = schema.source.filename
    if filename is None:
        raise ValueError(f"CSV filename not specified for {schema.id}")

    path = data_dir / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}. "
            f"Download it from: {schema.source.url or 'the dataset source'}"
        )

    df = pd.read_csv(path)
    logger.info(
        "csv_loaded", path=str(path), rows=len(df), columns=df.shape[1]
    )

    # Apply column mapping if defined
    if schema.source.column_mapping:
        df = df.rename(columns=schema.source.column_mapping)

    # Extract target
    target_col = schema.target.column
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {path}")

    y = _extract_target(df[[target_col]], schema)
    X = df.drop(columns=[target_col], errors="ignore")

    return X, y


def _load_parquet(
    schema: DatasetSchema, data_dir: Path
) -> tuple[pd.DataFrame, pd.Series]:
    """Load from a Parquet file."""
    filename = schema.source.filename
    if filename is None:
        raise ValueError(f"Parquet filename not specified for {schema.id}")

    path = data_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    df = pd.read_parquet(path)

    if schema.source.column_mapping:
        df = df.rename(columns=schema.source.column_mapping)

    target_col = schema.target.column
    y = _extract_target(df[[target_col]], schema)
    X = df.drop(columns=[target_col], errors="ignore")

    return X, y


def _extract_target(y_df: pd.DataFrame, schema: DatasetSchema) -> pd.Series:
    """Extract and map the target variable to binary 0/1.

    Args:
        y_df: DataFrame containing the target column.
        schema: Dataset schema with target mapping.

    Returns:
        Series named 'target' with binary 0/1 values.
    """
    # Use the first (or only) column
    col = y_df.columns[0]
    y = y_df[col].copy()

    # Apply value mapping if defined
    if schema.target.mapping:
        # Build a mapping with flexible key types (int/str matching)
        mapping = {}
        for k, v in schema.target.mapping.items():
            mapping[k] = v
            # Also try numeric conversion for robustness
            with contextlib.suppress(ValueError, TypeError):
                mapping[int(k)] = v
            with contextlib.suppress(ValueError, TypeError):
                mapping[float(k)] = v

        y = y.map(mapping)

    # Ensure binary int
    y = y.astype(int)
    y.name = "target"

    return y
