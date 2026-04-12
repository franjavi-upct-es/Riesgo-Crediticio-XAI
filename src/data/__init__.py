# src/data/__init__.py
"""Data loading, schema management, and preprocessing modules."""

from src.data.adapter import load_dataset
from src.data.preprocessing import (
    COLUMN_MAPPING,
    encode_features,
    preprocess_input,
)
from src.data.schema import (
    DatasetSchema,
    list_available_datasets,
    load_dataset_schema,
)

__all__ = [
    "COLUMN_MAPPING",
    "DatasetSchema",
    "encode_features",
    "list_available_datasets",
    "load_dataset",
    "load_dataset_schema",
    "preprocess_input",
]
