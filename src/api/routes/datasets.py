# src/api/routes/datasets.py
"""Dataset discovery and schema endpoints.

Allows the dashboard (or any client) to discover available datasets,
fetch their schemas for dynamic form rendering, and generate random
sample inputs for demo purposes.
"""

from typing import Any

import structlog
from fastapi import APIRouter, HTTPException

from src.api.dynamic_schema import build_defaults, build_random_sample
from src.data.schema import list_available_datasets, load_dataset_schema

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.get("/")
def list_datasets() -> dict[str, Any]:
    """List all available dataset configurations.

    Returns a summary of each dataset: id, name, description, and
    feature/target counts.
    """
    dataset_ids = list_available_datasets()
    datasets = []

    for ds_id in dataset_ids:
        try:
            schema = load_dataset_schema(ds_id)
            datasets.append(
                {
                    "id": schema.id,
                    "name": schema.name,
                    "description": schema.description,
                    "n_features": len(schema.features),
                    "n_categorical": len(schema.categorical_features),
                    "n_numerical": len(schema.numerical_features),
                    "n_protected": len(schema.protected_features),
                    "target_labels": schema.target.labels,
                }
            )
        except Exception as exc:
            logger.warning(
                "dataset_schema_load_failed", dataset_id=ds_id, error=str(exc)
            )

    return {"datasets": datasets, "count": len(datasets)}


@router.get("/{dataset_id}/schema")
def get_dataset_schema(dataset_id: str) -> dict[str, Any]:
    """Return the full schema for a dataset (for dynamic form rendering).

    The response includes all features with types, constraints, options,
    defaults, and protected-attribute flags.
    """
    try:
        schema = load_dataset_schema(dataset_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{dataset_id}' not found. Available: {list_available_datasets()}",
        ) from None

    return schema.to_api_schema()


@router.get("/{dataset_id}/defaults")
def get_dataset_defaults(dataset_id: str) -> dict[str, Any]:
    """Return default form values for a dataset."""
    try:
        schema = load_dataset_schema(dataset_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"Dataset '{dataset_id}' not found."
        ) from None

    return {"dataset_id": dataset_id, "defaults": build_defaults(schema)}


@router.get("/{dataset_id}/random")
def get_random_sample(dataset_id: str) -> dict[str, Any]:
    """Generate a random input sample within the schema constraints."""
    try:
        schema = load_dataset_schema(dataset_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"Dataset '{dataset_id}' not found."
        ) from None

    return {"dataset_id": dataset_id, "sample": build_random_sample(schema)}
