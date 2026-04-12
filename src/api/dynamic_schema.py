# src/api/dynamic_schema.py
"""Dynamic Pydantic model generation from dataset schemas.

Generates FastAPI-compatible request validation models at runtime
based on the YAML dataset definitions. This eliminates the need for
hardcoded Pydantic models per dataset — adding a new dataset is just
adding a new YAML file.

Usage::

    from src.api.dynamic_schema import build_request_validator

    schema = load_dataset_schema("lending_club")
    validator = build_request_validator(schema)
    # validator is now a Pydantic BaseModel class
"""

from typing import Any

from pydantic import BaseModel, Field, create_model

from src.data.schema import DatasetSchema, FeatureSchema


def _feature_to_field(feature: FeatureSchema) -> tuple[type, Any]:
    """Convert a FeatureSchema to a Pydantic field definition.

    Returns:
        Tuple of (python_type, Field(...)) for create_model().
    """
    if feature.is_numerical:
        kwargs: dict[str, Any] = {"description": feature.description}
        if feature.min is not None:
            kwargs["ge"] = feature.min
        if feature.max is not None:
            kwargs["le"] = feature.max
        return (float, Field(**kwargs))

    # Categorical
    return (str, Field(description=feature.description))


def build_request_validator(schema: DatasetSchema) -> type[BaseModel]:
    """Dynamically create a Pydantic BaseModel for a dataset's input schema.

    Args:
        schema: Dataset schema with feature definitions.

    Returns:
        A Pydantic BaseModel class with validated fields matching the schema.
    """
    field_definitions: dict[str, Any] = {}

    for feature in schema.features:
        python_type, field_info = _feature_to_field(feature)
        field_definitions[feature.name] = (python_type, field_info)

    model_name = f"{schema.id.title().replace('_', '')}Request"
    return create_model(model_name, **field_definitions)


def build_defaults(schema: DatasetSchema) -> dict[str, Any]:
    """Build a dictionary of default values for all features.

    Used by the dashboard to populate the initial form state.
    """
    return {f.name: f.default_value for f in schema.features}


def build_random_sample(schema: DatasetSchema) -> dict[str, Any]:
    """Generate a random sample within the schema constraints.

    Used by the dashboard's 'Random' button for quick demos.
    """
    import random

    sample: dict[str, Any] = {}
    for f in schema.features:
        if f.is_categorical and f.options:
            sample[f.name] = random.choice(f.options)
        elif f.is_numerical:
            lo = f.min if f.min is not None else 0
            hi = f.max if f.max is not None else 100
            if isinstance(lo, float) or isinstance(hi, float):
                sample[f.name] = round(random.uniform(lo, hi), 2)
            else:
                sample[f.name] = random.randint(int(lo), int(hi))
        else:
            sample[f.name] = f.default_value

    return sample
