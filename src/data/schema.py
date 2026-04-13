# src/data/schema.py
"""Dataset schema definitions loaded from YAML configuration.

Provides typed dataclasses that describe a dataset's features, target,
source, and metadata. The schema is the single source of truth for:
  - Which features exist and their types/constraints
  - How to load and preprocess the data
  - How to build API request validation
  - How to render the dashboard form
  - Which features are protected (for fairness auditing)

Schemas are loaded from YAML files in configs/datasets/.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog
import yaml

logger = structlog.get_logger(__name__)

DATASETS_DIR = Path("configs/datasets")


@dataclass(frozen=True)
class FeatureSchema:
    """Schema for a single feature column."""

    name: str
    type: str  # "numerical" or "categorical"
    description: str = ""
    options: list[str] = field(default_factory=list)
    min: float | None = None
    max: float | None = None
    protected: bool = False

    @property
    def is_categorical(self) -> bool:
        return self.type == "categorical"

    @property
    def is_numerical(self) -> bool:
        return self.type == "numerical"

    @property
    def default_value(self) -> Any:
        """Generate a sensible default value for form rendering."""
        if self.is_categorical and self.options:
            return self.options[0]
        if self.is_numerical:
            if self.min is not None and self.max is not None:
                return int((self.min + self.max) / 2)
            return 0
        return ""


@dataclass(frozen=True)
class TargetSchema:
    """Schema for the target/label column."""

    column: str
    mapping: dict[str, int] = field(default_factory=dict)
    positive_label: int = 1
    labels: dict[int, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SourceSchema:
    """How to load the raw data."""

    type: str  # "uci", "csv", "parquet"
    uci_dataset_id: int | None = None
    filename: str | None = None
    url: str | None = None
    kaggle_dataset: str | None = None
    kaggle_filename: str | None = None
    column_mapping: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetSchema:
    """Complete schema for a credit risk dataset.

    Loaded from a YAML file and used throughout the system to:
    - Load and validate raw data
    - Build preprocessing pipelines
    - Generate API request schemas
    - Render dashboard forms
    - Identify protected attributes for fairness auditing
    """

    id: str
    name: str
    description: str
    source: SourceSchema
    target: TargetSchema
    features: list[FeatureSchema]

    @property
    def feature_names(self) -> list[str]:
        """Ordered list of feature names."""
        return [f.name for f in self.features]

    @property
    def categorical_features(self) -> list[str]:
        """Names of categorical features."""
        return [f.name for f in self.features if f.is_categorical]

    @property
    def numerical_features(self) -> list[str]:
        """Names of numerical features."""
        return [f.name for f in self.features if f.is_numerical]

    @property
    def protected_features(self) -> list[str]:
        """Names of features marked as protected (for fairness)."""
        return [f.name for f in self.features if f.protected]

    def get_feature(self, name: str) -> FeatureSchema | None:
        """Look up a feature by name."""
        for f in self.features:
            if f.name == name:
                return f
        return None

    def to_api_schema(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict for the dashboard."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "features": [
                {
                    "name": f.name,
                    "type": f.type,
                    "description": f.description,
                    "options": f.options,
                    "min": f.min,
                    "max": f.max,
                    "protected": f.protected,
                    "default_value": f.default_value,
                }
                for f in self.features
            ],
            "target": {
                "labels": self.target.labels,
            },
        }


def load_dataset_schema(dataset_id: str, datasets_dir: Path | None = None) -> DatasetSchema:
    """Load a dataset schema from its YAML definition file.

    Args:
        dataset_id: Identifier matching the YAML filename (without extension).
        datasets_dir: Directory containing YAML files. Defaults to configs/datasets/.

    Returns:
        Parsed DatasetSchema.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        ValueError: If the YAML is malformed.
    """
    base_dir = datasets_dir or DATASETS_DIR
    path = base_dir / f"{dataset_id}.yml"

    if not path.exists():
        raise FileNotFoundError(f"Dataset schema not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if not raw:
        raise ValueError(f"Empty or invalid YAML in {path}")

    logger.info("dataset_schema_loaded", dataset_id=dataset_id, path=str(path))

    source_raw = raw.get("source", {})
    target_raw = raw.get("target", {})

    # Parse target mapping — YAML may give int or str keys
    target_mapping = {}
    for k, v in target_raw.get("mapping", {}).items():
        target_mapping[str(k)] = int(v)

    target_labels = {}
    for k, v in target_raw.get("labels", {}).items():
        target_labels[int(k)] = str(v)

    features = []
    for feat_raw in raw.get("features", []):
        features.append(
            FeatureSchema(
                name=feat_raw["name"],
                type=feat_raw["type"],
                description=feat_raw.get("description", ""),
                options=feat_raw.get("options", []),
                min=feat_raw.get("min"),
                max=feat_raw.get("max"),
                protected=feat_raw.get("protected", False),
            )
        )

    return DatasetSchema(
        id=raw["id"],
        name=raw["name"],
        description=raw.get("description", ""),
        source=SourceSchema(
            type=source_raw.get("type", "csv"),
            uci_dataset_id=source_raw.get("uci_dataset_id"),
            filename=source_raw.get("filename"),
            url=source_raw.get("url"),
            kaggle_dataset=source_raw.get("kaggle_dataset"),
            kaggle_filename=source_raw.get("kaggle_filename"),
            column_mapping=source_raw.get("column_mapping", {}),
        ),
        target=TargetSchema(
            column=target_raw.get("column", "target"),
            mapping=target_mapping,
            positive_label=target_raw.get("positive_label", 1),
            labels=target_labels,
        ),
        features=features,
    )


def list_available_datasets(datasets_dir: Path | None = None) -> list[str]:
    """List all available dataset IDs (from YAML filenames).

    Args:
        datasets_dir: Directory to scan. Defaults to configs/datasets/.

    Returns:
        Sorted list of dataset identifiers.
    """
    base_dir = datasets_dir or DATASETS_DIR
    if not base_dir.exists():
        return []
    return sorted(p.stem for p in base_dir.glob("*.yml"))
