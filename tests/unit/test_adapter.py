# tests/unit/test_adapter.py
"""Unit tests for src.data.adapter.

Covers all three loaders (UCI, CSV, Parquet), target extraction with
various mapping types, feature selection, missing-feature warnings,
and error paths.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.adapter import (
    _extract_target,
    _load_csv,
    _load_parquet,
    _load_uci,
    load_dataset,
)
from src.data.schema import (
    DatasetSchema,
    FeatureSchema,
    SourceSchema,
    TargetSchema,
)


def _make_schema(
    source_type="csv",
    filename="test.csv",
    uci_id=None,
    column_mapping=None,
    target_col="label",
    target_mapping=None,
    features=None,
) -> DatasetSchema:
    return DatasetSchema(
        id="test",
        name="Test",
        description="",
        source=SourceSchema(
            type=source_type,
            filename=filename,
            uci_dataset_id=uci_id,
            column_mapping=column_mapping or {},
        ),
        target=TargetSchema(
            column=target_col,
            mapping=target_mapping or {"0": 0, "1": 1},
            positive_label=1,
            labels={0: "Good", 1: "Bad"},
        ),
        features=features
        or [
            FeatureSchema(name="age", type="numerical"),
            FeatureSchema(name="income", type="numerical"),
        ],
    )


class TestLoadDataset:
    def test_csv_source(self, tmp_path):
        df = pd.DataFrame(
            {"age": [25, 30], "income": [5000, 6000], "label": [0, 1]}
        )
        df.to_csv(tmp_path / "test.csv", index=False)

        schema = _make_schema()
        X, y = load_dataset(schema, data_dir=tmp_path)

        assert len(X) == 2
        assert list(X.columns) == ["age", "income"]
        assert list(y) == [0, 1]

    def test_unsupported_source_type(self):
        schema = _make_schema(source_type="hdf5")
        with pytest.raises(ValueError, match="Unsupported"):
            load_dataset(schema)

    def test_missing_feature_warning(self, tmp_path):
        df = pd.DataFrame({"age": [25], "label": [0]})
        df.to_csv(tmp_path / "test.csv", index=False)

        schema = _make_schema(
            features=[
                FeatureSchema(name="age", type="numerical"),
                FeatureSchema(name="missing_col", type="numerical"),
            ]
        )
        X, _y = load_dataset(schema, data_dir=tmp_path)
        assert "age" in X.columns
        assert "missing_col" not in X.columns

    def test_parquet_source(self, tmp_path):
        df = pd.DataFrame(
            {"age": [25, 30], "income": [5000, 6000], "label": [0, 1]}
        )
        df.to_parquet(tmp_path / "test.parquet")

        schema = _make_schema(source_type="parquet", filename="test.parquet")
        X, _y = load_dataset(schema, data_dir=tmp_path)
        assert len(X) == 2


class TestLoadCSV:
    def test_loads_csv_and_splits_target(self, tmp_path):
        df = pd.DataFrame({"age": [25], "income": [5000], "label": [0]})
        df.to_csv(tmp_path / "test.csv", index=False)

        schema = _make_schema()
        X, y = _load_csv(schema, tmp_path)
        assert "label" not in X.columns
        assert len(y) == 1

    def test_raises_when_no_filename(self, tmp_path):
        schema = _make_schema(filename=None)
        with pytest.raises(ValueError, match="filename"):
            _load_csv(schema, tmp_path)

    def test_raises_when_file_missing(self, tmp_path):
        schema = _make_schema(filename="nonexistent.csv")
        with pytest.raises(FileNotFoundError):
            _load_csv(schema, tmp_path)

    def test_raises_when_target_column_missing(self, tmp_path):
        df = pd.DataFrame({"age": [25], "income": [5000]})
        df.to_csv(tmp_path / "test.csv", index=False)

        schema = _make_schema(target_col="nonexistent")
        with pytest.raises(ValueError, match="Target column"):
            _load_csv(schema, tmp_path)

    def test_applies_column_mapping(self, tmp_path):
        df = pd.DataFrame({"A": [25], "B": [5000], "label": [0]})
        df.to_csv(tmp_path / "test.csv", index=False)

        schema = _make_schema(
            column_mapping={"A": "age", "B": "income"},
            features=[
                FeatureSchema(name="age", type="numerical"),
                FeatureSchema(name="income", type="numerical"),
            ],
        )
        X, _y = _load_csv(schema, tmp_path)
        assert "age" in X.columns
        assert "income" in X.columns


class TestLoadParquet:
    def test_loads_parquet(self, tmp_path):
        df = pd.DataFrame({"age": [25], "income": [5000], "label": [0]})
        df.to_parquet(tmp_path / "test.parquet")

        schema = _make_schema(source_type="parquet", filename="test.parquet")
        X, _y = _load_parquet(schema, tmp_path)
        assert len(X) == 1
        assert "label" not in X.columns

    def test_raises_when_no_filename(self, tmp_path):
        schema = _make_schema(source_type="parquet", filename=None)
        with pytest.raises(ValueError, match="filename"):
            _load_parquet(schema, tmp_path)

    def test_raises_when_file_missing(self, tmp_path):
        schema = _make_schema(source_type="parquet", filename="nope.parquet")
        with pytest.raises(FileNotFoundError):
            _load_parquet(schema, tmp_path)

    def test_applies_column_mapping(self, tmp_path):
        df = pd.DataFrame({"A": [25], "label": [0]})
        df.to_parquet(tmp_path / "test.parquet")

        schema = _make_schema(
            source_type="parquet",
            filename="test.parquet",
            column_mapping={"A": "age"},
            features=[FeatureSchema(name="age", type="numerical")],
        )
        X, _y = _load_parquet(schema, tmp_path)
        assert "age" in X.columns


class TestLoadUCI:
    @patch("ucimlrepo.fetch_ucirepo")
    def test_loads_uci_dataset(self, mock_fetch):
        mock_dataset = MagicMock()
        mock_dataset.data.features = pd.DataFrame({"A1": [1, 2], "A2": [3, 4]})
        mock_dataset.data.targets = pd.DataFrame({"target": [0, 1]})
        mock_fetch.return_value = mock_dataset

        schema = _make_schema(
            source_type="uci",
            uci_id=144,
            column_mapping={"A1": "age", "A2": "income"},
            target_col="target",
            target_mapping={"0": 0, "1": 1},
        )
        X, y = _load_uci(schema)
        assert "age" in X.columns
        assert len(y) == 2

    def test_raises_when_no_uci_id(self):
        schema = _make_schema(source_type="uci", uci_id=None)
        with pytest.raises(ValueError, match="UCI dataset ID"):
            _load_uci(schema)

    @patch("ucimlrepo.fetch_ucirepo", side_effect=Exception("network"))
    def test_raises_on_fetch_failure(self, mock_fetch):
        schema = _make_schema(source_type="uci", uci_id=999)
        with pytest.raises(RuntimeError, match="Failed to fetch"):
            _load_uci(schema)


class TestExtractTarget:
    def test_maps_string_keys(self):
        schema = _make_schema(target_mapping={"good": 0, "bad": 1})
        y_df = pd.DataFrame({"label": ["good", "bad", "good"]})
        y = _extract_target(y_df, schema)
        assert list(y) == [0, 1, 0]
        assert y.name == "target"

    def test_maps_integer_keys(self):
        schema = _make_schema(target_mapping={"1": 0, "2": 1})
        y_df = pd.DataFrame({"label": [1, 2, 1]})
        y = _extract_target(y_df, schema)
        assert list(y) == [0, 1, 0]

    def test_no_mapping_passthrough(self):
        schema = _make_schema(target_mapping={})
        y_df = pd.DataFrame({"label": [0, 1, 0]})
        y = _extract_target(y_df, schema)
        assert list(y) == [0, 1, 0]
