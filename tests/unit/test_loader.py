# tests/unit/test_loader.py
"""Unit tests for src.data.loader (legacy UCI loader)."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.loader import load_uci_dataset


class TestLoadUCIDataset:
    @patch("src.data.loader.fetch_ucirepo")
    def test_loads_and_maps_columns(self, mock_fetch):
        mock_dataset = MagicMock()
        mock_dataset.data.features = pd.DataFrame(
            {
                "Attribute1": ["a"],
                "Attribute2": [12],
                "Attribute3": ["b"],
                "Attribute4": ["c"],
                "Attribute5": [5000],
                "Attribute6": ["d"],
                "Attribute7": ["e"],
                "Attribute8": [4],
                "Attribute9": ["f"],
                "Attribute10": ["g"],
                "Attribute11": [1],
                "Attribute12": ["h"],
                "Attribute13": [35],
                "Attribute14": ["i"],
                "Attribute15": ["j"],
                "Attribute16": [1],
                "Attribute17": ["k"],
                "Attribute18": [1],
                "Attribute19": ["l"],
                "Attribute20": ["m"],
            }
        )
        mock_dataset.data.targets = pd.DataFrame({"class": [1]})
        mock_fetch.return_value = mock_dataset

        X, y = load_uci_dataset()

        assert "checking_status" in X.columns
        assert "age" in X.columns
        assert len(X) == 1
        assert y.iloc[0] == 0  # 1 maps to 0

    @patch("src.data.loader.fetch_ucirepo")
    def test_target_mapping(self, mock_fetch):
        mock_dataset = MagicMock()
        mock_dataset.data.features = pd.DataFrame(
            {
                f"Attribute{i}": [f"v{i}"]
                if i in [1, 3, 4, 6, 7, 9, 10, 12, 14, 15, 17, 19, 20]
                else [i]
                for i in range(1, 21)
            }
        )
        mock_dataset.data.targets = pd.DataFrame({"class": [2]})
        mock_fetch.return_value = mock_dataset

        _, y = load_uci_dataset()
        assert y.iloc[0] == 1  # 2 maps to 1

    @patch(
        "src.data.loader.fetch_ucirepo", side_effect=Exception("Network error")
    )
    def test_raises_on_fetch_failure(self, mock_fetch):
        with pytest.raises(RuntimeError, match="Failed to fetch"):
            load_uci_dataset()
