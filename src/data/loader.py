# src/data/loader.py
"""Dataset loading utilities.

Handles fetching the UCI Statlog German Credit dataset and applying
the standard column mapping for human-readable feature names.
"""

import pandas as pd
import structlog
from ucimlrepo import fetch_ucirepo

from src.config import settings
from src.data.preprocessing import COLUMN_MAPPING

logger = structlog.get_logger(__name__)


def load_uci_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Fetch the UCI German Credit dataset and return features + target.

    Returns:
        Tuple of (X, y) where:
            - X: DataFrame with human-readable column names (raw, before encoding).
            - y: Series named 'risk_flag' with binary labels (0 = good, 1 = bad).

    Raises:
        RuntimeError: If the dataset cannot be fetched from UCI.
    """
    dataset_id = settings.data.uci_dataset_id
    logger.info("loading_uci_dataset", dataset_id=dataset_id)

    try:
        dataset = fetch_ucirepo(id=dataset_id)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to fetch UCI dataset (id={dataset_id}). "
            "Check your network connection and the ucimlrepo package."
        ) from exc

    data = dataset.data
    if data is None or data.features is None or data.targets is None:
        raise RuntimeError(f"UCI dataset (id={dataset_id}) returned no data.")

    X = data.features.rename(columns=COLUMN_MAPPING)
    y = data.targets.copy()

    y.columns = ["risk_flag"]
    # UCI encoding: 1 = good credit, 2 = bad credit → remap to 0/1
    y["risk_flag"] = y["risk_flag"].map({1: 0, 2: 1})

    logger.info(
        "dataset_loaded",
        n_samples=len(X),
        n_features=X.shape[1],
        class_distribution=y["risk_flag"].value_counts().to_dict(),
    )

    return X, y["risk_flag"]
