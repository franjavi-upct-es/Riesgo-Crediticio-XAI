# tests/conftest.py
"""Shared test fixtures for unit and integration tests.

Provides sample credit data payloads, mock model artifacts, and a
pre-configured FastAPI TestClient with dependency overrides.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.api.schemas import CreditDataRequest

# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------

VALID_CREDIT_PAYLOAD: dict = {
    "checking_status": "no_checking",
    "duration": 12,
    "credit_history": "critical/other",
    "purpose": "radio/television",
    "credit_amount": 5000,
    "savings_status": "no_savings",
    "employment": "unemployed",
    "installment_commitment": 4,
    "personal_status": "male single",
    "other_parties": "none",
    "residence_since": 4,
    "property_magnitude": "real estate",
    "age": 35,
    "other_payment_plans": "none",
    "housing": "own",
    "existing_credits": 1,
    "job": "skilled",
    "num_dependents": 1,
    "own_telephone": "yes",
    "foreign_worker": "no",
}


@pytest.fixture
def valid_payload() -> dict:
    """Return a valid credit data payload dictionary."""
    return VALID_CREDIT_PAYLOAD.copy()


@pytest.fixture
def valid_credit_request() -> CreditDataRequest:
    """Return a validated CreditDataRequest instance."""
    return CreditDataRequest(**VALID_CREDIT_PAYLOAD)


# ---------------------------------------------------------------------------
# Sample feature names (subset matching get_dummies output for the payload)
# ---------------------------------------------------------------------------

SAMPLE_FEATURE_NAMES: list[str] = [
    "duration",
    "credit_amount",
    "installment_commitment",
    "residence_since",
    "age",
    "existing_credits",
    "num_dependents",
    "checking_status_no_checking",
    "credit_history_critical/other",
    "purpose_radio/television",
    "savings_status_no_savings",
    "employment_unemployed",
    "personal_status_male single",
    "other_parties_none",
    "property_magnitude_real estate",
    "other_payment_plans_none",
    "housing_own",
    "job_skilled",
    "own_telephone_yes",
    "foreign_worker_no",
]


@pytest.fixture
def sample_feature_names() -> list[str]:
    """Return a sample list of feature names after one-hot encoding."""
    return SAMPLE_FEATURE_NAMES.copy()


# ---------------------------------------------------------------------------
# Mock model fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_xgb_model():
    """Return a mock XGBoost model with predict_proba support."""
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.7, 0.3]])
    model.n_features_in_ = len(SAMPLE_FEATURE_NAMES)
    return model
