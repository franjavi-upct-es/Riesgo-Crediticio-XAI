# tests/unit/test_schemas.py
"""Unit tests for API request/response schemas.

Verifies that business-rule validators (age bounds, credit amount
positivity, duration limits) reject invalid inputs and that valid
payloads pass validation cleanly.
"""

import pytest
from pydantic import ValidationError
from src.api.schemas import CreditDataRequest, PredictionResponse


class TestCreditDataRequest:
    """Validate business-rule enforcement on the request schema."""

    def test_valid_payload_passes(self, valid_payload):
        req = CreditDataRequest(**valid_payload)
        assert req.age == 35
        assert req.credit_amount == 5000

    def test_rejects_negative_credit_amount(self, valid_payload):
        valid_payload["credit_amount"] = -1000
        with pytest.raises(ValidationError) as exc_info:
            CreditDataRequest(**valid_payload)
        assert "credit_amount" in str(exc_info.value)

    def test_rejects_zero_credit_amount(self, valid_payload):
        valid_payload["credit_amount"] = 0
        with pytest.raises(ValidationError):
            CreditDataRequest(**valid_payload)

    def test_rejects_age_below_18(self, valid_payload):
        valid_payload["age"] = 17
        with pytest.raises(ValidationError) as exc_info:
            CreditDataRequest(**valid_payload)
        assert "age" in str(exc_info.value)

    def test_rejects_age_above_120(self, valid_payload):
        valid_payload["age"] = 121
        with pytest.raises(ValidationError):
            CreditDataRequest(**valid_payload)

    def test_accepts_edge_age_18(self, valid_payload):
        valid_payload["age"] = 18
        req = CreditDataRequest(**valid_payload)
        assert req.age == 18

    def test_rejects_duration_zero(self, valid_payload):
        valid_payload["duration"] = 0
        with pytest.raises(ValidationError):
            CreditDataRequest(**valid_payload)

    def test_rejects_duration_above_120(self, valid_payload):
        valid_payload["duration"] = 121
        with pytest.raises(ValidationError):
            CreditDataRequest(**valid_payload)

    def test_rejects_installment_commitment_out_of_range(self, valid_payload):
        valid_payload["installment_commitment"] = 5
        with pytest.raises(ValidationError):
            CreditDataRequest(**valid_payload)

    def test_rejects_missing_required_field(self, valid_payload):
        del valid_payload["age"]
        with pytest.raises(ValidationError):
            CreditDataRequest(**valid_payload)

    def test_model_dump_returns_all_fields(self, valid_payload):
        req = CreditDataRequest(**valid_payload)
        dumped = req.model_dump()
        assert set(dumped.keys()) == set(valid_payload.keys())

    def test_rejects_wrong_type(self, valid_payload):
        valid_payload["age"] = "thirty-five"
        with pytest.raises(ValidationError):
            CreditDataRequest(**valid_payload)


class TestPredictionResponse:
    """Validate response schema constraints."""

    def test_valid_response(self):
        resp = PredictionResponse(
            prediction="Low Risk (No Default)",
            probability_of_risk=0.25,
            xai_interpretation={  # type: ignore[arg-type]
                "base_risk_score": 0.35,
                "detailed_explanation": [],
            },
        )
        assert resp.status == "success"

    def test_rejects_probability_above_1(self):
        with pytest.raises(ValidationError):
            PredictionResponse(
                prediction="High Risk (Default)",
                probability_of_risk=1.5,
                xai_interpretation={  # type: ignore[arg-type]
                    "base_risk_score": 0.35,
                    "detailed_explanation": [],
                },
            )

    def test_rejects_negative_probability(self):
        with pytest.raises(ValidationError):
            PredictionResponse(
                prediction="Low Risk (No Default)",
                probability_of_risk=-0.1,
                xai_interpretation={  # type: ignore[arg-type]
                    "base_risk_score": 0.35,
                    "detailed_explanation": [],
                },
            )
