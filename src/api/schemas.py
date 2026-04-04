# src/api/schemas.py
"""Pydantic request and response schemas for the credit risk API.

Includes business-rule validation on input fields (e.g., age >= 18,
credit_amount > 0) beyond basic type coercion.
"""

from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class CreditDataRequest(BaseModel):
    """Input schema for a credit risk prediction.

    Field names match the UCI German Credit dataset after column mapping.
    Business-rule validators prevent nonsensical inputs that would produce
    misleading predictions.
    """

    checking_status: str = Field(..., description="Status of the existing checking account")
    duration: int = Field(..., ge=1, le=120, description="Credit duration in months (1-120)")
    credit_history: str = Field(..., description="Credit history category")
    purpose: str = Field(..., description="Purpose of the credit")
    credit_amount: int = Field(
        ...,
        gt=0,
        le=100_000_000,
        description="Credit amount in currency units",
    )
    savings_status: str = Field(..., description="Savings account/bonds status")
    employment: str = Field(..., description="Employment duration category")
    installment_commitment: int = Field(
        ...,
        ge=1,
        le=4,
        description="Installment rate as % of disposable income (1-4)",
    )
    personal_status: str = Field(..., description="Personal status and sex")
    other_parties: str = Field(..., description="Other debtors / guarantors")
    residence_since: int = Field(..., ge=1, le=4, description="Present residence since (1-4)")
    property_magnitude: str = Field(..., description="Property type")
    age: int = Field(..., ge=18, le=120, description="Age in years (18-120)")
    other_payment_plans: str = Field(..., description="Other installment plans")
    housing: str = Field(..., description="Housing status (rent/own/free)")
    existing_credits: int = Field(
        ...,
        ge=1,
        le=10,
        description="Number of existing credits at this bank (1-10)",
    )
    job: str = Field(..., description="Job category")
    num_dependents: int = Field(..., ge=1, le=10, description="Number of dependents (1-10)")
    own_telephone: str = Field(..., description="Whether the applicant has a telephone registered")
    foreign_worker: str = Field(..., description="Whether the applicant is a foreign worker")


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------


class ShapFactorResponse(BaseModel):
    """Single factor in the SHAP explanation."""

    factor: str
    risk_impact: Literal["increases", "reduces"]
    shap_magnitude: float
    input_value: float | int | str


class XAIInterpretation(BaseModel):
    """SHAP-based explanation of the prediction."""

    base_risk_score: float
    detailed_explanation: list[ShapFactorResponse]


class PredictionResponse(BaseModel):
    """Full prediction response including risk assessment and explanation."""

    prediction: str
    probability_of_risk: float = Field(..., ge=0.0, le=1.0)
    xai_interpretation: XAIInterpretation
    status: Literal["success"] = "success"


class HealthResponse(BaseModel):
    """Health check response."""

    status: Literal["healthy", "degraded"]
    model_loaded: bool
    version: str
