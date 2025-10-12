from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import shap
import numpy as np
from typing import List, Dict, Any
import logging
import traceback

# --- CONFIGURACIÓN Y CARGA DE ARTEFACTOS ---
app = FastAPI(
    title="API de Riesgo Crediticio Explicable (XAI)",
    description="Predicción de riesgo crediticio con justificación SHAP."
)

# Logging config
logging.basicConfig(level=logging.INFO)

# Cargar el modelo y los nombres de las características
try:
    model = joblib.load('models/xgb_model.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    explainer = shap.TreeExplainer(model)
except Exception:
    model = None
    feature_names = None
    explainer = None


# --- DEFINICIÓN DE ESTRUCTURA DE DATOS (Pydantic) ---
class CreditData(BaseModel):
    checking_status: str
    duration: int
    credit_history: str
    purpose: str
    credit_amount: int
    savings_status: str
    employment: str
    installment_commitment: int
    personal_status: str
    other_parties: str
    residence_since: int
    property_magnitude: str
    age: int
    other_payment_plans: str
    housing: str
    existing_credits: int
    job: str
    num_dependents: int
    own_telephone: str
    foreign_worker: str


def preprocess_input(input_data: CreditData, feature_names: List[str]):
    input_df = pd.DataFrame([input_data.dict()])
    original_categorical_cols = input_df.select_dtypes(include=['object']).columns
    input_encoded = pd.get_dummies(input_df, columns=original_categorical_cols, drop_first=True)

    final_features = pd.DataFrame(0, index=[0], columns=feature_names)
    for col in input_encoded.columns:
        if col in final_features.columns:
            final_features[col] = input_encoded[col].iloc[0]
    return final_features


@app.post("/predict_risk/", response_model=Dict[str, Any])
def predict_risk(data: CreditData):
    """
    Endpoint to predict risk and return SHAP explanation.
    Wrapped with robust handling for different SHAP output formats and logging.
    """
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Modelo no cargado. Ejecute model_train.py primero.")

        if feature_names is None:
            raise HTTPException(status_code=500, detail="feature_names no cargado. Ejecute model_train.py primero.")

        if explainer is None:
            raise HTTPException(status_code=500, detail="Explainer SHAP no inicializado. Re-entrene o regenere artefactos.")

        # Preprocess input
        X_processed = preprocess_input(data, feature_names)

        # Ensure processed shape matches expected features
        if X_processed.shape[1] != len(feature_names):
            raise HTTPException(status_code=500, detail=f"Incompatibilidad de características: se esperaban {len(feature_names)} features, pero se procesaron {X_processed.shape[1]}.")

        # Inference
        proba_risk = model.predict_proba(X_processed)[:, 1][0]
        prediction = 1 if proba_risk > 0.5 else 0

        # SHAP explanation: handle multiple possible return formats from explainer.shap_values
        shap_values_raw = explainer.shap_values(X_processed)
        # Determine shap_values as 2D array (n_samples, n_features) for the positive class when possible
        if isinstance(shap_values_raw, (list, tuple)):
            if len(shap_values_raw) > 1:
                shap_values = np.asarray(shap_values_raw[1])
            else:
                shap_values = np.asarray(shap_values_raw[0])
        elif isinstance(shap_values_raw, np.ndarray):
            if shap_values_raw.ndim == 3:
                # (n_classes, n_samples, n_features)
                shap_values = shap_values_raw[1] if shap_values_raw.shape[0] > 1 else shap_values_raw[0]
            elif shap_values_raw.ndim == 2:
                shap_values = shap_values_raw
            else:
                shap_values = np.asarray(shap_values_raw)[0]
        else:
            # fallback attempts
            try:
                shap_values = np.asarray(shap_values_raw)[1]
            except Exception:
                shap_values = np.asarray(shap_values_raw)[0]

        # Expected value handling
        expected_value_raw = explainer.expected_value
        if isinstance(expected_value_raw, (list, tuple, np.ndarray)):
            try:
                base_value = expected_value_raw[1] if len(expected_value_raw) > 1 else expected_value_raw[0]
            except Exception:
                base_value = expected_value_raw[0]
        else:
            base_value = expected_value_raw
        base_value_py = float(base_value)

        # Build explanation list
        explanation_list = []
        cols = X_processed.columns.tolist()
        # shap_values should be (n_samples, n_features); take first sample
        shap_row = np.asarray(shap_values)[0] if np.asarray(shap_values).ndim == 2 else np.asarray(shap_values)
        values_row = X_processed.values[0]

        # Ensure shap_row length matches cols
        if len(shap_row) != len(cols):
            # Try transposing or reshaping if mismatch
            shap_row = np.ravel(shap_row)[:len(cols)]

        for feature, shap_val, val in zip(cols, shap_row, values_row):
            # convert numpy scalars/arrays to native Python types
            if isinstance(val, np.generic):
                try:
                    val_py = val.item()
                except Exception:
                    try:
                        val_py = float(val)
                    except Exception:
                        val_py = str(val)
            elif isinstance(val, np.ndarray):
                try:
                    val_py = val.tolist()
                except Exception:
                    val_py = [x.item() if isinstance(x, np.generic) else x for x in val]
            else:
                val_py = val

            if abs(float(shap_val)) > 0.001:
                explanation_list.append({
                    "factor": feature,
                    "impacto_riesgo": "Aumenta" if float(shap_val) > 0 else "Reduce",
                    "magnitud_shap": round(float(shap_val), 4),
                    "valor_input": val_py
                })

        return {
            "prediction": "Alto Riesgo (Impago)" if prediction == 1 else "Bajo Riesgo (No Impago)",
            "probability_of_risk": round(float(proba_risk), 4),
            "interpretacion_xai": {
                "base_risk_score": round(base_value_py, 4),
                "explicacion_detallada": explanation_list
            },
            "status": "success"
        }

    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        logging.error("Error interno en /predict_risk/: %s\n%s", str(e), tb)
        raise HTTPException(status_code=500, detail=f"Error interno en la API: {str(e)}")
