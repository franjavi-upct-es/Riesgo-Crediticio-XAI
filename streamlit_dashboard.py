import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
import requests
import json
import time
from ucimlrepo import fetch_ucirepo

# --- CONFIGURACIÓN Y CARGA DE DATOS ---
st.set_page_config(layout="wide", page_title="Riesgo Crediticio XAI")
API_URL = "http://127.0.0.1:8000/predict_risk/"

# Mapeo de nombres (IDÉNTICO a model_train.py para asegurar la consistencia)
COLUMN_MAPPING = {
    'Attribute1': 'checking_status', 'Attribute2': 'duration',
    'Attribute3': 'credit_history', 'Attribute4': 'purpose',
    'Attribute5': 'credit_amount', 'Attribute6': 'savings_status',
    'Attribute7': 'employment', 'Attribute8': 'installment_commitment',
    'Attribute9': 'personal_status', 'Attribute10': 'other_parties',
    'Attribute11': 'residence_since', 'Attribute12': 'property_magnitude',
    'Attribute13': 'age', 'Attribute14': 'other_payment_plans',
    'Attribute15': 'housing', 'Attribute16': 'existing_credits',
    'Attribute17': 'job', 'Attribute18': 'num_dependents',
    'Attribute19': 'own_telephone', 'Attribute20': 'foreign_worker',
}

# Campos que deben ser enteros para el payload JSON de la API
INT_FIELDS = ['duration', 'credit_amount', 'installment_commitment',
              'residence_since', 'age', 'existing_credits', 'num_dependents']


@st.cache_data
def load_data_artifacts():
    """Carga todos los artefactos necesarios."""
    try:
        model = joblib.load('models/xgb_model.pkl')

        synthetic_test_data = pd.read_csv('data/synthetic_test_set.csv')
        X_test_synthetic = synthetic_test_data.drop('Risk_Flag', axis=1)
        y_test_synthetic = synthetic_test_data['Risk_Flag']

        statlog_german_credit_data = fetch_ucirepo(id=144)
        X_orig_unencoded = statlog_german_credit_data.data.features
        X_orig_unencoded = X_orig_unencoded.rename(columns=COLUMN_MAPPING)

        return model, X_test_synthetic, y_test_synthetic, X_orig_unencoded
    except Exception as e:
        st.error(f"Error al cargar archivos. Asegúrese de que run_all.py se ejecutó. Error: {e}")
        return None, None, None, None


model, X_test_synthetic, y_test_synthetic, X_orig_unencoded = load_data_artifacts()

if model is None:
    st.stop()


@st.cache_data
def perform_evaluation(_model, X, y):
    y_pred_proba = _model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_pred_proba)
    f1 = f1_score(y, y_pred_proba > 0.5)
    conf_matrix = confusion_matrix(y, y_pred_proba > 0.5)

    explainer = shap.TreeExplainer(_model)
    shap_values = explainer.shap_values(X)
    return auc, f1, conf_matrix, shap_values


def plot_shap_summary(shap_values, X):
    st.subheader("Gráfico de Importancia Global de Variables (Top 10)")

    # Calcular la importancia media absoluta de cada feature
    shap_importance = pd.DataFrame({
        'Feature': X.columns,
        'SHAP_Importance': np.abs(shap_values).mean(axis=0)
    }).sort_values(by='SHAP_Importance', ascending=False).head(10)

    # Crear gráfico de barras interactivo con Plotly
    fig = px.bar(
        shap_importance.sort_values('SHAP_Importance', ascending=True),
        x='SHAP_Importance',
        y='Feature',
        orientation='h',
        color='SHAP_Importance',
        color_continuous_scale='Reds',
        title="Importancia Media Absoluta (Top 10 Variables)"
    )

    fig.update_layout(
        xaxis_title="Media |SHAP|",
        yaxis_title="Variable",
        title_x=0.5,
        margin=dict(l=80, r=40, t=60, b=40),
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_local_explanation(explanation_list, proba_risk):
    if not explanation_list:
        st.info("No hay factores significativos para la explicación local.")
        return

    df = pd.DataFrame(explanation_list)
    df = df.sort_values('magnitud_shap', ascending=True)

    st.subheader("Justificación de la Decisión (SHAP Local)")

    fig = px.bar(
        df, x='magnitud_shap', y='factor',
        color='impacto_riesgo', orientation='h',
        title=f"Impacto de cada factor en la Probabilidad de Riesgo ({proba_risk * 100:.2f}%)",
        color_discrete_map={'Aumenta': 'red', 'Reduce': 'blue'}
    )
    st.plotly_chart(fig, use_container_width=True)


def page_global_evaluation():
    st.title("🛡️ 1. Evaluación Global (Datos Sintéticos)")
    auc, f1, conf_matrix, shap_values = perform_evaluation(model, X_test_synthetic, y_test_synthetic)

    col1, col2 = st.columns(2)
    col1.metric("AUC Score", f"{auc:.4f}")
    col2.metric("F1 Score", f"{f1:.4f}")

    st.markdown("---")
    plot_shap_summary(shap_values, X_test_synthetic)
    st.markdown("---")

    st.subheader("Matriz de Confusión (Datos Sintéticos Balanceados)")
    cm_df = pd.DataFrame(conf_matrix,
                         index=['Real No Riesgo (0)', 'Real Riesgo (1)'],
                         columns=['Predicho No Riesgo (0)', 'Predicho Riesgo (1)'])
    fig_cm = px.imshow(cm_df, text_auto=True, color_continuous_scale='Blues')
    st.plotly_chart(fig_cm, use_container_width=False)


def page_local_prediction():
    st.title("👤 2. Predicción en Tiempo Real (Demo Aleatoria)")
    st.info(
        f"El dashboard selecciona un caso aleatorio del dataset original y lo envía a la API de FastAPI ({API_URL}) para demostrar la justificación de la decisión.")
    st.markdown("---")

    if 'last_payload' not in st.session_state:
        st.session_state.last_payload = None
        st.session_state.last_result = None

    if st.button("Generar Caso y Predecir", type="primary"):

        random_case_df = X_orig_unencoded.sample(n=1, random_state=int(time.time()))
        payload = random_case_df.iloc[0].to_dict()

        # ---------- Conversión robusta de tipos NumPy a nativos ----------
        def _to_native(v):
            if isinstance(v, np.generic):
                try:
                    return v.item()
                except Exception:
                    try:
                        return float(v)
                    except Exception:
                        return str(v)
            if isinstance(v, np.ndarray):
                return v.tolist()
            return v

        payload = {k: _to_native(v) for k, v in payload.items()}

        for field in INT_FIELDS:
            if field in payload and payload[field] is not None:
                try:
                    payload[field] = int(payload[field])
                except Exception:
                    pass

        try:
            response = requests.post(API_URL, json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()
            st.session_state.last_payload = payload
            st.session_state.last_result = result
            st.success(f"Predicción recibida con éxito (Status: {result.get('status', 'OK')})")

            if 'interpretacion_xai' in result and 'explicacion_detallada' in result['interpretacion_xai']:
                explanation_data = result['interpretacion_xai']['explicacion_detallada']
                plot_local_explanation(explanation_data, result.get('probability_of_risk'))

        except requests.exceptions.ConnectionError:
            st.error("❌ ERROR: No se pudo conectar a la API. Asegúrese de que run_all.py la haya iniciado.")
            return
        except requests.exceptions.Timeout:
            st.error("❌ ERROR: Tiempo de espera agotado al contactar la API.")
            return
        except requests.exceptions.HTTPError as e:
            st.error(f"❌ ERROR de la API: {e}. Revise el log de FastAPI para el detalle.")
            return
        except Exception as e:
            st.error(f"❌ ERROR inesperado: {e}")
            return

    if st.session_state.last_result:
        result = st.session_state.last_result
        payload = st.session_state.last_payload

        col_payload, col_result = st.columns([1, 2])
        with col_payload:
            st.markdown("### Datos del Solicitante Enviados:")
            st.json(payload)

        with col_result:
            st.markdown(f"## 🛑 Predicción: **{result['prediction']}**")
            st.markdown(f"### Probabilidad de Riesgo: **{result['probability_of_risk'] * 100:.2f}%**")

    else:
        st.info("Presione el botón para generar una predicción de demostración de un cliente aleatorio.")


page = st.sidebar.radio("Seleccione el Modo", ["Evaluación Global", "Predicción Local"])
if page == "Evaluación Global":
    page_global_evaluation()
else:
    page_local_prediction()
