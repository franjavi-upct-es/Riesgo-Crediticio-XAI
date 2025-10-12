import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
from imblearn.over_sampling import SMOTE
import joblib
import os
import sys

# --- CONFIGURACIÓN DE NOMBRES ---
# Mapeo de nombres genéricos de UCI a nombres descriptivos para el proyecto
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


# --- CARGA Y PREPROCESAMIENTO ---
def load_and_preprocess_data():
    """Carga el dataset, mapea nombres, limpia y aplica One-Hot Encoding."""
    print("Iniciando carga y preprocesamiento de datos...")

    # 1. Cargar datos desde UCI
    try:
        statlog_german_credit_data = fetch_ucirepo(id=144)
        X = statlog_german_credit_data.data.features
        # Usamos .copy() para evitar SettingWithCopyWarning
        y = statlog_german_credit_data.data.targets.copy()
    except Exception as e:
        print(f"❌ ERROR al cargar datos de UCI: {e}")
        sys.exit(1)

    # 2. Renombrar columnas ANTES de la codificación
    X = X.rename(columns=COLUMN_MAPPING)

    # 3. Preparar la variable objetivo (Risk_Flag)
    y.columns = ['Risk_Flag']
    y['Risk_Flag'] = y['Risk_Flag'].map({1: 0, 2: 1})  # Mapeo: 1 (Buen) -> 0, 2 (Mal) -> 1

    # 4. Ingeniería de Características (One-Hot Encoding)
    categorical_cols = X.select_dtypes(include=['object']).columns
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    feature_names = X_encoded.columns.tolist()

    return X_encoded, y, feature_names


# --- ENTRENAMIENTO Y GUARDADO ---
def train_and_save_model(X_encoded, y, feature_names):
    """Entrena el modelo XGBoost y guarda los artefactos."""

    # Dividir el conjunto original para obtener una porción de prueba
    X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )

    # Ajuste por desequilibrio de clases original (70/30)
    scale_pos_weight_value = (y_train_orig.value_counts()[0] / y_train_orig.value_counts()[1])

    # Inicializar y entrenar XGBoost
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        n_estimators=100,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight_value
    )

    print("Entrenando modelo XGBoost con datos ORIGINALES...")
    model.fit(X_train_orig, y_train_orig)

    # Guardar artefactos
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/xgb_model.pkl')
    joblib.dump(feature_names, 'models/feature_names.pkl')
    print("✅ Modelo y Nombres de características guardados.")

    return X_test_orig, y_test_orig


# --- GENERACIÓN DEL CONJUNTO DE PRUEBA SINTÉTICO ---
def generate_synthetic_test_set(X_test_orig, y_test_orig):
    """Genera datos sintéticos a partir del conjunto de prueba original."""

    sm = SMOTE(sampling_strategy='minority', random_state=42)

    # Generar muestras sintéticas de la clase minoritaria (Riesgo=1)
    X_synthetic, y_synthetic = sm.fit_resample(X_test_orig, y_test_orig)

    # Verificar si y_synthetic es Series o DataFrame y manejar apropiadamente
    if isinstance(y_synthetic, pd.Series):
        y_synthetic_df = y_synthetic.to_frame(name='Risk_Flag')
    else:
        # Si ya es DataFrame, simplemente renombrar la columna si es necesario
        y_synthetic_df = y_synthetic.copy()
        if y_synthetic_df.columns[0] != 'Risk_Flag':
            y_synthetic_df.columns = ['Risk_Flag']
    
    # Crear un nuevo DataFrame con los datos sintéticos y originales de prueba
    synthetic_test_data = pd.concat([X_synthetic, y_synthetic_df], axis=1)

    os.makedirs('data', exist_ok=True)
    synthetic_test_data.to_csv('data/synthetic_test_set.csv', index=False)
    print("✅ Conjunto de prueba sintético (balanceado 50/50) guardado en data/synthetic_test_set.csv.")


# --- EJECUCIÓN PRINCIPAL ---
if __name__ == "__main__":
    X_encoded, y, feature_names = load_and_preprocess_data()
    X_test_orig, y_test_orig = train_and_save_model(X_encoded, y, feature_names)
    generate_synthetic_test_set(X_test_orig, y_test_orig)