// dashboard/src/types/api.ts

// --- Prediction ---

export interface CreditDataInput {
  checking_status: string;
  duration: number;
  credit_history: string;
  purpose: string;
  credit_amount: number;
  saving_status: string;
  employment: string;
  installment_commitment: number;
  personal_status: string;
  other_parties: string;
  residence_since: number;
  property_magnitude: string;
  age: number;
  other_payment_plans: string;
  housing: string;
  existing_credits: number;
  job: string;
  num_dependents: number;
  own_telephone: string;
  foreign_worker: string;
}

export interface ShapFactor {
  factor: string;
  risk_impact: "increases" | "reduces";
  shap_magnitude: number;
  input_value: number | string;
}

export interface XAIInterpretation {
  base_risk_score: number;
  detailed_explanation: ShapFactor[];
}

export interface PredictionResponse {
  prediction: string;
  probability_of_risk: number;
  xai_interpretation: XAIInterpretation;
  status: "success";
}

// --- Evaluation ---

export interface EvaluationMetrics {
  auc: number;
  f1: number;
  precision: number;
  recall: number;
}

export interface ConfusionMatrixData {
  matrix: number[][];
  labels: string[];
}

export interface RocPoint {
  fpr: number;
  tpr: number;
}

export interface DistributionBin {
  bin_start: number;
  bin_end: number;
  count: number;
}

export interface ShapImportance {
  feature: string;
  importance: number;
}

export interface DatasetInfo {
  n_samples: number;
  n_features: number;
  class_distribution: Record<string, number>;
}

export interface FullEvaluation {
  metrics: EvaluationMetrics;
  confusion_matrix: ConfusionMatrixData;
  roc_curve: RocPoint[];
  prediction_distribution: DistributionBin[];
  shap_importance: ShapImportance[];
  dataset_info: DatasetInfo;
}

// --- Health ---

export interface HealthResponse {
  status: "healthy" | "degraded";
  model_loaded: boolean;
  version: string;
}
