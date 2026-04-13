// dashboard/src/types/api.ts

// --- Datasets ---

export interface FeatureDefinition {
  name: string;
  type: "numerical" | "categorical";
  description: string;
  options: string[];
  min: number | null;
  max: number | null;
  protected: boolean;
  default_value: string | number;
}

export interface DatasetSchemaResponse {
  id: string;
  name: string;
  description: string;
  features: FeatureDefinition[];
  target: {
    labels: Record<number, string>;
  };
}

export interface DatasetSummary {
  id: string;
  name: string;
  description: string;
  n_features: number;
  n_categorical: number;
  n_numerical: number;
  n_protected: number;
  target_labels: Record<number, string>;
}

export interface DatasetsListResponse {
  datasets: DatasetSummary[];
  count: number;
}

// --- Prediction ---

/**
 * Validated prediction payload sent to the API.
 *
 * Keys are dataset-driven (resolved at runtime from the schema), but
 * each value is narrowed to `number | string` by the Zod resolver before
 * the mutation fires — no `unknown` reaches the wire.
 */
export type PredictionPayload = Record<string, number | string>;

export interface RandomSampleResponse {
  dataset_id: string;
  sample: PredictionPayload;
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
  loaded_datasets: string[];
}
