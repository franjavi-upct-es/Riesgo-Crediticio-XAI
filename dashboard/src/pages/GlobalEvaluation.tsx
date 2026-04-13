// dashboard/src/pages/GlobalEvaluation.tsx

import { Target, TrendingUp, Crosshair, Percent } from "lucide-react";
import { useEvaluation } from "@/hooks/useApi";
import { useActiveDataset } from "@/hooks/useDatasetContext";
import MetricCard from "@/components/ui/MetricCard";
import { LoadingState, ErrorState } from "@/components/ui/StatusStates";
import RocCurveChart from "@/components/charts/RocCurveChart";
import ConfusionMatrixChart from "@/components/charts/ConfusionMatrixChart";
import ShapImportanceChart from "@/components/charts/ShapImportanceChart";
import DistributionChart from "@/components/charts/DistributionChart";

export default function GlobalEvaluation() {
  const { activeDatasetId } = useActiveDataset();
  const { data, isLoading, isError, error, refetch } = useEvaluation(activeDatasetId);

  if (!activeDatasetId) {
    return (
      <ErrorState
        title="No dataset selected"
        message="Select a dataset from the sidebar to view evaluation metrics."
      />
    );
  }

  if (isLoading) return <LoadingState message="Loading evaluation metrics…" />;
  if (isError || !data) {
    return (
      <ErrorState
        title="Evaluation data unavailable"
        message={
          (error as Error)?.message ??
          "Run the training pipeline to generate evaluation metrics."
        }
        onRetry={refetch}
      />
    );
  }

  const {
    metrics,
    confusion_matrix,
    roc_curve,
    shap_importance,
    prediction_distribution,
    dataset_info,
  } = data;

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-bold tracking-tight text-foreground">
          Global evaluation
        </h2>
        <p className="mt-1 text-sm text-muted-foreground">
          Model performance on the synthetic balanced test set (
          {dataset_info.n_samples.toLocaleString()} samples,{" "}
          {dataset_info.n_features} features)
        </p>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          label="AUC-ROC"
          value={metrics.auc.toFixed(4)}
          icon={<TrendingUp size={16} />}
          variant={
            metrics.auc > 0.8 ? "success" : metrics.auc > 0.6 ? "info" : "danger"
          }
        />
        <MetricCard
          label="F1 score"
          value={metrics.f1.toFixed(4)}
          icon={<Target size={16} />}
          variant={metrics.f1 > 0.7 ? "success" : "info"}
        />
        <MetricCard
          label="Precision"
          value={metrics.precision.toFixed(4)}
          icon={<Crosshair size={16} />}
          variant="default"
        />
        <MetricCard
          label="Recall"
          value={metrics.recall.toFixed(4)}
          icon={<Percent size={16} />}
          variant="default"
        />
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <RocCurveChart data={roc_curve} auc={metrics.auc} />
        <ConfusionMatrixChart data={confusion_matrix} />
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <ShapImportanceChart data={shap_importance} />
        <DistributionChart data={prediction_distribution} />
      </div>
    </div>
  );
}
