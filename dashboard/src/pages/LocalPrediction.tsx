// dashboard/src/pages/LocalPrediction.tsx

import { useState } from "react";
import { Send } from "lucide-react";
import Card from "@/components/ui/Card";
import { LoadingState, ErrorState } from "@/components/ui/StatusStates";
import PredictionForm from "@/components/prediction/PredictionForm";
import PredictionResults from "@/components/prediction/PredictionResults";
import PredictionResultsSkeleton from "@/components/prediction/PredictionResultsSkeleton";
import { useDatasetSchema, usePrediction } from "@/hooks/useApi";
import { useActiveDataset } from "@/hooks/useDatasetContext";
import type { PredictionPayload, PredictionResponse } from "@/types/api";

export default function LocalPrediction() {
  const { activeDatasetId } = useActiveDataset();
  const {
    data: schema,
    isLoading: schemaLoading,
    isError: schemaError,
  } = useDatasetSchema(activeDatasetId);

  const mutation = usePrediction();
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [submittedPayload, setSubmittedPayload] =
    useState<PredictionPayload | null>(null);

  async function handleSubmit(payload: PredictionPayload) {
    if (!activeDatasetId) return;
    setSubmittedPayload(payload);
    try {
      const resp = await mutation.mutateAsync({
        data: payload,
        datasetId: activeDatasetId,
      });
      setResult(resp);
    } catch {
      setResult(null);
    }
  }

  if (!activeDatasetId) {
    return (
      <ErrorState
        title="No dataset selected"
        message="Select a dataset from the sidebar to start making predictions."
      />
    );
  }

  if (schemaLoading) return <LoadingState message="Loading dataset schema…" />;
  if (schemaError || !schema) {
    return (
      <ErrorState
        title="Schema unavailable"
        message={`Could not load schema for '${activeDatasetId}'.`}
      />
    );
  }

  const submitError = mutation.isError
    ? ((mutation.error as Error)?.message ?? "Prediction failed.")
    : null;

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-bold tracking-tight text-foreground">
          Local prediction
        </h2>
        <p className="mt-1 text-sm text-muted-foreground">
          {schema.name} — {schema.features.length} features · Submit an
          applicant profile for risk prediction with SHAP explanation
        </p>
      </div>

      <div className="grid gap-6 xl:grid-cols-5">
        <PredictionForm
          schema={schema}
          datasetId={activeDatasetId}
          isSubmitting={mutation.isPending}
          submitError={submitError}
          onSubmit={handleSubmit}
        />

        <div className="xl:col-span-3">
          {mutation.isPending ? (
            <PredictionResultsSkeleton />
          ) : result && submittedPayload ? (
            <PredictionResults
              result={result}
              submittedPayload={submittedPayload}
            />
          ) : (
            <Card>
              <div className="flex flex-col items-center justify-center py-16 text-muted-foreground">
                <Send size={32} className="mb-3 opacity-30" />
                <p className="text-sm">
                  Fill in the form and click <strong>Predict</strong> to see the
                  risk assessment and SHAP explanation.
                </p>
              </div>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
