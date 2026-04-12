// dashboard/src/pages/LocalPrediction.tsx

import { useEffect, useState } from "react";
import { Dices, Loader2, Send } from "lucide-react";
import { clsx } from "clsx";
import {
  useDatasetSchema,
  usePrediction,
  useRandomSample,
} from "@/hooks/useApi";
import { useActiveDataset } from "@/hooks/useDatasetContext";
import RiskBadge from "@/components/ui/RiskBadge";
import Card from "@/components/ui/Card";
import { LoadingState, ErrorState } from "@/components/ui/StatusStates";
import ShapWaterfallChart from "@/components/charts/ShapWaterfallChart";
import type { FeatureDefinition, PredictionResponse } from "@/types/api";

export default function LocalPrediction() {
  const { activeDatasetId } = useActiveDataset();
  const {
    data: schema,
    isLoading: schemaLoading,
    isError: schemaError,
  } = useDatasetSchema(activeDatasetId);

  const [formData, setFormData] = useState<Record<string, unknown>>({});
  const [result, setResult] = useState<PredictionResponse | null>(null);

  const mutation = usePrediction();
  const randomMutation = useRandomSample(activeDatasetId);

  // Initialize form with defaults when schema loads
  useEffect(() => {
    if (schema?.features) {
      const defaults: Record<string, unknown> = {};
      for (const f of schema.features) {
        defaults[f.name] = f.default_value;
      }
      setFormData(defaults);
      setResult(null);
    }
  }, [schema]);

  function updateField(name: string, value: unknown) {
    setFormData((prev) => ({ ...prev, [name]: value }));
  }

  async function handleRandomize() {
    if (!activeDatasetId) return;
    const resp = await randomMutation.mutateAsync();
    setFormData(resp.sample);
    setResult(null);
  }

  async function handleSubmit() {
    if (!activeDatasetId || !schema) return;
    const payload: Record<string, unknown> = {};
    for (const f of schema.features) {
      const raw = formData[f.name];
      payload[f.name] = f.type === "numerical" ? Number(raw) : String(raw);
    }
    const resp = await mutation.mutateAsync({
      data: payload,
      datasetId: activeDatasetId,
    });
    setResult(resp);
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

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-xl font-bold tracking-tight text-slate-900">
          Local prediction
        </h2>
        <p className="mt-1 text-sm text-accent-muted">
          {schema.name} — {schema.features.length} features · Submit an
          applicant profile for risk prediction with SHAP explanation
        </p>
      </div>

      <div className="grid gap-6 xl:grid-cols-5">
        {/* Dynamic form */}
        <Card className="xl:col-span-2" title="Applicant data">
          <div className="grid gap-3 sm:grid-cols-2">
            {schema.features.map((feature: FeatureDefinition) => (
              <label key={feature.name} className="block">
                <span className="mb-1 flex items-center gap-1 text-xs font-medium text-accent-subtle">
                  {feature.description || feature.name}
                  {feature.protected && (
                    <span className="rounded bg-amber-100 px-1 text-[9px] font-bold text-amber-700">
                      Protected
                    </span>
                  )}
                </span>
                {feature.type === "categorical" && feature.options.length > 0 ? (
                  <select
                    value={String(formData[feature.name] ?? "")}
                    onChange={(e) => updateField(feature.name, e.target.value)}
                    className="w-full rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-sm text-slate-900 outline-none transition-colors focus:border-violet-400 focus:ring-2 focus:ring-violet-100"
                  >
                    {feature.options.map((opt) => (
                      <option key={opt} value={opt}>
                        {opt}
                      </option>
                    ))}
                  </select>
                ) : (
                  <input
                    type="number"
                    value={String(formData[feature.name] ?? "")}
                    onChange={(e) =>
                      updateField(feature.name, Number(e.target.value))
                    }
                    min={feature.min ?? undefined}
                    max={feature.max ?? undefined}
                    className="w-full rounded-lg border border-slate-300 bg-white px-3 py-1.5 font-mono text-sm text-slate-900 outline-none transition-colors focus:border-violet-400 focus:ring-2 focus:ring-violet-100"
                  />
                )}
              </label>
            ))}
          </div>

          <div className="mt-5 flex gap-3">
            <button
              onClick={handleSubmit}
              disabled={mutation.isPending}
              className={clsx(
                "inline-flex items-center gap-2 rounded-lg px-5 py-2 text-sm font-semibold text-white transition-colors",
                mutation.isPending
                  ? "cursor-wait bg-slate-400"
                  : "bg-slate-900 hover:bg-slate-700 active:bg-slate-800",
              )}
            >
              {mutation.isPending ? (
                <Loader2 size={15} className="animate-spin" />
              ) : (
                <Send size={15} />
              )}
              Predict
            </button>
            <button
              onClick={handleRandomize}
              disabled={randomMutation.isPending}
              className="inline-flex items-center gap-2 rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-700 transition-colors hover:bg-slate-50"
            >
              <Dices size={15} />
              Random
            </button>
          </div>

          {mutation.isError && (
            <p className="mt-3 text-sm text-red-600">
              {(mutation.error as Error)?.message ?? "Prediction failed."}
            </p>
          )}
        </Card>

        {/* Result */}
        <div className="space-y-5 xl:col-span-3">
          {result ? (
            <>
              <Card>
                <div className="flex flex-wrap items-center gap-4">
                  <RiskBadge
                    label={result.prediction}
                    probability={result.probability_of_risk}
                  />
                  <div className="text-sm text-accent-subtle">
                    Probability:{" "}
                    <span className="font-mono font-semibold text-slate-900">
                      {(result.probability_of_risk * 100).toFixed(2)}%
                    </span>
                  </div>
                </div>
              </Card>

              <ShapWaterfallChart
                factors={result.xai_interpretation.detailed_explanation}
                baseValue={result.xai_interpretation.base_risk_score}
                probability={result.probability_of_risk}
              />

              <Card title="Submitted payload">
                <pre className="max-h-48 overflow-auto rounded-lg bg-slate-50 p-3 font-mono text-xs text-slate-700">
                  {JSON.stringify(formData, null, 2)}
                </pre>
              </Card>
            </>
          ) : (
            <Card>
              <div className="flex flex-col items-center justify-center py-16 text-accent-muted">
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
