// dashboard/src/pages/LocalPrediction.tsx

import { useState } from "react";
import { Dices, Loader2, Send } from "lucide-react";
import { clsx } from "clsx";
import { usePrediction } from "@/hooks/useApi";
import {
  CREDIT_FIELDS,
  buildDefaults,
  randomize,
} from "@/components/forms/creditFormConfig";
import RiskBadge from "@/components/ui/RiskBadge";
import Card from "@/components/ui/Card";
import ShapWaterfallChart from "@/components/charts/ShapWaterfallChart";
import type { CreditDataInput, PredictionResponse } from "@/types/api";

export default function LocalPrediction() {
  const [formData, setFormData] =
    useState<Record<string, string | number>>(buildDefaults());
  const [result, setResult] = useState<PredictionResponse | null>(null);

  const mutation = usePrediction();

  function updateField(name: string, value: string | number) {
    setFormData((prev) => ({ ...prev, [name]: value }));
  }

  function handleRandomize() {
    setFormData(randomize());
    setResult(null);
  }

  async function handleSubmit() {
    const payload: Record<string, string | number> = {};
    for (const field of CREDIT_FIELDS) {
      const raw = formData[field.name];
      payload[field.name] = field.type === "number" ? Number(raw) : String(raw);
    }

    const resp = await mutation.mutateAsync(
      payload as unknown as CreditDataInput,
    );
    setResult(resp);
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-xl font-bold tracking-tight text-slate-900">
          Local prediction
        </h2>
        <p className="mt-1 text-sm text-accent-muted">
          Submit an applicant profile and get a risk prediction with SHAP
          explanation
        </p>
      </div>

      <div className="grid gap-6 xl:grid-cols-5">
        {/* Form */}
        <Card className="xl:col-span-2" title="Applicant data">
          <div className="grid gap-3 sm:grid-cols-2">
            {CREDIT_FIELDS.map((field) => (
              <label key={field.name} className="block">
                <span className="mb-1 block text-xs font-medium text-accent-subtle">
                  {field.label}
                </span>
                {field.type === "select" ? (
                  <select
                    value={String(formData[field.name])}
                    onChange={(e) => updateField(field.name, e.target.value)}
                    className="w-full rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-sm text-slate-900 outline-none transition-colors focus:border-violet-400 focus:ring-2 focus:ring-violet-100"
                  >
                    {field.options?.map((opt) => (
                      <option key={opt} value={opt}>
                        {opt}
                      </option>
                    ))}
                  </select>
                ) : (
                  <input
                    type="number"
                    value={formData[field.name]}
                    onChange={(e) =>
                      updateField(field.name, Number(e.target.value))
                    }
                    min={field.min}
                    max={field.max}
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
              {/* Risk badge + probability */}
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

              {/* SHAP waterfall */}
              <ShapWaterfallChart
                factors={result.xai_interpretation.detailed_explanation}
                baseValue={result.xai_interpretation.base_risk_score}
                probability={result.probability_of_risk}
              />

              {/* Raw payload preview */}
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
