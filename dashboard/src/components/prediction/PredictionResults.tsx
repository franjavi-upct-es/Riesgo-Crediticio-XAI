// dashboard/src/components/prediction/PredictionResults.tsx

import Card from "@/components/ui/Card";
import RiskBadge from "@/components/ui/RiskBadge";
import ShapWaterfallChart from "@/components/charts/ShapWaterfallChart";
import type { PredictionPayload, PredictionResponse } from "@/types/api";

interface Props {
  result: PredictionResponse;
  submittedPayload: PredictionPayload;
}

export default function PredictionResults({ result, submittedPayload }: Props) {
  return (
    <div className="animate-fade-in-up space-y-5">
      <Card>
        <div className="flex flex-wrap items-center gap-4">
          <RiskBadge
            label={result.prediction}
            probability={result.probability_of_risk}
          />
          <div className="text-sm text-muted-foreground">
            Probability:{" "}
            <span className="font-mono font-semibold text-foreground">
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
        <pre className="max-h-48 overflow-auto rounded-lg bg-muted p-3 font-mono text-xs text-foreground/80">
          {JSON.stringify(submittedPayload, null, 2)}
        </pre>
      </Card>
    </div>
  );
}
