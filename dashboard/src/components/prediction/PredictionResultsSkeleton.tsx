// dashboard/src/components/prediction/PredictionResultsSkeleton.tsx

import Card from "@/components/ui/Card";
import { Skeleton } from "@/components/ui/skeleton";

/**
 * Placeholder for the results column while the prediction request is in flight.
 *
 * Mirrors the layout of `PredictionResults` (risk badge → SHAP chart → payload)
 * so the page doesn't reflow when real data arrives.
 */
export default function PredictionResultsSkeleton() {
  return (
    <div className="animate-fade-in space-y-5">
      <Card>
        <div className="flex flex-wrap items-center gap-4">
          <Skeleton className="h-10 w-40 rounded-full" />
          <Skeleton className="h-4 w-32" />
        </div>
      </Card>

      <Card
        title="Prediction explanation (SHAP)"
        subtitle="Computing feature contributions…"
      >
        <div className="space-y-2 py-2">
          {Array.from({ length: 8 }).map((_, i) => (
            <div key={i} className="flex items-center gap-3">
              <Skeleton className="h-3 w-32 shrink-0" />
              <Skeleton
                className="h-4"
                style={{ width: `${20 + ((i * 13) % 60)}%` }}
              />
            </div>
          ))}
        </div>
      </Card>

      <Card title="Submitted payload">
        <Skeleton className="h-24 w-full" />
      </Card>
    </div>
  );
}
