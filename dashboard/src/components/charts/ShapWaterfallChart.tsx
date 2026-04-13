// dashboard/src/components/charts/ShapWaterfallChart.tsx

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import Card from "@/components/ui/Card";
import type { ShapFactor } from "@/types/api";

interface Props {
  factors: ShapFactor[];
  baseValue: number;
  probability: number;
}

// Softer, less-saturated palette
const COLOR_INCREASE = "#f87171"; // red-400
const COLOR_DECREASE = "#34d399"; // emerald-400

export default function ShapWaterfallChart({
  factors,
  baseValue,
  probability,
}: Props) {
  const sorted = [...factors].sort(
    (a, b) => Math.abs(b.shap_magnitude) - Math.abs(a.shap_magnitude),
  );

  const chartData = sorted.map((f) => ({
    name: f.factor,
    value: f.shap_magnitude,
    impact: f.risk_impact,
    inputValue: f.input_value,
  }));

  return (
    <Card
      title="Prediction explanation (SHAP)"
      subtitle={`Base risk score: ${baseValue.toFixed(4)} → Predicted probability: ${(probability * 100).toFixed(1)}%`}
    >
      {chartData.length === 0 ? (
        <p className="py-10 text-center text-sm text-muted-foreground">
          No significant factors for this prediction.
        </p>
      ) : (
        <ResponsiveContainer
          width="100%"
          height={Math.max(260, chartData.length * 32)}
        >
          <BarChart
            data={chartData}
            layout="vertical"
            margin={{ top: 5, right: 20, bottom: 5, left: 10 }}
          >
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="hsl(var(--border))"
              strokeOpacity={0.5}
              horizontal={false}
            />
            <XAxis
              type="number"
              tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
              stroke="hsl(var(--border))"
              label={{
                value: "SHAP value (impact on risk)",
                position: "insideBottom",
                offset: -2,
                fontSize: 12,
                fill: "hsl(var(--muted-foreground))",
              }}
            />
            <YAxis
              dataKey="name"
              type="category"
              width={160}
              tick={{ fontSize: 11, fill: "hsl(var(--foreground))" }}
              stroke="hsl(var(--border))"
            />
            <Tooltip
              cursor={{ fill: "hsl(var(--muted))", opacity: 0.4 }}
              content={({ active, payload }) => {
                if (!active || !payload?.[0]) return null;
                const d = payload[0].payload as (typeof chartData)[number];
                return (
                  <div className="surface-glass rounded-lg px-3 py-2 text-xs shadow-lg">
                    <p className="font-semibold text-foreground">{d.name}</p>
                    <p className="text-muted-foreground">
                      Input:{" "}
                      <span className="font-mono text-foreground">
                        {String(d.inputValue)}
                      </span>
                    </p>
                    <p
                      className={
                        d.value > 0
                          ? "font-medium text-red-500 dark:text-red-400"
                          : "font-medium text-emerald-600 dark:text-emerald-400"
                      }
                    >
                      SHAP: {d.value > 0 ? "+" : ""}
                      {d.value.toFixed(4)} ({d.impact} risk)
                    </p>
                  </div>
                );
              }}
            />
            <ReferenceLine
              x={0}
              stroke="hsl(var(--muted-foreground))"
              strokeOpacity={0.5}
              strokeWidth={1}
            />
            <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={18}>
              {chartData.map((entry, idx) => (
                <Cell
                  key={idx}
                  fill={entry.value > 0 ? COLOR_INCREASE : COLOR_DECREASE}
                  fillOpacity={0.85}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      )}
    </Card>
  );
}
