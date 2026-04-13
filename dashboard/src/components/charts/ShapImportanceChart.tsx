// dashboard/src/components/charts/ShapImportanceChart.tsx

import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import Card from "@/components/ui/Card";
import type { ShapImportance } from "@/types/api";

interface Props {
  data: ShapImportance[];
  maxFeatures?: number;
}

export default function ShapImportanceChart({
  data,
  maxFeatures = 12,
}: Props) {
  const sorted = [...data]
    .sort((a, b) => a.importance - b.importance)
    .slice(-maxFeatures);

  return (
    <Card
      title="Global feature importance"
      subtitle={`Top ${maxFeatures} features by mean |SHAP|`}
    >
      <ResponsiveContainer
        width="100%"
        height={Math.max(280, sorted.length * 28)}
      >
        <BarChart
          data={sorted}
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
              value: "Mean |SHAP value|",
              position: "insideBottom",
              offset: -2,
              fontSize: 12,
              fill: "hsl(var(--muted-foreground))",
            }}
          />
          <YAxis
            dataKey="feature"
            type="category"
            width={160}
            tick={{ fontSize: 11, fill: "hsl(var(--foreground))" }}
            stroke="hsl(var(--border))"
          />
          <Tooltip
            cursor={{ fill: "hsl(var(--muted))", opacity: 0.4 }}
            content={({ active, payload }) => {
              if (!active || !payload?.[0]) return null;
              const d = payload[0].payload as ShapImportance;
              return (
                <div className="surface-glass rounded-lg px-3 py-2 text-xs shadow-lg">
                  <p className="font-semibold text-foreground">{d.feature}</p>
                  <p className="text-muted-foreground">
                    Importance:{" "}
                    <span className="font-mono text-foreground">
                      {d.importance.toFixed(6)}
                    </span>
                  </p>
                </div>
              );
            }}
          />
          <Bar
            dataKey="importance"
            fill="hsl(var(--primary))"
            fillOpacity={0.8}
            radius={[0, 4, 4, 0]}
            barSize={16}
          />
        </BarChart>
      </ResponsiveContainer>
    </Card>
  );
}
