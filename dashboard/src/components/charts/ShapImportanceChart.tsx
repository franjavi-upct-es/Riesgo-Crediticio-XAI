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

export default function ShapImportanceChart({ data, maxFeatures = 12 }: Props) {
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
            stroke="#e2e8f0"
            horizontal={false}
          />
          <XAxis
            type="number"
            tick={{ fontSize: 11, fill: "#94a3b8" }}
            label={{
              value: "Mean |SHAP value|",
              position: "insideBottom",
              offset: -2,
              fontSize: 12,
              fill: "#64748b",
            }}
          />
          <YAxis
            dataKey="feature"
            type="category"
            width={160}
            tick={{ fontSize: 11, fill: "#475569" }}
          />
          <Tooltip
            formatter={(v: number) => v.toFixed(6)}
            contentStyle={{
              fontSize: 12,
              borderRadius: 8,
              border: "1px solid #e2e8f0",
            }}
          />
          <Bar
            dataKey="importance"
            fill="#7c3aed"
            radius={[0, 4, 4, 0]}
            barSize={16}
          />
        </BarChart>
      </ResponsiveContainer>
    </Card>
  );
}
