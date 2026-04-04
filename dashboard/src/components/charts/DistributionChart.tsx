// dashboard/src/components/charts/DistributionChart.tsx

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
import type { DistributionBin } from "@/types/api";

interface Props {
  data: DistributionBin[];
}

export default function DistributionChart({ data }: Props) {
  const chartData = data.map((bin) => ({
    label: `${bin.bin_start.toFixed(2)}`,
    count: bin.count,
    midpoint: (bin.bin_start + bin.bin_end) / 2,
  }));

  return (
    <Card
      title="Prediction distribution"
      subtitle="Risk probability histogram across the test set"
    >
      <ResponsiveContainer width="100%" height={260}>
        <BarChart
          data={chartData}
          margin={{ top: 5, right: 10, bottom: 5, left: 0 }}
        >
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="#e2e8f0"
            vertical={false}
          />
          <XAxis
            dataKey="label"
            tick={{ fontSize: 10, fill: "#94a3b8" }}
            label={{
              value: "Predicted probability",
              position: "insideBottom",
              offset: -2,
              fontSize: 12,
              fill: "#64748b",
            }}
          />
          <YAxis
            tick={{ fontSize: 11, fill: "#94a3b8" }}
            label={{
              value: "Count",
              angle: -90,
              position: "insideLeft",
              offset: 10,
              fontSize: 12,
              fill: "#64748b",
            }}
          />
          <Tooltip
            formatter={(v: number) => [v, "Samples"]}
            contentStyle={{
              fontSize: 12,
              borderRadius: 8,
              border: "1px solid #e2e8f0",
            }}
          />
          <ReferenceLine
            x="0.50"
            stroke="#94a3b8"
            strokeDasharray="4 4"
            label={{
              value: "θ=0.5",
              position: "top",
              fontSize: 10,
              fill: "#64748b",
            }}
          />
          <Bar dataKey="count" radius={[3, 3, 0, 0]} barSize={20}>
            {chartData.map((entry, idx) => (
              <Cell
                key={idx}
                fill={entry.midpoint > 0.5 ? "#fca5a5" : "#6ee7b7"}
                fillOpacity={0.85}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </Card>
  );
}
