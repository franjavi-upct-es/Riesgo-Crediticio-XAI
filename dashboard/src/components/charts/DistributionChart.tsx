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
            stroke="hsl(var(--border))"
            strokeOpacity={0.5}
            vertical={false}
          />
          <XAxis
            dataKey="label"
            tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
            stroke="hsl(var(--border))"
            label={{
              value: "Predicted probability",
              position: "insideBottom",
              offset: -2,
              fontSize: 12,
              fill: "hsl(var(--muted-foreground))",
            }}
          />
          <YAxis
            tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
            stroke="hsl(var(--border))"
            label={{
              value: "Count",
              angle: -90,
              position: "insideLeft",
              offset: 10,
              fontSize: 12,
              fill: "hsl(var(--muted-foreground))",
            }}
          />
          <Tooltip
            cursor={{ fill: "hsl(var(--muted))", opacity: 0.4 }}
            content={({ active, payload, label }) => {
              if (!active || !payload?.[0]) return null;
              const count = payload[0].value as number;
              return (
                <div className="surface-glass rounded-lg px-3 py-2 text-xs shadow-lg">
                  <p className="text-muted-foreground">
                    Bin start:{" "}
                    <span className="font-mono text-foreground">{label}</span>
                  </p>
                  <p className="text-muted-foreground">
                    Samples:{" "}
                    <span className="font-mono text-foreground">{count}</span>
                  </p>
                </div>
              );
            }}
          />
          <ReferenceLine
            x="0.50"
            stroke="hsl(var(--muted-foreground))"
            strokeOpacity={0.5}
            strokeDasharray="4 4"
            label={{
              value: "θ=0.5",
              position: "top",
              fontSize: 10,
              fill: "hsl(var(--muted-foreground))",
            }}
          />
          <Bar dataKey="count" radius={[3, 3, 0, 0]} barSize={20}>
            {chartData.map((entry, idx) => (
              <Cell
                key={idx}
                fill={entry.midpoint > 0.5 ? "#fca5a5" : "#86efac"}
                fillOpacity={0.8}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </Card>
  );
}
