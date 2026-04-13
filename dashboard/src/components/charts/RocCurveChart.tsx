// dashboard/src/components/charts/RocCurveChart.tsx

import {
  Area,
  AreaChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import Card from "@/components/ui/Card";
import type { RocPoint } from "@/types/api";

interface Props {
  data: RocPoint[];
  auc: number;
}

export default function RocCurveChart({ data, auc }: Props) {
  return (
    <Card title="ROC curve" subtitle={`AUC = ${auc.toFixed(4)}`}>
      <ResponsiveContainer width="100%" height={300}>
        <AreaChart data={data} margin={{ top: 5, right: 10, bottom: 5, left: 0 }}>
          <defs>
            <linearGradient id="rocGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#38bdf8" stopOpacity={0.35} />
              <stop offset="100%" stopColor="#38bdf8" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="hsl(var(--border))"
            strokeOpacity={0.5}
          />
          <XAxis
            dataKey="fpr"
            type="number"
            domain={[0, 1]}
            tickFormatter={(v: number) => v.toFixed(1)}
            stroke="hsl(var(--border))"
            label={{
              value: "False positive rate",
              position: "insideBottom",
              offset: -2,
              fontSize: 12,
              fill: "hsl(var(--muted-foreground))",
            }}
            tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
          />
          <YAxis
            dataKey="tpr"
            type="number"
            domain={[0, 1]}
            tickFormatter={(v: number) => v.toFixed(1)}
            stroke="hsl(var(--border))"
            label={{
              value: "True positive rate",
              angle: -90,
              position: "insideLeft",
              offset: 10,
              fontSize: 12,
              fill: "hsl(var(--muted-foreground))",
            }}
            tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
          />
          <Tooltip
            cursor={{ stroke: "hsl(var(--muted-foreground))", strokeOpacity: 0.4 }}
            content={({ active, payload, label }) => {
              if (!active || !payload?.[0]) return null;
              const tpr = payload[0].value as number;
              return (
                <div className="surface-glass rounded-lg px-3 py-2 text-xs shadow-lg">
                  <p className="text-muted-foreground">
                    FPR:{" "}
                    <span className="font-mono text-foreground">
                      {Number(label).toFixed(4)}
                    </span>
                  </p>
                  <p className="text-muted-foreground">
                    TPR:{" "}
                    <span className="font-mono text-foreground">
                      {tpr.toFixed(4)}
                    </span>
                  </p>
                </div>
              );
            }}
          />
          <ReferenceLine
            segment={[
              { x: 0, y: 0 },
              { x: 1, y: 1 },
            ]}
            stroke="hsl(var(--muted-foreground))"
            strokeOpacity={0.4}
            strokeDasharray="4 4"
          />
          <Area
            type="monotone"
            dataKey="tpr"
            stroke="#38bdf8"
            strokeWidth={2}
            fill="url(#rocGrad)"
            dot={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </Card>
  );
}
