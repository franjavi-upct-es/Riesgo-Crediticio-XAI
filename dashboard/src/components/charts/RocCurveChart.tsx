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
        <AreaChart
          data={data}
          margin={{ top: 5, right: 10, bottom: 5, left: 0 }}
        >
          <defs>
            <linearGradient id="rocGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#0ea5e9" stopOpacity={0.2} />
              <stop offset="100%" stopColor="#0ea5e9" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis
            dataKey="fpr"
            type="number"
            domain={[0, 1]}
            tickFormatter={(v: number) => v.toFixed(1)}
            label={{
              value: "False positive rate",
              position: "insideBottom",
              offset: -2,
              fontSize: 12,
              fill: "#64748b",
            }}
            tick={{ fontSize: 11, fill: "#94a3b8" }}
          />
          <YAxis
            dataKey="tpr"
            type="number"
            domain={[0, 1]}
            tickFormatter={(v: number) => v.toFixed(1)}
            label={{
              value: "True positive rate",
              angle: -90,
              position: "insideLeft",
              offset: 10,
              fontSize: 12,
              fill: "#64748b",
            }}
            tick={{ fontSize: 11, fill: "#94a3b8" }}
          />
          <Tooltip
            formatter={(v: number) => v.toFixed(4)}
            labelFormatter={(l: number) => `FPR: ${l.toFixed(4)}`}
            contentStyle={{
              fontSize: 12,
              borderRadius: 8,
              border: "1px solid #e2e8f0",
            }}
          />
          <ReferenceLine
            segment={[
              { x: 0, y: 0 },
              { x: 1, y: 1 },
            ]}
            stroke="#cbd5e1"
            strokeDasharray="4 4"
          />
          <Area
            type="monotone"
            dataKey="tpr"
            stroke="#0ea5e9"
            strokeWidth={2}
            fill="url(#rocGrad)"
            dot={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </Card>
  );
}
