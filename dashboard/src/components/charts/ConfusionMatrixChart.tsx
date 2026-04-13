// dashboard/src/components/charts/ConfusionMatrixChart.tsx

import Card from "@/components/ui/Card";
import { clsx } from "clsx";
import type { ConfusionMatrixData } from "@/types/api";

interface Props {
  data: ConfusionMatrixData;
}

export default function ConfusionMatrixChart({ data }: Props) {
  const { matrix, labels } = data;
  const max = Math.max(...matrix.flat());

  return (
    <Card title="Confusion matrix" subtitle="Synthetic balanced test set">
      <div className="flex justify-center">
        <table className="border-collapse text-sm">
          <thead>
            <tr>
              <th className="p-2" />
              {labels.map((l) => (
                <th
                  key={l}
                  className="px-3 py-2 text-center text-xs font-medium text-muted-foreground"
                >
                  Pred: {l}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, i) => (
              <tr key={i}>
                <td className="px-3 py-2 text-right text-xs font-medium text-muted-foreground">
                  True: {labels[i]}
                </td>
                {row.map((cell, j) => {
                  const intensity = max > 0 ? cell / max : 0;
                  const isDiagonal = i === j;
                  return (
                    <td key={j} className="p-1">
                      <div
                        className={clsx(
                          "flex h-16 w-24 items-center justify-center rounded-lg font-mono text-lg font-semibold transition-colors",
                          isDiagonal
                            ? "text-emerald-900 dark:text-emerald-200"
                            : "text-red-900 dark:text-red-200",
                        )}
                        style={{
                          backgroundColor: isDiagonal
                            ? `rgba(52, 211, 153, ${0.12 + intensity * 0.4})`
                            : `rgba(248, 113, 113, ${0.08 + intensity * 0.3})`,
                        }}
                      >
                        {cell}
                      </div>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}
