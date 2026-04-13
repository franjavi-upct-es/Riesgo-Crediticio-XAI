// dashboard/src/components/ui/RiskBadge.tsx

import { clsx } from "clsx";
import { ShieldAlert, ShieldCheck } from "lucide-react";

interface Props {
  label: string;
  probability: number;
}

export default function RiskBadge({ label, probability }: Props) {
  const isHigh = probability > 0.5;

  return (
    <div
      className={clsx(
        "inline-flex items-center gap-2.5 rounded-full px-5 py-2.5 text-sm font-semibold ring-1 transition-colors",
        isHigh
          ? "bg-red-100 text-red-800 ring-red-200 dark:bg-red-950/40 dark:text-red-200 dark:ring-red-900/60"
          : "bg-emerald-100 text-emerald-800 ring-emerald-200 dark:bg-emerald-950/40 dark:text-emerald-200 dark:ring-emerald-900/60",
      )}
    >
      {isHigh ? <ShieldAlert size={18} /> : <ShieldCheck size={18} />}
      {label}
      <span className="font-mono text-xs opacity-70">
        {(probability * 100).toFixed(1)}%
      </span>
    </div>
  );
}
