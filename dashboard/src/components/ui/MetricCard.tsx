// dashboard/src/components/ui/MetricCard.tsx

import { clsx } from "clsx";
import type { ReactNode } from "react";

interface Props {
  label: string;
  value: string | number;
  subtitle?: string;
  icon?: ReactNode;
  variant?: "default" | "success" | "danger" | "info";
}

const variantStyles = {
  default: "border-slate-200 bg-white",
  success: "border-emerald-200 bg-emerald-50/50",
  danger: "border-red-200 bg-red-50/50",
  info: "border-sky-200 bg-sky-50/50",
};

const valueStyles = {
  default: "text-slate-900",
  success: "text-emerald-700",
  danger: "text-red-700",
  info: "text-sky-700",
};

export default function MetricCard({
  label,
  value,
  subtitle,
  icon,
  variant = "default",
}: Props) {
  return (
    <div
      className={clsx(
        "rounded-card border px-5 py-4 transition-shadow hover:shadow-sm",
        variantStyles[variant],
      )}
    >
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium text-accent-subtle">{label}</span>
        {icon && (
          <span className="text-accent-muted">{icon}</span>
        )}
      </div>
      <p
        className={clsx(
          "mt-1 text-2xl font-semibold tracking-tight",
          valueStyles[variant],
        )}
      >
        {value}
      </p>
      {subtitle && (
        <p className="mt-0.5 text-xs text-accent-muted">{subtitle}</p>
      )}
    </div>
  );
}
