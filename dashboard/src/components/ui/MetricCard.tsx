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
  default: "border-border bg-card",
  success:
    "border-emerald-200 bg-emerald-50/50 dark:border-emerald-900/40 dark:bg-emerald-950/20",
  danger:
    "border-red-200 bg-red-50/50 dark:border-red-900/40 dark:bg-red-950/20",
  info: "border-sky-200 bg-sky-50/50 dark:border-sky-900/40 dark:bg-sky-950/20",
};

const valueStyles = {
  default: "text-foreground",
  success: "text-emerald-700 dark:text-emerald-300",
  danger: "text-red-700 dark:text-red-300",
  info: "text-sky-700 dark:text-sky-300",
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
        <span className="text-sm font-medium text-muted-foreground">
          {label}
        </span>
        {icon && <span className="text-muted-foreground">{icon}</span>}
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
        <p className="mt-0.5 text-xs text-muted-foreground">{subtitle}</p>
      )}
    </div>
  );
}
