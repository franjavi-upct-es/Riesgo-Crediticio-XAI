// dashboard/src/components/ui/Card.tsx

import { clsx } from "clsx";
import type { ReactNode } from "react";

interface Props {
  title?: string;
  subtitle?: string;
  children: ReactNode;
  className?: string;
}

export default function Card({ title, subtitle, children, className }: Props) {
  return (
    <div
      className={clsx(
        "rounded-card border border-slate-200 bg-white p-5",
        className,
      )}
    >
      {title && (
        <div className="mb-4">
          <h3 className="text-sm font-semibold text-slate-800">{title}</h3>
          {subtitle && (
            <p className="mt-0.5 text-xs text-accent-muted">{subtitle}</p>
          )}
        </div>
      )}
      {children}
    </div>
  );
}
