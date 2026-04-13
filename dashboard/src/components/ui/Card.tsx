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
        "rounded-card border border-border bg-card p-5 text-card-foreground shadow-sm transition-colors",
        className,
      )}
    >
      {title && (
        <div className="mb-4">
          <h3 className="text-sm font-semibold text-foreground">{title}</h3>
          {subtitle && (
            <p className="mt-0.5 text-xs text-muted-foreground">{subtitle}</p>
          )}
        </div>
      )}
      {children}
    </div>
  );
}
