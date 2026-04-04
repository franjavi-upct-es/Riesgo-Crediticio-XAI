// dashboard/src/components/ui/StatusStates.tsx

import { AlertTriangle, Loader2 } from "lucide-react";

export function LoadingState({ message = "Loading…" }: { message?: string }) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 py-20 text-accent-muted">
      <Loader2 size={28} className="animate-spin" />
      <span className="text-sm">{message}</span>
    </div>
  );
}

export function ErrorState({
  title = "Something went wrong",
  message,
  onRetry,
}: {
  title?: string;
  message?: string;
  onRetry?: () => void;
}) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 py-20">
      <AlertTriangle size={28} className="text-red-500" />
      <p className="font-medium text-slate-800">{title}</p>
      {message && (
        <p className="max-w-md text-center text-sm text-accent-muted">
          {message}
        </p>
      )}
      {onRetry && (
        <button
          onClick={onRetry}
          className="mt-2 rounded-lg bg-slate-900 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-slate-700"
        >
          Try again
        </button>
      )}
    </div>
  );
}
