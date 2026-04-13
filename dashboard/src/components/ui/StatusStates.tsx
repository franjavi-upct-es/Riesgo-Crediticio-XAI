// dashboard/src/components/ui/StatusStates.tsx

import { AlertTriangle, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";

export function LoadingState({ message = "Loading…" }: { message?: string }) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 py-20 text-muted-foreground">
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
      <AlertTriangle size={28} className="text-destructive" />
      <p className="font-medium text-foreground">{title}</p>
      {message && (
        <p className="max-w-md text-center text-sm text-muted-foreground">
          {message}
        </p>
      )}
      {onRetry && (
        <Button onClick={onRetry} className="mt-2">
          Try again
        </Button>
      )}
    </div>
  );
}
