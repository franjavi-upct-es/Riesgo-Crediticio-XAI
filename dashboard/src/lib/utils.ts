// dashboard/src/lib/utils.ts

import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/**
 * Merge Tailwind classes intelligently, deduplicating conflicting utilities.
 * Used by every shadcn/ui primitive and by app code that composes class lists.
 */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}
