// dashboard/src/hooks/useApi.ts

import { useMutation, useQuery } from "@tanstack/react-query";
import { fetchFullEvaluation, fetchHealth, postPrediction } from "@/api/client";
import type { CreditDataInput } from "@/types/api";

export function useHealth() {
  return useQuery({
    queryKey: ["health"],
    queryFn: fetchHealth,
    refetchInterval: 30_000,
    retry: 1,
  });
}

export function useEvaluation() {
  return useQuery({
    queryKey: ["evaluation"],
    queryFn: fetchFullEvaluation,
    staleTime: 5 * 60_000,
    retry: 1,
  });
}

export function usePrediction() {
  return useMutation({
    mutationFn: (data: CreditDataInput) => postPrediction(data),
  });
}
