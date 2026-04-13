// dashboard/src/hooks/useApi.ts

import { useMutation, useQuery } from "@tanstack/react-query";
import {
  fetchDatasetSchema,
  fetchDatasets,
  fetchFullEvaluation,
  fetchHealth,
  fetchRandomSample,
  postPrediction,
} from "@/api/client";
import type { PredictionPayload } from "@/types/api";

export function useHealth() {
  return useQuery({
    queryKey: ["health"],
    queryFn: fetchHealth,
    refetchInterval: 30_000,
    retry: 1,
  });
}

export function useDatasets() {
  return useQuery({
    queryKey: ["datasets"],
    queryFn: fetchDatasets,
    staleTime: 60_000,
    retry: 1,
  });
}

export function useDatasetSchema(datasetId: string | null) {
  return useQuery({
    queryKey: ["dataset-schema", datasetId],
    queryFn: () => fetchDatasetSchema(datasetId!),
    enabled: !!datasetId,
    staleTime: 5 * 60_000,
  });
}

export function useRandomSample(datasetId: string | null) {
  return useMutation({
    mutationFn: () => fetchRandomSample(datasetId!),
  });
}

export function useEvaluation(datasetId: string | null) {
  return useQuery({
    queryKey: ["evaluation", datasetId],
    queryFn: () => fetchFullEvaluation(datasetId!),
    enabled: !!datasetId,
    staleTime: 5 * 60_000,
    retry: 1,
  });
}

export interface PredictionMutationInput {
  data: PredictionPayload;
  datasetId?: string;
}

export function usePrediction() {
  return useMutation({
    mutationFn: ({ data, datasetId }: PredictionMutationInput) =>
      postPrediction(data, datasetId),
  });
}
