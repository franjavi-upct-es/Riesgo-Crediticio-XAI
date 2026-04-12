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
    mutationFn: ({
      data,
      datasetId,
    }: {
      data: Record<string, unknown>;
      datasetId?: string;
    }) => postPrediction(data, datasetId),
  });
}
