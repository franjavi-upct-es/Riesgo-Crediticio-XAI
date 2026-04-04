// dashboard/src/api/client.ts

import type {
  CreditDataInput,
  FullEvaluation,
  HealthResponse,
  PredictionResponse,
} from "@/types/api";

const BASE_URL = import.meta.env.VITE_API_URL ?? "";

class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json", ...options?.headers },
    ...options,
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new ApiError(res.status, body.detail ?? res.statusText);
  }

  return res.json() as Promise<T>;
}

// --- Health ---

export function fetchHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/health");
}

// -- Prediction ---

export function postPrediction(
  data: CreditDataInput,
): Promise<PredictionResponse> {
  return request<PredictionResponse>("/predict_risk/", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

// --- Evaluation ---

export function fetchFullEvaluation(): Promise<FullEvaluation> {
  return request<FullEvaluation>("/evaluation/full");
}

export { ApiError };
