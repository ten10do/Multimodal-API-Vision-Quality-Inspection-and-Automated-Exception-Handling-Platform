// REST client. All fetch URLs live here, never scattered in components.

import type {
  DriftReport,
  HumanDecision,
  HumanFeedback,
  Inspection,
  InspectionFilters,
  ModelMetrics,
  RealtimeStatus,
  RegistryModel,
  ReviewMetrics,
  ReviewTask,
  TrainingCandidate,
} from "../types";

const API_BASE = "/api/v1";

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    message: string,
    public readonly code?: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      headers: { Accept: "application/json", ...(init?.headers ?? {}) },
      ...init,
    });
  } catch (err) {
    throw new ApiError(0, `backend unreachable: ${String(err)}`);
  }
  if (!response.ok) {
    let code: string | undefined;
    let message = `HTTP ${response.status}`;
    try {
      const body = (await response.json()) as { error?: { code?: string; message?: string } };
      code = body.error?.code;
      message = body.error?.message ?? message;
    } catch {
      // non-JSON error body
    }
    throw new ApiError(response.status, message, code);
  }
  return (await response.json()) as T;
}

export function toQuery(filters: object): string {
  const params = new URLSearchParams();
  for (const [k, v] of Object.entries(filters)) {
    if (v !== undefined && v !== null && v !== "") params.set(k, String(v));
  }
  const qs = params.toString();
  return qs ? `?${qs}` : "";
}

export interface ReviewFilters {
  status?: string;
  priority?: number;
  defect_type?: string;
  production_line?: string;
  station?: string;
  batch_id?: string;
  limit?: number;
  offset?: number;
}

function jsonRequest<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export const api = {
  realtimeStatus: (): Promise<RealtimeStatus> => request("/realtime/status"),

  listInspections: (filters: InspectionFilters = {}): Promise<Inspection[]> =>
    request(`/inspections${toQuery(filters)}`),

  getInspection: (id: string): Promise<Inspection> => request(`/inspections/${id}`),

  getProduct: (productId: string): Promise<{ product_id: string; production_line: string; station: string; created_at: string }> =>
    request(`/products/${productId}`),

  productInspections: (productId: string): Promise<Inspection[]> =>
    request(`/products/${productId}/inspections`),

  // ---- Phase 5: human review ----
  listReviews: (filters: ReviewFilters = {}): Promise<ReviewTask[]> =>
    request(`/reviews${toQuery(filters)}`),

  getReview: (taskId: string): Promise<ReviewTask> => request(`/reviews/${taskId}`),

  claimReview: (taskId: string, reviewer: string): Promise<ReviewTask> =>
    jsonRequest(`/reviews/${taskId}/claim`, { reviewer }),

  resolveReview: (
    taskId: string,
    reviewer: string,
    humanDecision: HumanDecision,
    humanLabel: string | null,
    reason: string | null,
  ): Promise<ReviewTask> =>
    jsonRequest(`/reviews/${taskId}/resolve`, {
      reviewer,
      human_decision: humanDecision,
      human_label: humanLabel,
      reason,
    }),

  reviewMetrics: (): Promise<ReviewMetrics> => request("/reviews-metrics"),

  trainingCandidates: (kind: "all" | "corrected" | "disagreed" | "low_confidence" = "all"): Promise<TrainingCandidate[]> =>
    request(`/training-candidates?kind=${kind}`),

  // ---- Phase 8 MLOps ----
  listModels: (status?: string, modelType?: string): Promise<RegistryModel[]> => {
    const p = new URLSearchParams();
    if (status) p.set("status", status);
    if (modelType) p.set("model_type", modelType);
    const qs = p.toString();
    return request(`/models${qs ? `?${qs}` : ""}`);
  },

  registerModel: (body: Record<string, unknown>): Promise<RegistryModel> =>
    jsonRequest(`/models`, body),

  promoteModel: (id: string, requiredDomain: string, thresholds?: Record<string, number> | null): Promise<RegistryModel & { gate: { passed: boolean; blocked: string[]; checks: unknown[] } }> =>
    jsonRequest(`/models/${id}/promote`, { required_domain: requiredDomain, thresholds: thresholds ?? null }),

  gateModel: (id: string, requiredDomain: string): Promise<{ model: string; gate: { passed: boolean; blocked: string[]; checks: unknown[] } }> =>
    jsonRequest(`/models/${id}/gate`, { required_domain: requiredDomain }),

  rollbackModel: (modelName: string, modelVersion: string): Promise<RegistryModel> =>
    jsonRequest(`/models/rollback`, { model_name: modelName, model_version: modelVersion }),

  modelMetrics: (modelVersion?: string): Promise<ModelMetrics> => {
    const p = new URLSearchParams();
    if (modelVersion) p.set("model_version", modelVersion);
    const qs = p.toString();
    return request(`/model-metrics${qs ? `?${qs}` : ""}`);
  },

  humanFeedback: (params: { model_version?: string; defect_type?: string; line?: string; station?: string } = {}): Promise<HumanFeedback> => {
    const p = new URLSearchParams();
    for (const [k, v] of Object.entries(params)) {
      if (v) p.set(k, v);
    }
    const qs = p.toString();
    return request(`/human-feedback${qs ? `?${qs}` : ""}`);
  },

  driftReport: (modelVersion?: string): Promise<DriftReport> => {
    const p = new URLSearchParams();
    if (modelVersion) p.set("model_version", modelVersion);
    const qs = p.toString();
    return request(`/drift${qs ? `?${qs}` : ""}`);
  },
};
