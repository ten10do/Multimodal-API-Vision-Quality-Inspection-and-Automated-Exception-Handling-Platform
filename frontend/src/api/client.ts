// REST client. All fetch URLs live here, never scattered in components.

import type { Inspection, InspectionFilters, RealtimeStatus } from "../types";

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

export function toQuery(filters: InspectionFilters): string {
  const params = new URLSearchParams();
  for (const [k, v] of Object.entries(filters)) {
    if (v !== undefined && v !== null && v !== "") params.set(k, String(v));
  }
  const qs = params.toString();
  return qs ? `?${qs}` : "";
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
};
