import type {
  DashboardStats,
  Inspection,
  InspectionListItem,
} from "@/lib/types";

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000/api/v1";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init);
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    const message =
      payload?.error?.message ??
      payload?.detail?.message ??
      payload?.error?.code ??
      payload?.detail?.code ??
      "请求处理失败";
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

export async function getInspections(): Promise<{
  items: InspectionListItem[];
  total: number;
}> {
  return request("/inspections");
}

export async function getInspection(id: string): Promise<Inspection> {
  return request(`/inspections/${id}`);
}

export async function getStats(): Promise<DashboardStats> {
  return request("/dashboard/stats");
}

export async function createInspection(form: FormData): Promise<Inspection> {
  return request("/inspections", {
    method: "POST",
    body: form,
    headers: { "Idempotency-Key": crypto.randomUUID() },
  });
}

export async function decideApproval(
  id: string,
  body: { decision: "approve" | "reject"; reviewer: string; comment: string },
): Promise<Inspection> {
  return request(`/inspections/${id}/approval`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export async function sendFeedback(
  id: string,
  body: {
    reviewer: string;
    comment: string;
    corrected_risk?: string;
    corrected_disposition?: string;
  },
): Promise<Inspection> {
  return request(`/inspections/${id}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}
