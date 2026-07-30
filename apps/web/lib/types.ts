export type InspectionStatus =
  | "queued"
  | "vision_analyzing"
  | "reasoning"
  | "executing"
  | "awaiting_approval"
  | "completed"
  | "manual_review"
  | "failed";

export type RiskLevel = "low" | "medium" | "high" | "critical";
export type Disposition = "release" | "manual_review" | "reject" | "stop_line";

export interface InspectionListItem {
  id: string;
  product_code: string;
  batch_code: string;
  status: InspectionStatus;
  risk_level: RiskLevel | null;
  disposition: Disposition | null;
  created_at: string;
}

export interface WorkflowAction {
  id: string;
  action_type: string;
  status: "succeeded" | "pending_approval" | "rejected" | "failed";
  result_payload: Record<string, unknown> | null;
  created_at: string;
}

export interface AuditLog {
  id: string;
  actor_type: string;
  actor_id: string;
  event_type: string;
  previous_state: Record<string, unknown> | null;
  new_state: Record<string, unknown> | null;
  detail: Record<string, unknown>;
  created_at: string;
}

export interface Inspection extends InspectionListItem {
  original_filename: string;
  content_type: string;
  vision_result: Record<string, unknown> | null;
  analysis_result: Record<string, unknown> | null;
  error_code: string | null;
  error_message: string | null;
  updated_at: string;
  actions: WorkflowAction[];
  audit_logs: AuditLog[];
  feedback: Array<{
    id: string;
    reviewer: string;
    comment: string;
    corrected_risk: RiskLevel | null;
    corrected_disposition: Disposition | null;
    created_at: string;
  }>;
}

export interface DashboardStats {
  total: number;
  completed: number;
  awaiting_approval: number;
  manual_review: number;
  defect_rate: number;
  by_risk: Record<RiskLevel, number>;
}
