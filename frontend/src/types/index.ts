// Backend API contract types (mirror of the backend schemas, strict).

export type WsConnectionState = "connecting" | "connected" | "disconnected" | "reconnecting" | "reconnected";

export type QualityResult = "PASS" | "REVIEW" | "FAIL";
export type ProcessStatus = "PENDING" | "COMPLETED" | "FAILED";
export type Severity = "low" | "medium" | "high" | "critical";
export type WsEventType = "inspection.completed" | "inspection.failed";

export interface Defect {
  id: string;
  class_id: number;
  class_name: string;
  confidence: number;
  bbox_xyxy: [number, number, number, number];
  bbox_normalized: [number, number, number, number];
  defect_area_px: number;
  defect_area_ratio: number;
  severity: Severity | null;
  matched_rule: string | null;
}

export interface Product {
  product_id: string;
  production_line: string;
  station: string;
  created_at: string;
}

export interface Inspection {
  inspection_id: string;
  product_id: string;
  batch_id: string | null;
  image_url: string | null;
  status: ProcessStatus;
  quality_result: QualityResult | null;
  severity: Severity | null;
  model_name: string | null;
  model_version: string | null;
  rule_version: number | null;
  inference_latency_ms: number | null;
  error_message: string | null;
  created_at: string;
  defects: Defect[];
  product?: Product;
}

export interface RealtimeStatus {
  // quality / persisted facts (DB-owned, single coherent snapshot)
  completed_total: number;
  failed_total: number;
  pass_total: number;
  review_total: number;
  fail_total: number;
  total_inspected: number;
  yield_rate: number | null;
  // runtime telemetry (pipeline view, refreshed independently)
  captured_total: number;
  queued_current: number;
  processing_current: number;
  queue_depth: number;
  throughput: number;
  // freshness
  snapshot_at: string;
  telemetry_at: string | null;
  // remaining
  queue_peak_depth: number;
  simulator_running: boolean;
  simulator_interval_ms: number | null;
  worker_count: number | null;
  queue_size: number | null;
  current_throughput: number;
  average_processing_latency_ms: number | null;
  p50_latency_ms: number | null;
  p95_latency_ms: number | null;
  average_inference_latency_ms: number | null;
  uptime_seconds: number;
  ws_client_count: number;
}

export interface InspectionEvent {
  event_id: string;
  event_type: WsEventType;
  timestamp: string;
  product_id: string;
  inspection_id: string;
  batch_id: string | null;
  production_line: string;
  station: string;
  process_status: ProcessStatus;
  quality_result: QualityResult | null;
  severity: Severity | null;
  defect_count: number;
  inference_latency_ms: number | null;
  model_version: string | null;
  error_message: string | null;
}

export interface InspectionFilters {
  product_id?: string;
  inspection_id?: string;
  batch_id?: string;
  quality_result?: QualityResult;
  status?: ProcessStatus;
  defect_type?: string;
  production_line?: string;
  station?: string;
  date_from?: string;
  date_to?: string;
  limit?: number;
  offset?: number;
}
