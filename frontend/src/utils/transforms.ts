// Unified statistics transforms and metric invariants (4G).
// The dashboard derives every number from these functions so the header
// cards, charts and lists can never disagree with each other.

import type { Inspection, InspectionEvent, QualityResult, RealtimeStatus } from "../types";

export interface OverviewStats {
  totalInspected: number; // completed + system failed (all processed)
  completed: number;
  systemFailed: number;
  pass: number;
  review: number;
  fail: number;
  yieldRate: number | null; // PASS / COMPLETED; null when no completions
  // runtime telemetry (separate, may lag the DB snapshot)
  captured: number;
  queueDepth: number;
  processing: number;
  throughput: number;
  avgLatencyMs: number | null;
  p95LatencyMs: number | null;
  modelVersion: string | null;
  // freshness
  snapshotAt: string | null;
  telemetryAt: string | null;
}

export function overviewStats(status: RealtimeStatus): OverviewStats {
  const completed = status.completed_total;
  const pass = status.pass_total;
  const yieldRate = completed > 0 ? pass / completed : null;
  return {
    // DB-owned quality facts; never mix in telemetry values here (Gate 1)
    totalInspected: status.completed_total + status.failed_total,
    completed,
    systemFailed: status.failed_total,
    pass,
    review: status.review_total,
    fail: status.fail_total,
    yieldRate,
    // runtime telemetry (pipeline view, refreshed independently)
    captured: status.captured_total,
    queueDepth: status.queued_current,
    processing: status.processing_current,
    throughput: status.throughput ?? status.current_throughput,
    avgLatencyMs: status.average_processing_latency_ms,
    p95LatencyMs: status.p95_latency_ms,
    modelVersion: null,
    snapshotAt: status.snapshot_at ?? null,
    telemetryAt: status.telemetry_at ?? null,
  };
}

/** The dashboard invariant (Gate 1): PASS + REVIEW + FAIL == COMPLETED. */
export function assertQualityInvariant(status: RealtimeStatus): boolean {
  return status.pass_total + status.review_total + status.fail_total === status.completed_total;
}

/** total_inspected must equal completed + failed in every snapshot (Gate 1). */
export function assertTotalInspected(status: RealtimeStatus): boolean {
  return status.total_inspected === status.completed_total + status.failed_total;
}

export type QualityKey = QualityResult;

export interface QualityDistribution {
  pass: number;
  review: number;
  fail: number;
}

export function qualityDistribution(inspections: Inspection[]): QualityDistribution {
  const d: QualityDistribution = { pass: 0, review: 0, fail: 0 };
  for (const i of inspections) {
    if (i.quality_result === "PASS") d.pass += 1;
    else if (i.quality_result === "REVIEW") d.review += 1;
    else if (i.quality_result === "FAIL") d.fail += 1;
  }
  return d;
}

export function defectTypeDistribution(inspections: Inspection[]): Array<{ name: string; count: number }> {
  const counts = new Map<string, number>();
  for (const i of inspections) {
    for (const d of i.defects) {
      counts.set(d.class_name, (counts.get(d.class_name) ?? 0) + 1);
    }
  }
  return [...counts.entries()]
    .map(([name, count]) => ({ name, count }))
    .sort((a, b) => b.count - a.count);
}

export interface TrendPoint {
  ts: number;
  completed: number;
  failed: number;
  latencyMs: number | null;
}

/** Sliding time-window trend (4H): fixed bucket size, bounded memory. */
export function qualityTrend(
  inspections: Inspection[],
  bucketMs = 60_000,
): Array<{ bucket: string; pass: number; review: number; fail: number; failed: number }> {
  const buckets = new Map<number, { pass: number; review: number; fail: number; failed: number }>();
  for (const i of inspections) {
    const ts = new Date(i.created_at).getTime();
    const b = Math.floor(ts / bucketMs) * bucketMs;
    const entry = buckets.get(b) ?? { pass: 0, review: 0, fail: 0, failed: 0 };
    if (i.status === "FAILED") entry.failed += 1;
    else if (i.quality_result === "PASS") entry.pass += 1;
    else if (i.quality_result === "REVIEW") entry.review += 1;
    else if (i.quality_result === "FAIL") entry.fail += 1;
    buckets.set(b, entry);
  }
  return [...buckets.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([ts, v]) => ({
      bucket: new Date(ts).toISOString(),
      pass: v.pass,
      review: v.review,
      fail: v.fail,
      failed: v.failed,
    }));
}

export interface LatencyPoint {
  ts: number;
  latencyMs: number;
}

export function latencySeries(inspections: Inspection[]): LatencyPoint[] {
  return inspections
    .filter((i) => i.inference_latency_ms !== null && i.status === "COMPLETED")
    .map((i) => ({
      ts: new Date(i.created_at).getTime(),
      latencyMs: i.inference_latency_ms as number,
    }))
    .sort((a, b) => a.ts - b.ts);
}

/** WS event -> InspectionEvent with validation (4K: parsing + dedup-ready). */
export function parseWsEvent(raw: unknown): InspectionEvent | null {
  if (typeof raw !== "object" || raw === null) return null;
  const e = raw as Record<string, unknown>;
  if (typeof e.event_type !== "string") return null;
  if (e.event_type !== "inspection.completed" && e.event_type !== "inspection.failed") return null;
  if (typeof e.inspection_id !== "string" || typeof e.product_id !== "string") return null;
  return {
    event_id: typeof e.event_id === "string" ? e.event_id : `evt-${e.inspection_id}`,
    event_type: e.event_type,
    timestamp: typeof e.timestamp === "string" ? e.timestamp : new Date().toISOString(),
    product_id: e.product_id,
    inspection_id: e.inspection_id,
    batch_id: typeof e.batch_id === "string" ? e.batch_id : null,
    production_line: typeof e.production_line === "string" ? e.production_line : "",
    station: typeof e.station === "string" ? e.station : "",
    process_status: e.process_status === "FAILED" ? "FAILED" : "COMPLETED",
    quality_result: e.quality_result === "PASS" || e.quality_result === "REVIEW" || e.quality_result === "FAIL"
      ? e.quality_result
      : null,
    severity: typeof e.severity === "string" ? (e.severity as InspectionEvent["severity"]) : null,
    defect_count: typeof e.defect_count === "number" ? e.defect_count : 0,
    inference_latency_ms: typeof e.inference_latency_ms === "number" ? e.inference_latency_ms : null,
    model_version: typeof e.model_version === "string" ? e.model_version : null,
    error_message: typeof e.error_message === "string" ? e.error_message : null,
  };
}

/** Bounded live list (4H): keep the latest N events, dropping the oldest. */
export function pushBounded<T>(list: T[], item: T, max: number): T[] {
  const next = [...list, item];
  return next.length > max ? next.slice(next.length - max) : next;
}

/** Deduplicate by inspection_id + event_type (4K). */
export function pushDeduped(list: InspectionEvent[], item: InspectionEvent, max: number): InspectionEvent[] {
  const filtered = list.filter(
    (e) => !(e.inspection_id === item.inspection_id && e.event_type === item.event_type),
  );
  return pushBounded(filtered, item, max);
}
