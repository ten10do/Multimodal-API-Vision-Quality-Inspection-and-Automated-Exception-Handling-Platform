import { describe, expect, it } from "vitest";
import type { Inspection, RealtimeStatus } from "../src/types";
import {
  assertQualityInvariant,
  defectTypeDistribution,
  overviewStats,
  parseWsEvent,
  pushBounded,
  pushDeduped,
  qualityDistribution,
  qualityTrend,
} from "../src/utils/transforms";

const status = (patch: Partial<RealtimeStatus>): RealtimeStatus => ({
  captured_total: 10,
  queued_current: 0,
  processing_current: 0,
  completed_total: 8,
  failed_total: 2,
  pass_total: 5,
  review_total: 2,
  fail_total: 1,
  queue_peak_depth: 4,
  simulator_running: true,
  simulator_interval_ms: 200,
  worker_count: 2,
  queue_size: 20,
  current_throughput: 3.5,
  average_processing_latency_ms: 60,
  p50_latency_ms: 55,
  p95_latency_ms: 90,
  average_inference_latency_ms: 14,
  uptime_seconds: 100,
  ws_client_count: 1,
  ...patch,
});

function inspection(patch: Partial<Inspection>): Inspection {
  return {
    inspection_id: "i1",
    product_id: "p1",
    batch_id: null,
    image_url: null,
    status: "COMPLETED",
    quality_result: "PASS",
    severity: "low",
    model_name: "yolov8s",
    model_version: "phase1-baseline",
    rule_version: 1,
    inference_latency_ms: 12,
    error_message: null,
    created_at: "2026-08-04T10:00:00Z",
    defects: [],
    ...patch,
  };
}

describe("overviewStats", () => {
  it("yield rate is PASS / COMPLETED and excludes system FAILED", () => {
    const s = overviewStats(status({}));
    expect(s.yieldRate).toBeCloseTo(5 / 8);
    expect(s.completed).toBe(8);
    expect(s.systemFailed).toBe(2);
    expect(s.totalInspected).toBe(10);
  });

  it("yield rate is null when nothing completed", () => {
    const s = overviewStats(status({ completed_total: 0, pass_total: 0 }));
    expect(s.yieldRate).toBeNull();
  });
});

describe("metric invariants (4G)", () => {
  it("PASS + REVIEW + FAIL == COMPLETED", () => {
    expect(assertQualityInvariant(status({}))).toBe(true);
    expect(assertQualityInvariant(status({ fail_total: 2 }))).toBe(false);
  });
});

describe("qualityDistribution", () => {
  it("counts by quality result", () => {
    const d = qualityDistribution([
      inspection({ quality_result: "PASS" }),
      inspection({ quality_result: "REVIEW" }),
      inspection({ quality_result: "FAIL" }),
      inspection({ quality_result: "PASS" }),
      inspection({ status: "FAILED", quality_result: null }),
    ]);
    expect(d).toEqual({ pass: 2, review: 1, fail: 1 });
  });
});

describe("defectTypeDistribution", () => {
  it("aggregates defect classes across inspections", () => {
    const d = defectTypeDistribution([
      inspection({ defects: [{ class_name: "crazing" } as Inspection["defects"][0], { class_name: "scratches" } as Inspection["defects"][0]] }),
      inspection({ defects: [{ class_name: "crazing" } as Inspection["defects"][0]] }),
    ]);
    expect(d).toEqual([
      { name: "crazing", count: 2 },
      { name: "scratches", count: 1 },
    ]);
  });
});

describe("qualityTrend", () => {
  it("buckets by fixed window and separates system failed from product fail", () => {
    const trend = qualityTrend(
      [
        inspection({ created_at: "2026-08-04T10:00:10Z", quality_result: "PASS" }),
        inspection({ created_at: "2026-08-04T10:00:20Z", quality_result: "FAIL" }),
        inspection({ created_at: "2026-08-04T10:00:30Z", status: "FAILED", quality_result: null }),
      ],
      60_000,
    );
    expect(trend).toHaveLength(1);
    expect(trend[0]).toMatchObject({ pass: 1, fail: 1, failed: 1 });
  });
});

describe("bounded live list (4H)", () => {
  it("keeps at most N items", () => {
    let list: number[] = [];
    for (let i = 0; i < 120; i++) list = pushBounded(list, i, 100);
    expect(list).toHaveLength(100);
    expect(list[0]).toBe(20);
    expect(list[99]).toBe(119);
  });
});

describe("event dedup (4K)", () => {
  it("deduplicates by inspection_id + event_type", () => {
    const ev = parseWsEvent({
      event_type: "inspection.completed",
      inspection_id: "i1",
      product_id: "p1",
    })!;
    const list = pushDeduped(pushDeduped([], ev, 10), ev, 10);
    expect(list).toHaveLength(1);
  });
});

describe("parseWsEvent", () => {
  it("parses a valid completed event", () => {
    const ev = parseWsEvent({
      event_type: "inspection.completed",
      timestamp: "2026-08-04T10:00:00Z",
      product_id: "p1",
      inspection_id: "i1",
      batch_id: "b1",
      production_line: "line-a",
      station: "qc-01",
      process_status: "COMPLETED",
      quality_result: "PASS",
      severity: "low",
      defect_count: 2,
      inference_latency_ms: 15,
      model_version: "v1",
    });
    expect(ev?.inspection_id).toBe("i1");
    expect(ev?.quality_result).toBe("PASS");
    expect(ev?.process_status).toBe("COMPLETED");
  });

  it("rejects malformed or unknown event types", () => {
    expect(parseWsEvent(null)).toBeNull();
    expect(parseWsEvent({ event_type: "other" })).toBeNull();
    expect(parseWsEvent({ event_type: "inspection.completed" })).toBeNull();
  });

  it("maps process_status FAILED without a quality result", () => {
    const ev = parseWsEvent({
      event_type: "inspection.failed",
      inspection_id: "i2",
      product_id: "p2",
      process_status: "FAILED",
    });
    expect(ev?.process_status).toBe("FAILED");
    expect(ev?.quality_result).toBeNull();
  });
});
