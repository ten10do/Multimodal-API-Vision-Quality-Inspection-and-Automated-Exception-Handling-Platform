import { useMemo } from "react";
import { useRealtimeStatus, useRecentInspections } from "../../hooks/queries";
import { overviewStats, defectTypeDistribution, qualityDistribution, qualityTrend, latencySeries } from "../../utils/transforms";
import { MetricCard } from "../../components/MetricCard";
import { Chart } from "../../components/Chart";
import { EmptyState, ErrorState, LoadingState } from "../../components/StateViews";

const QUALITY_COLORS: Record<string, string> = {
  PASS: "#22c55e",
  REVIEW: "#eab308",
  FAIL: "#ef4444",
};

export function OverviewPage() {
  const statusQ = useRealtimeStatus();
  const recentQ = useRecentInspections(300);

  const stats = useMemo(() => (statusQ.data ? overviewStats(statusQ.data) : null), [statusQ.data]);
  const inspections = recentQ.data ?? [];

  const dist = useMemo(() => qualityDistribution(inspections), [inspections]);
  const defects = useMemo(() => defectTypeDistribution(inspections), [inspections]);
  const trend = useMemo(() => qualityTrend(inspections), [inspections]);
  const latencies = useMemo(() => latencySeries(inspections), [inspections]);

  // Phase 7: industrial final states over the recent window (13)
  const industrial = useMemo(() => {
    const c = { released: 0, rejected: 0, held: 0, safeHold: 0, notIntegrated: 0, mesFailed: 0, plcFault: 0 };
    for (const i of inspections) {
      const st = i.industrial_final_state ?? i.industrial_state;
      if (st === "RELEASED") c.released += 1;
      else if (st === "REJECTED") c.rejected += 1;
      else if (st === "HELD") c.held += 1;
      else if (st === "SAFE_HOLD" || st === "COMMAND_FAILED") {
        c.safeHold += 1;
        c.plcFault += 1;
      } else if (st === "NOT_INTEGRATED") c.notIntegrated += 1;
      if (i.mes_sync_status === "FAILED") c.mesFailed += 1;
    }
    return c;
  }, [inspections]);

  if (statusQ.isLoading || recentQ.isLoading) return <LoadingState label="加载实时指标…" />;
  if (statusQ.isError || recentQ.isError) {
    const msg = (statusQ.error as Error)?.message ?? "backend unavailable";
    return <ErrorState message={msg} onRetry={() => void (statusQ.refetch(), recentQ.refetch())} />;
  }
  if (!stats) return <EmptyState />;

  const yieldPct = stats.yieldRate === null ? "—" : `${(stats.yieldRate * 100).toFixed(1)}%`;
  const modelVersion = inspections.find((i) => i.model_version)?.model_version ?? "—";
  const fmtTime = (iso: string | null) =>
    iso ? new Date(iso).toLocaleTimeString("zh-CN", { hour12: false }) : "—";

  return (
    <div className="page">
      <div className="freshness-bar">
        <span>质量快照 snapshot_at：{fmtTime(stats.snapshotAt)}</span>
        <span>管线采集 telemetry_at：{fmtTime(stats.telemetryAt)}</span>
        <span className="freshness-note">两类指标独立刷新；captured 为实时采集，不参与质量守恒</span>
      </div>

      <section className="metric-grid">
        {/* quality / persisted facts: single coherent snapshot */}
        <MetricCard
          label="Total Inspected"
          value={stats.totalInspected}
          hint={`= completed ${stats.completed} + failed ${stats.systemFailed}`}
        />
        <MetricCard label="Completed" value={stats.completed} />
        <MetricCard label="System Failed" value={stats.systemFailed} tone="danger" hint="系统处理失败，不计入良率" />
        <MetricCard label="PASS" value={stats.pass} tone="success" />
        <MetricCard label="REVIEW" value={stats.review} tone="warn" />
        <MetricCard label="FAIL" value={stats.fail} tone="danger" hint="产品质量不合格" />
        <MetricCard label="Yield Rate" value={yieldPct} hint="PASS / COMPLETED" />
        {/* runtime telemetry (independent freshness) */}
        <MetricCard label="Captured (Pipeline)" value={stats.captured} hint="实时采集，独立于质量快照" />
        <MetricCard label="Throughput" value={`${stats.throughput.toFixed(2)}/s`} />
        <MetricCard label="Queue Depth" value={stats.queueDepth} hint={`processing ${stats.processing}`} />
        <MetricCard
          label="Avg E2E Latency"
          value={stats.avgLatencyMs === null ? "—" : `${stats.avgLatencyMs.toFixed(0)} ms`}
        />
        <MetricCard label="P95 E2E Latency" value={stats.p95LatencyMs === null ? "—" : `${stats.p95LatencyMs.toFixed(0)} ms`} />
        <MetricCard label="Model Version" value={modelVersion} />
      </section>

      <section className="metric-grid">
        <span className="freshness-note" style={{ gridColumn: "1 / -1" }}>
          工业执行（Phase 7，最近 {inspections.length} 次）：NOT INTEGRATED / HELD / SAFE HOLD / REJECTED / RELEASED 语义互斥
        </span>
        <MetricCard label="Released" value={industrial.released} tone="success" />
        <MetricCard label="Rejected" value={industrial.rejected} tone="danger" />
        <MetricCard label="Held" value={industrial.held} tone="warn" />
        <MetricCard label="Safe Hold" value={industrial.safeHold} tone="danger" hint="PLC 故障/离线/系统失败" />
        <MetricCard label="Not Integrated" value={industrial.notIntegrated} hint="PLC 未启用，非故障" />
        <MetricCard label="PLC Fault" value={industrial.plcFault} tone="danger" />
        <MetricCard label="MES Sync Failed" value={industrial.mesFailed} tone="danger" />
      </section>

      <section className="chart-grid">
        <div className="panel">
          <h3>Quality Result Distribution <span className="win-note">(最近 {inspections.length} 次)</span></h3>
          {inspections.length === 0 ? (
            <EmptyState message="暂无质检数据" />
          ) : (
            <Chart
              option={{
                xAxis: ["PASS", "REVIEW", "FAIL"],
                series: [
                  {
                    name: "count",
                    type: "bar",
                    data: [
                      { value: dist.pass, itemStyle: { color: QUALITY_COLORS.PASS } },
                      { value: dist.review, itemStyle: { color: QUALITY_COLORS.REVIEW } },
                      { value: dist.fail, itemStyle: { color: QUALITY_COLORS.FAIL } },
                    ] as never,
                  },
                ],
                yLabel: "count",
              }}
            />
          )}
        </div>
        <div className="panel">
          <h3>Defect Type Distribution <span className="win-note">(最近 {inspections.length} 次)</span></h3>
          {defects.length === 0 ? (
            <EmptyState message="暂无缺陷数据" />
          ) : (
            <Chart
              option={{
                xAxis: defects.map((d) => d.name),
                series: [{ name: "defects", type: "bar", data: defects.map((d) => d.count) }],
                yLabel: "count",
              }}
            />
          )}
        </div>
        <div className="panel">
          <h3>Quality Trend <span className="win-note">(最近 {inspections.length} 次)</span></h3>
          {trend.length === 0 ? (
            <EmptyState message="暂无趋势数据" />
          ) : (
            <Chart
              option={{
                xAxis: trend.map((t) => new Date(t.bucket).toLocaleTimeString("zh-CN", { hour12: false })),
                series: [
                  { name: "PASS", type: "line", data: trend.map((t) => t.pass), color: QUALITY_COLORS.PASS },
                  { name: "REVIEW", type: "line", data: trend.map((t) => t.review), color: QUALITY_COLORS.REVIEW },
                  { name: "FAIL", type: "line", data: trend.map((t) => t.fail), color: QUALITY_COLORS.FAIL },
                  { name: "SYSTEM FAILED", type: "line", data: trend.map((t) => t.failed), color: "#64748b" },
                ],
                yLabel: "count / min",
              }}
            />
          )}
        </div>
        <div className="panel">
          <h3>Throughput / Latency Trend <span className="win-note">(最近 {inspections.length} 次)</span></h3>
          {latencies.length === 0 ? (
            <EmptyState message="暂无延迟数据" />
          ) : (
            <Chart
              option={{
                xAxis: latencies.map((l) => new Date(l.ts).toLocaleTimeString("zh-CN", { hour12: false })),
                series: [{ name: "inference ms", type: "line", data: latencies.map((l) => l.latencyMs), color: "#38bdf8" }],
                yLabel: "ms",
              }}
            />
          )}
        </div>
      </section>
    </div>
  );
}
