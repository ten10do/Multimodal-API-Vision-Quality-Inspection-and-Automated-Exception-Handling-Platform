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

  if (statusQ.isLoading || recentQ.isLoading) return <LoadingState label="加载实时指标…" />;
  if (statusQ.isError || recentQ.isError) {
    const msg = (statusQ.error as Error)?.message ?? "backend unavailable";
    return <ErrorState message={msg} onRetry={() => void (statusQ.refetch(), recentQ.refetch())} />;
  }
  if (!stats) return <EmptyState />;

  const yieldPct = stats.yieldRate === null ? "—" : `${(stats.yieldRate * 100).toFixed(1)}%`;
  const modelVersion = inspections.find((i) => i.model_version)?.model_version ?? "—";

  return (
    <div className="page">
      <section className="metric-grid">
        <MetricCard label="Total Inspected" value={stats.totalInspected} hint={`captured ${stats.captured}`} />
        <MetricCard label="Completed" value={stats.completed} />
        <MetricCard label="System Failed" value={stats.systemFailed} tone="danger" hint="系统处理失败，不计入良率" />
        <MetricCard label="PASS" value={stats.pass} tone="success" />
        <MetricCard label="REVIEW" value={stats.review} tone="warn" />
        <MetricCard label="FAIL" value={stats.fail} tone="danger" hint="产品质量不合格" />
        <MetricCard label="Yield Rate" value={yieldPct} hint="PASS / COMPLETED" />
        <MetricCard label="Throughput" value={`${stats.throughput.toFixed(2)}/s`} />
        <MetricCard label="Queue Depth" value={stats.queueDepth} hint={`processing ${stats.processing}`} />
        <MetricCard
          label="Avg E2E Latency"
          value={stats.avgLatencyMs === null ? "—" : `${stats.avgLatencyMs.toFixed(0)} ms`}
        />
        <MetricCard label="P95 E2E Latency" value={stats.p95LatencyMs === null ? "—" : `${stats.p95LatencyMs.toFixed(0)} ms`} />
        <MetricCard label="Model Version" value={modelVersion} />
      </section>

      <section className="chart-grid">
        <div className="panel">
          <h3>Quality Result Distribution</h3>
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
          <h3>Defect Type Distribution</h3>
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
          <h3>Quality Trend</h3>
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
          <h3>Throughput / Latency Trend</h3>
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
