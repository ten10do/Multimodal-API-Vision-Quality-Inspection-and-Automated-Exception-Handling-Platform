import { useCallback } from "react";
import { useInspectionSocket } from "../../hooks/useInspectionSocket";
import { useRealtimeStatus, useRecentInspections } from "../../hooks/queries";
import { StatusBadge } from "../../components/StatusBadge";
import { ErrorState, LoadingState } from "../../components/StateViews";

export function LivePage() {
  const statusQ = useRealtimeStatus(2000);
  const recentQ = useRecentInspections(50, 5000);

  // Reconciliation: after (re)connect, refresh the REST list because the WS
  // channel may have missed events while disconnected (4F).
  const reconcile = useCallback(() => {
    void recentQ.refetch();
  }, [recentQ]);

  const { events, state } = useInspectionSocket(reconcile);

  return (
    <div className="page">
      <div className="ws-status-bar">
        <span className={`ws-dot ws-${state}`} />
        实时连接：<b>{state}</b>
        {state === "reconnecting" || state === "disconnected" ? (
          <span className="ws-hint">重连中，恢复后将从 REST 拉取最新数据（reconciliation）</span>
        ) : null}
      </div>

      <section className="panel">
        <h3>Live Inspections（最近 {Math.max(events.length, 1)} / 100）</h3>
        {events.length === 0 ? (
          <div className="state-block empty">等待实时事件…（启动 Simulator 后自动刷新）</div>
        ) : (
          <table className="table">
            <thead>
              <tr>
                <th>结果</th>
                <th>product_id</th>
                <th>inspection_id</th>
                <th>时间</th>
                <th>line / station</th>
                <th>缺陷数</th>
                <th>延迟 ms</th>
                <th>model</th>
              </tr>
            </thead>
            <tbody>
              {events.map((e) => (
                <tr key={`${e.inspection_id}-${e.event_type}`}>
                  <td>
                    <StatusBadge
                      status={e.process_status === "FAILED" ? "FAILED" : "COMPLETED"}
                      quality={e.quality_result}
                    />
                    {e.quality_result !== null && e.process_status !== "FAILED" ? (
                      <span className={`sev sev-${e.severity ?? ""}`}>{e.severity}</span>
                    ) : null}
                  </td>
                  <td>{e.product_id}</td>
                  <td className="mono">{e.inspection_id}</td>
                  <td>{new Date(e.timestamp).toLocaleTimeString("zh-CN", { hour12: false })}</td>
                  <td>
                    {e.production_line} / {e.station}
                  </td>
                  <td>{e.defect_count}</td>
                  <td>{e.inference_latency_ms === null ? "—" : Math.round(e.inference_latency_ms)}</td>
                  <td className="mono">{e.model_version ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      <section className="panel">
        <h3>Backend 视图（REST，事实来源）</h3>
        {statusQ.isLoading || recentQ.isLoading ? (
          <LoadingState />
        ) : statusQ.isError ? (
          <ErrorState message={(statusQ.error as Error).message ?? "backend unavailable"} onRetry={() => void statusQ.refetch()} />
        ) : (
          <table className="table">
            <thead>
              <tr>
                <th>captured</th>
                <th>queued</th>
                <th>processing</th>
                <th>completed</th>
                <th>system failed</th>
                <th>PASS</th>
                <th>REVIEW</th>
                <th>FAIL</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>{statusQ.data?.captured_total ?? "—"}</td>
                <td>{statusQ.data?.queued_current ?? "—"}</td>
                <td>{statusQ.data?.processing_current ?? "—"}</td>
                <td>{statusQ.data?.completed_total ?? "—"}</td>
                <td>{statusQ.data?.failed_total ?? "—"}</td>
                <td>{statusQ.data?.pass_total ?? "—"}</td>
                <td>{statusQ.data?.review_total ?? "—"}</td>
                <td>{statusQ.data?.fail_total ?? "—"}</td>
              </tr>
            </tbody>
          </table>
        )}
      </section>
    </div>
  );
}
