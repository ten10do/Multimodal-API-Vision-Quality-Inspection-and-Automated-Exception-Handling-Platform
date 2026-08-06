import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../../api/client";
import { MetricCard } from "../../components/MetricCard";
import { EmptyState, ErrorState, LoadingState } from "../../components/StateViews";

const STATUS_TONE: Record<string, string> = {
  PRODUCTION: "badge-ok",
  STAGING: "badge-warn",
  CANDIDATE: "badge-pending",
  ARCHIVED: "badge-neutral",
};

function fmtRate(v: number | null | undefined): string {
  return v === null || v === undefined ? "—" : `${(v * 100).toFixed(1)}%`;
}

export function ModelOpsPage() {
  const queryClient = useQueryClient();
  const [actionMsg, setActionMsg] = useState<string | null>(null);

  const modelsQ = useQuery({
    queryKey: ["models"],
    queryFn: () => api.listModels(),
  });
  const metricsQ = useQuery({
    queryKey: ["model-metrics"],
    queryFn: () => api.modelMetrics(),
  });
  const feedbackQ = useQuery({
    queryKey: ["human-feedback"],
    queryFn: () => api.humanFeedback(),
  });
  const driftQ = useQuery({
    queryKey: ["drift"],
    queryFn: () => api.driftReport(),
  });

  const models = modelsQ.data ?? [];
  const production = models.filter((m) => m.status === "PRODUCTION");

  const runAction = async (fn: () => Promise<unknown>, okMsg: string) => {
    try {
      await fn();
      await queryClient.invalidateQueries({ queryKey: ["models"] });
      setActionMsg(okMsg);
    } catch (e) {
      const err = e as Error;
      try {
        const body = JSON.parse(err.message);
        setActionMsg(`失败：${body.detail?.message ?? body.message ?? err.message}`);
      } catch {
        setActionMsg(`失败：${err.message}`);
      }
    }
  };

  if (modelsQ.isLoading || metricsQ.isLoading) return <LoadingState label="加载 Model Operations…" />;
  if (modelsQ.isError) {
    return <ErrorState message={(modelsQ.error as Error)?.message ?? "backend unavailable"} onRetry={() => void modelsQ.refetch()} />;
  }

  return (
    <div className="page">
      <div className="freshness-bar">
        <span>Model Registry（Phase 8）</span>
        <span>deployment_version 2026.08.1</span>
        <span className="freshness-note">生产模型由 registry 唯一指定；禁止无版本启动</span>
      </div>

      {actionMsg ? (
        <div className="state-block" style={{ marginBottom: 12 }}>
          {actionMsg}
        </div>
      ) : null}

      <section>
        <h3>Current Production Models</h3>
        {production.length === 0 ? (
          <EmptyState message="暂无 PRODUCTION 模型（backfill 为 CANDIDATE）" />
        ) : (
          <div className="metric-grid">
            {production.map((m) => (
              <MetricCard
                key={m.id}
                label={`${m.model_name} @ ${m.model_version}`}
                value={m.status}
                tone={m.status === "PRODUCTION" ? "success" : "warn"}
                hint={`${m.model_type} · ${m.dataset_version ?? "no dataset"} · promoted ${m.promoted_at ? new Date(m.promoted_at).toLocaleString("zh-CN", { hour12: false }) : "—"}`}
              />
            ))}
          </div>
        )}
      </section>

      <section>
        <h3>Registry</h3>
        {models.length === 0 ? (
          <EmptyState message="运行 scripts/backfill_mlflow.py 登记基线模型" />
        ) : (
          <table className="table">
            <thead>
              <tr>
                <th>model</th>
                <th>version</th>
                <th>type</th>
                <th>status</th>
                <th>dataset</th>
                <th>domain</th>
                <th>key metrics</th>
                <th>actions</th>
              </tr>
            </thead>
            <tbody>
              {models.map((m) => (
                <tr key={m.id}>
                  <td className="mono">{m.model_name}</td>
                  <td className="mono">{m.model_version}</td>
                  <td>{m.model_type}</td>
                  <td>
                    <span className={`badge ${STATUS_TONE[m.status] ?? "badge-neutral"}`}>{m.status}</span>
                  </td>
                  <td className="mono">{m.dataset_version ?? "—"}</td>
                  <td>{m.domain_validated ? "validated" : "NOT validated"}</td>
                  <td className="mono">
                    {m.model_type === "yolo"
                      ? `mAP50 ${m.metrics?.mAP50 ?? "—"} · r ${m.metrics?.recall ?? "—"}`
                      : `imgAUROC ${m.metrics?.image_auroc ?? "—"} · pix ${m.metrics?.pixel_auroc ?? "—"}`}
                  </td>
                  <td>
                    <button
                      className="btn"
                      disabled={m.status === "PRODUCTION" || !m.domain_validated}
                      onClick={() =>
                        void runAction(
                          () => api.promoteModel(m.id, "steel"),
                          `${m.model_name}@${m.model_version} 已晋升 PRODUCTION（门禁通过）`,
                        )
                      }
                    >
                      Promote
                    </button>{" "}
                    <button
                      className="btn"
                      disabled={m.status !== "PRODUCTION"}
                      onClick={() =>
                        void runAction(
                          () => api.rollbackModel(m.model_name, "1.0.0"),
                          `已回滚 ${m.model_name} 到 1.0.0`,
                        )
                      }
                    >
                      Rollback→1.0.0
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        <div className="state-block" style={{ marginTop: 8 }}>
          提示：Promote 会运行可配置门禁（mAP/recall/latency 或 AUROC/latency）+ domain 校验；MVTec PatchCore 因 steel_domain_validated=false 无法晋升。
        </div>
      </section>

      <section className="chart-grid">
        <div className="panel">
          <h3>Production Metrics</h3>
          {!metricsQ.data ? (
            <EmptyState message="暂无监控数据" />
          ) : (
            <div className="metric-grid">
              <MetricCard label="Inference" value={metricsQ.data.inference_count} />
              <MetricCard label="Errors" value={metricsQ.data.error_count} tone={metricsQ.data.error_count ? "danger" : "success"} />
              <MetricCard label="Error Rate" value={fmtRate(metricsQ.data.error_rate)} />
              <MetricCard label="Latency P95" value={metricsQ.data.inference_latency_p95_ms === null ? "—" : `${Math.round(metricsQ.data.inference_latency_p95_ms)} ms`} />
              <MetricCard label="Review Rate" value={fmtRate(metricsQ.data.review_rate)} />
            </div>
          )}
        </div>
        <div className="panel">
          <h3>Drift（8I）</h3>
          {!driftQ.data ? (
            <EmptyState message="暂无 drift 数据" />
          ) : (
            <>
              <div className="metric-grid">
                <MetricCard label="Overall" value={driftQ.data.overall} tone={driftQ.data.overall === "NORMAL" ? "success" : driftQ.data.overall === "WARNING" ? "warn" : "danger"} />
                {Object.entries(driftQ.data.signals).map(([k, s]) => (
                  <MetricCard
                    key={k}
                    label={k}
                    value={s.level}
                    tone={s.level === "NORMAL" ? "success" : s.level === "WARNING" ? "warn" : "danger"}
                    hint={s.score !== undefined ? `score ${s.score}` : s.max_delta !== undefined ? `Δ ${s.max_delta}` : undefined}
                  />
                ))}
              </div>
              <div className="state-block">{driftQ.data.note}</div>
            </>
          )}
        </div>
      </section>

      <section className="panel" style={{ marginTop: 12 }}>
        <h3>Human Feedback（8H，按 model_version 关联）</h3>
        {!feedbackQ.data ? (
          <EmptyState message="暂无 human review 数据" />
        ) : (
          <>
            <div className="metric-grid">
              <MetricCard label="Resolved" value={feedbackQ.data.resolved} />
              <MetricCard label="Defect Confirmation" value={fmtRate(feedbackQ.data.defect_confirmation_rate)} />
              <MetricCard label="Label Agreement" value={fmtRate(feedbackQ.data.ai_human_label_agreement_rate)} />
              <MetricCard label="Pass Override" value={fmtRate(feedbackQ.data.pass_override_rate)} />
              <MetricCard label="Corrected" value={fmtRate(feedbackQ.data.corrected_label_rate)} />
            </div>
            {Object.keys(feedbackQ.data.per_defect).length > 0 && (
              <table className="table" style={{ marginTop: 8 }}>
                <thead>
                  <tr>
                    <th>defect</th>
                    <th>confirmation</th>
                    <th>label agreement</th>
                    <th>pass override</th>
                    <th>corrected</th>
                    <th>n</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(feedbackQ.data.per_defect).map(([k, v]) => (
                    <tr key={k}>
                      <td>{k}</td>
                      <td>{fmtRate(v.defect_confirmation_rate)}</td>
                      <td>{fmtRate(v.ai_human_label_agreement_rate)}</td>
                      <td>{fmtRate(v.pass_override_rate)}</td>
                      <td>{fmtRate(v.corrected_label_rate)}</td>
                      <td>{v.resolved}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </>
        )}
      </section>
    </div>
  );
}
