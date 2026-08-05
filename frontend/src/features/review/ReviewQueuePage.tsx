import { useCallback, useEffect, useMemo, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { api } from "../../api/client";
import { useReviews, useReviewMetrics } from "../../hooks/queries";
import { useInspectionSocket } from "../../hooks/useInspectionSocket";
import type { HumanDecision, ReviewTask } from "../../types";
import {
  reviewWaitSeconds,
  topDefect,
  validateReviewDecision,
} from "../../utils/transforms";
import { StatusBadge } from "../../components/StatusBadge";
import { BBoxImage } from "../../components/BBoxImage";
import { EmptyState, ErrorState, LoadingState } from "../../components/StateViews";

const REVIEWER = "qc-worker-01"; // first version: explicit reviewer identifier (5C)

const STATUS_LABEL: Record<string, string> = {
  PENDING: "待认领",
  IN_REVIEW: "复核中",
  RESOLVED: "已复核",
};

export function ReviewQueuePage() {
  const queryClient = useQueryClient();
  const filters = useMemo(() => ({ limit: 200 }), []);
  const [selected, setSelected] = useState<ReviewTask | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const reviewsQ = useReviews(filters, 5000);
  const metricsQ = useReviewMetrics(5000);

  // WS review events + REST reconciliation on (re)connect (5I / 4F)
  const reconcile = useCallback(() => {
    void reviewsQ.refetch();
    void metricsQ.refetch();
  }, [reviewsQ, metricsQ]);
  const { events: wsEvents, state } = useInspectionSocket(reconcile);

  // Live patch: apply review.* events onto the queue list and metrics.
  useEffect(() => {
    const reviewEvents = wsEvents.filter((e) => e.event_type.startsWith("review."));
    if (reviewEvents.length === 0) return;
    void reviewsQ.refetch();
    void metricsQ.refetch();
  }, [wsEvents, reviewsQ, metricsQ]);

  const tasks = useMemo(() => {
    const map = new Map<string, ReviewTask>();
    for (const t of reviewsQ.data ?? []) map.set(t.review_task_id, t);
    for (const e of wsEvents) {
      if (!e.event_type.startsWith("review.")) continue;
      // WS 是通知，最终以 REST reconciliation 的结果为准；这里仅做轻量提示
    }
    return [...map.values()].sort((a, b) => a.priority - b.priority || +new Date(a.created_at) - +new Date(b.created_at));
  }, [reviewsQ.data, wsEvents]);

  const claim = async (task: ReviewTask) => {
    setBusy(task.review_task_id);
    setError(null);
    try {
      const updated = await api.claimReview(task.review_task_id, REVIEWER);
      setSelected((prev) => (prev?.review_task_id === updated.review_task_id ? updated : prev));
      void reviewsQ.refetch();
      void metricsQ.refetch();
    } catch (e) {
      setError(conflictMessage(e));
    } finally {
      setBusy(null);
    }
  };

  const resolve = async (
    task: ReviewTask,
    decision: HumanDecision,
    label: string | null,
    reason: string | null,
  ) => {
    const validation = validateReviewDecision(decision, label);
    if (validation) {
      setError(validation);
      return;
    }
    setBusy(task.review_task_id);
    setError(null);
    try {
      const updated = await api.resolveReview(task.review_task_id, REVIEWER, decision, label, reason);
      setSelected((prev) => (prev?.review_task_id === updated.review_task_id ? updated : prev));
      void reviewsQ.refetch();
      void metricsQ.refetch();
      void queryClient.refetchQueries({ queryKey: ["inspections"] });
    } catch (e) {
      setError(conflictMessage(e));
    } finally {
      setBusy(null);
    }
  };

  const now = Date.now();
  const m = metricsQ.data;

  return (
    <div className="page">
      <div className="ws-status-bar">
        <span className={`ws-dot ws-${state}`} />
        实时连接：<b>{state}</b>
        {state === "reconnecting" || state === "disconnected" ? (
          <span className="ws-hint">重连中，恢复后将从 REST 拉取最新复核队列</span>
        ) : null}
      </div>

      <section className="metric-grid">
        <div className="metric-card">
          <div className="metric-label">Pending Reviews</div>
          <div className="metric-value">{m?.pending_review_count ?? "—"}</div>
          <div className="metric-hint">pending {m?.pending ?? 0} / in-review {m?.in_review ?? 0}</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Resolved</div>
          <div className="metric-value">{m?.resolved ?? "—"}</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Avg Wait</div>
          <div className="metric-value">
            {m?.average_review_wait_time_s === null || m === undefined ? "—" : `${Math.max(0, Math.round((m.average_review_wait_time_s ?? 0) / 1000)).toFixed(0)}s`}
          </div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Review Rate</div>
          <div className="metric-value">{m?.review_rate === null || m === undefined ? "—" : `${((m.review_rate ?? 0) * 100).toFixed(1)}%`}</div>
          <div className="metric-hint">AI REVIEW / completed</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">AI-Human Agreement</div>
          <div className="metric-value">{m?.ai_human_agreement_rate === null || m === undefined ? "—" : `${((m.ai_human_agreement_rate ?? 0) * 100).toFixed(1)}%`}</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Override Rate</div>
          <div className="metric-value">{m?.override_rate === null || m === undefined ? "—" : `${((m.override_rate ?? 0) * 100).toFixed(1)}%`}</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Corrected Labels</div>
          <div className="metric-value">{m?.corrected_label_count ?? "—"}</div>
          <div className="metric-hint">PASS overrides {m?.pass_overrides ?? 0}</div>
        </div>
      </section>

      {error ? <div className="state-block error" role="alert">{error}</div> : null}

      <section className="panel">
        <h3>Review Queue（{tasks.length}）</h3>
        {reviewsQ.isLoading ? (
          <LoadingState />
        ) : reviewsQ.isError ? (
          <ErrorState message={(reviewsQ.error as Error).message} onRetry={() => void reviewsQ.refetch()} />
        ) : tasks.length === 0 ? (
          <EmptyState message="当前没有复核任务（REVIEW 质检结果会自动进入此队列）" />
        ) : (
          <table className="table clickable">
            <thead>
              <tr>
                <th>状态</th>
                <th>product_id</th>
                <th>图像</th>
                <th>AI defect</th>
                <th>conf</th>
                <th>severity</th>
                <th>rule</th>
                <th>line / station</th>
                <th>等待</th>
                <th>assigned</th>
              </tr>
            </thead>
            <tbody>
              {tasks.map((t) => {
                const top = topDefect(t);
                return (
                  <tr key={t.review_task_id} onClick={() => setSelected(t)}>
                    <td>
                      <StatusBadge
                        status={t.status === "RESOLVED" ? "FAILED" : "COMPLETED"}
                        quality={t.status === "RESOLVED" ? (t.decision?.final_quality_result ?? "FAIL") : t.status === "IN_REVIEW" ? "REVIEW" : null}
                      />
                      <span className="sev">{STATUS_LABEL[t.status]}</span>
                    </td>
                    <td>{t.product_id}</td>
                    <td>
                      {t.image_url ? (
                        <img src={t.image_url} alt="" width={40} height={40} style={{ objectFit: "cover", borderRadius: 3 }} />
                      ) : (
                        "—"
                      )}
                    </td>
                    <td>{top ? `${top.name}` : "—"}</td>
                    <td>{top ? `${Math.round(top.confidence * 100)}%` : "—"}</td>
                    <td>{t.ai_severity ?? "—"}</td>
                    <td className="mono">{t.ai_rule_version ?? "—"}</td>
                    <td>
                      {t.production_line} / {t.station}
                    </td>
                    <td>{reviewWaitSeconds(t, now)}s</td>
                    <td>{t.assigned_to ?? "—"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </section>

      {selected ? (
        <ReviewDetailModal
          task={selected}
          busy={busy === selected.review_task_id}
          onClaim={() => void claim(selected)}
          onResolve={(d, label, reason) => void resolve(selected, d, label, reason)}
          onClose={() => setSelected(null)}
        />
      ) : null}
    </div>
  );
}

function ReviewDetailModal({
  task,
  busy,
  onClaim,
  onResolve,
  onClose,
}: {
  task: ReviewTask;
  busy: boolean;
  onClaim: () => void;
  onResolve: (d: HumanDecision, label: string | null, reason: string | null) => void;
  onClose: () => void;
}) {
  const [decision, setDecision] = useState<HumanDecision>("PASS");
  const [label, setLabel] = useState("");
  const [reason, setReason] = useState("");
  const needsLabel = decision !== "PASS";
  const claimedByMe = task.assigned_to === REVIEWER;
  const resolved = task.status === "RESOLVED";

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <h3>Review {task.review_task_id}</h3>
          <button className="btn" onClick={onClose}>×</button>
        </div>
        <div className="modal-body">
          <div className="detail-grid">
            <div className="detail-col">
              <h4>Original Image + Bounding Boxes</h4>
              {task.image_url ? (
                <BBoxImage
                  imageUrl={task.image_url}
                  defects={(task.inspection?.defects ?? []).map((d) => ({
                    class_name: d.class_name,
                    confidence: d.confidence,
                    bbox_normalized: d.bbox_normalized,
                    bbox_xyxy: d.bbox_xyxy,
                    severity: d.severity,
                    matched_rule: d.matched_rule,
                    defect_area_ratio: d.defect_area_ratio,
                    id: d.id,
                    class_id: d.class_id,
                    defect_area_px: d.defect_area_px,
                  }))}
                />
              ) : (
                <div className="state-block empty">图像不可用</div>
              )}

              <h4>AI Prediction（固化快照）</h4>
              <table className="table">
                <thead>
                  <tr><th>class</th><th>confidence</th><th>severity</th><th>rule</th></tr>
                </thead>
                <tbody>
                  {task.ai_defects_snapshot.length === 0 ? (
                    <tr><td colSpan={4}>无缺陷</td></tr>
                  ) : (
                    task.ai_defects_snapshot.map((d, i) => (
                      <tr key={i}>
                        <td>{d.class_name}</td>
                        <td>{Math.round(d.confidence * 100)}%</td>
                        <td>{d.severity ?? "—"}</td>
                        <td className="mono">{d.matched_rule ?? "—"}</td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
              <div className="kv" style={{ marginTop: 8 }}>
                <dt>AI quality_result</dt>
                <dd>{task.ai_quality_result}</dd>
                <dt>model_version</dt>
                <dd className="mono">{task.ai_model_version ?? "—"}</dd>
                <dt>rule_version</dt>
                <dd>{task.ai_rule_version ?? "—"}</dd>
                <dt>priority</dt>
                <dd>{task.priority}</dd>
              </div>
            </div>

            <div className="detail-col">
              <h4>Product Metadata</h4>
              <dl className="kv">
                <dt>product_id</dt><dd>{task.product_id}</dd>
                <dt>batch_id</dt><dd>{task.batch_id ?? "—"}</dd>
                <dt>production_line</dt><dd>{task.production_line}</dd>
                <dt>station</dt><dd>{task.station}</dd>
                <dt>created_at</dt><dd>{new Date(task.created_at).toLocaleString("zh-CN", { hour12: false })}</dd>
                <dt>assigned_to</dt><dd>{task.assigned_to ?? "—"}</dd>
                <dt>status</dt><dd>{STATUS_LABEL[task.status]}</dd>
              </dl>

              <h4>Review Controls</h4>
              {resolved ? (
                <div className="state-block empty">
                  已复核：{task.decision?.human_decision} → {task.decision?.final_quality_result}
                  {task.decision?.reason ? <div className="metric-hint">reason：{task.decision.reason}</div> : null}
                </div>
              ) : claimedByMe ? (
                <>
                  <div className="filter-row" style={{ marginBottom: 8 }}>
                    {(["PASS", "CONFIRM_DEFECT", "CORRECT_DEFECT", "OTHER_DEFECT"] as HumanDecision[]).map((d) => (
                      <label key={d} style={{ display: "flex", gap: 4, alignItems: "center" }}>
                        <input type="radio" checked={decision === d} onChange={() => setDecision(d)} />
                        {d}
                      </label>
                    ))}
                  </div>
                  {needsLabel ? (
                    <input
                      placeholder="human_label（如 crazing / scratches）"
                      value={label}
                      onChange={(e) => setLabel(e.target.value)}
                      style={{ width: "100%", marginBottom: 8 }}
                    />
                  ) : null}
                  <textarea
                    placeholder="reason / comment"
                    value={reason}
                    onChange={(e) => setReason(e.target.value)}
                    rows={2}
                    style={{ width: "100%", marginBottom: 8, background: "#0b1220", color: "#e2e8f0", border: "1px solid #22304d", borderRadius: 4 }}
                  />
                  <button className="btn" disabled={busy} onClick={() => onResolve(decision, needsLabel ? label : null, reason || null)}>
                    {busy ? "提交中…" : "Resolve"}
                  </button>
                </>
              ) : (
                <button className="btn" disabled={busy || task.status !== "PENDING"} onClick={onClaim}>
                  {busy ? "认领中…" : task.status === "IN_REVIEW" ? `已被 ${task.assigned_to} 认领` : "Claim"}
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function conflictMessage(e: unknown): string {
  if (e instanceof Error && "code" in e) {
    const code = (e as { code?: string }).code;
    if (code === "already_claimed") return "冲突：该任务已被其他质检员认领（409）";
    if (code === "already_resolved") return "冲突：该任务已被复核完成（409）";
    if (code === "not_owner") return "冲突：该任务不属于当前质检员（409）";
    if (code === "not_claimed") return "该任务必须先认领才能复核";
  }
  return e instanceof Error ? e.message : String(e);
}
