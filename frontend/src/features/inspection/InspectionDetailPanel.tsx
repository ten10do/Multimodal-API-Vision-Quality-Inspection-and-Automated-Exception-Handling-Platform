import { useEffect } from "react";
import type { Inspection } from "../../types";
import { BBoxImage } from "../../components/BBoxImage";
import { StatusBadge } from "../../components/StatusBadge";

/** Inspection detail: original image + BBoxes + defects + full traceability. */
export function InspectionDetailPanel({ inspection, onClose }: { inspection: Inspection; onClose: () => void }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <h3>Inspection {inspection.inspection_id}</h3>
          <button className="btn" onClick={onClose}>
            ×
          </button>
        </div>

        <div className="modal-body">
          <div className="detail-grid">
            <div className="detail-col">
              <h4>原始图像与 Bounding Box</h4>
              {inspection.image_url ? (
                <BBoxImage imageUrl={inspection.image_url} defects={inspection.defects} />
              ) : (
                <div className="state-block empty">图像不可用</div>
              )}

              <h4>Defects</h4>
              {inspection.defects.length === 0 ? (
                <div className="state-block empty">无缺陷</div>
              ) : (
                <table className="table">
                  <thead>
                    <tr>
                      <th>class</th>
                      <th>confidence</th>
                      <th>bbox_xyxy</th>
                      <th>area_ratio</th>
                      <th>severity</th>
                      <th>matched_rule</th>
                    </tr>
                  </thead>
                  <tbody>
                    {inspection.defects.map((d, idx) => (
                      <tr key={idx}>
                        <td>{d.class_name}</td>
                        <td>{Math.round(d.confidence * 100)}%</td>
                        <td className="mono">
                          [{d.bbox_xyxy.map((v) => Math.round(v)).join(", ")}]
                        </td>
                        <td>{(d.defect_area_ratio * 100).toFixed(2)}%</td>
                        <td>{d.severity ?? "—"}</td>
                        <td className="mono">{d.matched_rule ?? "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>

            <div className="detail-col">
              <h4>质检信息</h4>
              <dl className="kv">
                <dt>product_id</dt>
                <dd>{inspection.product_id}</dd>
                <dt>batch_id</dt>
                <dd>{inspection.batch_id ?? "—"}</dd>
                <dt>production_line</dt>
                <dd>{inspection.product?.production_line ?? "—"}</dd>
                <dt>station</dt>
                <dd>{inspection.product?.station ?? "—"}</dd>
                <dt>process_status</dt>
                <dd>
                  <StatusBadge
                    status={inspection.status === "FAILED" ? "FAILED" : "COMPLETED"}
                    quality={inspection.quality_result}
                  />
                </dd>
                <dt>quality_result</dt>
                <dd>{inspection.quality_result ?? "—"}</dd>
                <dt>severity</dt>
                <dd>{inspection.severity ?? "—"}</dd>
                <dt>model_version</dt>
                <dd className="mono">{inspection.model_version ?? "—"}</dd>
                <dt>rule_version</dt>
                <dd>{inspection.rule_version ?? "—"}</dd>
                <dt>inference_latency_ms</dt>
                <dd>{inspection.inference_latency_ms === null ? "—" : Math.round(inspection.inference_latency_ms)}</dd>
                <dt>created_at</dt>
                <dd>{new Date(inspection.created_at).toLocaleString("zh-CN", { hour12: false })}</dd>
                {inspection.error_message ? (
                  <>
                    <dt>error_message</dt>
                    <dd className="mono err">{inspection.error_message}</dd>
                  </>
                ) : null}
              </dl>

              <h4>工业执行（Phase 7）</h4>
              <dl className="kv">
                <dt>Desired Command</dt>
                <dd>{inspection.desired_command ?? "—"}</dd>
                <dt>Execution Status</dt>
                <dd className="mono">{inspection.execution_status ?? "—"}</dd>
                <dt>Industrial State</dt>
                <dd>
                  <IndustrialBadge state={inspection.industrial_final_state ?? inspection.industrial_state ?? null} />
                </dd>
                <dt>PLC Adapter</dt>
                <dd className="mono">{inspection.plc_adapter_type ?? "—"}</dd>
                <dt>PLC Latency</dt>
                <dd>{inspection.plc_latency_ms == null ? "—" : `${Math.round(inspection.plc_latency_ms)} ms`}</dd>
                <dt>MES Sync</dt>
                <dd>
                  <MesBadge status={inspection.mes_sync_status ?? null} />
                </dd>
                <dt>Reason Code</dt>
                <dd className="mono">{inspection.plc_reason_code ?? "—"}</dd>
              </dl>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

const STATE_STYLE: Record<string, string> = {
  NOT_INTEGRATED: "badge-neutral",
  HELD: "badge-warn",
  SAFE_HOLD: "badge-danger",
  REJECTED: "badge-danger",
  RELEASED: "badge-ok",
  COMMAND_FAILED: "badge-danger",
};

function IndustrialBadge({ state }: { state: string | null }) {
  if (!state) return <span>—</span>;
  const cls = STATE_STYLE[state] ?? "badge-neutral";
  return <span className={`badge ${cls}`}>{state}</span>;
}

function MesBadge({ status }: { status: string | null }) {
  if (!status) return <span>—</span>;
  const cls = status === "SYNCED" ? "badge-ok" : status === "PENDING" ? "badge-warn" : "badge-danger";
  return <span className={`badge ${cls}`}>{status}</span>;
}
