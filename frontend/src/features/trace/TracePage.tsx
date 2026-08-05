import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "../../api/client";
import type { Inspection, InspectionFilters, QualityResult } from "../../types";
import { StatusBadge } from "../../components/StatusBadge";
import { EmptyState, ErrorState, LoadingState } from "../../components/StateViews";
import { InspectionDetailPanel } from "../inspection/InspectionDetailPanel";

export function TracePage() {
  const [filters, setFilters] = useState<InspectionFilters>({ limit: 100 });
  const [selected, setSelected] = useState<Inspection | null>(null);

  const q = useQuery({
    queryKey: ["inspections", "search", filters],
    queryFn: () => api.listInspections(filters),
    retry: 1,
  });

  const set = (patch: Partial<InspectionFilters>) => setFilters((f) => ({ ...f, ...patch }));

  return (
    <div className="page">
      <section className="panel">
        <h3>质量追溯查询</h3>
        <div className="filter-row">
          <input placeholder="product_id" value={filters.product_id ?? ""} onChange={(e) => set({ product_id: e.target.value || undefined })} />
          <input placeholder="inspection_id" value={filters.inspection_id ?? ""} onChange={(e) => set({ inspection_id: e.target.value || undefined })} />
          <input placeholder="batch_id" value={filters.batch_id ?? ""} onChange={(e) => set({ batch_id: e.target.value || undefined })} />
          <select value={filters.quality_result ?? ""} onChange={(e) => set({ quality_result: (e.target.value || undefined) as QualityResult | undefined })}>
            <option value="">quality: all</option>
            <option value="PASS">PASS</option>
            <option value="REVIEW">REVIEW</option>
            <option value="FAIL">FAIL</option>
          </select>
          <input placeholder="defect_type" value={filters.defect_type ?? ""} onChange={(e) => set({ defect_type: e.target.value || undefined })} />
          <input placeholder="production_line" value={filters.production_line ?? ""} onChange={(e) => set({ production_line: e.target.value || undefined })} />
          <input placeholder="station" value={filters.station ?? ""} onChange={(e) => set({ station: e.target.value || undefined })} />
          <input type="datetime-local" onChange={(e) => set({ date_from: e.target.value ? new Date(e.target.value).toISOString() : undefined })} />
          <input type="datetime-local" onChange={(e) => set({ date_to: e.target.value ? new Date(e.target.value).toISOString() : undefined })} />
          <button className="btn" onClick={() => void q.refetch()}>
            查询
          </button>
        </div>
      </section>

      <section className="panel">
        <h3>查询结果（{q.data?.length ?? 0}）</h3>
        {q.isLoading ? (
          <LoadingState />
        ) : q.isError ? (
          <ErrorState message={(q.error as Error).message ?? "查询失败"} onRetry={() => void q.refetch()} />
        ) : q.data && q.data.length === 0 ? (
          <EmptyState message="无匹配记录" />
        ) : (
          <table className="table clickable">
            <thead>
              <tr>
                <th>结果</th>
                <th>product_id</th>
                <th>inspection_id</th>
                <th>batch</th>
                <th>line / station</th>
                <th>缺陷</th>
                <th>model</th>
                <th>时间</th>
              </tr>
            </thead>
            <tbody>
              {(q.data ?? []).map((i) => (
                <tr key={i.inspection_id} onClick={() => setSelected(i)}>
                  <td>
                    <StatusBadge status={i.status === "FAILED" ? "FAILED" : "COMPLETED"} quality={i.quality_result} />
                  </td>
                  <td>{i.product_id}</td>
                  <td className="mono">{i.inspection_id}</td>
                  <td>{i.batch_id ?? "—"}</td>
                  <td>
                    {i.product?.production_line ?? "—"} / {i.product?.station ?? "—"}
                  </td>
                  <td>{i.defects.length}</td>
                  <td className="mono">{i.model_version ?? "—"}</td>
                  <td>{new Date(i.created_at).toLocaleString("zh-CN", { hour12: false })}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      {selected ? <InspectionDetailPanel inspection={selected} onClose={() => setSelected(null)} /> : null}
    </div>
  );
}
