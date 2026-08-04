import type { InspectionStatus, RiskLevel } from "@/lib/types";
import { cn } from "@/lib/utils";

const statusLabel: Record<InspectionStatus, string> = {
  queued: "排队中",
  vision_analyzing: "视觉分析",
  reasoning: "异常研判",
  executing: "执行处置",
  awaiting_approval: "等待审批",
  completed: "已完成",
  manual_review: "人工复检",
  failed: "失败",
};

const riskLabel: Record<RiskLevel, string> = {
  low: "低风险",
  medium: "中风险",
  high: "高风险",
  critical: "严重",
};

export function StatusBadge({ status }: { status: InspectionStatus }) {
  return (
    <span
      className={cn(
        "inline-flex rounded-full px-2.5 py-1 text-xs font-semibold",
        status === "completed" && "bg-emerald-50 text-emerald-700",
        status === "awaiting_approval" && "bg-red-50 text-red-700",
        status === "manual_review" && "bg-amber-50 text-amber-700",
        !["completed", "awaiting_approval", "manual_review"].includes(status) &&
          "bg-slate-100 text-slate-700",
      )}
    >
      {statusLabel[status]}
    </span>
  );
}

export function RiskBadge({ risk }: { risk: RiskLevel | null }) {
  if (!risk) return <span className="text-sm text-slate-400">—</span>;
  return (
    <span
      className={cn(
        "inline-flex rounded-full px-2.5 py-1 text-xs font-semibold",
        risk === "low" && "bg-emerald-50 text-emerald-700",
        risk === "medium" && "bg-yellow-50 text-yellow-700",
        risk === "high" && "bg-orange-50 text-orange-700",
        risk === "critical" && "bg-red-50 text-red-700",
      )}
    >
      {riskLabel[risk]}
    </span>
  );
}
