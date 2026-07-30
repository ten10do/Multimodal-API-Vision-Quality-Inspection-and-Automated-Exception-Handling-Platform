"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  ArrowLeft,
  Bot,
  CheckCircle2,
  Clock3,
  GitBranch,
  ShieldAlert,
  UserCheck,
} from "lucide-react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { useState } from "react";
import { Shell } from "@/components/shell";
import { RiskBadge, StatusBadge } from "@/components/status-badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { decideApproval, getInspection, sendFeedback } from "@/lib/api";

const actionLabels: Record<string, string> = {
  release_product: "产品放行",
  manual_review: "转人工复检",
  reject_product: "不良品剔除",
  create_ticket: "创建异常工单",
  send_notification: "发送告警通知",
  request_line_stop: "申请模拟停线",
  execute_line_stop: "执行模拟停线",
};
const actionStatusLabels: Record<string, string> = {
  succeeded: "已完成",
  pending_approval: "待审批",
  rejected: "已拒绝",
  failed: "执行失败",
};

export default function InspectionDetailPage() {
  const { id } = useParams<{ id: string }>();
  const client = useQueryClient();
  const [reviewer, setReviewer] = useState("值班主管");
  const [comment, setComment] = useState("已核对当前批次和设备状态");
  const canApproveLineStop =
    (process.env.NEXT_PUBLIC_OPERATOR_ROLE ?? "supervisor") === "supervisor";
  const inspection = useQuery({
    queryKey: ["inspection", id],
    queryFn: () => getInspection(id),
    refetchInterval: (query) =>
      ["queued", "vision_analyzing", "reasoning", "executing"].includes(
        query.state.data?.status ?? "",
      )
        ? 2_000
        : false,
  });
  const approval = useMutation({
    mutationFn: (decision: "approve" | "reject") =>
      decideApproval(id, { decision, reviewer, comment }),
    onSuccess: (data) => client.setQueryData(["inspection", id], data),
  });
  const feedback = useMutation({
    mutationFn: () => sendFeedback(id, { reviewer, comment }),
    onSuccess: (data) => client.setQueryData(["inspection", id], data),
  });

  if (inspection.isLoading) {
    return (
      <Shell>
        <div className="mx-auto grid min-h-[60vh] max-w-7xl place-items-center text-sm text-slate-500">
          加载检测闭环…
        </div>
      </Shell>
    );
  }
  if (!inspection.data) {
    return (
      <Shell>
        <div className="mx-auto grid min-h-[60vh] max-w-7xl place-items-center text-red-600">
          任务不存在或 API 不可用
        </div>
      </Shell>
    );
  }
  const data = inspection.data;
  const vision = data.vision_result as {
    summary?: string;
    defects?: Array<Record<string, unknown>>;
  } | null;
  const analysis = data.analysis_result as {
    rationale?: string;
    probable_causes?: string[];
    recommended_actions?: string[];
  } | null;

  return (
    <Shell>
      <main className="mx-auto w-full max-w-7xl px-5 py-8 lg:px-8">
        <Link
          href="/"
          className="mb-5 inline-flex items-center gap-2 text-sm font-semibold text-slate-600 hover:text-slate-900"
        >
          <ArrowLeft size={16} /> 返回检测台
        </Link>
        <section className="mb-6 flex flex-col justify-between gap-4 rounded-2xl bg-[#233d2f] p-6 text-white md:flex-row md:items-center">
          <div>
            <p className="mono text-xs text-[#d8ff65]">
              INSPECTION / {data.id}
            </p>
            <h1 className="mt-2 text-2xl font-semibold">{data.product_code}</h1>
            <p className="mt-1 text-sm text-emerald-50/70">
              批次 {data.batch_code} · {data.original_filename}
            </p>
          </div>
          <div className="flex items-center gap-3">
            <RiskBadge risk={data.risk_level} />
            <StatusBadge status={data.status} />
          </div>
        </section>

        {data.error_message && (
          <Card className="mb-6 border-amber-200 bg-amber-50 p-4 text-sm text-amber-800">
            Provider 错误：{data.error_message}
          </Card>
        )}

        {data.status === "awaiting_approval" && (
          <Card className="mb-6 border-red-200 bg-red-50 p-5">
            <div className="flex items-start gap-4">
              <span className="grid size-10 shrink-0 place-items-center rounded-xl bg-red-100 text-red-700">
                <ShieldAlert size={20} />
              </span>
              <div className="w-full">
                <h2 className="font-semibold text-red-900">
                  严重操作等待人工确认
                </h2>
                <p className="mt-1 text-sm text-red-700">
                  AI 已建议停线，但系统不会自动执行。请核对现场情况后审批。
                </p>
                <div className="mt-4 grid gap-3 md:grid-cols-2">
                  <input
                    value={reviewer}
                    onChange={(event) => setReviewer(event.target.value)}
                    aria-label="审批人"
                    className="h-10 rounded-lg border border-red-200 bg-white px-3 text-sm text-slate-900"
                  />
                  <input
                    value={comment}
                    onChange={(event) => setComment(event.target.value)}
                    aria-label="审批说明"
                    className="h-10 rounded-lg border border-red-200 bg-white px-3 text-sm text-slate-900"
                  />
                </div>
                <div className="mt-3 flex gap-3">
                  <Button
                    variant="danger"
                    disabled={approval.isPending || !canApproveLineStop}
                    onClick={() => approval.mutate("approve")}
                  >
                    批准模拟停线
                  </Button>
                  <Button
                    variant="outline"
                    disabled={approval.isPending || !canApproveLineStop}
                    onClick={() => approval.mutate("reject")}
                  >
                    拒绝并转复检
                  </Button>
                </div>
                {!canApproveLineStop && (
                  <p className="mt-2 text-xs font-semibold text-red-700">
                    当前角色无停线审批权限
                  </p>
                )}
              </div>
            </div>
          </Card>
        )}

        <div className="grid gap-6 lg:grid-cols-[1fr_.9fr]">
          <div className="grid gap-6">
            <Card className="p-5">
              <div className="mb-4 flex items-center gap-3">
                <Bot size={19} className="text-[#233d2f]" />
                <div>
                  <h2 className="font-semibold">AI 检测与研判</h2>
                  <p className="text-xs text-slate-500">
                    严格 Schema 校验后的结构化结果
                  </p>
                </div>
              </div>
              <div className="grid gap-4 md:grid-cols-2">
                <div className="rounded-xl bg-slate-50 p-4">
                  <p className="text-xs font-semibold tracking-wider text-slate-400 uppercase">
                    Bailian vision
                  </p>
                  <p className="mt-2 text-sm leading-6">
                    {vision?.summary ?? "等待视觉结果"}
                  </p>
                  <p className="mt-3 text-xs text-slate-500">
                    缺陷数量：{vision?.defects?.length ?? 0}
                  </p>
                </div>
                <div className="rounded-xl bg-[#f3f8e7] p-4">
                  <p className="text-xs font-semibold tracking-wider text-[#657b35] uppercase">
                    DeepSeek reasoning
                  </p>
                  <p className="mt-2 text-sm leading-6">
                    {analysis?.rationale ?? "等待异常研判"}
                  </p>
                </div>
              </div>
              <div className="mt-5 grid gap-4 md:grid-cols-2">
                <div>
                  <h3 className="text-xs font-semibold text-slate-500">
                    可能根因
                  </h3>
                  <ul className="mt-2 grid gap-2 text-sm">
                    {analysis?.probable_causes?.map((cause) => (
                      <li key={cause} className="flex gap-2">
                        <GitBranch
                          size={15}
                          className="mt-0.5 text-slate-400"
                        />
                        {cause}
                      </li>
                    )) ?? <li>暂无</li>}
                  </ul>
                </div>
                <div>
                  <h3 className="text-xs font-semibold text-slate-500">
                    推荐动作
                  </h3>
                  <ul className="mt-2 grid gap-2 text-sm">
                    {analysis?.recommended_actions?.map((action) => (
                      <li key={action} className="flex gap-2">
                        <CheckCircle2
                          size={15}
                          className="mt-0.5 text-emerald-600"
                        />
                        {action}
                      </li>
                    )) ?? <li>暂无</li>}
                  </ul>
                </div>
              </div>
            </Card>

            <Card className="p-5">
              <div className="mb-4 flex items-center gap-3">
                <UserCheck size={19} />
                <div>
                  <h2 className="font-semibold">人工反馈</h2>
                  <p className="text-xs text-slate-500">
                    保存复核意见，形成质量闭环
                  </p>
                </div>
              </div>
              <div className="grid gap-3 sm:grid-cols-[.7fr_1.3fr_auto]">
                <input
                  value={reviewer}
                  onChange={(event) => setReviewer(event.target.value)}
                  aria-label="复核人"
                  className="h-10 rounded-lg border px-3 text-sm"
                />
                <input
                  value={comment}
                  onChange={(event) => setComment(event.target.value)}
                  aria-label="反馈内容"
                  className="h-10 rounded-lg border px-3 text-sm"
                />
                <Button
                  variant="outline"
                  disabled={!reviewer || !comment || feedback.isPending}
                  onClick={() => feedback.mutate()}
                >
                  保存反馈
                </Button>
              </div>
              {data.feedback.length > 0 && (
                <div className="mt-4 border-t pt-4 text-sm text-slate-600">
                  最近反馈：{data.feedback.at(-1)?.reviewer} ·{" "}
                  {data.feedback.at(-1)?.comment}
                </div>
              )}
            </Card>
          </div>

          <div className="grid content-start gap-6">
            <Card className="p-5">
              <h2 className="font-semibold">处置动作</h2>
              <div className="mt-4 grid gap-3">
                {data.actions.map((action, index) => (
                  <div key={action.id} className="flex items-start gap-3">
                    <span className="mono grid size-7 shrink-0 place-items-center rounded-full bg-[#233d2f] text-[11px] text-white">
                      {String(index + 1).padStart(2, "0")}
                    </span>
                    <div className="flex-1 border-b pb-3">
                      <div className="flex items-center justify-between gap-3">
                        <p className="text-sm font-semibold">
                          {actionLabels[action.action_type] ??
                            action.action_type}
                        </p>
                        <span className="text-xs text-slate-400">
                          {actionStatusLabels[action.status] ?? action.status}
                        </span>
                      </div>
                      <p className="mono mt-1 text-[10px] text-slate-400">
                        {String(action.result_payload?.reference ?? "SIM")}
                      </p>
                    </div>
                  </div>
                ))}
                {!data.actions.length && (
                  <p className="text-sm text-slate-400">等待工作流执行</p>
                )}
              </div>
            </Card>

            <Card className="p-5">
              <h2 className="font-semibold">审计时间线</h2>
              <div className="mt-4 grid gap-4">
                {data.audit_logs.map((entry) => (
                  <div key={entry.id} className="flex gap-3">
                    <Clock3
                      size={15}
                      className="mt-0.5 shrink-0 text-slate-400"
                    />
                    <div>
                      <p className="text-sm font-medium">{entry.event_type}</p>
                      <p className="mt-0.5 text-xs text-slate-400">
                        {entry.actor_type}/{entry.actor_id} ·{" "}
                        {new Date(entry.created_at).toLocaleString("zh-CN")}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          </div>
        </div>
      </main>
    </Shell>
  );
}
