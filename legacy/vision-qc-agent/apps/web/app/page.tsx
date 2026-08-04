"use client";

import { useQuery } from "@tanstack/react-query";
import {
  AlertTriangle,
  CheckCircle2,
  ClipboardCheck,
  ScanLine,
} from "lucide-react";
import Link from "next/link";
import { RiskChart } from "@/components/risk-chart";
import { Shell } from "@/components/shell";
import { RiskBadge, StatusBadge } from "@/components/status-badge";
import { Card } from "@/components/ui/card";
import { UploadPanel } from "@/components/upload-panel";
import { getInspections, getStats } from "@/lib/api";

export default function DashboardPage() {
  const inspections = useQuery({
    queryKey: ["inspections"],
    queryFn: getInspections,
    refetchInterval: 5_000,
  });
  const stats = useQuery({ queryKey: ["stats"], queryFn: getStats });

  return (
    <Shell>
      <main className="mx-auto w-full max-w-7xl px-5 py-8 lg:px-8">
        <section className="mb-8 grid gap-5 lg:grid-cols-[1.55fr_.85fr]">
          <div className="rounded-3xl bg-[#233d2f] p-7 text-white lg:p-9">
            <div className="mb-10 flex items-center gap-2 text-xs font-semibold tracking-[0.16em] text-[#d8ff65] uppercase">
              <ScanLine size={16} /> Inspection station / A-03
            </div>
            <h1 className="max-w-2xl text-3xl font-semibold tracking-tight lg:text-5xl">
              上传一张图，
              <br />
              让异常处置走完整个闭环。
            </h1>
            <p className="mt-5 max-w-xl text-sm leading-6 text-emerald-50/75">
              百炼负责看见缺陷，DeepSeek
              负责研判风险；平台负责把放行、剔除、工单、通知与停线审批落到可追溯状态。
            </p>
          </div>
          <Card className="p-6">
            <div className="mb-2 flex items-center justify-between">
              <div>
                <p className="text-xs font-semibold tracking-widest text-slate-400 uppercase">
                  Risk profile
                </p>
                <h2 className="mt-1 text-lg font-semibold">风险分布</h2>
              </div>
              <span className="rounded-full bg-[#eff7d9] px-2.5 py-1 text-xs font-semibold text-[#405b20]">
                实时
              </span>
            </div>
            {stats.data ? (
              <RiskChart stats={stats.data} />
            ) : (
              <div className="grid h-44 place-items-center text-sm text-slate-400">
                等待检测数据
              </div>
            )}
          </Card>
        </section>

        <Card className="mb-8 p-5 lg:p-6">
          <div className="mb-5 flex items-center gap-3">
            <div className="grid size-9 place-items-center rounded-lg bg-[#eff7d9] text-[#233d2f]">
              <ScanLine size={18} />
            </div>
            <div>
              <h2 className="font-semibold">新建质检任务</h2>
              <p className="text-xs text-slate-500">
                文件会先经过扩展名、MIME、大小和可解码性校验
              </p>
            </div>
          </div>
          <UploadPanel />
        </Card>

        <section className="mb-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {[
            {
              label: "累计检测",
              value: stats.data?.total ?? 0,
              icon: ScanLine,
              tone: "bg-slate-100 text-slate-700",
            },
            {
              label: "闭环完成",
              value: stats.data?.completed ?? 0,
              icon: CheckCircle2,
              tone: "bg-emerald-50 text-emerald-700",
            },
            {
              label: "待停线审批",
              value: stats.data?.awaiting_approval ?? 0,
              icon: AlertTriangle,
              tone: "bg-red-50 text-red-700",
            },
            {
              label: "人工复检",
              value: stats.data?.manual_review ?? 0,
              icon: ClipboardCheck,
              tone: "bg-amber-50 text-amber-700",
            },
          ].map((item) => (
            <Card
              key={item.label}
              className="flex items-center justify-between p-5"
            >
              <div>
                <p className="text-xs text-slate-500">{item.label}</p>
                <p className="mt-1 text-2xl font-semibold">{item.value}</p>
              </div>
              <span
                className={`grid size-10 place-items-center rounded-xl ${item.tone}`}
              >
                <item.icon size={19} />
              </span>
            </Card>
          ))}
        </section>

        <Card className="overflow-hidden">
          <div className="flex items-center justify-between border-b px-5 py-4">
            <div>
              <h2 className="font-semibold">最近检测任务</h2>
              <p className="text-xs text-slate-500">
                模型、工作流和人工动作统一追踪
              </p>
            </div>
            <span className="mono text-xs text-slate-400">
              {inspections.data?.total ?? 0} RECORDS
            </span>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[760px] text-left text-sm">
              <thead className="bg-slate-50 text-xs text-slate-500">
                <tr>
                  <th className="px-5 py-3 font-medium">任务 / 产品</th>
                  <th className="px-5 py-3 font-medium">批次</th>
                  <th className="px-5 py-3 font-medium">风险</th>
                  <th className="px-5 py-3 font-medium">状态</th>
                  <th className="px-5 py-3 font-medium">创建时间</th>
                  <th className="px-5 py-3" />
                </tr>
              </thead>
              <tbody className="divide-y">
                {inspections.data?.items.map((item) => (
                  <tr key={item.id} className="hover:bg-slate-50/70">
                    <td className="px-5 py-4">
                      <p className="font-semibold">{item.product_code}</p>
                      <p className="mono mt-0.5 text-[11px] text-slate-400">
                        {item.id.slice(0, 8)}
                      </p>
                    </td>
                    <td className="px-5 py-4 text-slate-600">
                      {item.batch_code}
                    </td>
                    <td className="px-5 py-4">
                      <RiskBadge risk={item.risk_level} />
                    </td>
                    <td className="px-5 py-4">
                      <StatusBadge status={item.status} />
                    </td>
                    <td className="px-5 py-4 text-slate-500">
                      {new Date(item.created_at).toLocaleString("zh-CN")}
                    </td>
                    <td className="px-5 py-4 text-right">
                      <Link
                        href={`/inspections/${item.id}`}
                        className="font-semibold text-[#233d2f] underline-offset-4 hover:underline"
                      >
                        查看闭环
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {!inspections.isLoading && !inspections.data?.items.length && (
              <div className="grid h-36 place-items-center text-sm text-slate-400">
                上传第一张产品图片开始演示
              </div>
            )}
            {inspections.error && (
              <div className="grid h-36 place-items-center px-5 text-center text-sm text-red-600">
                无法连接 API，请先启动后端服务。
              </div>
            )}
          </div>
        </Card>
      </main>
    </Shell>
  );
}
