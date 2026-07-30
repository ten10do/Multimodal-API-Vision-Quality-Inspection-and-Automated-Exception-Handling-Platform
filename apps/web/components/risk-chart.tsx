"use client";

import {
  Bar,
  BarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { DashboardStats } from "@/lib/types";

export function RiskChart({ stats }: { stats: DashboardStats }) {
  const data = [
    { label: "低", value: stats.by_risk.low },
    { label: "中", value: stats.by_risk.medium },
    { label: "高", value: stats.by_risk.high },
    { label: "严重", value: stats.by_risk.critical },
  ];
  return (
    <div className="h-44 w-full" aria-label="风险等级分布图">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={data}
          margin={{ top: 8, right: 8, left: -30, bottom: 0 }}
        >
          <XAxis
            dataKey="label"
            axisLine={false}
            tickLine={false}
            fontSize={12}
          />
          <YAxis
            allowDecimals={false}
            axisLine={false}
            tickLine={false}
            fontSize={11}
          />
          <Tooltip cursor={{ fill: "#f2f5f3" }} />
          <Bar dataKey="value" fill="#233d2f" radius={[5, 5, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
