import { Activity, Boxes, ShieldCheck } from "lucide-react";
import Link from "next/link";

export function Shell({ children }: { children: React.ReactNode }) {
  return (
    <div className="grid-lines min-h-screen">
      <header className="border-b bg-white/90 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-5 py-4 lg:px-8">
          <Link href="/" className="flex items-center gap-3">
            <span className="grid size-10 place-items-center rounded-xl bg-[#233d2f] text-[#d8ff65]">
              <ShieldCheck size={21} />
            </span>
            <span>
              <span className="block text-sm font-bold tracking-tight">
                VISION QC AGENT
              </span>
              <span className="block text-[11px] text-slate-500">
                异常闭环自动化平台
              </span>
            </span>
          </Link>
          <div className="hidden items-center gap-6 text-xs text-slate-500 sm:flex">
            <span className="flex items-center gap-2">
              <Activity size={14} className="text-emerald-600" />双 Provider
              编排
            </span>
            <span className="flex items-center gap-2">
              <Boxes size={14} />
              模拟产线 A-03
            </span>
          </div>
        </div>
      </header>
      {children}
    </div>
  );
}
