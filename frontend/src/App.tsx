import { useState } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { OverviewPage } from "./features/overview/OverviewPage";
import { LivePage } from "./features/live/LivePage";
import { TracePage } from "./features/trace/TracePage";

type Tab = "overview" | "live" | "trace";

const TABS: Array<{ key: Tab; label: string }> = [
  { key: "overview", label: "Production Overview" },
  { key: "live", label: "Live Inspection" },
  { key: "trace", label: "Quality Traceability" },
];

const queryClient = new QueryClient({
  defaultOptions: { queries: { staleTime: 1500, refetchOnWindowFocus: false } },
});

function Dashboard() {
  const [tab, setTab] = useState<Tab>("overview");
  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">IVQC</span>
          <span className="brand-title">Industrial Vision · Quality Control</span>
        </div>
        <nav className="tabs">
          {TABS.map((t) => (
            <button key={t.key} className={`tab ${tab === t.key ? "active" : ""}`} onClick={() => setTab(t.key)}>
              {t.label}
            </button>
          ))}
        </nav>
      </header>
      <main className="main">
        {tab === "overview" ? <OverviewPage /> : null}
        {tab === "live" ? <LivePage /> : null}
        {tab === "trace" ? <TracePage /> : null}
      </main>
    </div>
  );
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Dashboard />
    </QueryClientProvider>
  );
}
