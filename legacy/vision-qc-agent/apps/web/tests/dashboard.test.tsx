import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import DashboardPage from "@/app/page";

const mocks = vi.hoisted(() => ({
  getInspections: vi.fn(),
  getStats: vi.fn(),
}));

vi.mock("@/lib/api", () => ({
  getInspections: mocks.getInspections,
  getStats: mocks.getStats,
  createInspection: vi.fn(),
}));
vi.mock("@/components/risk-chart", () => ({
  RiskChart: () => <div aria-label="风险等级分布图">chart</div>,
}));
vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn() }),
}));

function renderPage() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <DashboardPage />
    </QueryClientProvider>,
  );
}

describe("dashboard", () => {
  beforeEach(() => {
    mocks.getInspections.mockReset();
    mocks.getStats.mockReset();
  });

  it("renders persisted metrics and recent inspection status", async () => {
    mocks.getStats.mockResolvedValue({
      total: 12,
      completed: 8,
      awaiting_approval: 2,
      manual_review: 1,
      defect_rate: 0.5,
      by_risk: { low: 6, medium: 2, high: 2, critical: 2 },
    });
    mocks.getInspections.mockResolvedValue({
      total: 1,
      items: [
        {
          id: "inspection-12345678",
          product_code: "AX-240",
          batch_code: "B-01",
          status: "manual_review",
          risk_level: "medium",
          disposition: "manual_review",
          created_at: "2026-07-30T08:00:00Z",
        },
      ],
    });
    renderPage();
    expect(await screen.findByText("12")).toBeInTheDocument();
    expect(screen.getByText("8")).toBeInTheDocument();
    expect(screen.getByText("2")).toBeInTheDocument();
    expect(screen.getByText("AX-240")).toBeInTheDocument();
    expect(screen.getAllByText("人工复检")).toHaveLength(2);
    expect(screen.getByText("中风险")).toBeInTheDocument();
    expect(screen.getByLabelText("风险等级分布图")).toBeInTheDocument();
  });
});
