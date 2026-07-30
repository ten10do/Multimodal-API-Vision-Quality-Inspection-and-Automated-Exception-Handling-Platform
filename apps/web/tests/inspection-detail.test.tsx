import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import InspectionDetailPage from "@/app/inspections/[id]/page";
import type { Inspection, WorkflowAction } from "@/lib/types";

const mocks = vi.hoisted(() => ({
  getInspection: vi.fn(),
  decideApproval: vi.fn(),
  sendFeedback: vi.fn(),
}));

vi.mock("@/lib/api", () => ({
  getInspection: mocks.getInspection,
  decideApproval: mocks.decideApproval,
  sendFeedback: mocks.sendFeedback,
}));
vi.mock("next/navigation", () => ({
  useParams: () => ({ id: "inspection-1" }),
}));

const now = "2026-07-30T08:00:00Z";

function action(
  actionType: string,
  status: WorkflowAction["status"] = "succeeded",
): WorkflowAction {
  return {
    id: `${actionType}-id`,
    action_type: actionType,
    status,
    result_payload: { reference: `SIM-${actionType}` },
    created_at: now,
  };
}

function inspection(overrides: Partial<Inspection> = {}): Inspection {
  return {
    id: "inspection-1",
    product_code: "AX-240",
    batch_code: "B-01",
    original_filename: "part.png",
    content_type: "image/png",
    status: "awaiting_approval",
    risk_level: "critical",
    disposition: "stop_line",
    vision_result: {
      summary: "发现结构损伤",
      defects: [{ defect_type: "structural_damage" }],
    },
    analysis_result: {
      rationale: "存在批次性风险",
      probable_causes: ["工艺参数漂移"],
      recommended_actions: ["隔离产品"],
    },
    error_code: null,
    error_message: null,
    created_at: now,
    updated_at: now,
    actions: [
      action("create_ticket"),
      action("request_line_stop", "pending_approval"),
    ],
    audit_logs: [
      {
        id: "audit-1",
        actor_type: "system",
        actor_id: "workflow",
        event_type: "workflow_actions_executed",
        previous_state: null,
        new_state: { status: "awaiting_approval" },
        detail: {},
        created_at: now,
      },
    ],
    feedback: [],
    ...overrides,
  };
}

function renderPage() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <InspectionDetailPage />
    </QueryClientProvider>,
  );
}

describe("inspection detail", () => {
  beforeEach(() => {
    process.env.NEXT_PUBLIC_OPERATOR_ROLE = "supervisor";
    mocks.getInspection.mockReset();
    mocks.decideApproval.mockReset();
    mocks.sendFeedback.mockReset();
  });

  it("renders status, model details and work order state", async () => {
    mocks.getInspection.mockResolvedValue(inspection());
    renderPage();
    expect(await screen.findByText("发现结构损伤")).toBeInTheDocument();
    expect(screen.getByText("等待审批")).toBeInTheDocument();
    expect(screen.getByText("严重")).toBeInTheDocument();
    expect(screen.getByText("创建异常工单")).toBeInTheDocument();
    expect(screen.getByText("已完成")).toBeInTheDocument();
    expect(screen.getByText("待审批")).toBeInTheDocument();
  });

  it("disables stop approval for a non-supervisor role", async () => {
    process.env.NEXT_PUBLIC_OPERATOR_ROLE = "operator";
    mocks.getInspection.mockResolvedValue(inspection());
    renderPage();
    const approve = await screen.findByRole("button", {
      name: "批准模拟停线",
    });
    expect(approve).toBeDisabled();
    expect(screen.getByText("当前角色无停线审批权限")).toBeInTheDocument();
  });

  it("updates the action list after human stop approval", async () => {
    const user = userEvent.setup();
    mocks.getInspection.mockResolvedValue(inspection());
    mocks.decideApproval.mockResolvedValue(
      inspection({
        status: "completed",
        actions: [
          action("create_ticket"),
          action("request_line_stop"),
          action("execute_line_stop"),
        ],
      }),
    );
    renderPage();
    await user.click(
      await screen.findByRole("button", { name: "批准模拟停线" }),
    );
    expect(await screen.findByText("执行模拟停线")).toBeInTheDocument();
    expect(mocks.decideApproval).toHaveBeenCalledWith(
      "inspection-1",
      expect.objectContaining({ decision: "approve" }),
    );
  });

  it("shows Provider fallback errors", async () => {
    mocks.getInspection.mockResolvedValue(
      inspection({
        status: "manual_review",
        risk_level: "medium",
        disposition: "manual_review",
        error_code: "provider_unavailable",
        error_message: "Provider 调用失败，已转人工复检",
      }),
    );
    renderPage();
    expect(
      await screen.findByText("Provider 错误：Provider 调用失败，已转人工复检"),
    ).toBeInTheDocument();
  });

  it("saves and displays human review feedback", async () => {
    const user = userEvent.setup();
    mocks.getInspection.mockResolvedValue(inspection({ status: "completed" }));
    mocks.sendFeedback.mockResolvedValue(
      inspection({
        status: "completed",
        feedback: [
          {
            id: "feedback-1",
            reviewer: "值班主管",
            comment: "已核对当前批次和设备状态",
            corrected_risk: null,
            corrected_disposition: null,
            created_at: now,
          },
        ],
      }),
    );
    renderPage();
    await user.click(await screen.findByRole("button", { name: "保存反馈" }));
    expect(await screen.findByText(/最近反馈：值班主管/)).toBeInTheDocument();
    expect(mocks.sendFeedback).toHaveBeenCalled();
  });

  it("renders a failed work order status update", async () => {
    mocks.getInspection.mockResolvedValue(
      inspection({ actions: [action("create_ticket", "failed")] }),
    );
    renderPage();
    expect(await screen.findByText("执行失败")).toBeInTheDocument();
  });
});
