import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { fireEvent } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ReviewQueuePage } from "../src/features/review/ReviewQueuePage";
import type { ReviewTask } from "../src/types";

vi.mock("../src/api/client", () => ({
  api: {
    listReviews: vi.fn(),
    reviewMetrics: vi.fn(),
    claimReview: vi.fn(),
    resolveReview: vi.fn(),
  },
}));

import { api } from "../src/api/client";

const task = (patch: Partial<ReviewTask> = {}): ReviewTask => ({
  review_task_id: "rt-1",
  inspection_id: "insp-1",
  inspection: null,
  status: "PENDING",
  priority: 200,
  assigned_to: null,
  claimed_at: null,
  resolved_at: null,
  version: 1,
  ai_quality_result: "REVIEW",
  ai_defects_snapshot: [{ class_name: "crazing", confidence: 0.42 }] as unknown as ReviewTask["ai_defects_snapshot"],
  ai_model_version: "v1",
  ai_rule_version: 1,
  ai_severity: "medium",
  product_id: "P-1",
  production_line: "line-a",
  station: "qc-01",
  batch_id: "b1",
  image_url: "/api/v1/inspections/insp-1/image",
  decision: null,
  created_at: "2026-08-05T10:00:00Z",
  updated_at: "2026-08-05T10:00:00Z",
  ...patch,
});

const metrics = {
  pending_review_count: 1,
  pending: 1,
  in_review: 0,
  resolved: 0,
  average_review_wait_time_s: null,
  review_rate: 0.1,
  ai_human_agreement_rate: null,
  override_rate: null,
  corrected_label_count: 0,
  pass_overrides: 0,
};

function renderPage() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <ReviewQueuePage />
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  (api.listReviews as ReturnType<typeof vi.fn>).mockResolvedValue([task()]);
  (api.reviewMetrics as ReturnType<typeof vi.fn>).mockResolvedValue(metrics);
});

describe("ReviewQueuePage", () => {
  it("renders the queue row with AI defect and waiting time", async () => {
    renderPage();
    await screen.findByText("P-1");
    expect(screen.getByText("crazing")).toBeTruthy();
    expect(screen.getByText("待认领")).toBeTruthy();
    expect(screen.getByText(/Pending Reviews/)).toBeTruthy();
  });

  it("claims a task and shows claim conflict message on 409", async () => {
    (api.claimReview as ReturnType<typeof vi.fn>)
      .mockRejectedValueOnce(Object.assign(new Error("conflict"), { code: "already_claimed", status: 409 }));
    renderPage();
    await screen.findByText("P-1");
    fireEvent.click(screen.getByText("P-1"));
    await screen.findByText("Review rt-1");
    fireEvent.click(screen.getByText("Claim"));
    await screen.findByText(/已被其他质检员认领/);
  });

  it("shows review controls after claim (resolve update path)", async () => {
    (api.claimReview as ReturnType<typeof vi.fn>).mockResolvedValue(
      task({ status: "IN_REVIEW", assigned_to: "qc-worker-01", version: 2 }),
    );
    (api.resolveReview as ReturnType<typeof vi.fn>).mockResolvedValue(
      task({ status: "RESOLVED", decision: { human_decision: "PASS", final_quality_result: "PASS" } as never }),
    );
    renderPage();
    await screen.findByText("P-1");
    fireEvent.click(screen.getByText("P-1"));
    await screen.findByText("Review rt-1");
    fireEvent.click(screen.getByText("Claim"));
    await screen.findByText("Resolve");
    // decision validation: CONFIRM_DEFECT without a label must be blocked
    fireEvent.click(screen.getByLabelText("CONFIRM_DEFECT"));
    fireEvent.click(screen.getByText("Resolve"));
    await screen.findByText(/需要填写缺陷类别/);
    expect(api.resolveReview).not.toHaveBeenCalled();
    // valid PASS resolve goes through
    fireEvent.click(screen.getByLabelText("PASS"));
    fireEvent.click(screen.getByText("Resolve"));
    await waitFor(() => expect(api.resolveReview).toHaveBeenCalledWith("rt-1", "qc-worker-01", "PASS", null, null));
  });
});
