import { afterEach, describe, expect, it, vi } from "vitest";
import {
  createInspection,
  decideApproval,
  getStats,
  sendFeedback,
} from "@/lib/api";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("API client", () => {
  it("parses dashboard responses", async () => {
    const payload = {
      total: 1,
      completed: 1,
      awaiting_approval: 0,
      manual_review: 0,
      defect_rate: 0,
      by_risk: { low: 1, medium: 0, high: 0, critical: 0 },
    };
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify(payload), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
      ),
    );
    await expect(getStats()).resolves.toEqual(payload);
  });

  it("surfaces the unified Provider error message", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({
            error: {
              code: "provider_unavailable",
              message: "Provider 已安全降级",
              request_id: "request-1",
            },
          }),
          { status: 503, headers: { "Content-Type": "application/json" } },
        ),
      ),
    );
    await expect(getStats()).rejects.toThrow("Provider 已安全降级");
  });

  it("sends idempotency, approval and feedback requests", async () => {
    const inspection = { id: "inspection-1" };
    const fetchMock = vi.fn().mockImplementation(() =>
      Promise.resolve(
        new Response(JSON.stringify(inspection), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
      ),
    );
    vi.stubGlobal("fetch", fetchMock);
    vi.stubGlobal("crypto", { randomUUID: () => "idempotency-1" });

    const form = new FormData();
    await createInspection(form);
    expect(fetchMock.mock.calls[0][1].headers).toEqual({
      "Idempotency-Key": "idempotency-1",
    });

    await decideApproval("inspection-1", {
      decision: "approve",
      reviewer: "supervisor",
      comment: "confirmed",
    });
    expect(fetchMock.mock.calls[1][1].method).toBe("POST");

    await sendFeedback("inspection-1", {
      reviewer: "reviewer",
      comment: "verified",
    });
    expect(fetchMock.mock.calls[2][1].body).toContain("verified");
  });
});
