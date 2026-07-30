import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { RiskBadge, StatusBadge } from "@/components/status-badge";

describe("status badges", () => {
  it("renders Chinese workflow labels", () => {
    render(
      <>
        <StatusBadge status="awaiting_approval" />
        <RiskBadge risk="critical" />
      </>,
    );
    expect(screen.getByText("等待审批")).toBeInTheDocument();
    expect(screen.getByText("严重")).toBeInTheDocument();
  });
});
