import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { StatusBadge, statusVariant } from "../src/components/StatusBadge";
import { EmptyState, ErrorState } from "../src/components/StateViews";

describe("statusVariant", () => {
  it("maps process status and quality to distinct labels", () => {
    expect(statusVariant("COMPLETED", "PASS")).toMatchObject({ label: "PASS" });
    expect(statusVariant("COMPLETED", "REVIEW")).toMatchObject({ label: "REVIEW" });
    expect(statusVariant("COMPLETED", "FAIL")).toMatchObject({ label: "FAIL" });
  });

  it("SYSTEM FAILED is distinct from product FAIL (4C)", () => {
    const failed = statusVariant("FAILED", null);
    const productFail = statusVariant("COMPLETED", "FAIL");
    expect(failed.label).toBe("SYSTEM FAILED");
    expect(productFail.label).toBe("FAIL");
    expect(failed.cls).not.toBe(productFail.cls);
  });
});

describe("StatusBadge rendering", () => {
  it("renders SYSTEM FAILED for failed process status", () => {
    render(<StatusBadge status="FAILED" quality={null} />);
    expect(screen.getByText("SYSTEM FAILED")).toBeTruthy();
  });

  it("renders FAIL (product) separately", () => {
    render(<StatusBadge status="COMPLETED" quality="FAIL" />);
    expect(screen.getByText("FAIL")).toBeTruthy();
    expect(screen.queryByText("SYSTEM FAILED")).toBeNull();
  });
});

describe("empty / error states (4I)", () => {
  it("renders empty and error states", () => {
    const { unmount } = render(<ErrorState message="backend unavailable" />);
    expect(screen.getByText("数据获取失败")).toBeTruthy();
    expect(screen.getByText("backend unavailable")).toBeTruthy();
    unmount();
    render(<EmptyState message="暂无数据" />);
    expect(screen.getByText("暂无数据")).toBeTruthy();
  });
});
