import type { QualityResult } from "../types";

// SYSTEM FAILED is rendered distinctly from product FAIL (4C): different
// label, color and icon semantics.
export function statusVariant(
  status: "COMPLETED" | "FAILED",
  quality: QualityResult | null,
): { label: string; cls: string } {
  if (status === "FAILED") return { label: "SYSTEM FAILED", cls: "badge-system-failed" };
  if (quality === "PASS") return { label: "PASS", cls: "badge-pass" };
  if (quality === "REVIEW") return { label: "REVIEW", cls: "badge-review" };
  if (quality === "FAIL") return { label: "FAIL", cls: "badge-fail" };
  return { label: "PENDING", cls: "badge-pending" };
}

export function StatusBadge({ status, quality }: { status: "COMPLETED" | "FAILED"; quality: QualityResult | null }) {
  const { label, cls } = statusVariant(status, quality);
  return <span className={`badge ${cls}`}>{label}</span>;
}
