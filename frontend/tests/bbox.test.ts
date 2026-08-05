import { describe, expect, it } from "vitest";
import { clampBox, normalizedAreaRatio, normalizedToDisplay, pixelToNormalized } from "../src/utils/bbox";

describe("normalizedToDisplay", () => {
  it("scales normalized bbox to display pixels", () => {
    const box = normalizedToDisplay([0.1, 0.2, 0.4, 0.5], 1000, 600);
    expect(box).toEqual({ x: 100, y: 120, width: 300, height: 180 });
  });
});

describe("pixelToNormalized", () => {
  it("converts and clamps into [0,1]", () => {
    expect(pixelToNormalized([100, 120, 400, 420], 1000, 600)).toEqual([0.1, 0.2, 0.4, 0.7]);
    // out-of-bounds clamped
    const n = pixelToNormalized([-50, 0, 1200, 800], 1000, 600);
    expect(n[0]).toBe(0);
    expect(n[2]).toBe(1);
    expect(n[3]).toBe(1);
  });
});

describe("normalizedAreaRatio", () => {
  it("computes box area ratio", () => {
    expect(normalizedAreaRatio([0, 0, 0.5, 0.5])).toBeCloseTo(0.25);
  });
});

describe("clampBox", () => {
  it("keeps a valid box inside display bounds", () => {
    const box = clampBox({ x: 950, y: 100, width: 200, height: 400 }, 1000, 600);
    expect(box.x + box.width).toBeLessThanOrEqual(1000);
    expect(box.x).toBe(950);
    expect(box.width).toBe(50);
  });
});
