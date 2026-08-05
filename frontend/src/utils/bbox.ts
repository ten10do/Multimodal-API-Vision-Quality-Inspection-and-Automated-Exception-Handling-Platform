// Bounding Box coordinate conversion (4K).
// The Vision Contract carries pixel bbox (bbox_xyxy) plus image dimensions in
// the Inspection? Actually dimensions are not in the persisted defect rows;
// the dashboard normalizes with bbox_normalized when available, otherwise with
// the rendered image size. All conversion helpers are pure and unit-tested.

export interface Box {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** Pixel bbox -> display box for a canvas/svg of (w, h) using normalized coords. */
export function normalizedToDisplay(
  bboxNormalized: [number, number, number, number],
  width: number,
  height: number,
): Box {
  const [x1, y1, x2, y2] = bboxNormalized;
  return {
    x: Math.round(x1 * width),
    y: Math.round(y1 * height),
    width: Math.round((x2 - x1) * width),
    height: Math.round((y2 - y1) * height),
  };
}

/** Pixel bbox -> normalized [0,1], clamped to valid ranges. */
export function pixelToNormalized(bbox: [number, number, number, number], width: number, height: number) {
  const [x1, y1, x2, y2] = bbox;
  const nx1 = Math.min(1, Math.max(0, x1 / width));
  const ny1 = Math.min(1, Math.max(0, y1 / height));
  const nx2 = Math.min(1, Math.max(0, x2 / width));
  const ny2 = Math.min(1, Math.max(0, y2 / height));
  return [nx1, ny1, nx2, ny2] as [number, number, number, number];
}

/** Area ratio of a normalized box against the full image. */
export function normalizedAreaRatio(bboxNormalized: [number, number, number, number]): number {
  const [x1, y1, x2, y2] = bboxNormalized;
  return Math.max(0, (x2 - x1) * (y2 - y1));
}

/** Keep a box within the display bounds. */
export function clampBox(box: Box, width: number, height: number): Box {
  return {
    x: Math.min(Math.max(0, box.x), width),
    y: Math.min(Math.max(0, box.y), height),
    width: Math.min(box.width, width - Math.min(Math.max(0, box.x), width)),
    height: Math.min(box.height, height - Math.min(Math.max(0, box.y), height)),
  };
}
