import { useState } from "react";
import type { Defect } from "../types";
import { clampBox, normalizedToDisplay } from "../utils/bbox";

const SEVERITY_COLOR: Record<string, string> = {
  low: "#22c55e",
  medium: "#eab308",
  high: "#f97316",
  critical: "#ef4444",
};

/** Raw inspection image + SVG Bounding Box overlay (4D).
 *  BBox is computed client-side from the Vision Contract; no server-side
 *  annotation image is needed. */
export function BBoxImage({ imageUrl, defects }: { imageUrl: string; defects: Defect[] }) {
  const [size, setSize] = useState<{ w: number; h: number } | null>(null);
  const [error, setError] = useState(false);

  if (error) {
    return <div className="state-block error">图像不可用（image endpoint 404 或后端不可达）</div>;
  }

  return (
    <div className="bbox-wrap">
      <img
        src={imageUrl}
        alt="inspection"
        onLoad={(e) => {
          const el = e.currentTarget;
          setSize({ w: el.clientWidth, h: el.clientHeight });
          setError(false);
        }}
        onError={() => setError(true)}
        className="bbox-image"
      />
      {size ? (
        <svg className="bbox-overlay" viewBox={`0 0 ${size.w} ${size.h}`} width={size.w} height={size.h}>
          {defects.map((d, idx) => {
            const box = clampBox(normalizedToDisplay(d.bbox_normalized, size.w, size.h), size.w, size.h);
            const color = SEVERITY_COLOR[d.severity ?? "medium"] ?? "#eab308";
            return (
              <g key={idx}>
                <rect
                  x={box.x}
                  y={box.y}
                  width={Math.max(box.width, 1)}
                  height={Math.max(box.height, 1)}
                  fill="none"
                  stroke={color}
                  strokeWidth={2}
                />
                <text x={box.x} y={Math.max(box.y - 4, 10)} fill={color} fontSize={12} className="bbox-label">
                  {d.class_name} {Math.round(d.confidence * 100)}%
                </text>
              </g>
            );
          })}
        </svg>
      ) : null}
    </div>
  );
}
