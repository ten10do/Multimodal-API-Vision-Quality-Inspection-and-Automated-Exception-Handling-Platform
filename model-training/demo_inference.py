"""Demo inference: run the baseline model on a few real test images and save
annotated images (bounding boxes + class + confidence) plus the raw contract JSON.

Used as Phase 1 acceptance evidence that the pipeline can output bounding boxes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference-service"))

from app.yolo_predictor import YoloPredictor  # noqa: E402

COLORS = [(220, 60, 60), (60, 160, 220), (60, 200, 120), (220, 180, 60), (180, 120, 220), (240, 140, 90)]


def draw_detections(image, result) -> None:
    for i, d in enumerate(result.detections):
        x1, y1, x2, y2 = [int(v) for v in d.bbox_xyxy]
        color = COLORS[d.class_id % len(COLORS)]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"{d.class_name} {d.confidence:.2f}"
        cv2.putText(image, label, (x1, max(y1 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=Path("model-training/runs/neu-det-yolov8s-baseline-2/weights/best.pt"))
    parser.add_argument("--images", type=Path, default=Path("model-training/datasets/neu-det-yolo/test/images"))
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--out", type=Path, default=Path("model-training/runs/neu-det-yolov8s-baseline-2/demo"))
    args = parser.parse_args()

    predictor = YoloPredictor(args.weights, model_version="phase1-baseline")
    image_paths = sorted(args.images.glob("*.jpg"))[: args.n]
    args.out.mkdir(parents=True, exist_ok=True)

    for i, path in enumerate(image_paths):
        result = predictor.predict(path, inspection_id=f"phase1-demo-{i + 1}")
        image = cv2.imread(str(path))
        draw_detections(image, result)
        out_path = args.out / f"demo_{i + 1:02d}_{path.stem}.png"
        cv2.imwrite(str(out_path), image)
        (args.out / f"demo_{i + 1:02d}_{path.stem}.json").write_text(
            result.model_dump_json(indent=2), encoding="utf-8"
        )
        print(f"{path.name}: {len(result.detections)} detections, latency {result.inference_latency_ms:.1f} ms -> {out_path.name}")


if __name__ == "__main__":
    main()
