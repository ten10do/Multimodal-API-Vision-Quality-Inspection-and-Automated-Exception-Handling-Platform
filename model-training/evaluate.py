"""Evaluate the trained YOLO baseline on the independent test split.

Reports Precision, Recall, mAP@50, mAP@50:95, per-class AP, and writes the
confusion matrix image already produced by ultralytics. Metrics are saved as
JSON for the Phase 1 report.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=Path("model-training/runs/neu-det-yolov8s-baseline/weights/best.pt"))
    parser.add_argument("--data", type=Path, default=Path("model-training/datasets/neu-det-yolo/data.yaml"))
    parser.add_argument("--out", type=Path, default=Path("model-training/runs/neu-det-yolov8s-baseline/test_metrics.json"))
    args = parser.parse_args()

    assert args.weights.exists(), f"weights missing: {args.weights}"

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    start = time.time()
    metrics = model.val(data=str(args.data), split="test", imgsz=256, verbose=True, plots=True, save_json=False)
    elapsed = time.time() - start

    box = metrics.box
    report = {
        "evaluation_set": "test",
        "precision": float(box.mp),
        "recall": float(box.mr),
        "mAP50": float(box.map50),
        "mAP50-95": float(box.map),
        "per_class_ap50": {name: float(v) for name, v in zip(metrics.names.values(), box.ap50)},
        "per_class_ap50_95": {name: float(v) for name, v in zip(metrics.names.values(), box.ap)},
        "evaluation_seconds": round(elapsed, 1),
        "device": str(model.device),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
