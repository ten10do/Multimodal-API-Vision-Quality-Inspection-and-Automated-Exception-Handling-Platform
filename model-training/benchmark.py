"""RTX 5060 inference benchmark for the YOLO baseline.

Measures single-image latency (mean / P50 / P95), throughput and GPU memory.
Warm-up iterations are excluded from the reported numbers.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=Path("model-training/runs/neu-det-yolov8s-baseline/weights/best.pt"))
    parser.add_argument("--images", type=Path, default=Path("model-training/datasets/neu-det-yolo/test/images"))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--out", type=Path, default=Path("model-training/runs/neu-det-yolov8s-baseline/benchmark.json"))
    args = parser.parse_args()

    assert args.weights.exists(), f"weights missing: {args.weights}"
    images = sorted(args.images.glob("*.jpg"))
    assert images, f"no images in {args.images}"
    image_paths = [images[i % len(images)] for i in range(args.warmup + args.iters)]

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)

    latencies = []
    for i, path in enumerate(image_paths):
        img = cv2.imread(str(path))
        start = time.perf_counter()
        model.predict(img, imgsz=args.imgsz, conf=0.25, verbose=False)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000.0)

    measured = latencies[args.warmup:]
    measured_sorted = sorted(measured)

    def pct(p: float) -> float:
        idx = min(len(measured_sorted) - 1, int(p * len(measured_sorted)))
        return measured_sorted[idx]

    vram_allocated = 0.0
    vram_reserved = 0.0
    if device.startswith("cuda"):
        vram_allocated = torch.cuda.memory_allocated() / (1024**2)
        vram_reserved = torch.cuda.memory_reserved() / (1024**2)

    report = {
        "device": device,
        "model": str(args.weights),
        "imgsz": args.imgsz,
        "warmup_iters": args.warmup,
        "measured_iters": args.iters,
        "latency_ms_mean": round(statistics.mean(measured), 2),
        "latency_ms_p50": round(pct(0.50), 2),
        "latency_ms_p95": round(pct(0.95), 2),
        "latency_ms_min": round(min(measured), 2),
        "latency_ms_max": round(max(measured), 2),
        "throughput_fps": round(1000.0 / statistics.mean(measured), 1),
        "gpu_mem_allocated_mb": round(vram_allocated, 1),
        "gpu_mem_reserved_mb": round(vram_reserved, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
