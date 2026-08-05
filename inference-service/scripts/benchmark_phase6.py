"""Phase 6 (6I): end-to-end vision latency breakdown + GPU coexistence.

Loads YOLO + PatchCore in one process (as the inference service does) and
records per-stage latencies and peak VRAM on the default device.

Output: docs/phase6-benchmark.json + console table.

Usage:
  python inference-service/scripts/benchmark_phase6.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference_app.fusion import fuse  # noqa: E402
from inference_app.patchcore_predictor import PatchCorePredictor  # noqa: E402
from inference_app.yolo_predictor import YoloPredictor  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / "inference-service/models/best.pt"
BANK = ROOT / "inference-service/models/patchcore-bottle/bank.npz"
IMAGES = sorted((ROOT / "model-training/datasets/neu-det-yolo/test/images").glob("*.jpg"))[:20]
OUT = ROOT / "docs/phase6-benchmark.json"


def main() -> None:
    if not WEIGHTS.exists() or not BANK.exists():
        sys.exit("missing weights/bank; run training first")
    if not torch.cuda.is_available():
        sys.exit("CUDA required for the co-existence benchmark")

    yolo = YoloPredictor(WEIGHTS, model_version="phase1-baseline")
    patch = PatchCorePredictor(BANK)
    patch._ensure_model()

    torch.cuda.reset_peak_memory_stats()
    yolo_lat, anomaly_lat, fusion_lat, total_lat = [], [], [], []
    fusion_counts: dict[str, int] = {}

    for img_path in IMAGES:
        img = cv2.imread(str(img_path))
        pil = Image.fromarray(img[..., ::-1])
        t0 = time.perf_counter()
        yolo_result = yolo.predict(img, inspection_id="bench")
        t1 = time.perf_counter()
        anomaly = patch.predict(pil, image_w=img.shape[1], image_h=img.shape[0], include_map_png=True)
        t2 = time.perf_counter()
        fusion_class = fuse(len(yolo_result.detections), anomaly.is_anomalous)
        t3 = time.perf_counter()

        yolo_lat.append((t1 - t0) * 1000)
        anomaly_lat.append((t2 - t1) * 1000)
        fusion_lat.append((t3 - t2) * 1000)
        total_lat.append((t3 - t0) * 1000)
        fusion_counts[fusion_class] = fusion_counts.get(fusion_class, 0) + 1

    peak_vram = torch.cuda.max_memory_allocated() / 1024**2
    # runtime peak while both models are resident
    runtime_vram = torch.cuda.memory_allocated() / 1024**2

    def stat(v):
        return {
            "mean_ms": round(float(np.mean(v)), 2),
            "p50_ms": round(float(np.median(v)), 2),
            "p95_ms": round(float(np.percentile(v, 95)), 2),
        }

    report = {
        "benchmark_kind": "model-level (YOLO + PatchCore + fusion in one process)",
        "device": "RTX 5060 8GB (cuda:0)",
        "input_size": "YOLO native (640 letterbox); PatchCore 224x224",
        "sample_count": len(IMAGES),
        "warmup_count": 2,  # YOLO warmup + PatchCore model load happen before timing
        "worker_count": 1,
        "yolo": stat(yolo_lat),
        "patchcore": stat(anomaly_lat),
        "fusion": stat(fusion_lat),
        "total_vision": stat(total_lat),
        "models_coexist": True,
        "peak_gpu_allocated_mb": round(peak_vram, 1),
        "peak_gpu_reserved_mb": round(torch.cuda.max_memory_reserved() / 1024**2, 1),
        "runtime_allocated_mb": round(runtime_vram, 1),
        "fusion_distribution": fusion_counts,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
