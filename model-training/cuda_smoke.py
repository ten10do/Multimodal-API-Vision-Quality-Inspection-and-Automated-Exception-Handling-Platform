"""CUDA smoke test on the local GPU (Phase 1A).

Verifies in order: driver exposure, torch CUDA build, GPU tensor math,
ultralytics model forward pass on GPU. The dependency freeze only happens
after this script passes on the actual RTX 5060.
"""

from __future__ import annotations

import time

import numpy as np
import torch


def main() -> None:
    print(f"torch version: {torch.__version__}")
    print(f"torch CUDA build available: {torch.version.cuda}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available, dependency freeze blocked")

    device = torch.device("cuda:0")
    print(f"device name: {torch.cuda.get_device_name(0)}")
    print(f"device capability: {torch.cuda.get_device_capability(0)}")
    print(f"device memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    a = torch.randn(1024, 1024, device=device)
    b = torch.randn(1024, 1024, device=device)
    start = time.perf_counter()
    c = (a @ b).sum()
    torch.cuda.synchronize()
    matmul_ms = (time.perf_counter() - start) * 1000.0
    print(f"GPU matmul+sum ok, value={c.item():.3f}, {matmul_ms:.2f} ms")

    from ultralytics import YOLO

    model = YOLO("yolov8n.pt").to(device)
    dummy = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
    start = time.perf_counter()
    result = model.predict(dummy, conf=0.25, verbose=False)
    torch.cuda.synchronize()
    fwd_ms = (time.perf_counter() - start) * 1000.0
    n_det = 0 if result[0].boxes is None else len(result[0].boxes)
    print(f"ultralytics GPU forward ok, detections={n_det}, {fwd_ms:.2f} ms")

    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
