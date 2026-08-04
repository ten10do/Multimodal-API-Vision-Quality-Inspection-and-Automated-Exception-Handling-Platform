"""Train a YOLOv8s baseline on NEU-DET.

Deliberately avoids hyperparameter tuning. The first run only establishes a
reliable, reproducible baseline. Model weights land under runs/ which is
gitignored.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("model-training/datasets/neu-det-yolo/data.yaml"))
    parser.add_argument("--model", default="yolov8s.pt")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", type=Path, default=(Path(__file__).parent / "runs").resolve())
    parser.add_argument("--name", default="neu-det-yolov8s-baseline")
    args = parser.parse_args()

    assert args.data.exists(), f"data.yaml missing: {args.data}"
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    from ultralytics import YOLO

    model = YOLO(args.model)
    start = time.time()
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0 if torch.cuda.is_available() else "cpu",
        project=str(args.project),
        name=args.name,
        seed=args.seed,
        patience=20,
        workers=4,
        verbose=True,
        plots=True,
        val=True,
    )
    elapsed = time.time() - start
    print(f"training finished in {elapsed / 60:.1f} minutes")
    print(f"artifacts under {args.project}/{args.name}")


if __name__ == "__main__":
    main()
