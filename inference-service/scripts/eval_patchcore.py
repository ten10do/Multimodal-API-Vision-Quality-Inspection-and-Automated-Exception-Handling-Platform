"""Evaluate the PatchCore baseline on MVTec AD 'bottle'.

Metrics (6C):
- Image-level AUROC
- Pixel-level AUROC
- PRO / AUPRO (approximation over FPR integration)
- anomaly threshold
- normal / anomaly score distributions
- latency + peak VRAM

Saves heatmap examples (normal, true anomaly, false positive, false negative)
and a metrics.json into docs/patchcore-eval/.

Usage:
  python inference-service/scripts/eval_patchcore.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference_app.patchcore_predictor import PatchCorePredictor  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BOTTLE = ROOT / "model-training/datasets/mvtec/bottle"
BANK = ROOT / "inference-service/models/patchcore-bottle/bank.npz"
OUT_DIR = ROOT / "docs/patchcore-eval"


def auroc(scores: list[float], labels: list[int]) -> float:
    from sklearn.metrics import roc_auc_score

    try:
        return float(roc_auc_score(labels, scores))
    except ValueError:
        return float("nan")


def aup_pro(pro_map: np.ndarray, mask: np.ndarray, max_fpr: float = 0.3) -> float:
    """Approximate AUPRO: average per-region recall over integrated FPR."""
    if mask.sum() == 0:
        return float("nan")
    fprs = np.linspace(0.0, max_fpr, 101)
    recalls = []
    for fpr in fprs:
        thresh = np.quantile(pro_map[mask == 0], 1 - fpr) if (mask == 0).sum() else 1.0
        pred = pro_map > thresh
        recalls.append((pred & mask).sum() / mask.sum())
    return float(np.trapezoid(recalls, fprs) / max_fpr)


def main() -> None:
    if not BANK.exists():
        sys.exit("bank not trained; run scripts/train_patchcore.py first")
    predictor = PatchCorePredictor(BANK)
    predictor._ensure_model()

    test_images: list[Path] = []
    for defect_dir in sorted((BOTTLE / "test").iterdir()):
        if defect_dir.is_dir():
            test_images += sorted(defect_dir.glob("*.png"))
    label_of = {p: 0 for p in test_images}
    for defect_dir in sorted((BOTTLE / "test").iterdir()):
        if not defect_dir.is_dir():
            continue
        if defect_dir.name == "good":
            continue
        for p in defect_dir.glob("*.png"):
            label_of[p] = 1

    torch.cuda.reset_peak_memory_stats()
    image_scores: list[float] = []
    labels: list[int] = []
    pixel_scores: list[float] = []
    pixel_labels: list[int] = []
    pro_maps: list[np.ndarray] = []
    pro_masks: list[np.ndarray] = []
    latencies: list[float] = []
    examples: dict[str, Path] = {}
    false_pos: list[tuple[Path, float]] = []
    false_neg: list[tuple[Path, float]] = []

    for p in test_images:
        img = Image.open(p)
        started = time.perf_counter()
        result = predictor.predict(img, include_map_png=False)
        latencies.append(result.latency_ms)
        image_scores.append(result.anomaly_score)
        labels.append(label_of[p])
        if label_of[p] == 0 and result.is_anomalous:
            false_pos.append((p, result.anomaly_score))
        if label_of[p] == 1 and not result.is_anomalous:
            false_neg.append((p, result.anomaly_score))

        anomaly_map, _ = predictor.score(img)
        mask_path = _mask_for(p, label_of[p])
        if mask_path is not None:
            mask = np.asarray(Image.open(mask_path).convert("L").resize((224, 224))) > 128
            pixel_scores.extend(anomaly_map[mask].tolist())
            pixel_labels.extend([1] * int(mask.sum()))
            pixel_scores.extend(anomaly_map[~mask].tolist())
            pixel_labels.extend([0] * int((~mask).sum()))
            pro_maps.append(anomaly_map)
            pro_masks.append(mask)

        if label_of[p] == 0:
            examples.setdefault("normal", p)
        else:
            examples.setdefault(f"anomaly-{p.parent.name}", p)
    if not false_pos:
        false_pos.append((sorted(test_images)[0], 0.0))
    if not false_neg:
        false_neg.append((sorted(test_images)[0], 0.0))
    examples["false_positive"] = false_pos[0][0]
    examples["false_negative"] = false_neg[0][0]

    peak_vram = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
    image_auroc = auroc(image_scores, labels)
    pixel_auroc = auroc(pixel_scores, pixel_labels) if pixel_labels else float("nan")
    pro = aup_pro(np.concatenate(pro_maps), np.concatenate(pro_masks)) if pro_maps else float("nan")

    normal_scores = [s for s, l in zip(image_scores, labels) if l == 0]
    anomaly_scores = [s for s, l in zip(image_scores, labels) if l == 1]

    metrics = {
        "dataset": "MVTec AD bottle",
        "domain_note": "benchmark only; NOT a steel-production-domain evaluation (see docs/08)",
        "model": predictor.model_name,
        "model_version": predictor.model_version,
        "n_normal_train": int(np.load(BANK)["train_images"]),
        "n_test": len(test_images),
        "n_anomaly": sum(labels),
        "threshold": predictor._threshold,
        "image_auroc": round(image_auroc, 4),
        "pixel_auroc": round(pixel_auroc, 4) if not np.isnan(pixel_auroc) else None,
        "aup_pro": round(pro, 4) if not np.isnan(pro) else None,
        "normal_score_dist": {"min": round(min(normal_scores), 4), "max": round(max(normal_scores), 4), "mean": round(float(np.mean(normal_scores)), 4)},
        "anomaly_score_dist": {"min": round(min(anomaly_scores), 4), "max": round(max(anomaly_scores), 4), "mean": round(float(np.mean(anomaly_scores)), 4)},
        "false_positives": [(str(p.name), round(s, 4)) for p, s in false_pos],
        "false_negatives": [(str(p.name), round(s, 4)) for p, s in false_neg],
        "latency_ms_mean": round(float(np.mean(latencies)), 2),
        "latency_ms_p95": round(float(np.percentile(latencies, 95)), 2),
        "peak_vram_mb": round(peak_vram, 1),
        "device": predictor.device,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    _save_examples(predictor, examples, OUT_DIR)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def _mask_for(image_path: Path, is_anomaly: int) -> Path | None:
    if not is_anomaly:
        return None
    mask = BOTTLE / "ground_truth" / image_path.parent.name / f"{image_path.stem}_mask.png"
    return mask if mask.exists() else None


def _save_examples(predictor: PatchCorePredictor, examples: dict[str, Path], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for key, path in examples.items():
        img = Image.open(path)
        anomaly_map, _ = predictor.score(img)
        fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
        axes[0].imshow(img)
        axes[0].set_title(path.parent.name)
        axes[0].axis("off")
        axes[1].imshow(anomaly_map, cmap="jet", vmin=0.0, vmax=1.0)
        axes[1].set_title("anomaly map")
        axes[1].axis("off")
        fig.tight_layout()
        fig.savefig(out / f"{key}.png", dpi=110)
        plt.close(fig)
        print("saved example:", key)


if __name__ == "__main__":
    main()
