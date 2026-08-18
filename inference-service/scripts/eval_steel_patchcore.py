"""Original-image evaluation of steel PatchCore (S/T items).

Formal metrics are computed at ORIGINAL-IMAGE level:
  * image anomaly score = max over the 7 tile scores
  * pixel anomaly map = stitch of 7 tile maps, overlap region = MEAN
  * Image AUROC, Pixel AUROC, AUPRO
  * operating point at baseline threshold: TP/TN/FP/FN, precision, recall,
    F1, normal FPR, anomaly recall, confusion matrix
  * score distributions for train_normal / validation_normal / test_normal
    / test_anomaly (min/p50/p95/p99/max) with the threshold marked

Memory strategy: per-image Pixel AUROC and per-image AUPRO are averaged
(standard MVTec-style per-region recall), unlike Phase 6's pooled
np.concatenate approximation which is infeasible at this scale (6.6k x
409600 px maps would exceed RAM). Difference is documented.

Usage:
  python inference-service/scripts/eval_steel_patchcore.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "model-training"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from steel_patchcore.rle import rle_decode  # noqa: E402
from steel_patchcore.tile import TILE_X0, tile_coords, stitch_scores  # noqa: E402

if TYPE_CHECKING:
    from inference_app.patchcore_predictor import PatchCorePredictor

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "model-training/datasets/severstal-steel"
CSV = DS / "raw/train.csv"
SPLIT = DS / "split_manifest.json"
IMG_DIR = DS / "raw/train_images"
BANK = ROOT / "inference-service/models/steel-patchcore/bank.npz"
TRAIN_SCORES = DS / "train_normal_scores.json"
OUT_DIR = ROOT / "docs/steel-patchcore-eval"

FORMAL_SPLITS = ("test_normal", "test_anomaly")
FORMAL_EXPECTED_COUNTS = {"test_normal": 591, "test_anomaly": 6666}


def auroc(scores, labels) -> float:
    try:
        return float(roc_auc_score(labels, scores))
    except ValueError:
        return float("nan")


def pending_original_ids(image_ids: list[str], completed: dict) -> list[str]:
    """Return only originals not already committed to the checkpoint."""
    return [image_id for image_id in image_ids if image_id not in completed]


def normalize_image_id(image_id: str) -> str:
    """Use the extension-free ID shared by the split, images and checkpoint."""
    return Path(str(image_id)).stem


def pending_pixel_evidence_ids(image_ids: list[str], completed: dict) -> list[str]:
    """Return completed anomaly scores whose required pixel evidence is absent."""
    required = {"pixel_auc", "aupro"}
    return [
        image_id
        for image_id in image_ids
        if image_id in completed and not required.issubset(completed[image_id])
    ]


def aggregate_evaluation(
    results: dict[str, dict],
    expected_counts: dict[str, int] | None = None,
) -> dict:
    """Keep validation diagnostics separate from the formal test allow-list."""
    expected_counts = expected_counts or FORMAL_EXPECTED_COUNTS
    formal_results = {name: results[name] for name in FORMAL_SPLITS}
    formal_counts = {
        name: len(formal_results[name]["scores"])
        for name in FORMAL_SPLITS
    }
    if formal_counts != expected_counts:
        raise RuntimeError(
            f"FORMAL_SAMPLE_COUNT_MISMATCH: expected={expected_counts} actual={formal_counts}"
        )

    normal = formal_results["test_normal"]
    anomaly = formal_results["test_anomaly"]
    tn, fp = normal["tn"], normal["fp"]
    tp, fn = anomaly["tp"], anomaly["fn"]
    if tn + fp != formal_counts["test_normal"]:
        raise RuntimeError("FORMAL_NORMAL_CONFUSION_COUNT_MISMATCH")
    if tp + fn != formal_counts["test_anomaly"]:
        raise RuntimeError("FORMAL_ANOMALY_CONFUSION_COUNT_MISMATCH")

    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0
        else 0.0
    )
    normal_fpr = fp / (fp + tn) if (fp + tn) else float("nan")

    validation = results["validation_normal"]
    validation_tn, validation_fp = validation["tn"], validation["fp"]
    validation_fpr = (
        validation_fp / (validation_fp + validation_tn)
        if (validation_fp + validation_tn)
        else float("nan")
    )

    pixel_aucs = [
        value
        for name in FORMAL_SPLITS
        for value in formal_results[name].get("pixel_aucs", [])
        if np.isfinite(value)
    ]
    aupros = [
        value
        for name in FORMAL_SPLITS
        for value in formal_results[name].get("aupros", [])
        if np.isfinite(value)
    ]
    return {
        "formal_counts": {
            "test_normal": formal_counts["test_normal"],
            "test_anomaly": formal_counts["test_anomaly"],
            "total": sum(formal_counts.values()),
        },
        "image_auroc": auroc(
            normal["scores"] + anomaly["scores"],
            [0] * formal_counts["test_normal"] + [1] * formal_counts["test_anomaly"],
        ),
        "operating_point": {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "normal_fpr": normal_fpr,
            "anomaly_recall": recall,
            "confusion_matrix": [[tn, fp], [fn, tp]],
        },
        "validation_diagnostic": {
            "n": len(validation["scores"]),
            "tn": validation_tn,
            "fp": validation_fp,
            "fpr": validation_fpr,
        },
        "pixel_auroc_mean": float(np.mean(pixel_aucs)) if pixel_aucs else None,
        "aup_pro_mean": float(np.mean(aupros)) if aupros else None,
    }


def aup_pro_per_image(pro_map: np.ndarray, mask: np.ndarray, max_fpr: float = 0.3) -> float:
    if mask.sum() == 0:
        return float("nan")
    fprs = np.linspace(0.0, max_fpr, 101)
    recalls = []
    neg = mask == 0
    if neg.sum() == 0:
        return float("nan")
    for fpr in fprs:
        thresh = np.quantile(pro_map[neg], 1 - fpr)
        pred = pro_map > thresh
        recalls.append((pred & mask).sum() / mask.sum())
    return float(np.trapezoid(recalls, fprs) / max_fpr)


def pixel_evidence(stitched: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Compute the frozen per-image Pixel AUROC and AUPRO evidence."""
    m_flat = stitched[mask > 0]
    b_flat = stitched[mask == 0]
    if not m_flat.size or not b_flat.size:
        raise RuntimeError("PIXEL_EVIDENCE_REQUIRES_FOREGROUND_AND_BACKGROUND")
    return {
        "pixel_auc": auroc(
            np.concatenate([m_flat, b_flat]),
            np.concatenate([np.ones(len(m_flat)), np.zeros(len(b_flat))]),
        ),
        "aupro": aup_pro_per_image(stitched, mask),
    }


def load_tiles(image_id: str) -> list[Image.Image]:
    img = Image.open(IMG_DIR / f"{image_id}.jpg")
    if img.mode != "RGB":
        img = img.convert("RGB")
    tiles = []
    for tid in range(len(TILE_X0)):
        x0, y0, w, h = tile_coords(tid)
        tiles.append(img.crop((x0, y0, x0 + w, y0 + h)))
    return tiles


def score_original(predictor: "PatchCorePredictor", image_id: str) -> tuple[float, np.ndarray]:
    """Return (image_score=max tile score, stitched 256x1600 anomaly map)."""
    maps = []
    tile_scores = []
    for tile in load_tiles(image_id):
        m, s = predictor.score(tile)
        maps.append(m)
        tile_scores.append(s)
    return float(max(tile_scores)), stitch_scores(maps)


def _bank_sha256(p: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


from steel_patchcore.threshold_verify import load_and_verify_threshold  # noqa: E402


def main() -> int:
    from inference_app.patchcore_predictor import PatchCorePredictor

    sm = json.load(open(SPLIT, encoding="utf-8"))
    splits = sm["splits"]

    # ---- Process Lifecycle Gate (shared with trainer) ----
    from steel_patchcore.lifecycle import lifecycle_enter
    EVAL_CKPT_PATH = DS / "raw/steel_eval_ckpt.json"
    _rc = lifecycle_enter("evaluation", _bank_sha256(BANK), EVAL_CKPT_PATH, EVAL_CKPT_PATH.parent)
    if _rc is not None:
        return _rc

    predictor = PatchCorePredictor(BANK, image_size=256)
    predictor._ensure_model()
    print("device:", predictor.device, "model:", predictor.model_name, predictor.model_version)
    # threshold lives in threshold.json (bank.npz is SEALED and carries 0.0)
    bank_sha = _bank_sha256(BANK)
    THRESH = DS / "threshold.json"
    if THRESH.exists():
        predictor._threshold, _tmeta = load_and_verify_threshold(THRESH, bank_sha)
        print("threshold from threshold.json:", predictor._threshold)
    else:
        print("WARNING: threshold.json missing; using bank.npz threshold")
    threshold = predictor._threshold

    # masks for anomaly images
    df = pd.read_csv(CSV)
    mask_of: dict[str, np.ndarray] = {}
    for img_id, group in df.groupby(df["ImageId"].astype(str)):
        m = np.zeros((256, 1600), dtype=np.uint8)
        for rle in group["EncodedPixels"].astype(str):
            m = np.maximum(m, rle_decode(rle))
        mask_of[normalize_image_id(img_id)] = m

    t0 = time.perf_counter()
    results: dict[str, dict] = {}

    # ---- original-image level checkpoint (survive session lifecycle) ----
    EVAL_CKPT = DS / "raw/steel_eval_ckpt.json"
    ckpt: dict = {}
    if EVAL_CKPT.exists():
        try:
            ckpt = json.load(open(EVAL_CKPT, encoding="utf-8"))
        except Exception:  # noqa: BLE001
            ckpt = {}

    def save_ckpt() -> None:
        json.dump(ckpt, open(EVAL_CKPT, "w"))

    for split_name in ("validation_normal", "test_normal", "test_anomaly"):
        ids = splits[split_name]
        label = 1 if split_name == "test_anomaly" else 0
        done = ckpt.get(split_name, {})
        pixel_todo = pending_pixel_evidence_ids(ids, done) if label == 1 else []
        todo = pending_original_ids(ids, done)
        print(
            f"{split_name}: total={len(ids)} done={len(done)} "
            f"pixel_todo={len(pixel_todo)} todo={len(todo)}",
            flush=True,
        )
        for i, img_id in enumerate(pixel_todo, 1):
            sc, stitched = score_original(predictor, img_id)
            entry = done[img_id]
            if not np.isclose(sc, float(entry["score"]), rtol=0.0, atol=1e-6):
                raise RuntimeError(f"PIXEL_BACKFILL_SCORE_MISMATCH:{img_id}")
            if int(sc >= threshold) != int(entry["pred"]):
                raise RuntimeError(f"PIXEL_BACKFILL_PREDICTION_MISMATCH:{img_id}")
            if img_id not in mask_of:
                raise RuntimeError(f"MISSING_ANOMALY_MASK:{img_id}")
            entry.update(pixel_evidence(stitched, mask_of[img_id]))
            done[img_id] = entry
            ckpt[split_name] = done
            save_ckpt()
            if i % 20 == 0:
                print(
                    f"...{split_name} pixel_backfill {i}/{len(pixel_todo)} "
                    f"el={time.perf_counter()-t0:.0f}s",
                    flush=True,
                )
        for i, img_id in enumerate(todo, 1):
            sc, stitched = score_original(predictor, img_id)
            pred = 1 if sc >= threshold else 0
            entry = {"score": sc, "pred": pred}
            if label == 1:
                if img_id not in mask_of:
                    raise RuntimeError(f"MISSING_ANOMALY_MASK:{img_id}")
                entry.update(pixel_evidence(stitched, mask_of[img_id]))
            done[img_id] = entry
            ckpt[split_name] = done
            if (len(done)) % 200 == 0:
                save_ckpt()
                print(f"...{split_name} {len(done)}/{len(ids)} el={time.perf_counter()-t0:.0f}s", flush=True)
        ckpt[split_name] = done
        save_ckpt()
        # aggregate this split
        scores = [v["score"] for v in done.values()]
        tps = fps = tns = fns = 0
        for v in done.values():
            if label == 1:
                if v["pred"] == 1:
                    tps += 1
                else:
                    fns += 1
            else:
                if v["pred"] == 1:
                    fps += 1
                else:
                    tns += 1
        pixel_aucs = [v["pixel_auc"] for v in done.values() if "pixel_auc" in v]
        aupros = [v["aupro"] for v in done.values() if "aupro" in v]
        results[split_name] = {
            "n": len(ids),
            "scores": scores,
            "tp": tps, "fp": fps, "tn": tns, "fn": fns,
            "pixel_aucs": pixel_aucs,
            "aupros": aupros,
            "pixel_auroc_mean": float(np.nanmean(pixel_aucs)) if pixel_aucs else None,
            "aup_pro_mean": float(np.nanmean(aupros)) if aupros else None,
        }
        print(f"{split_name}: n={len(ids)} scores[min={min(scores):.4f} max={max(scores):.4f}] "
              f"tp={tps} fp={fps} tn={tns} fn={fns}", flush=True)

    # ---- formal metrics: explicit allow-list, validation remains diagnostic ----
    aggregate = aggregate_evaluation(results)
    operating_point = aggregate["operating_point"]
    test_n_scores = results["test_normal"]["scores"]
    test_a_scores = results["test_anomaly"]["scores"]

    # ---- score distributions ----
    def dist(ss):
        arr = np.asarray(ss, dtype=float)
        return {
            "n": int(len(arr)),
            "min": round(float(arr.min()), 6) if arr.size else None,
            "p50": round(float(np.median(arr)), 6) if arr.size else None,
            "p95": round(float(np.percentile(arr, 95)), 6) if arr.size else None,
            "p99": round(float(np.percentile(arr, 99)), 6) if arr.size else None,
            "max": round(float(arr.max()), 6) if arr.size else None,
        }

    train_n_scores = json.load(open(TRAIN_SCORES, encoding="utf-8"))["original_image_scores"]
    dists = {
        "train_normal": dist(train_n_scores),
        "validation_normal": dist(results["validation_normal"]["scores"]),
        "test_normal": dist(test_n_scores),
        "test_anomaly": dist(test_a_scores),
        "threshold": round(threshold, 6),
    }

    metrics = {
        "dataset": "Severstal Steel Defect Detection (train subset)",
        "domain_note": "steel hot-rolled flat sheet, Severstal production-line camera distribution only",
        "model": predictor.model_name,
        "model_version": predictor.model_version,
        "bank": str(BANK),
        "n_train_normal": len(splits["train_normal"]),
        "n_validation_normal": len(splits["validation_normal"]),
        "n_test_normal": len(splits["test_normal"]),
        "n_test_anomaly": len(splits["test_anomaly"]),
        "threshold": round(threshold, 6),
        "formal_test": aggregate["formal_counts"],
        "validation_diagnostic": {
            **aggregate["validation_diagnostic"],
            "fpr": round(aggregate["validation_diagnostic"]["fpr"], 4),
        },
        "image_auroc": round(aggregate["image_auroc"], 4),
        "pixel_auroc_mean_per_image": round(aggregate["pixel_auroc_mean"], 4)
        if aggregate["pixel_auroc_mean"] is not None else None,
        "aup_pro_mean_per_image": round(aggregate["aup_pro_mean"], 4)
        if aggregate["aup_pro_mean"] is not None else None,
        "operating_point": {
            **operating_point,
            "precision": round(operating_point["precision"], 4),
            "recall": round(operating_point["recall"], 4),
            "f1": round(operating_point["f1"], 4),
            "normal_fpr": round(operating_point["normal_fpr"], 4),
            "anomaly_recall": round(operating_point["anomaly_recall"], 4),
        },
        "score_distributions": dists,
        "aggregation": {
            "image_score": "max over 7 tile scores",
            "pixel_overlap": "mean",
            "formal_split_allowlist": list(FORMAL_SPLITS),
            "auroc_pooled_or_per_image": "image AUROC pooled over images; pixel AUROC and AUPRO are per-image means (documented difference from Phase 6 pooled approx)",
        },
        "device": predictor.device,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("EVAL_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
