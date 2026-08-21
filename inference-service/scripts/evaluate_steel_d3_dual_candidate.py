"""Sealed evaluation for D3 1.3 dual-branch candidate integration."""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model-training"))
sys.path.insert(0, str(ROOT / "inference-service"))

from inference_app.d3_dual_branch_predictor import D3DualBranchPredictor  # noqa: E402
from steel_patchcore.aggregation import auroc  # noqa: E402
from steel_patchcore.candidate_registry import sha256_file  # noqa: E402
from steel_patchcore.d3_localization_representation import D3_IMAGE_AUROC  # noqa: E402
from steel_patchcore.d3_operational import atomic_write_json, pixel_localization_metrics  # noqa: E402
from steel_patchcore.dual_candidate_registry import DualCandidateRegistry  # noqa: E402
from steel_patchcore.rle import rle_decode  # noqa: E402

MANIFEST = ROOT / "model-training/registry/steel-patchcore-d3-candidate/1.3.0-candidate.1/manifest.json"
DATASET = ROOT / "model-training/datasets/severstal-steel"
IMAGE_DIR = DATASET / "raw/train_images"
CSV_PATH = DATASET / "raw/train.csv"
SOURCE_SPLIT = DATASET / "split_manifest.json"
RECOVERY_SPLIT = DATASET / "recovery_split_manifest.json"
SHADOW_LOG = ROOT / "docs/d3-shadow-prediction-log.json"
RUN_ROOT = ROOT / "model-training/runs/steel-d3-dual-candidate"
CHECKPOINT = RUN_ROOT / "evaluation-checkpoint.json"
REPORT = ROOT / "docs/dual-branch-evaluation-report.json"
REPORT_MD = ROOT / "docs/dual-branch-evaluation-report.md"
CHECKPOINT_EVERY = 10


class DualEvaluationError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def evaluation_ids() -> tuple[list[str], list[str]]:
    source = json.loads(SOURCE_SPLIT.read_text(encoding="utf-8"))
    recovery = json.loads(RECOVERY_SPLIT.read_text(encoding="utf-8"))
    normal = list(source["splits"]["test_normal"])
    anomaly = list(recovery["recovery_holdout_anomaly"])
    if len(normal) != 591 or len(anomaly) != 3333 or set(normal) & set(anomaly):
        raise DualEvaluationError("SEALED_EVALUATION_MEMBERSHIP_INVALID")
    return normal, anomaly


def mask_rows(image_ids: list[str]) -> dict[str, list[str]]:
    wanted = set(image_ids)
    frame = pd.read_csv(CSV_PATH, keep_default_na=True)
    selected = frame[frame["ImageId"].astype(str).map(lambda value: Path(value).stem in wanted)]
    result = {}
    for raw_id, group in selected.groupby(selected["ImageId"].astype(str)):
        rows = [str(value) for value in group["EncodedPixels"] if not pd.isna(value)]
        if rows:
            result[Path(raw_id).stem] = rows
    if set(result) != wanted:
        raise DualEvaluationError("SEALED_MASK_MEMBERSHIP_INVALID")
    return result


def decode_mask(rows: list[str]) -> np.ndarray:
    mask = np.zeros((256, 1600), dtype=np.uint8)
    for row in rows:
        mask = np.maximum(mask, rle_decode(row))
    return mask


def checkpoint_identity(manifest: dict, hashes: dict[str, str]) -> dict:
    return {
        "schema_version": "steel_patchcore_d3_dual_evaluation_checkpoint_v1",
        "manifest_sha256": sha256_file(MANIFEST),
        "model_version": manifest["model_version"],
        "artifact_version": manifest["artifact_version"],
        "threshold": manifest["image_branch"]["threshold"],
        "artifact_hashes": hashes,
    }


def load_checkpoint(identity: dict) -> dict:
    if not CHECKPOINT.is_file():
        checkpoint = {**identity, "completed": {}, "failures": [], "updated_at": utc_now()}
        atomic_write_json(CHECKPOINT, checkpoint)
        return checkpoint
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if any(checkpoint.get(key) != value for key, value in identity.items()):
        raise DualEvaluationError("CHECKPOINT_LINEAGE_MISMATCH")
    return checkpoint


def validate_report(report: dict) -> None:
    if report.get("schema_version") != "steel_patchcore_d3_dual_branch_evaluation_v1":
        raise DualEvaluationError("REPORT_SCHEMA_MISMATCH")
    if report.get("candidate_status") != "CANDIDATE" or report.get("production_promotion") is not False:
        raise DualEvaluationError("REPORT_CANDIDATE_ONLY_VIOLATION")
    metrics = report.get("metrics", {})
    if not math.isclose(metrics.get("image_auroc", -1), D3_IMAGE_AUROC, rel_tol=0.0, abs_tol=1e-12):
        raise DualEvaluationError("REPORT_IMAGE_AUROC_CHANGED")
    if metrics.get("threshold") != 0.8471092581748962 or metrics.get("image_score_mismatch_count") != 0:
        raise DualEvaluationError("REPORT_IMAGE_BRANCH_CHANGED")
    if metrics.get("pixel_auroc", 0.0) < 0.75 or metrics.get("aupro", 0.0) < 0.50:
        raise DualEvaluationError("REPORT_LOCALIZATION_GATE_FAILED")
    if report.get("artifact_unchanged") is not True or report.get("verdict") != "PASS":
        raise DualEvaluationError("REPORT_INTEGRATION_GATE_FAILED")


def finalize(manifest: dict, before: dict[str, str], after: dict[str, str], order: list[tuple[str, str]], checkpoint: dict) -> dict:
    completed = checkpoint["completed"]
    if set(completed) != {image_id for _, image_id in order}:
        raise DualEvaluationError(f"EVALUATION_INCOMPLETE:{len(completed)}/{len(order)}")
    labels = np.asarray([0 if role == "test_normal" else 1 for role, _ in order], dtype=np.int8)
    scores = np.asarray([completed[image_id]["image_score"] for _, image_id in order], dtype=np.float64)
    anomaly_ids = [image_id for role, image_id in order if role == "recovery_holdout_anomaly"]
    pixel = float(np.mean([completed[image_id]["pixel_auroc"] for image_id in anomaly_ids]))
    aupro = float(np.mean([completed[image_id]["aupro"] for image_id in anomaly_ids]))
    metrics = {
        "image_auroc": auroc(scores, labels),
        "threshold": manifest["image_branch"]["threshold"],
        "image_score_mismatch_count": sum(not row["image_score_identical"] for row in completed.values()),
        "pixel_auroc": pixel,
        "aupro": aupro,
    }
    report = {
        "schema_version": "steel_patchcore_d3_dual_branch_evaluation_v1",
        "candidate_status": manifest["status"],
        "model_name": manifest["model_name"],
        "model_version": manifest["model_version"],
        "artifact_version": manifest["artifact_version"],
        "branches": {"image": "D3-ZCA A0", "localization": "R-L3 multi-scale fusion"},
        "dataset": {"test_normal": 591, "recovery_holdout_anomaly": 3333, "total": 3924},
        "metrics": metrics,
        "expected_metrics": manifest["evaluation_gate"],
        "image_score_immutable": metrics["image_score_mismatch_count"] == 0,
        "threshold_changed": False,
        "artifact_hashes_before": before,
        "artifact_hashes_after": after,
        "artifact_unchanged": before == after,
        "failures": checkpoint["failures"],
        "verdict": "PASS" if (
            metrics["image_score_mismatch_count"] == 0
            and math.isclose(metrics["image_auroc"], D3_IMAGE_AUROC, rel_tol=0.0, abs_tol=1e-12)
            and pixel >= 0.75 and aupro >= 0.50 and before == after and not checkpoint["failures"]
        ) else "FAILED",
        "production_promotion": False,
        "generated_at": utc_now(),
    }
    validate_report(report)
    atomic_write_json(REPORT, report)
    REPORT_MD.write_text(
        "\n".join([
            "# D3 Dual-Branch Evaluation Report", "", f"Verdict: **`{report['verdict']}`**", "",
            f"- Image AUROC: `{metrics['image_auroc']:.12f}`",
            f"- Frozen threshold: `{metrics['threshold']!r}`",
            f"- Per-image D3 score mismatches: `{metrics['image_score_mismatch_count']}`",
            f"- Pixel AUROC: `{metrics['pixel_auroc']:.12f}`",
            f"- AUPRO: `{metrics['aupro']:.12f}`",
            "- Candidate artifacts unchanged: `true`", "- Production promotion: `false`", "",
        ]), encoding="utf-8"
    )
    return report


def evaluate() -> dict:
    if not torch.cuda.is_available():
        raise DualEvaluationError("DUAL_EVALUATION_REQUIRES_GPU")
    registry = DualCandidateRegistry(ROOT)
    manifest = registry.load_manifest(MANIFEST)
    _, before = registry.verify_artifacts(manifest)
    predictor = D3DualBranchPredictor.from_manifest(MANIFEST, project_root=ROOT, device="cuda:0")
    normal_ids, anomaly_ids = evaluation_ids()
    order = [("test_normal", image_id) for image_id in normal_ids] + [
        ("recovery_holdout_anomaly", image_id) for image_id in anomaly_ids
    ]
    masks = mask_rows(anomaly_ids)
    shadow = {row["image_id"]: row for row in json.loads(SHADOW_LOG.read_text(encoding="utf-8"))["records"]}
    if set(shadow) != {image_id for _, image_id in order}:
        raise DualEvaluationError("SHADOW_MEMBERSHIP_MISMATCH")
    checkpoint = load_checkpoint(checkpoint_identity(manifest, before))
    since_write = 0
    for role, image_id in order:
        if image_id in checkpoint["completed"]:
            continue
        try:
            with Image.open(IMAGE_DIR / f"{image_id}.jpg") as opened:
                image = opened.convert("RGB")
            output = predictor.infer(image)
            repeated = predictor.infer(image) if len(checkpoint["completed"]) < 3 else None
            if repeated is not None and (
                repeated.image_score != output.image_score or not np.array_equal(repeated.heatmap, output.heatmap)
            ):
                raise DualEvaluationError(f"INFERENCE_REPRODUCIBILITY_FAILED:{image_id}")
            reference_score = float(shadow[image_id]["score"])
            if output.image_score != reference_score:
                raise DualEvaluationError(f"IMAGE_SCORE_CHANGED:{image_id}:{reference_score}:{output.image_score}")
            record = {
                "split_role": role,
                "image_score": output.image_score,
                "image_score_identical": True,
                "anomaly_label": output.anomaly_label,
                "localization_branch": output.localization_metadata["branch"],
            }
            if role == "recovery_holdout_anomaly":
                record.update(pixel_localization_metrics(output.heatmap, decode_mask(masks[image_id])))
            checkpoint["completed"][image_id] = record
            since_write += 1
        except Exception as exc:
            checkpoint["failures"].append({"image_id": image_id, "error": f"{type(exc).__name__}:{exc}"})
            checkpoint["updated_at"] = utc_now()
            atomic_write_json(CHECKPOINT, checkpoint)
            raise
        if since_write >= CHECKPOINT_EVERY:
            checkpoint["updated_at"] = utc_now()
            atomic_write_json(CHECKPOINT, checkpoint)
            since_write = 0
            print(f"dual evaluation completed={len(checkpoint['completed'])}/{len(order)}", flush=True)
    checkpoint["updated_at"] = utc_now()
    atomic_write_json(CHECKPOINT, checkpoint)
    _, after = registry.verify_artifacts(manifest)
    report = finalize(manifest, before, after, order, checkpoint)
    print(json.dumps({"verdict": report["verdict"], "metrics": report["metrics"]}), flush=True)
    return report


if __name__ == "__main__":
    try:
        evaluate()
    except Exception as exc:
        print(f"D3_DUAL_EVALUATION_BLOCKED:{type(exc).__name__}:{exc}", flush=True)
        raise SystemExit(3)
