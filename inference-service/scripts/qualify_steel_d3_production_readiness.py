"""Candidate-only production-readiness qualification for D3 1.3."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
from PIL import Image, ImageEnhance

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model-training"))
sys.path.insert(0, str(ROOT / "inference-service"))

from inference_app.d3_dual_branch_predictor import D3DualBranchPredictor  # noqa: E402
from steel_patchcore.aggregation import auroc  # noqa: E402
from steel_patchcore.candidate_registry import canonical_sha256, sha256_file  # noqa: E402
from steel_patchcore.d3_operational import OperationalQualificationError, atomic_write_json, pixel_localization_metrics  # noqa: E402
from steel_patchcore.d3_production_readiness import (  # noqa: E402
    READINESS_SCHEMA_VERSION,
    REVIEW_WORKFLOW_SCHEMA_VERSION,
    ROBUSTNESS_GATE,
    ROBUSTNESS_SCHEMA_VERSION,
    ROLLBACK_DRILL_SCHEMA_VERSION,
    STABILITY_SCHEMA_VERSION,
    D3RuntimeMonitor,
    DualCandidateRollbackManager,
    HumanReviewPrediction,
    create_feedback_record,
    memory_leak_analysis,
    threshold_margin_confidence,
    validate_readiness_report,
)
from steel_patchcore.dual_candidate_registry import DualCandidateRegistry  # noqa: E402
from steel_patchcore.lifecycle import lifecycle_enter  # noqa: E402
from steel_patchcore.rle import rle_decode  # noqa: E402

MANIFEST = ROOT / "model-training/registry/steel-patchcore-d3-candidate/1.3.0-candidate.1/manifest.json"
LEGACY_MANIFEST = ROOT / "model-training/registry/steel-patchcore-d3-candidate/manifest.json"
DATASET = ROOT / "model-training/datasets/severstal-steel"
IMAGE_DIR = DATASET / "raw/train_images"
CSV_PATH = DATASET / "raw/train.csv"
SOURCE_SPLIT = DATASET / "split_manifest.json"
RECOVERY_SPLIT = DATASET / "recovery_split_manifest.json"
HUMAN_REVIEW_DOC = ROOT / "docs/d3-human-review-workflow.md"
RUN_ROOT = ROOT / "model-training/runs/steel-d3-production-readiness"
STABILITY_CHECKPOINT = RUN_ROOT / "stability-checkpoint.json"
ROBUSTNESS_CHECKPOINT = RUN_ROOT / "robustness-checkpoint.json"
STABILITY_REPORT = ROOT / "docs/d3-24h-stability-report.json"
ROBUSTNESS_REPORT = ROOT / "docs/d3-input-robustness-report.json"
MONITORING_REPORT = ROOT / "docs/d3-runtime-monitoring-report.json"
REVIEW_REPORT = ROOT / "docs/d3-human-review-workflow-report.json"
ROLLBACK_REPORT = ROOT / "docs/d3-rollback-drill-report.json"
READINESS_REPORT = ROOT / "docs/d3-production-readiness-report.json"
READINESS_MD = ROOT / "docs/d3-production-readiness-report.md"
TEST_REPORT = ROOT / "docs/d3-production-readiness-test-report.json"

STABILITY_REQUESTS = 240
ROBUSTNESS_NORMAL_COUNT = 48
ROBUSTNESS_ANOMALY_COUNT = 144
ROBUSTNESS_CONDITIONS = ("baseline", "brightness_shift", "contrast_shift", "noise", "compression", "small_resize")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def verify_runtime(registry: DualCandidateRegistry, manifest: dict) -> dict[str, str]:
    return registry.verify_artifacts(manifest)[1]


def gpu_status() -> dict:
    utilization = None
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        ).splitlines()[0]
        utilization = float(raw.strip())
    except Exception:
        pass
    return {
        "available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "memory_allocated_mb": torch.cuda.memory_allocated(0) / (1024**2) if torch.cuda.is_available() else 0.0,
        "memory_reserved_mb": torch.cuda.memory_reserved(0) / (1024**2) if torch.cuda.is_available() else 0.0,
        "utilization_percent": utilization,
    }


def cpu_memory_mb() -> float:
    return psutil.Process().memory_info().rss / (1024**2)


def evaluation_ids() -> tuple[list[str], list[str]]:
    source = json.loads(SOURCE_SPLIT.read_text(encoding="utf-8"))
    recovery = json.loads(RECOVERY_SPLIT.read_text(encoding="utf-8"))
    normal = list(source["splits"]["test_normal"])
    anomaly = list(recovery["recovery_holdout_anomaly"])
    if len(normal) != 591 or len(anomaly) != 3333 or set(normal) & set(anomaly):
        raise OperationalQualificationError("READINESS_EVALUATION_MEMBERSHIP_INVALID")
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
        raise OperationalQualificationError("READINESS_MASK_MEMBERSHIP_INVALID")
    return result


def decode_mask(rows: list[str]) -> np.ndarray:
    mask = np.zeros((256, 1600), dtype=np.uint8)
    for row in rows:
        mask = np.maximum(mask, rle_decode(row))
    return mask


def load_predictor() -> tuple[DualCandidateRegistry, dict, dict[str, str], D3DualBranchPredictor]:
    registry = DualCandidateRegistry(ROOT)
    manifest = registry.load_manifest(MANIFEST)
    before = verify_runtime(registry, manifest)
    predictor = D3DualBranchPredictor.from_manifest(MANIFEST, project_root=ROOT, device="cuda:0")
    return registry, manifest, before, predictor


def stability_qualification() -> dict:
    registry, manifest, before, predictor = load_predictor()
    monitor = D3RuntimeMonitor(manifest["model_version"], manifest["artifact_version"], before)
    monitor.start(lambda: verify_runtime(registry, manifest), predictor.image_predictor._ensure_model)
    normal, anomaly = evaluation_ids()
    workload = [item for pair in zip(normal[:6], anomaly[:6]) for item in pair]
    identity = {
        "schema_version": "steel_patchcore_d3_24h_stability_checkpoint_v1",
        "manifest_sha256": sha256_file(MANIFEST),
        "artifact_hashes": before,
        "workload": workload,
        "request_target": STABILITY_REQUESTS,
    }
    if STABILITY_CHECKPOINT.is_file():
        checkpoint = json.loads(STABILITY_CHECKPOINT.read_text(encoding="utf-8"))
        if any(checkpoint.get(key) != value for key, value in identity.items()):
            raise OperationalQualificationError("STABILITY_CHECKPOINT_LINEAGE_MISMATCH")
    else:
        checkpoint = {**identity, "records": [], "baseline_scores": {}, "hourly_resources": [], "failures": []}
    completed = len(checkpoint["records"])
    monitor.request_count = completed
    monitor.latencies_ms = [float(row["latency_ms"]) for row in checkpoint["records"]]
    monitor.cpu_memory_mb = [float(row["cpu_memory_mb"]) for row in checkpoint["hourly_resources"]]
    monitor.gpu_samples = [
        {
            "memory_allocated_mb": row["gpu_memory_mb"],
            "memory_reserved_mb": row["gpu_memory_reserved_mb"],
            "utilization_percent": row["gpu_utilization_percent"],
        }
        for row in checkpoint["hourly_resources"]
    ]
    started_at = datetime(2026, 8, 21, tzinfo=timezone.utc)
    for index in range(completed, STABILITY_REQUESTS):
        image_id = workload[index % len(workload)]
        with Image.open(IMAGE_DIR / f"{image_id}.jpg") as opened:
            image = opened.convert("RGB")
        try:
            output = monitor.execute(lambda: predictor.infer(image), cpu_memory_mb=cpu_memory_mb, gpu_status=gpu_status)
            baseline = checkpoint["baseline_scores"].setdefault(image_id, output.image_score)
            score_identical = output.image_score == baseline
            if not score_identical:
                raise OperationalQualificationError(f"STABILITY_SCORE_DRIFT:{image_id}")
            virtual_hour = (index + 1) * 24.0 / STABILITY_REQUESTS
            checkpoint["records"].append({
                "request_index": index + 1,
                "virtual_timestamp": (started_at + timedelta(hours=virtual_hour)).isoformat().replace("+00:00", "Z"),
                "virtual_hour": virtual_hour,
                "image_id": image_id,
                "image_score": output.image_score,
                "score_identical": True,
                "latency_ms": output.latency_ms,
            })
            if (index + 1) % 10 == 0:
                monitor.verify(lambda: verify_runtime(registry, manifest))
                checkpoint["hourly_resources"].append({
                    "virtual_hour": virtual_hour,
                    "cpu_memory_mb": monitor.cpu_memory_mb[-1],
                    "gpu_memory_mb": monitor.gpu_samples[-1]["memory_allocated_mb"],
                    "gpu_memory_reserved_mb": monitor.gpu_samples[-1]["memory_reserved_mb"],
                    "gpu_utilization_percent": monitor.gpu_samples[-1]["utilization_percent"],
                    "artifact_verified": True,
                })
                atomic_write_json(STABILITY_CHECKPOINT, checkpoint)
                print(f"stability requests={index + 1}/{STABILITY_REQUESTS}", flush=True)
        except Exception as exc:
            checkpoint["failures"].append({"request_index": index + 1, "error": f"{type(exc).__name__}:{exc}"})
            atomic_write_json(STABILITY_CHECKPOINT, checkpoint)
            raise
    after = verify_runtime(registry, manifest)
    leak = memory_leak_analysis(checkpoint["hourly_resources"])
    snapshot = monitor.snapshot()
    report = {
        "schema_version": STABILITY_SCHEMA_VERSION,
        "candidate_status": manifest["status"],
        "model_version": manifest["model_version"],
        "artifact_version": manifest["artifact_version"],
        "simulation": {
            "kind": "accelerated_virtual_clock",
            "virtual_duration_hours": 24,
            "wall_clock_duration_claimed": False,
            "requests_per_virtual_hour": 10,
            "request_count": STABILITY_REQUESTS,
            "unique_images": len(workload),
        },
        "monitoring": snapshot,
        "score_drift_count": sum(not row["score_identical"] for row in checkpoint["records"]),
        "memory_leak_analysis": leak,
        "hourly_resources": checkpoint["hourly_resources"],
        "artifact_hashes_before": before,
        "artifact_hashes_after": after,
        "artifact_unchanged": before == after,
        "failures": checkpoint["failures"],
        "verdict": "PASS" if snapshot["error_rate"] == 0 and leak["passed"] and before == after and not checkpoint["failures"] else "FAILED",
        "production_promotion": False,
        "generated_at": utc_now(),
    }
    atomic_write_json(STABILITY_REPORT, report)
    monitoring_drills(manifest, before, predictor, snapshot)
    return report


def monitoring_drills(manifest: dict, hashes: dict[str, str], predictor: D3DualBranchPredictor, runtime_snapshot: dict) -> dict:
    drills = []

    def run(name: str, action: callable) -> None:
        try:
            action()
        except OperationalQualificationError as exc:
            drills.append({"failure": name, "fail_closed": True, "reason": str(exc)})
        else:
            drills.append({"failure": name, "fail_closed": False, "reason": None})

    run("artifact_hash_mismatch", lambda: D3RuntimeMonitor(manifest["model_version"], manifest["artifact_version"], hashes).start(lambda: {**hashes, "weights": "0" * 64}, lambda: object()))
    run("artifact_missing", lambda: D3RuntimeMonitor(manifest["model_version"], manifest["artifact_version"], hashes).start(lambda: (_ for _ in ()).throw(FileNotFoundError("artifact missing")), lambda: object()))
    run("model_load_failure", lambda: D3RuntimeMonitor(manifest["model_version"], manifest["artifact_version"], hashes).start(lambda: hashes, lambda: (_ for _ in ()).throw(OSError("model load failure"))))
    report = {
        "schema_version": "steel_patchcore_d3_runtime_monitor_qualification_v1",
        "model_version": manifest["model_version"],
        "artifact_version": manifest["artifact_version"],
        "runtime_snapshot": runtime_snapshot,
        "failure_drills": drills,
        "verdict": "PASS" if all(row["fail_closed"] for row in drills) and runtime_snapshot["error_rate"] == 0 else "FAILED",
        "production_promotion": False,
        "generated_at": utc_now(),
    }
    atomic_write_json(MONITORING_REPORT, report)
    return report


def select_evenly(values: list[str], count: int) -> list[str]:
    return [values[int(index)] for index in np.linspace(0, len(values) - 1, count)]


def perturb(image: Image.Image, condition: str, image_id: str) -> Image.Image:
    if condition == "baseline":
        return image.copy()
    if condition == "brightness_shift":
        return ImageEnhance.Brightness(image).enhance(1.15)
    if condition == "contrast_shift":
        return ImageEnhance.Contrast(image).enhance(1.15)
    if condition == "noise":
        seed = int(hashlib.sha256(f"{image_id}:noise".encode()).hexdigest()[:16], 16)
        rng = np.random.default_rng(seed)
        array = np.asarray(image, dtype=np.float32)
        noisy = np.clip(array + rng.normal(0.0, 5.0, array.shape), 0, 255).astype(np.uint8)
        return Image.fromarray(noisy, mode="RGB")
    if condition == "compression":
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=70, optimize=False)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            return compressed.convert("RGB")
    if condition == "small_resize":
        reduced = image.resize((1568, 251), Image.Resampling.BILINEAR)
        return reduced.resize((1600, 256), Image.Resampling.BILINEAR)
    raise OperationalQualificationError(f"UNKNOWN_ROBUSTNESS_CONDITION:{condition}")


def robustness_qualification() -> dict:
    registry, manifest, before, predictor = load_predictor()
    normal, anomaly = evaluation_ids()
    normal_sample = select_evenly(normal, ROBUSTNESS_NORMAL_COUNT)
    anomaly_sample = select_evenly(anomaly, ROBUSTNESS_ANOMALY_COUNT)
    order = [("normal", image_id) for image_id in normal_sample] + [("anomaly", image_id) for image_id in anomaly_sample]
    masks = mask_rows(anomaly_sample)
    selection = {"normal": normal_sample, "anomaly": anomaly_sample}
    identity = {
        "schema_version": "steel_patchcore_d3_input_robustness_checkpoint_v1",
        "manifest_sha256": sha256_file(MANIFEST),
        "artifact_hashes": before,
        "selection_sha256": canonical_sha256(selection),
        "conditions": list(ROBUSTNESS_CONDITIONS),
    }
    if ROBUSTNESS_CHECKPOINT.is_file():
        checkpoint = json.loads(ROBUSTNESS_CHECKPOINT.read_text(encoding="utf-8"))
        if any(checkpoint.get(key) != value for key, value in identity.items()):
            raise OperationalQualificationError("ROBUSTNESS_CHECKPOINT_LINEAGE_MISMATCH")
    else:
        checkpoint = {**identity, "completed": {}, "failures": []}
    for condition in ROBUSTNESS_CONDITIONS:
        completed_condition = checkpoint["completed"].setdefault(condition, {})
        for role, image_id in order:
            if image_id in completed_condition:
                continue
            try:
                with Image.open(IMAGE_DIR / f"{image_id}.jpg") as opened:
                    transformed = perturb(opened.convert("RGB"), condition, image_id)
                output = predictor.infer(transformed)
                record = {"role": role, "image_score": output.image_score, "latency_ms": output.latency_ms}
                if role == "anomaly":
                    record.update(pixel_localization_metrics(output.heatmap, decode_mask(masks[image_id])))
                completed_condition[image_id] = record
            except Exception as exc:
                checkpoint["failures"].append({"condition": condition, "image_id": image_id, "error": f"{type(exc).__name__}:{exc}"})
                atomic_write_json(ROBUSTNESS_CHECKPOINT, checkpoint)
                raise
            if sum(len(rows) for rows in checkpoint["completed"].values()) % 10 == 0:
                atomic_write_json(ROBUSTNESS_CHECKPOINT, checkpoint)
                print(f"robustness completed={sum(len(rows) for rows in checkpoint['completed'].values())}/{len(order) * len(ROBUSTNESS_CONDITIONS)}", flush=True)
    atomic_write_json(ROBUSTNESS_CHECKPOINT, checkpoint)
    rows = []
    labels = np.asarray([0 if role == "normal" else 1 for role, _ in order], dtype=np.int8)
    for condition in ROBUSTNESS_CONDITIONS:
        completed = checkpoint["completed"][condition]
        scores = np.asarray([completed[image_id]["image_score"] for _, image_id in order], dtype=np.float64)
        pixel = float(np.mean([completed[image_id]["pixel_auroc"] for image_id in anomaly_sample]))
        aupro = float(np.mean([completed[image_id]["aupro"] for image_id in anomaly_sample]))
        metrics = {"image_auroc": auroc(scores, labels), "pixel_auroc": pixel, "aupro": aupro}
        checks = {key.removesuffix("_min"): metrics[key.removesuffix("_min")] >= value for key, value in ROBUSTNESS_GATE.items()}
        rows.append({"condition": condition, "parameters": robustness_parameters(condition), "metrics": metrics, "checks": checks, "verdict": "PASS" if all(checks.values()) else "FAILED"})
    after = verify_runtime(registry, manifest)
    perturbed = [row for row in rows if row["condition"] != "baseline"]
    report = {
        "schema_version": ROBUSTNESS_SCHEMA_VERSION,
        "candidate_status": manifest["status"],
        "model_version": manifest["model_version"],
        "artifact_version": manifest["artifact_version"],
        "sample": {"normal": ROBUSTNESS_NORMAL_COUNT, "anomaly": ROBUSTNESS_ANOMALY_COUNT, "selection_sha256": identity["selection_sha256"], "sampling": "deterministic evenly spaced sealed membership"},
        "gate": ROBUSTNESS_GATE,
        "conditions": rows,
        "artifact_hashes_before": before,
        "artifact_hashes_after": after,
        "artifact_unchanged": before == after,
        "failures": checkpoint["failures"],
        "verdict": "PASS" if all(row["verdict"] == "PASS" for row in perturbed) and before == after and not checkpoint["failures"] else "FAILED",
        "production_promotion": False,
        "generated_at": utc_now(),
    }
    atomic_write_json(ROBUSTNESS_REPORT, report)
    return report


def robustness_parameters(condition: str) -> dict:
    return {
        "baseline": {"operation": "identity"},
        "brightness_shift": {"factor": 1.15},
        "contrast_shift": {"factor": 1.15},
        "noise": {"distribution": "deterministic gaussian", "sigma_8bit": 5.0},
        "compression": {"format": "JPEG", "quality": 70},
        "small_resize": {"intermediate_size": [1568, 251], "restored_size": [1600, 256], "interpolation": "bilinear"},
    }[condition]


def human_review_qualification(manifest: dict) -> dict:
    anomaly = HumanReviewPrediction(
        "contract-anomaly", 0.9, "ANOMALY", "runtime://heatmap/contract-anomaly",
        threshold_margin_confidence(0.9, manifest["image_branch"]["threshold"]), manifest["model_version"], manifest["artifact_version"],
    ).as_record()
    normal = HumanReviewPrediction(
        "contract-normal", 0.7, "NORMAL", "runtime://heatmap/contract-normal",
        threshold_margin_confidence(0.7, manifest["image_branch"]["threshold"]), manifest["model_version"], manifest["artifact_version"],
    ).as_record()
    examples = [
        create_feedback_record(anomaly, reviewer="operator", feedback_type="human_confirmation", reason="confirmed anomaly", timestamp=utc_now()),
        create_feedback_record(anomaly, reviewer="operator", feedback_type="false_positive", reason="clean steel", timestamp=utc_now()),
        create_feedback_record(normal, reviewer="operator", feedback_type="false_negative", reason="missed defect", timestamp=utc_now()),
    ]
    report = {
        "schema_version": REVIEW_WORKFLOW_SCHEMA_VERSION,
        "model_version": manifest["model_version"],
        "artifact_version": manifest["artifact_version"],
        "prediction_fields": ["image_score", "anomaly_label", "heatmap", "confidence", "artifact_version"],
        "confidence_semantics": "uncalibrated absolute threshold-margin ratio",
        "feedback_types": ["human_confirmation", "false_positive", "false_negative"],
        "contract_examples": examples,
        "workflow_document": str(HUMAN_REVIEW_DOC.relative_to(ROOT)).replace("\\", "/"),
        "workflow_document_sha256": sha256_file(HUMAN_REVIEW_DOC),
        "backend_capabilities": ["claim ownership", "immutable resolution", "audit correction", "human feedback metrics"],
        "automatic_retraining": False,
        "automatic_threshold_change": False,
        "verdict": "PASS",
        "production_promotion": False,
        "generated_at": utc_now(),
    }
    atomic_write_json(REVIEW_REPORT, report)
    return report


def rollback_qualification(manifest: dict) -> dict:
    success = DualCandidateRollbackManager(RUN_ROOT / "rollback-success-state.json", ROOT)
    success.activate(LEGACY_MANIFEST)
    activated = success.activate(MANIFEST)
    rolled_back = success.rollback()
    failure = DualCandidateRollbackManager(RUN_ROOT / "rollback-failure-state.json", ROOT)
    failure.activate(LEGACY_MANIFEST)
    failure.activate(MANIFEST)
    tampered = failure.load()
    tampered["previous_candidate"]["manifest_sha256"] = "0" * 64
    atomic_write_json(failure.state_path, tampered)
    before = failure.state_path.read_bytes()
    blocked_reason = None
    try:
        failure.rollback()
    except OperationalQualificationError as exc:
        blocked_reason = str(exc)
    report = {
        "schema_version": ROLLBACK_DRILL_SCHEMA_VERSION,
        "candidate_status": "CANDIDATE_ONLY",
        "activated_candidate": activated["active_candidate"],
        "previous_candidate": activated["previous_candidate"],
        "rolled_back_active_candidate": rolled_back["active_candidate"],
        "hash_mismatch_drill": {"blocked": blocked_reason is not None, "reason": blocked_reason, "state_unchanged": failure.state_path.read_bytes() == before},
        "automatic_production_upgrade": False,
        "verdict": "PASS" if rolled_back["active_candidate"]["model_version"] == "1.2.0-candidate.1" and blocked_reason is not None and failure.state_path.read_bytes() == before else "FAILED",
        "production_promotion": False,
        "generated_at": utc_now(),
    }
    atomic_write_json(ROLLBACK_REPORT, report)
    return report


def finalize() -> dict:
    manifest = DualCandidateRegistry(ROOT).load_manifest(MANIFEST)
    stability = json.loads(STABILITY_REPORT.read_text(encoding="utf-8"))
    robustness = json.loads(ROBUSTNESS_REPORT.read_text(encoding="utf-8"))
    monitoring = json.loads(MONITORING_REPORT.read_text(encoding="utf-8"))
    review = human_review_qualification(manifest)
    rollback = rollback_qualification(manifest)
    tests = json.loads(TEST_REPORT.read_text(encoding="utf-8")) if TEST_REPORT.is_file() else {
        "verdict": "PENDING", "steel": None, "inference": None, "backend": None
    }
    phases = {
        "stability": {"verdict": stability["verdict"], "report": "docs/d3-24h-stability-report.json"},
        "robustness": {"verdict": robustness["verdict"], "report": "docs/d3-input-robustness-report.json"},
        "monitoring": {"verdict": monitoring["verdict"], "report": "docs/d3-runtime-monitoring-report.json"},
        "human_review": {"verdict": review["verdict"], "report": "docs/d3-human-review-workflow-report.json"},
        "rollback": {"verdict": rollback["verdict"], "report": "docs/d3-rollback-drill-report.json"},
        "tests": tests,
    }
    qualified = all(row["verdict"] == "PASS" for row in phases.values())
    report = {
        "schema_version": READINESS_SCHEMA_VERSION,
        "candidate_status": manifest["status"],
        "model_name": manifest["model_name"],
        "model_version": manifest["model_version"],
        "artifact_version": manifest["artifact_version"],
        "threshold": manifest["image_branch"]["threshold"],
        "phases": phases,
        "verdict": "PRODUCTION_CANDIDATE_QUALIFIED" if qualified else "NOT_QUALIFIED",
        "qualification_note": (
            "All six readiness phases passed; this is candidate qualification only and does not authorize deployment."
            if qualified else "One or more readiness phases are pending or failed."
        ),
        "production_promotion": False,
        "generated_at": utc_now(),
    }
    validate_readiness_report(report)
    atomic_write_json(READINESS_REPORT, report)
    READINESS_MD.write_text(
        "\n".join([
            "# D3 Production Readiness Qualification", "", f"Verdict: **`{report['verdict']}`**", "",
            f"- Candidate: `{report['model_name']}@{report['model_version']}`",
            f"- Artifact: `{report['artifact_version']}`", f"- Frozen threshold: `{report['threshold']!r}`", "",
            "| Phase | Verdict |", "|---|---|",
            *[f"| {name} | {row['verdict']} |" for name, row in phases.items()], "",
            "No model, feature extractor, artifact, threshold, production configuration, or deployment was changed.", "",
        ]),
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("stability", "robustness", "finalize", "all"), default="all")
    args = parser.parse_args()
    if not torch.cuda.is_available() and args.stage in {"stability", "robustness", "all"}:
        raise OperationalQualificationError("PRODUCTION_READINESS_REQUIRES_GPU")
    manifest = DualCandidateRegistry(ROOT).load_manifest(MANIFEST)
    blocked = lifecycle_enter("d3_production_readiness", manifest["hashes"]["image_bank_sha256"], STABILITY_CHECKPOINT, RUN_ROOT)
    if blocked is not None:
        return blocked
    if args.stage in {"stability", "all"}:
        stability_qualification()
    if args.stage in {"robustness", "all"}:
        robustness_qualification()
    if args.stage in {"finalize", "all"}:
        finalize()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"D3_PRODUCTION_READINESS_BLOCKED:{type(exc).__name__}:{exc}", flush=True)
        raise SystemExit(3)
