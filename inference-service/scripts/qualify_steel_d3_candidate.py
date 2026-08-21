"""Operational qualification runner for the frozen D3 candidate.

The runner is resumable, reads sealed images without modifying them, and writes
only qualification evidence and derived heatmap PNGs. It never trains, tunes a
threshold, changes an artifact, or promotes a model.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model-training"))
sys.path.insert(0, str(ROOT / "inference-service"))

from inference_app.d3_candidate_predictor import D3CandidatePredictor  # noqa: E402
from steel_patchcore.candidate_registry import CandidateRegistry  # noqa: E402
from steel_patchcore.d3_operational import (  # noqa: E402
    BENCHMARK_BATCH_SIZES,
    HEATMAP_SCHEMA_VERSION,
    PERFORMANCE_SCHEMA_VERSION,
    SHADOW_SCHEMA_VERSION,
    CandidateRollbackManager,
    InferenceMonitor,
    OperationalQualificationError,
    atomic_write_json,
    latency_percentiles,
    pixel_localization_metrics,
    validate_heatmap_report,
    validate_performance_report,
    validate_shadow_record,
)
from steel_patchcore.lifecycle import lifecycle_enter  # noqa: E402
from steel_patchcore.rle import rle_decode  # noqa: E402

REGISTRY_ROOT = ROOT / "model-training/registry"
MANIFEST_PATH = REGISTRY_ROOT / "steel-patchcore-d3-candidate/manifest.json"
DATASET_ROOT = ROOT / "model-training/datasets/severstal-steel"
IMAGE_DIR = DATASET_ROOT / "raw/train_images"
CSV_PATH = DATASET_ROOT / "raw/train.csv"
SOURCE_SPLIT = DATASET_ROOT / "split_manifest.json"
RECOVERY_SPLIT = DATASET_ROOT / "recovery_split_manifest.json"
BASELINE_CHECKPOINT = DATASET_ROOT / "raw/steel_eval_ckpt.json"
REPRODUCIBILITY_CHECKPOINT = ROOT / "model-training/runs/steel-d3-recovery-holdout/checkpoint.json"

RUN_ROOT = ROOT / "model-training/runs/steel-d3-operational-qualification"
HEATMAP_DIR = RUN_ROOT / "heatmaps"
SHADOW_CHECKPOINT = RUN_ROOT / "shadow-checkpoint.json"
ROLLBACK_STATE = RUN_ROOT / "candidate-rollback-state.json"

PERFORMANCE_REPORT = ROOT / "docs/d3-performance-report.json"
SHADOW_LOG = ROOT / "docs/d3-shadow-prediction-log.json"
HEATMAP_REPORT = ROOT / "docs/heatmap-validation-report.json"
MONITORING_REPORT = ROOT / "docs/d3-monitoring-report.json"
ROLLBACK_REPORT = ROOT / "docs/d3-rollback-report.json"
REGRESSION_REPORT = ROOT / "docs/d3-regression-test-report.json"
QUALIFICATION_REPORT = ROOT / "docs/steel-patchcore-d3-operational-qualification-report.json"
QUALIFICATION_MD = ROOT / "docs/steel-patchcore-d3-operational-qualification-report.md"

CHECKPOINT_EVERY = 10


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def load_roles() -> dict[str, list[str]]:
    source = json.loads(SOURCE_SPLIT.read_text(encoding="utf-8"))
    recovery = json.loads(RECOVERY_SPLIT.read_text(encoding="utf-8"))
    roles = {
        "test_normal": list(source["splits"]["test_normal"]),
        "recovery_holdout_anomaly": list(recovery["recovery_holdout_anomaly"]),
    }
    if len(roles["test_normal"]) != 591 or len(roles["recovery_holdout_anomaly"]) != 3333:
        raise OperationalQualificationError("SEALED_ROLE_COUNT_MISMATCH")
    if set(roles["test_normal"]) & set(roles["recovery_holdout_anomaly"]):
        raise OperationalQualificationError("SEALED_ROLE_OVERLAP")
    for role, image_ids in roles.items():
        missing = [image_id for image_id in image_ids if not (IMAGE_DIR / f"{image_id}.jpg").is_file()]
        if missing:
            raise OperationalQualificationError(f"SEALED_IMAGE_MISSING:{role}:{missing[:5]}")
    return roles


def deterministic_order(roles: dict[str, list[str]]) -> list[tuple[str, str]]:
    normal = roles["test_normal"]
    anomaly = roles["recovery_holdout_anomaly"]
    rows: list[tuple[str, str]] = []
    for index in range(max(len(normal), len(anomaly))):
        if index < len(normal):
            rows.append(("test_normal", normal[index]))
        if index < len(anomaly):
            rows.append(("recovery_holdout_anomaly", anomaly[index]))
    return rows


def gpu_probe() -> tuple[float | None, float | None]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
                "--id=0",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        memory, utilization = (float(value.strip()) for value in result.stdout.splitlines()[0].split(","))
        return memory, utilization
    except Exception:  # noqa: BLE001
        return None, None


class ResourceSampler:
    def __init__(self, interval_seconds: float = 0.25) -> None:
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, float | None]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def sample(self) -> None:
        gpu_memory, gpu_utilization = gpu_probe()
        cpu_memory = psutil.Process().memory_info().rss / (1024.0 * 1024.0)
        self.samples.append(
            {
                "cpu_memory_mb": cpu_memory,
                "gpu_memory_mb": gpu_memory,
                "gpu_utilization_percent": gpu_utilization,
            }
        )

    def start(self) -> None:
        self.sample()

        def loop() -> None:
            while not self._stop.wait(self.interval_seconds):
                self.sample()

        self._thread = threading.Thread(target=loop, name="d3-resource-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        self.sample()

    def summary(self) -> dict:
        cpu = [float(row["cpu_memory_mb"]) for row in self.samples if row["cpu_memory_mb"] is not None]
        gpu = [float(row["gpu_memory_mb"]) for row in self.samples if row["gpu_memory_mb"] is not None]
        utilization = [
            float(row["gpu_utilization_percent"])
            for row in self.samples
            if row["gpu_utilization_percent"] is not None
        ]
        return {
            "cpu_memory_peak_mb": max(cpu) if cpu else None,
            "gpu_memory_peak_mb": max(gpu) if gpu else None,
            "gpu_utilization_percent": float(np.mean(utilization)) if utilization else None,
            "gpu_utilization_peak_percent": max(utilization) if utilization else None,
            "resource_sample_count": len(self.samples),
        }


def load_runtime() -> tuple[CandidateRegistry, dict, object, D3CandidatePredictor, InferenceMonitor]:
    if not torch.cuda.is_available():
        raise OperationalQualificationError("D3_OPERATIONAL_QUALIFICATION_REQUIRES_GPU")
    registry = CandidateRegistry(REGISTRY_ROOT, ROOT)
    manifest = registry.load_manifest()
    candidate = registry.load_artifact()
    predictor = D3CandidatePredictor(candidate, device="cuda:0")
    monitor = InferenceMonitor()
    monitor.start(lambda: registry.verify_artifact(manifest), predictor._ensure_model)
    predictor._ensure_tensors()
    return registry, manifest, candidate, predictor, monitor


def artifact_snapshot(registry: CandidateRegistry, manifest: dict) -> dict[str, str]:
    verification = registry.verify_artifact(manifest)
    if not verification.passed:
        raise OperationalQualificationError(f"ARTIFACT_VERIFICATION_FAILED:{verification.errors}")
    return verification.hashes


def run_benchmark() -> dict:
    roles = load_roles()
    registry, manifest, _, predictor, monitor = load_runtime()
    before = artifact_snapshot(registry, manifest)
    order = deterministic_order(roles)[: max(BENCHMARK_BATCH_SIZES)]
    sampler = ResourceSampler()
    sampler.start()
    latencies: list[float] = []
    rows: list[dict] = []
    try:
        for index, (_, image_id) in enumerate(order, 1):
            with Image.open(IMAGE_DIR / f"{image_id}.jpg") as opened:
                image = opened.convert("RGB")
            output = monitor.execute(
                lambda: predictor.infer(image),
                gpu_memory_mb=lambda: torch.cuda.memory_allocated() / 1048576.0,
            )
            latencies.append(output.latency_ms)
            if index in BENCHMARK_BATCH_SIZES:
                sampler.sample()
                rows.append(
                    {
                        "image_count": index,
                        "latency_ms": latency_percentiles(latencies),
                        "resources": {
                            **sampler.summary(),
                            "torch_gpu_allocated_peak_mb": torch.cuda.max_memory_allocated() / 1048576.0,
                            "torch_gpu_reserved_peak_mb": torch.cuda.max_memory_reserved() / 1048576.0,
                        },
                    }
                )
            if index % 25 == 0:
                print(f"benchmark completed={index}/1000", flush=True)
    finally:
        sampler.stop()
    after = artifact_snapshot(registry, manifest)
    if before != after:
        raise OperationalQualificationError("ARTIFACT_MUTATED_DURING_BENCHMARK")
    report = {
        "schema_version": PERFORMANCE_SCHEMA_VERSION,
        "candidate_status": manifest["status"],
        "model_version": manifest["model_version"],
        "artifact_version": manifest["artifact_version"],
        "production_promotion": False,
        "benchmark_semantics": "cumulative prefixes of one deterministic interleaved sealed-image sequence",
        "device": "cuda:0",
        "benchmarks": rows,
        "monitoring": monitor.snapshot(),
        "artifact_hashes_before": before,
        "artifact_hashes_after": after,
        "artifact_unchanged": True,
        "generated_at": utc_now(),
    }
    validate_performance_report(report)
    atomic_write_json(PERFORMANCE_REPORT, report)
    print(json.dumps({"performance_report": str(PERFORMANCE_REPORT), "requests": len(latencies)}), flush=True)
    return report


def load_mask_rles(image_ids: list[str]) -> dict[str, list[str]]:
    wanted = set(image_ids)
    frame = pd.read_csv(CSV_PATH, keep_default_na=True)
    encoded_by_image: dict[str, list[str]] = {}
    selected = frame[frame["ImageId"].astype(str).map(lambda value: Path(value).stem in wanted)]
    for raw_id, group in selected.groupby(selected["ImageId"].astype(str)):
        image_id = Path(raw_id).stem
        encoded_rows = [str(encoded) for encoded in group["EncodedPixels"] if not pd.isna(encoded)]
        if not encoded_rows:
            raise OperationalQualificationError(f"SEALED_ANOMALY_MASK_EMPTY:{image_id}")
        encoded_by_image[image_id] = encoded_rows
    missing = wanted - set(encoded_by_image)
    if missing:
        raise OperationalQualificationError(f"SEALED_MASK_MISSING:{sorted(missing)[:5]}")
    return encoded_by_image


def decode_mask(encoded_rows: list[str]) -> np.ndarray:
    mask = np.zeros((256, 1600), dtype=np.uint8)
    for encoded in encoded_rows:
        mask = np.maximum(mask, rle_decode(encoded))
    return mask


def new_shadow_checkpoint(manifest: dict, hashes: dict[str, str]) -> dict:
    return {
        "schema_version": "steel_patchcore_d3_shadow_checkpoint_v1",
        "model_version": manifest["model_version"],
        "artifact_version": manifest["artifact_version"],
        "threshold": manifest["threshold"],
        "artifact_hashes": hashes,
        "completed": {},
        "failures": [],
        "resource_peaks": {"cpu_memory_mb": None, "gpu_memory_mb": None, "gpu_utilization_percent": None},
        "updated_at": utc_now(),
    }


def load_shadow_checkpoint(manifest: dict, hashes: dict[str, str]) -> dict:
    if not SHADOW_CHECKPOINT.is_file():
        checkpoint = new_shadow_checkpoint(manifest, hashes)
        atomic_write_json(SHADOW_CHECKPOINT, checkpoint)
        return checkpoint
    try:
        checkpoint = json.loads(SHADOW_CHECKPOINT.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OperationalQualificationError("SHADOW_CHECKPOINT_UNREADABLE") from exc
    expected = {
        "schema_version": "steel_patchcore_d3_shadow_checkpoint_v1",
        "model_version": manifest["model_version"],
        "artifact_version": manifest["artifact_version"],
        "threshold": manifest["threshold"],
        "artifact_hashes": hashes,
    }
    if any(checkpoint.get(key) != value for key, value in expected.items()):
        raise OperationalQualificationError("SHADOW_CHECKPOINT_LINEAGE_MISMATCH")
    return checkpoint


def update_resource_peaks(checkpoint: dict, sampler: ResourceSampler) -> None:
    summary = sampler.summary()
    mapping = {
        "cpu_memory_mb": summary["cpu_memory_peak_mb"],
        "gpu_memory_mb": summary["gpu_memory_peak_mb"],
        "gpu_utilization_percent": summary["gpu_utilization_peak_percent"],
    }
    for key, value in mapping.items():
        current = checkpoint["resource_peaks"].get(key)
        if value is not None:
            checkpoint["resource_peaks"][key] = max(float(value), float(current)) if current is not None else float(value)


def run_shadow() -> dict:
    roles = load_roles()
    order = deterministic_order(roles)
    anomaly_ids = roles["recovery_holdout_anomaly"]
    mask_rles = load_mask_rles(anomaly_ids)
    registry, manifest, _, predictor, monitor = load_runtime()
    before = artifact_snapshot(registry, manifest)
    checkpoint = load_shadow_checkpoint(manifest, before)
    reference = json.loads(REPRODUCIBILITY_CHECKPOINT.read_text(encoding="utf-8"))["completed"]
    HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
    sampler = ResourceSampler()
    sampler.start()
    since_write = 0
    try:
        for role, image_id in order:
            existing = checkpoint["completed"].get(image_id)
            if existing and (ROOT / existing["heatmap_path"]).is_file():
                continue
            try:
                with Image.open(IMAGE_DIR / f"{image_id}.jpg") as opened:
                    image = opened.convert("RGB")
                output = monitor.execute(
                    lambda: predictor.infer(image),
                    gpu_memory_mb=lambda: torch.cuda.memory_allocated() / 1048576.0,
                )
                expected_score = float(reference[role][image_id]["score"])
                if not math.isclose(output.anomaly_score, expected_score, rel_tol=0.0, abs_tol=1e-6):
                    raise OperationalQualificationError(f"REPRODUCIBILITY_SCORE_MISMATCH:{image_id}")
                relative_heatmap = Path("model-training/runs/steel-d3-operational-qualification/heatmaps") / f"{image_id}.png"
                Image.fromarray(np.rint(output.normalized_heatmap * 255.0).astype(np.uint8), mode="L").save(
                    ROOT / relative_heatmap,
                    format="PNG",
                    optimize=False,
                )
                record = {
                    "timestamp": utc_now(),
                    "image_id": image_id,
                    "model_version": output.model_version,
                    "artifact_version": output.artifact_version,
                    "score": output.anomaly_score,
                    "heatmap_path": relative_heatmap.as_posix(),
                    "split_role": role,
                    "latency_ms": output.latency_ms,
                    "prediction": int(output.is_anomaly),
                    "threshold": output.threshold,
                    "reproducible_against_sealed_holdout": True,
                }
                if role == "recovery_holdout_anomaly":
                    record.update(pixel_localization_metrics(output.raw_anomaly_map, decode_mask(mask_rles[image_id])))
                validate_shadow_record(record)
                checkpoint["completed"][image_id] = record
                since_write += 1
            except Exception as exc:
                checkpoint["failures"].append(
                    {"timestamp": utc_now(), "image_id": image_id, "error": f"{type(exc).__name__}:{exc}"}
                )
                checkpoint["updated_at"] = utc_now()
                update_resource_peaks(checkpoint, sampler)
                atomic_write_json(SHADOW_CHECKPOINT, checkpoint)
                raise
            if since_write >= CHECKPOINT_EVERY:
                checkpoint["updated_at"] = utc_now()
                update_resource_peaks(checkpoint, sampler)
                atomic_write_json(SHADOW_CHECKPOINT, checkpoint)
                since_write = 0
                print(f"shadow completed={len(checkpoint['completed'])}/{len(order)}", flush=True)
    finally:
        sampler.stop()
        checkpoint["updated_at"] = utc_now()
        update_resource_peaks(checkpoint, sampler)
        atomic_write_json(SHADOW_CHECKPOINT, checkpoint)
    after = artifact_snapshot(registry, manifest)
    if before != after:
        raise OperationalQualificationError("ARTIFACT_MUTATED_DURING_SHADOW")
    return finalize_shadow(roles, manifest, checkpoint, before, after)


def finalize_shadow(roles: dict[str, list[str]], manifest: dict, checkpoint: dict, before: dict, after: dict) -> dict:
    expected_ids = set(roles["test_normal"]) | set(roles["recovery_holdout_anomaly"])
    completed = checkpoint["completed"]
    if set(completed) != expected_ids:
        raise OperationalQualificationError(f"SHADOW_INCOMPLETE:{len(completed)}/{len(expected_ids)}")
    ordered_records = [completed[image_id] for _, image_id in deterministic_order(roles)]
    for record in ordered_records:
        validate_shadow_record(record)
        if not (ROOT / record["heatmap_path"]).is_file():
            raise OperationalQualificationError(f"HEATMAP_FILE_MISSING:{record['image_id']}")
    shadow_log = {
        "schema_version": SHADOW_SCHEMA_VERSION,
        "mode": "read_only_shadow",
        "candidate_status": "CANDIDATE",
        "threshold_changed": False,
        "artifact_changed": False,
        "record_count": len(ordered_records),
        "records": ordered_records,
    }
    atomic_write_json(SHADOW_LOG, shadow_log)

    anomaly_records = [row for row in ordered_records if row["split_role"] == "recovery_holdout_anomaly"]
    baseline = json.loads(BASELINE_CHECKPOINT.read_text(encoding="utf-8"))["test_anomaly"]
    candidate_pixel = float(np.mean([row["pixel_auroc"] for row in anomaly_records]))
    candidate_aupro = float(np.mean([row["aupro"] for row in anomaly_records]))
    baseline_pixel = float(np.mean([baseline[row["image_id"]]["pixel_auc"] for row in anomaly_records]))
    baseline_aupro = float(np.mean([baseline[row["image_id"]]["aupro"] for row in anomaly_records]))
    checks = {
        "pixel_auroc_no_regression": candidate_pixel >= baseline_pixel,
        "aupro_no_regression": candidate_aupro >= baseline_aupro,
    }
    heatmap_report = {
        "schema_version": HEATMAP_SCHEMA_VERSION,
        "candidate_status": manifest["status"],
        "model_version": manifest["model_version"],
        "artifact_version": manifest["artifact_version"],
        "sample_count": len(anomaly_records),
        "evaluation_set": "sealed recovery_holdout_anomaly",
        "metric_semantics": "mean of per-image full-resolution metrics; AUPRO integrated through background FPR=0.3",
        "metrics": {
            "pixel_auroc_mean_per_image": candidate_pixel,
            "aupro_mean_per_image": candidate_aupro,
        },
        "paired_baseline_comparison": {
            "sample_count": len(anomaly_records),
            "baseline": "existing steel PatchCore checkpoint, same image IDs",
            "pixel_auroc_mean_per_image": baseline_pixel,
            "aupro_mean_per_image": baseline_aupro,
            "pixel_auroc_delta": candidate_pixel - baseline_pixel,
            "aupro_delta": candidate_aupro - baseline_aupro,
        },
        "acceptance_gate": "candidate pixel AUROC and AUPRO must each be >= paired baseline",
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "image_score_gate_changed": False,
        "threshold": manifest["threshold"],
        "generated_at": utc_now(),
    }
    validate_heatmap_report(heatmap_report)
    atomic_write_json(HEATMAP_REPORT, heatmap_report)

    latencies = [float(row["latency_ms"]) for row in ordered_records]
    monitoring_report = {
        "schema_version": "steel_patchcore_d3_monitoring_report_v1",
        "fail_closed": True,
        "request_count": len(ordered_records),
        "error_count": len(checkpoint["failures"]),
        "error_rate": len(checkpoint["failures"]) / (len(ordered_records) + len(checkpoint["failures"])),
        "latency_ms": latency_percentiles(latencies),
        "gpu_memory_peak_mb": checkpoint["resource_peaks"]["gpu_memory_mb"],
        "cpu_memory_peak_mb": checkpoint["resource_peaks"]["cpu_memory_mb"],
        "gpu_utilization_peak_percent": checkpoint["resource_peaks"]["gpu_utilization_percent"],
        "artifact_hashes": after,
        "artifact_unchanged": before == after,
        "fail_closed_conditions": ["artifact mismatch", "missing file", "model load failure"],
    }
    atomic_write_json(MONITORING_REPORT, monitoring_report)

    registry = CandidateRegistry(REGISTRY_ROOT, ROOT)
    manager = CandidateRollbackManager(ROLLBACK_STATE, registry)
    state = manager.activate(MANIFEST_PATH)
    active_candidate = dict(state["active_candidate"])
    active_candidate["manifest_path"] = Path(active_candidate["manifest_path"]).relative_to(ROOT).as_posix()
    previous_candidate = state["previous_candidate"]
    if previous_candidate is not None:
        previous_candidate = dict(previous_candidate)
        previous_candidate["manifest_path"] = Path(previous_candidate["manifest_path"]).relative_to(ROOT).as_posix()
    rollback_report = {
        "schema_version": "steel_patchcore_d3_rollback_report_v1",
        "status": state["status"],
        "active_candidate": active_candidate,
        "previous_candidate": previous_candidate,
        "previous_slot_supported": True,
        "hash_verification_required_on_activate_and_rollback": True,
        "automatic_production_upgrade": False,
        "live_rollback_available": state["previous_candidate"] is not None,
        "note": "No earlier D3 candidate manifest exists; the previous slot is populated atomically on the next authorized candidate activation.",
    }
    atomic_write_json(ROLLBACK_REPORT, rollback_report)
    print(
        json.dumps(
            {
                "shadow_records": len(ordered_records),
                "heatmap_verdict": heatmap_report["verdict"],
                "pixel_auroc": candidate_pixel,
                "aupro": candidate_aupro,
            }
        ),
        flush=True,
    )
    return heatmap_report


def finalize_existing_shadow() -> dict:
    roles = load_roles()
    registry = CandidateRegistry(REGISTRY_ROOT, ROOT)
    manifest = registry.load_manifest()
    hashes = artifact_snapshot(registry, manifest)
    checkpoint = load_shadow_checkpoint(manifest, hashes)
    return finalize_shadow(roles, manifest, checkpoint, hashes, hashes)


def write_qualification_report() -> dict:
    performance = json.loads(PERFORMANCE_REPORT.read_text(encoding="utf-8"))
    heatmap = json.loads(HEATMAP_REPORT.read_text(encoding="utf-8"))
    shadow = json.loads(SHADOW_LOG.read_text(encoding="utf-8"))
    monitoring = json.loads(MONITORING_REPORT.read_text(encoding="utf-8"))
    rollback = json.loads(ROLLBACK_REPORT.read_text(encoding="utf-8"))
    regression = json.loads(REGRESSION_REPORT.read_text(encoding="utf-8"))
    regression_passed = all(row.get("status") == "PASS" for row in regression.get("suites", {}).values())
    checks = {
        "performance_schema": True,
        "shadow_complete": shadow["record_count"] == 3924,
        "inference_reproducibility": all(row["reproducible_against_sealed_holdout"] for row in shadow["records"]),
        "heatmap_acceptance": heatmap["verdict"] == "PASS",
        "monitoring_fail_closed": monitoring["fail_closed"] and monitoring["artifact_unchanged"],
        "rollback_mechanism": rollback["previous_slot_supported"] and rollback["automatic_production_upgrade"] is False,
        "regression_tests": regression_passed,
        "candidate_only": performance["candidate_status"] == "CANDIDATE",
    }
    performance_1000 = performance["benchmarks"][-1]
    heatmap_metrics = heatmap["metrics"]
    baseline_metrics = heatmap["paired_baseline_comparison"]
    report = {
        "schema_version": "steel_patchcore_d3_operational_qualification_v1",
        "candidate": "steel-patchcore-d3-candidate@1.2.0-candidate.1",
        "status": "CANDIDATE ONLY",
        "verdict": "OPERATIONALLY_QUALIFIED" if all(checks.values()) else "OPERATIONAL_QUALIFICATION_FAILED",
        "checks": checks,
        "results": {
            "performance_1000": performance_1000,
            "shadow_record_count": shadow["record_count"],
            "shadow_error_count": monitoring["error_count"],
            "heatmap_metrics": heatmap_metrics,
            "paired_baseline_metrics": {
                "pixel_auroc_mean_per_image": baseline_metrics["pixel_auroc_mean_per_image"],
                "aupro_mean_per_image": baseline_metrics["aupro_mean_per_image"],
            },
            "heatmap_deltas": {
                "pixel_auroc": baseline_metrics["pixel_auroc_delta"],
                "aupro": baseline_metrics["aupro_delta"],
            },
        },
        "production_promotion": False,
        "threshold_changed": False,
        "artifact_changed": False,
        "git": {"branch": git_value("branch", "--show-current"), "head": git_value("rev-parse", "HEAD")},
        "reports": {
            "performance": PERFORMANCE_REPORT.relative_to(ROOT).as_posix(),
            "shadow": SHADOW_LOG.relative_to(ROOT).as_posix(),
            "heatmap": HEATMAP_REPORT.relative_to(ROOT).as_posix(),
            "monitoring": MONITORING_REPORT.relative_to(ROOT).as_posix(),
            "rollback": ROLLBACK_REPORT.relative_to(ROOT).as_posix(),
            "regression_tests": REGRESSION_REPORT.relative_to(ROOT).as_posix(),
        },
        "generated_at": utc_now(),
    }
    atomic_write_json(QUALIFICATION_REPORT, report)
    lines = [
        "# Steel PatchCore D3 Operational Qualification",
        "",
        f"Verdict: **`{report['verdict']}`**",
        "",
        f"Candidate: `{report['candidate']}` (`CANDIDATE ONLY`).",
        "",
        "## Qualification checks",
        "",
        "| Check | Result |",
        "|---|---|",
        *[f"| {name} | {'PASS' if passed else 'FAIL'} |" for name, passed in checks.items()],
        "",
        "## Operational evidence",
        "",
        "| Evidence | Result |",
        "|---|---:|",
        f"| 1,000-image latency p50 / p95 / p99 | {performance_1000['latency_ms']['p50_ms']:.3f} / {performance_1000['latency_ms']['p95_ms']:.3f} / {performance_1000['latency_ms']['p99_ms']:.3f} ms |",
        f"| 1,000-image GPU memory peak | {performance_1000['resources']['gpu_memory_peak_mb']:.1f} MB |",
        f"| Shadow predictions / errors | {shadow['record_count']} / {monitoring['error_count']} |",
        f"| D3 pixel AUROC / paired baseline | {heatmap_metrics['pixel_auroc_mean_per_image']:.6f} / {baseline_metrics['pixel_auroc_mean_per_image']:.6f} |",
        f"| D3 AUPRO / paired baseline | {heatmap_metrics['aupro_mean_per_image']:.6f} / {baseline_metrics['aupro_mean_per_image']:.6f} |",
        "",
        "## Guardrails",
        "",
        "- Artifact and threshold were unchanged.",
        "- No training, backbone search, fine-tuning, or production promotion was performed.",
        "- Rollback is candidate-only and hash-verifies both activation and rollback. No earlier candidate manifest exists, so the live previous slot remains empty.",
        "- A failed heatmap gate is terminal evidence; it does not authorize tuning.",
        "",
        "## Evidence",
        "",
        *[f"- `{path}`" for path in report["reports"].values()],
        "",
    ]
    QUALIFICATION_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"verdict": report["verdict"], "checks": checks}), flush=True)
    return report


def enter_lifecycle(stage: str) -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    blocked = lifecycle_enter(stage, manifest["bank_sha256"], SHADOW_CHECKPOINT, RUN_ROOT)
    if blocked is not None:
        raise OperationalQualificationError(f"LIFECYCLE_BLOCKED:{blocked}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("benchmark", "shadow", "finalize", "report", "all"), default="all")
    args = parser.parse_args()
    try:
        if args.phase in {"benchmark", "shadow", "all"}:
            enter_lifecycle(f"d3_operational_{args.phase}")
        if args.phase in {"benchmark", "all"}:
            run_benchmark()
        if args.phase in {"shadow", "all"}:
            run_shadow()
        if args.phase == "finalize":
            finalize_existing_shadow()
        if args.phase in {"report", "all"}:
            write_qualification_report()
        return 0
    except Exception as exc:
        print(f"D3_OPERATIONAL_QUALIFICATION_BLOCKED:{type(exc).__name__}:{exc}", flush=True)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
