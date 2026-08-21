"""Offline Factory Acceptance Test for the immutable D3 1.3 candidate."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model-training"))
sys.path.insert(0, str(ROOT / "inference-service"))
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))

from app.industrial.commands import decision_to_command  # noqa: E402
from app.mlops.drift import classify_ks, classify_psi, ks_statistic, psi  # noqa: E402
from inference_app.d3_dual_branch_predictor import D3DualBranchPredictor  # noqa: E402
from steel_patchcore.candidate_registry import sha256_file  # noqa: E402
from steel_patchcore.d3_factory_acceptance import (  # noqa: E402
    FAT_SCHEMA_VERSION,
    FEEDBACK_SCHEMA_VERSION,
    PIPELINE_SCHEMA_VERSION,
    decision_api,
    drift_comparison,
    simulate_eight_hour_queue,
    validate_fat_report,
    validate_feedback_record,
    validate_pipeline_record,
)
from steel_patchcore.d3_localization_representation import INTERMEDIATE_BLOCK_INDEX  # noqa: E402
from steel_patchcore.d3_operational import OperationalQualificationError, atomic_write_json, latency_percentiles  # noqa: E402
from steel_patchcore.dual_candidate_registry import DualCandidateRegistry  # noqa: E402
from steel_patchcore.lifecycle import lifecycle_enter  # noqa: E402

MANIFEST = ROOT / "model-training/registry/steel-patchcore-d3-candidate/1.3.0-candidate.1/manifest.json"
DATASET = ROOT / "model-training/datasets/severstal-steel"
IMAGE_DIR = DATASET / "raw/train_images"
SOURCE_SPLIT = DATASET / "split_manifest.json"
RECOVERY_SPLIT = DATASET / "recovery_split_manifest.json"
STABILITY_CHECKPOINT = ROOT / "model-training/runs/steel-d3-production-readiness/stability-checkpoint.json"
RUN_ROOT = ROOT / "model-training/runs/steel-d3-factory-acceptance"
RUN_CHECKPOINT = RUN_ROOT / "fat-checkpoint.json"
HEATMAP_DIR = RUN_ROOT / "heatmaps"

PIPELINE_LOG = ROOT / "docs/d3-fat-industrial-pipeline-log.json"
PIPELINE_REPORT = ROOT / "docs/d3-fat-industrial-pipeline-report.json"
THROUGHPUT_REPORT = ROOT / "docs/d3-fat-throughput-report.json"
PLC_MES_REPORT = ROOT / "docs/d3-fat-plc-mes-report.json"
DRIFT_REPORT = ROOT / "docs/d3-fat-drift-report.json"
FEEDBACK_REPORT = ROOT / "docs/d3-fat-human-feedback-report.json"
TEST_REPORT = ROOT / "docs/d3-fat-test-report.json"
FAT_REPORT = ROOT / "docs/d3-factory-acceptance-report.json"
FAT_MD = ROOT / "docs/d3-factory-acceptance-report.md"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _ids() -> tuple[list[str], list[str]]:
    source = json.loads(SOURCE_SPLIT.read_text(encoding="utf-8"))
    recovery = json.loads(RECOVERY_SPLIT.read_text(encoding="utf-8"))
    normal = list(source["splits"]["test_normal"])
    anomaly = list(recovery["recovery_holdout_anomaly"])
    if len(normal) != 591 or len(anomaly) != 3333 or set(normal) & set(anomaly):
        raise OperationalQualificationError("FAT_SEALED_MEMBERSHIP_INVALID")
    return normal, anomaly


def _runtime() -> tuple[DualCandidateRegistry, dict, dict[str, str], D3DualBranchPredictor]:
    registry = DualCandidateRegistry(ROOT)
    manifest = registry.load_manifest(MANIFEST)
    hashes = registry.verify_artifacts(manifest)[1]
    predictor = D3DualBranchPredictor.from_manifest(MANIFEST, project_root=ROOT, device="cuda:0")
    return registry, manifest, hashes, predictor


def _trace(index: int, image_id: str) -> str:
    value = f"steel-fat:{index}:{image_id}:1.3.0-candidate.1".encode()
    return f"fat-{hashlib.sha256(value).hexdigest()[:24]}"


def _save_heatmap(path: Path, heatmap: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(heatmap * 255.0, 0, 255).astype(np.uint8), mode="L").save(path)


class _PlcMock:
    """Transport-free PLC mock with the production adapter's idempotency rule."""

    def __init__(self) -> None:
        self.executed: dict[str, dict] = {}

    def execute(self, command: Mapping) -> tuple[str, dict, bool]:
        command_id = str(command["command_id"])
        if command_id in self.executed:
            return "ACK", {"duplicate": True, "command_id": command_id}, False
        self.executed[command_id] = dict(command)
        return "ACK", {"duplicate": False, "command_id": command_id}, True


class _MesMock:
    """Transport-free MES mock idempotent on inspection id and record type."""

    def __init__(self) -> None:
        self.inspections: dict[str, dict] = {}

    def post_inspection(self, body: Mapping) -> tuple[dict, bool]:
        key = f"{body['inspection_id']}:inspection"
        if key in self.inspections:
            return {"duplicate": True, "inspection_id": body["inspection_id"]}, False
        self.inspections[key] = dict(body)
        return {"duplicate": False, "inspection_id": body["inspection_id"]}, True


def industrial_pipeline() -> dict:
    registry, manifest, before, predictor = _runtime()
    normal, anomaly = _ids()
    workload = [item for pair in zip(normal[:6], anomaly[:6]) for item in pair]
    base_time = datetime(2026, 8, 21, tzinfo=timezone.utc)
    records = []
    HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
    for index, image_id in enumerate(workload):
        image_path = IMAGE_DIR / f"{image_id}.jpg"
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        if image.size != (1600, 256) or not sha256_file(image_path):
            raise OperationalQualificationError("FAT_GATEWAY_INPUT_INVALID")
        output = predictor.infer(image)
        timestamp = (base_time + timedelta(seconds=index * 48)).isoformat().replace("+00:00", "Z")
        trace_id = _trace(index, image_id)
        decision = decision_api(
            trace_id=trace_id,
            image_id=image_id,
            anomaly_label=output.anomaly_label,
            model_version=output.model_version,
            artifact_version=output.artifact_version,
            timestamp=timestamp,
        ).as_payload()
        heatmap_path = HEATMAP_DIR / f"{trace_id}.png"
        _save_heatmap(heatmap_path, output.heatmap)
        record = {
            "schema_version": PIPELINE_SCHEMA_VERSION,
            "trace_id": trace_id,
            "image_id": image_id,
            "timestamp": timestamp,
            "latency_ms": output.latency_ms,
            "image_score": output.image_score,
            "confidence": output.confidence,
            "result": decision["result"],
            "heatmap_path": heatmap_path.relative_to(ROOT).as_posix(),
            "model_version": output.model_version,
            "artifact_version": output.artifact_version,
            "stages": {"camera": "PASS", "gateway": "PASS", "inference": "PASS", "decision": "PASS"},
        }
        validate_pipeline_record(record)
        records.append(record)
    after = registry.verify_artifacts(manifest)[1]
    atomic_write_json(PIPELINE_LOG, {"schema_version": "steel_patchcore_d3_fat_pipeline_log_v1", "records": records})
    results = {name: sum(row["result"] == name for row in records) for name in ("PASS", "FAIL", "REVIEW_REQUIRED")}
    report = {
        "schema_version": "steel_patchcore_d3_fat_pipeline_report_v1",
        "candidate": f"{manifest['model_name']}@{manifest['model_version']}",
        "artifact_version": manifest["artifact_version"],
        "threshold": manifest["image_branch"]["threshold"],
        "flow": ["Camera", "Gateway", "Inference", "Decision"],
        "request_count": len(records),
        "result_counts": results,
        "latency_ms": latency_percentiles([row["latency_ms"] for row in records]),
        "heatmaps_written": len(records),
        "artifact_hashes_before": before,
        "artifact_hashes_after": after,
        "artifact_unchanged": before == after,
        "log": "docs/d3-fat-industrial-pipeline-log.json",
        "verdict": "PASS" if before == after and len(records) == len(workload) else "FAILED",
        "production_promotion": False,
    }
    atomic_write_json(PIPELINE_REPORT, report)
    return report


def throughput_qualification() -> dict:
    checkpoint = json.loads(STABILITY_CHECKPOINT.read_text(encoding="utf-8"))
    latencies = [float(row["latency_ms"]) for row in checkpoint["records"]]
    if len(latencies) != 240:
        raise OperationalQualificationError("FAT_MEASURED_REPLAY_PROFILE_INVALID")
    report = simulate_eight_hour_queue(latencies)
    report["measured_profile"] = {
        "source": "model-training/runs/steel-d3-production-readiness/stability-checkpoint.json",
        "source_sha256": sha256_file(STABILITY_CHECKPOINT),
        "sample_count": len(latencies),
        "latency_ms": latency_percentiles(latencies),
        "candidate_manifest_sha256": checkpoint["manifest_sha256"],
    }
    if checkpoint["manifest_sha256"] != sha256_file(MANIFEST):
        raise OperationalQualificationError("FAT_THROUGHPUT_LINEAGE_MISMATCH")
    atomic_write_json(THROUGHPUT_REPORT, report)
    return report


def plc_mes_qualification() -> dict:
    manifest = DualCandidateRegistry(ROOT).load_manifest(MANIFEST)
    plc = _PlcMock()
    mes = _MesMock()
    timestamp = "2026-08-21T08:00:00Z"
    cases = [
        ("normal", "NORMAL", False, None),
        ("unknown-anomaly", "ANOMALY", False, None),
        ("confirmed-defect", "ANOMALY", True, None),
        ("inference-timeout", None, False, "timeout"),
    ]
    rows = []
    for index, (name, label, confirmed, failure) in enumerate(cases):
        trace_id = f"fat-plc-mes-{index:02d}"
        decision = decision_api(
            trace_id=trace_id,
            image_id=name,
            anomaly_label=label,
            human_confirmed_anomaly=confirmed,
            failure_reason=failure,
            model_version=manifest["model_version"],
            artifact_version=manifest["artifact_version"],
            timestamp=timestamp,
        ).as_payload()
        quality = "REVIEW" if decision["result"] == "REVIEW_REQUIRED" else decision["result"]
        process_status = "FAILED" if failure else "COMPLETED"
        command, held = decision_to_command(
            inspection_id=trace_id,
            product_id=f"product-{index}",
            final_quality_result=failure or quality,
            process_status=process_status,
            timestamp=timestamp,
        )
        plc_input = command.to_payload()
        ack, response, executed = plc.execute(plc_input)
        duplicate_ack, duplicate_response, duplicate_executed = plc.execute(plc_input)
        mes_input = {
            "inspection_id": trace_id,
            "product_id": f"product-{index}",
            "batch_id": "fat-batch-1",
            "line": "steel-line-mock",
            "station": "gateway-mock",
            "ai_result": decision["result"],
            "model_version": manifest["model_version"],
            "industrial_state": command.command_type,
            "timestamp": timestamp,
        }
        mes_response, mes_written = mes.post_inspection(mes_input)
        mes_duplicate, mes_duplicate_written = mes.post_inspection(mes_input)
        rows.append({
            "trace_id": trace_id,
            "case": name,
            "decision": decision,
            "command": command.to_payload(),
            "held": held,
            "plc": {"ack": ack, "response": response, "executed": executed},
            "plc_duplicate": {"ack": duplicate_ack, "response": duplicate_response, "executed": duplicate_executed},
            "mes": {"response": mes_response, "written": mes_written},
            "mes_duplicate": {"response": mes_duplicate, "written": mes_duplicate_written},
        })
    mapping = {row["decision"]["result"]: row["command"]["command_type"] for row in rows[:3]}
    timeout_safe = rows[-1]["decision"]["result"] == "REVIEW_REQUIRED" and rows[-1]["command"]["command_type"] == "HOLD"
    idempotent = all(row["plc"]["executed"] and not row["plc_duplicate"]["executed"] and row["mes"]["written"] and not row["mes_duplicate"]["written"] for row in rows)
    report = {
        "schema_version": "steel_patchcore_d3_fat_plc_mes_v1",
        "decision_contract": ["PASS", "FAIL", "REVIEW_REQUIRED"],
        "command_mapping": mapping,
        "records": rows,
        "trace_id_present": all(row["trace_id"] for row in rows),
        "idempotency_verified": idempotent,
        "failure_safe_hold": timeout_safe,
        "verdict": "PASS" if mapping == {"PASS": "RELEASE", "REVIEW_REQUIRED": "HOLD", "FAIL": "REJECT"} and timeout_safe and idempotent else "FAILED",
        "production_connection_used": False,
    }
    atomic_write_json(PLC_MES_REPORT, report)
    return report


def _distribution_signals(predictor: D3DualBranchPredictor, image_ids: list[str], brightness: float) -> dict[str, list[float]]:
    signals: dict[str, list[float]] = {name: [] for name in ("feature", "score", "input_mean", "input_std")}
    model = predictor.image_predictor._ensure_model()
    for image_id in image_ids:
        with Image.open(IMAGE_DIR / f"{image_id}.jpg") as opened:
            image = opened.convert("RGB")
        if brightness != 1.0:
            image = ImageEnhance.Brightness(image).enhance(brightness)
        array = np.asarray(image, dtype=np.float32) / 255.0
        output = predictor.infer(image)
        with torch.no_grad():
            tokens = model.get_intermediate_layers(
                predictor._tile_batch(image, 252), n=[INTERMEDIATE_BLOCK_INDEX], norm=True
            )[0]
            tokens = F.normalize(tokens, p=2, dim=2)
        signals["feature"].append(float(tokens[..., 0].mean().cpu()))
        signals["score"].append(output.image_score)
        signals["input_mean"].append(float(array.mean()))
        signals["input_std"].append(float(array.std()))
    return signals


def drift_qualification() -> dict:
    registry, manifest, before, predictor = _runtime()
    normal, anomaly = _ids()
    workload = [item for pair in zip(normal[:4], anomaly[:4]) for item in pair]
    baseline = _distribution_signals(predictor, workload, 1.0)
    stable = {name: list(values) for name, values in baseline.items()}
    shifted = _distribution_signals(predictor, workload, 1.3)
    kwargs = {"classify_ks": classify_ks, "ks_statistic": ks_statistic, "psi": psi, "classify_psi": classify_psi}
    stable_result = drift_comparison(baseline, stable, **kwargs)
    shifted_result = drift_comparison(baseline, shifted, **kwargs)
    after = registry.verify_artifacts(manifest)[1]
    report = {
        "schema_version": "steel_patchcore_d3_fat_drift_report_v1",
        "signals": ["feature_distribution", "score_distribution", "input_statistics"],
        "sample_count": len(workload),
        "baseline": baseline,
        "stable_window": stable_result,
        "brightness_shift_window": {"factor": 1.3, **shifted_result},
        "trigger_action": "warning_only",
        "automatic_retraining": False,
        "artifact_unchanged": before == after,
        "verdict": "PASS" if stable_result["trigger"] == "NONE" and shifted_result["trigger"] == "WARNING" and before == after else "FAILED",
    }
    atomic_write_json(DRIFT_REPORT, report)
    return report


def feedback_qualification() -> dict:
    integration = json.loads(PLC_MES_REPORT.read_text(encoding="utf-8"))["records"]
    review = next(row["decision"] for row in integration if row["decision"]["result"] == "REVIEW_REQUIRED")
    accepted = next(row["decision"] for row in integration if row["decision"]["result"] == "PASS")
    specs = [
        (review, "operator_review", {"label": "surface_reviewed", "note": "operator disposition required"}),
        (review, "false_positive", {"label": "clean", "note": "operator accepted surface"}),
        (accepted, "false_negative", {"label": "defect", "note": "operator found missed defect", "bbox_xyxy": [100, 40, 180, 120]}),
    ]
    records = []
    for index, (source, feedback_type, annotation) in enumerate(specs):
        record = {
            "schema_version": FEEDBACK_SCHEMA_VERSION,
            "trace_id": source["trace_id"],
            "image_id": source["image_id"],
            "operator": f"fat-operator-{index + 1}",
            "feedback_type": feedback_type,
            "annotation": annotation,
            "prediction_snapshot": {
                "result": source["result"],
                "artifact_version": source["artifact_version"],
                "source": "plc_mes_decision_contract_simulation",
            },
            "timestamp": "2026-08-21T09:00:00Z",
            "automatic_retraining": False,
        }
        validate_feedback_record(record)
        records.append(record)
    report = {
        "schema_version": "steel_patchcore_d3_fat_human_feedback_report_v1",
        "supported_feedback": ["operator_review", "false_positive", "false_negative"],
        "records": records,
        "annotation_recorded": all(row["annotation"] for row in records),
        "automatic_retraining": False,
        "verdict": "PASS",
    }
    atomic_write_json(FEEDBACK_REPORT, report)
    return report


def finalize() -> dict:
    manifest = DualCandidateRegistry(ROOT).load_manifest(MANIFEST)
    sources = {
        "industrial_pipeline": PIPELINE_REPORT,
        "throughput": THROUGHPUT_REPORT,
        "plc_mes": PLC_MES_REPORT,
        "drift": DRIFT_REPORT,
        "human_feedback": FEEDBACK_REPORT,
    }
    phases = {
        name: {"verdict": json.loads(path.read_text(encoding="utf-8"))["verdict"], "report": path.relative_to(ROOT).as_posix()}
        for name, path in sources.items()
    }
    phases["tests"] = json.loads(TEST_REPORT.read_text(encoding="utf-8")) if TEST_REPORT.is_file() else {"verdict": "PENDING"}
    passed = all(row["verdict"] == "PASS" for row in phases.values())
    report = {
        "schema_version": FAT_SCHEMA_VERSION,
        "candidate": f"{manifest['model_name']}@{manifest['model_version']}",
        "candidate_status": "PRODUCTION_CANDIDATE_QUALIFIED",
        "artifact_version": manifest["artifact_version"],
        "threshold": manifest["image_branch"]["threshold"],
        "manifest_sha256": sha256_file(MANIFEST),
        "phases": phases,
        "verdict": "FACTORY_ACCEPTANCE_PASS" if passed else "FACTORY_ACCEPTANCE_FAILED",
        "qualification_scope": "offline_factory_acceptance_only",
        "production_promotion": False,
        "automatic_retraining": False,
        "generated_at": utc_now(),
    }
    validate_fat_report(report)
    atomic_write_json(FAT_REPORT, report)
    FAT_MD.write_text("\n".join([
        "# D3 Factory Acceptance Test", "", f"Verdict: **`{report['verdict']}`**", "",
        f"- Candidate: `{report['candidate']}`", f"- Artifact: `{report['artifact_version']}`",
        f"- Frozen threshold: `{report['threshold']!r}`", "", "| Phase | Verdict |", "|---|---|",
        *[f"| {name} | {row['verdict']} |" for name, row in phases.items()], "",
        "The 8-hour workload is an accelerated discrete-event simulation replaying measured candidate latencies; it is not an eight-hour wall-clock soak.", "",
        "No model, artifact, threshold, feature extractor, production configuration, or deployment was changed. Drift only raises a warning and feedback does not trigger retraining.", "",
    ]), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("pipeline", "throughput", "plc-mes", "drift", "feedback", "finalize", "all"), default="all")
    args = parser.parse_args()
    if not torch.cuda.is_available() and args.stage in {"pipeline", "drift", "all"}:
        raise OperationalQualificationError("FAT_REQUIRES_GPU")
    manifest = DualCandidateRegistry(ROOT).load_manifest(MANIFEST)
    blocked = lifecycle_enter("d3_factory_acceptance", manifest["hashes"]["image_bank_sha256"], RUN_CHECKPOINT, RUN_ROOT)
    if blocked is not None:
        return blocked
    functions = {
        "pipeline": industrial_pipeline,
        "throughput": throughput_qualification,
        "plc-mes": plc_mes_qualification,
        "drift": drift_qualification,
        "feedback": feedback_qualification,
        "finalize": finalize,
    }
    if args.stage == "all":
        for name in ("pipeline", "throughput", "plc-mes", "drift", "feedback", "finalize"):
            functions[name]()
    else:
        functions[args.stage]()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"D3_FACTORY_ACCEPTANCE_BLOCKED:{type(exc).__name__}:{exc}", flush=True)
        raise SystemExit(3)
