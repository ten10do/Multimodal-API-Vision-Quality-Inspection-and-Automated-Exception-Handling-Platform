"""One-shot evaluation of frozen D3 on the sealed recovery holdout.

No fitting, recalibration, candidate search, or artifact mutation is permitted.
The only resumable state is a per-original score checkpoint bound to the full
frozen lineage and full-precision development threshold.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model-training"))
sys.path.insert(0, str(ROOT / "inference-service"))

from steel_patchcore.d3_recovery_holdout import (  # noqa: E402
    BOOTSTRAP_ITERATIONS,
    BOOTSTRAP_SEED,
    EXPECTED_LINEAGE,
    FROZEN_QUARTILE_BOUNDARIES,
    HOLDOUT_COUNTS,
    HOLDOUT_GATE,
    HOLDOUT_ROLES,
    PROTOCOL_VERSION,
    RESULTS_SCHEMA_VERSION,
    HoldoutBlocked,
    a0_global_max,
    assert_artifacts_unchanged,
    assign_frozen_quartiles,
    evaluate_holdout,
    gate_verdict,
    load_frozen_threshold,
    new_checkpoint,
    record_checkpoint_result,
    stratified_bootstrap_auroc,
    validate_checkpoint,
    validate_holdout_membership,
    verify_artifact_lineage,
)
from steel_patchcore.domain_representation import adapted_input_side  # noqa: E402
from steel_patchcore.rle import rle_decode  # noqa: E402

DS = ROOT / "model-training/datasets/severstal-steel"
IMG_DIR = DS / "raw/train_images"
CSV = DS / "raw/train.csv"
SOURCE_SPLIT = DS / "split_manifest.json"
RECOVERY_SPLIT = DS / "recovery_split_manifest.json"
EVIDENCE_MANIFEST = DS / "recovery_evidence_manifest.json"
QUARTILE_MANIFEST = DS / "representation_diagnostic_manifest.json"
BASELINE_BANK = ROOT / "inference-service/models/steel-patchcore/bank.npz"
WEIGHTS = Path.home() / ".cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth"
D3_DIR = ROOT / "model-training/runs/steel-d3-full-development/D3-full-development"
WHITENING = D3_DIR / "whitening.npz"
D3_BANK = D3_DIR / "bank.npz"
D3_RESULTS = ROOT / "docs/steel-patchcore-d3-full-development-results.json"

RUN_ROOT = ROOT / "model-training/runs/steel-d3-recovery-holdout"
CHECKPOINT = RUN_ROOT / "checkpoint.json"
RESULTS_JSON = ROOT / "docs/steel-patchcore-d3-recovery-holdout-results.json"
RESULTS_MD = ROOT / "docs/steel-patchcore-d3-recovery-holdout-results.md"
CHECKPOINT_EVERY = 25

ARTIFACT_PATHS = {
    "baseline_bank_sha256": BASELINE_BANK,
    "source_split_sha256": SOURCE_SPLIT,
    "recovery_split_sha256": RECOVERY_SPLIT,
    "evidence_manifest_sha256": EVIDENCE_MANIFEST,
    "dino_weights_sha256": WEIGHTS,
    "whitening_sha256": WHITENING,
    "d3_bank_sha256": D3_BANK,
    "quartile_manifest_sha256": QUARTILE_MANIFEST,
    "d3_results_sha256": D3_RESULTS,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _sanitize(value):
    if isinstance(value, dict):
        return {str(k): _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_sanitize(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_representation_runner():
    path = ROOT / "inference-service/scripts/run_steel_representation_experiment.py"
    spec = importlib.util.spec_from_file_location("_d3_holdout_representation", path)
    if spec is None or spec.loader is None:
        raise HoldoutBlocked("REPRESENTATION_LOADER_UNAVAILABLE")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_dino_model(device: torch.device):
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    model.eval().to(device)
    if int(getattr(model, "embed_dim", -1)) != 768 or int(getattr(model, "patch_size", -1)) != 14:
        raise HoldoutBlocked("DINO_MODEL_IDENTITY_MISMATCH")
    return model


def extract_patch_tokens(model, tile: torch.Tensor) -> torch.Tensor:
    side = adapted_input_side(tile.shape[-1], 14)
    if tile.shape[-1] != side:
        tile = F.interpolate(tile, size=(side, side), mode="bilinear", align_corners=False)
    with torch.no_grad():
        output = model.forward_features(tile)
    tokens = output["x_norm_patchtokens"][0]
    if tuple(tokens.shape) != (18 * 18, 768):
        raise HoldoutBlocked(f"PATCH_TOKEN_SHAPE_MISMATCH:{tuple(tokens.shape)}")
    return tokens


def whiten_normalize(tokens: torch.Tensor, mean: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    return F.normalize((tokens - mean) @ matrix, p=2, dim=1)


def score_original(model, device, bank, mean, matrix, rep, image_id: str) -> float:
    """Frozen cosine 1-NN patch distance with A0 over seven tiles."""
    tile_maxima = []
    for tile in rep.load_normalized_tiles(IMG_DIR / f"{image_id}.jpg", device):
        embedding = whiten_normalize(extract_patch_tokens(model, tile), mean, matrix)
        distance = 1.0 - (embedding @ bank.T).max(dim=1).values
        tile_maxima.append(float(distance.max()))
    if len(tile_maxima) != 7:
        raise HoldoutBlocked(f"TILE_COUNT_MISMATCH:{image_id}:{len(tile_maxima)}")
    return a0_global_max(tile_maxima)


def load_area_ratios(image_ids: list[str]) -> np.ndarray:
    """Load GT areas only; deliberately never calculate holdout quantiles."""
    wanted = set(image_ids)
    frame = pd.read_csv(CSV, keep_default_na=True)
    values: dict[str, float] = {}
    for raw_id, group in frame.groupby(frame["ImageId"].astype(str)):
        image_id = Path(str(raw_id)).stem
        if image_id not in wanted:
            continue
        mask = np.zeros((256, 1600), dtype=np.uint8)
        for encoded in group["EncodedPixels"]:
            if not pd.isna(encoded):
                mask = np.maximum(mask, rle_decode(str(encoded)))
        values[image_id] = float(mask.sum()) / float(256 * 1600)
    missing = wanted - set(values)
    if missing:
        raise HoldoutBlocked(f"HOLDOUT_MASK_MISSING:{sorted(missing)[:5]}")
    return np.asarray([values[image_id] for image_id in image_ids], dtype=np.float64)


def load_or_create_checkpoint(lineage: dict[str, str], threshold: float, roles: dict[str, list[str]]):
    if CHECKPOINT.exists():
        try:
            checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HoldoutBlocked("CHECKPOINT_UNREADABLE") from exc
        resumed = validate_checkpoint(checkpoint, lineage, roles, threshold)
        return checkpoint, resumed
    checkpoint = new_checkpoint(lineage, threshold, utc_now())
    atomic_write_json(CHECKPOINT, checkpoint)
    return checkpoint, {role: 0 for role in HOLDOUT_ROLES}


def run_role(model, device, bank, mean, matrix, rep, checkpoint, role: str, ids: list[str]) -> None:
    completed = checkpoint["completed"][role]
    pending_since_write = 0
    for image_id in ids:
        if image_id in completed:
            continue
        score = score_original(model, device, bank, mean, matrix, rep, image_id)
        record_checkpoint_result(checkpoint, role, image_id, score)
        pending_since_write += 1
        if pending_since_write >= CHECKPOINT_EVERY:
            checkpoint["updated_at"] = utc_now()
            atomic_write_json(CHECKPOINT, checkpoint)
            pending_since_write = 0
            access_count = sum(len(checkpoint["completed"][key]) for key in HOLDOUT_ROLES)
            print(f"checkpoint role={role} completed={len(checkpoint['completed'][role])} HOLDOUT_ACCESS_COUNT={access_count}", flush=True)
    checkpoint["updated_at"] = utc_now()
    atomic_write_json(CHECKPOINT, checkpoint)


def render_results(payload: dict) -> str:
    metrics = payload["metrics"]
    bootstrap = payload["bootstrap_ci"]
    op = metrics["operating_point"]
    lines = [
        "# Steel PatchCore D3 Recovery Holdout Results",
        "",
        f"Verdict: **`{payload['verdict']}`**",
        "",
        "## 1. Handoff Audit",
        "",
        f"- Branch: `{payload['git']['branch']}`",
        f"- Pre-holdout HEAD: `{payload['pre_holdout_freeze']['commit']}`",
        "- Working tree, worker, GPU-worker, and lifecycle-lock audit passed before implementation.",
        "",
        "## 2. Pre-Holdout Freeze",
        "",
        f"- Protocol: `{payload['protocol_version']}`",
        f"- Freeze commit: `{payload['pre_holdout_freeze']['commit']}`",
        "- Evaluator, protocol, and CPU tests were frozen before any holdout image was opened.",
        "",
        "## 3. Frozen D3 Lineage",
        "",
    ]
    for key, value in payload["artifact_lineage"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## 4. Holdout Dataset",
            "",
            f"- test_normal: {payload['holdout_dataset']['test_normal']} unique originals",
            f"- recovery_holdout_anomaly: {payload['holdout_dataset']['recovery_holdout_anomaly']} unique originals",
            f"- development intersection: {payload['holdout_dataset']['development_intersection']}",
            f"- HOLDOUT_ACCESS_COUNT: {payload['one_shot_execution']['holdout_access_count']}",
            "",
            "## 5. One-shot Execution",
            "",
            "- DINOv2 ViT-B/14 raw x_norm_patchtokens; CLS/register excluded; 18x18x768 per tile.",
            "- Frozen full-development ZCA, 50,000-row seed-42 bank, seven frozen tiles, cosine 1-NN, A0 global max.",
            f"- Checkpoint resume counts at launch: `{payload['one_shot_execution']['resumed_counts']}`",
            "- Each original has exactly one checkpoint row bound to its split role.",
            "",
            "## 6. Metrics",
            "",
            f"- Image AUROC: **{metrics['image_auroc']:.6f}**",
            f"- Normal distribution: `{metrics['normal_distribution']}`",
            f"- Anomaly distribution: `{metrics['anomaly_distribution']}`",
            f"- Anomaly median - normal median: **{metrics['anomaly_minus_normal_median']:.6f}**",
            "",
            "## 7. Bootstrap CI",
            "",
            f"- Stratified bootstrap: seed={bootstrap['seed']}, iterations={bootstrap['iterations']}",
            f"- Median AUROC: {bootstrap['median']:.6f}",
            f"- 95% percentile CI: [{bootstrap['percentile_95_ci'][0]:.6f}, {bootstrap['percentile_95_ci'][1]:.6f}]",
            "",
            "## 8. Threshold Diagnostics",
            "",
            f"- Frozen full-development threshold (loaded, not recalibrated): `{metrics['frozen_threshold']!r}`",
            f"- TP={op['tp']} TN={op['tn']} FP={op['fp']} FN={op['fn']}",
            f"- Precision={op['precision']:.6f} Recall={op['recall']:.6f} F1={op['f1']:.6f}",
            f"- Normal FPR={op['normal_fpr']:.6f} Anomaly Recall={op['anomaly_recall']:.6f}",
            "- Confusion metrics are report-only and do not participate in the gate.",
            "",
            "## 9. Q1-Q4",
            "",
            f"Frozen development area boundaries: `{payload['quartile_analysis']['boundaries']}`",
            "",
            "| Quartile | Count | Normal-vs-quartile AUROC |",
            "|---|---:|---:|",
        ]
    )
    for row in metrics["quartiles"]:
        lines.append(f"| Q{row['quartile']} | {row['count']} | {row['normal_vs_quartile_auroc']:.6f} |")
    lines.extend(
        [
            "",
            "## 10. Gate Verdict",
            "",
            f"- Gate: AUROC >= {HOLDOUT_GATE['image_auroc_min']} AND anomaly median > normal median.",
            f"- Verdict: **`{payload['verdict']}`**",
            "",
            "## 11. Tests",
            "",
            f"- Pre-holdout command: `{payload['tests']['pre_holdout_command']}`",
            "- Coverage includes membership/isolation, fail-closed lineage and split behavior, threshold-only loading, no recalibration, checkpoint resume and duplicate rejection, A0 and metrics, frozen quartiles, deterministic bootstrap, and artifact immutability.",
            "",
            "## 12. Git",
            "",
            f"- Branch: `{payload['git']['branch']}`",
            f"- Freeze commit: `{payload['pre_holdout_freeze']['commit']}`",
            "- Results are committed separately with precise staging; main is not merged.",
            "",
            "## 13. Limitations",
            "",
            "- This is a single sealed holdout evaluation, not a production estimate across sites or acquisition shifts.",
            "- The threshold confusion metrics are diagnostic only and can be poor even when the rank-based gate passes.",
            "- No post-holdout tuning, recalibration, candidate search, or alternative-model evaluation was performed.",
            "",
        ]
    )
    return "\n".join(lines)


def write_blocked(reason: str) -> None:
    payload = {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "verdict": "RECOVERY_HOLDOUT_BLOCKED",
        "reason": reason,
        "generated_at": utc_now(),
    }
    atomic_write_json(RESULTS_JSON, payload)
    headings = [
        "Handoff Audit", "Pre-Holdout Freeze", "Frozen D3 Lineage", "Holdout Dataset",
        "One-shot Execution", "Metrics", "Bootstrap CI", "Threshold Diagnostics", "Q1-Q4",
        "Gate Verdict", "Tests", "Git", "Limitations",
    ]
    lines = ["# Steel PatchCore D3 Recovery Holdout Results", "", "Verdict: **`RECOVERY_HOLDOUT_BLOCKED`**", ""]
    for index, heading in enumerate(headings, 1):
        lines.extend([f"## {index}. {heading}", "", f"Blocked: `{reason}`", ""])
    RESULTS_MD.write_text("\n".join(lines), encoding="utf-8")


def execute() -> str:
    if not torch.cuda.is_available():
        raise HoldoutBlocked("D3_RECOVERY_HOLDOUT_REQUIRES_GPU")
    from steel_patchcore.lifecycle import lifecycle_enter

    blocked = lifecycle_enter("d3_recovery_holdout", EXPECTED_LINEAGE["d3_bank_sha256"], CHECKPOINT, RUN_ROOT)
    if blocked is not None:
        raise HoldoutBlocked(f"LIFECYCLE_BLOCKED:{blocked}")

    lineage = verify_artifact_lineage(ARTIFACT_PATHS)
    d3_results = json.loads(D3_RESULTS.read_text(encoding="utf-8"))
    threshold = load_frozen_threshold(d3_results)
    source = json.loads(SOURCE_SPLIT.read_text(encoding="utf-8"))
    recovery = json.loads(RECOVERY_SPLIT.read_text(encoding="utf-8"))
    roles = validate_holdout_membership(source, recovery)
    checkpoint, resumed_counts = load_or_create_checkpoint(lineage, threshold, roles)
    starting_access_count = sum(resumed_counts.values())
    print(
        f"UNSEAL_HOLDOUT test_normal unique={len(set(roles['test_normal']))} "
        f"recovery_holdout_anomaly unique={len(set(roles['recovery_holdout_anomaly']))} "
        f"development_intersection=0 HOLDOUT_ACCESS_COUNT={starting_access_count}",
        flush=True,
    )
    for role, ids in roles.items():
        missing = [image_id for image_id in ids if not (IMG_DIR / f"{image_id}.jpg").is_file()]
        if missing:
            raise HoldoutBlocked(f"HOLDOUT_IMAGE_MISSING:{role}:{missing[:5]}")

    device = torch.device("cuda:0")
    model = build_dino_model(device)
    rep = load_representation_runner()
    whitening = np.load(WHITENING)
    mean = torch.from_numpy(whitening["mean"].astype(np.float32)).to(device)
    matrix = torch.from_numpy(whitening["whitening_matrix"].astype(np.float32)).to(device)
    bank_np = np.load(D3_BANK)["features"].astype(np.float32)
    if bank_np.shape != (50000, 768):
        raise HoldoutBlocked(f"D3_BANK_SHAPE_MISMATCH:{bank_np.shape}")
    bank = torch.from_numpy(bank_np).to(device)

    for role in HOLDOUT_ROLES:
        run_role(model, device, bank, mean, matrix, rep, checkpoint, role, roles[role])
    final_counts = validate_checkpoint(checkpoint, lineage, roles, threshold)
    if final_counts != HOLDOUT_COUNTS:
        raise HoldoutBlocked(f"CHECKPOINT_INCOMPLETE:{final_counts}")
    assert_artifacts_unchanged(ARTIFACT_PATHS, lineage)

    normal_scores = np.asarray(
        [checkpoint["completed"]["test_normal"][image_id]["score"] for image_id in roles["test_normal"]],
        dtype=np.float64,
    )
    anomaly_ids = roles["recovery_holdout_anomaly"]
    anomaly_scores = np.asarray(
        [checkpoint["completed"]["recovery_holdout_anomaly"][image_id]["score"] for image_id in anomaly_ids],
        dtype=np.float64,
    )
    ratios = load_area_ratios(anomaly_ids)
    quartiles = assign_frozen_quartiles(ratios)
    metrics = evaluate_holdout(normal_scores, anomaly_scores, threshold, quartiles)
    bootstrap = stratified_bootstrap_auroc(
        normal_scores, anomaly_scores, seed=BOOTSTRAP_SEED, iterations=BOOTSTRAP_ITERATIONS
    )
    verdict = gate_verdict(metrics["image_auroc"], metrics["anomaly_median"], metrics["normal_median"])
    commit = git_head()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    payload = {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "verdict": verdict,
        "gate": HOLDOUT_GATE,
        "pre_holdout_freeze": {"commit": commit},
        "artifact_lineage": lineage,
        "holdout_dataset": {
            "test_normal": len(roles["test_normal"]),
            "recovery_holdout_anomaly": len(anomaly_ids),
            "development_intersection": 0,
        },
        "one_shot_execution": {
            "checkpoint": str(CHECKPOINT),
            "resumed_counts": resumed_counts,
            "completed_counts": final_counts,
            "holdout_access_count": sum(final_counts.values()),
            "one_final_result_per_original": True,
        },
        "metrics": metrics,
        "bootstrap_ci": bootstrap,
        "quartile_analysis": {
            "boundaries": {"q1": FROZEN_QUARTILE_BOUNDARIES[0], "q2": FROZEN_QUARTILE_BOUNDARIES[1], "q3": FROZEN_QUARTILE_BOUNDARIES[2]},
            "source": "frozen development representation_diagnostic_manifest.json",
            "recomputed_on_holdout": False,
        },
        "threshold_diagnostics_report_only": True,
        "tests": {"pre_holdout_command": ".venv/Scripts/python.exe -m pytest inference-service/tests/test_steel_d3_recovery_holdout.py -q"},
        "git": {"branch": branch, "freeze_commit": commit},
        "generated_at": utc_now(),
    }
    assert_artifacts_unchanged(ARTIFACT_PATHS, lineage)
    atomic_write_json(RESULTS_JSON, payload)
    RESULTS_MD.write_text(render_results(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "verdict": verdict,
                "image_auroc": metrics["image_auroc"],
                "normal_median": metrics["normal_median"],
                "anomaly_median": metrics["anomaly_median"],
                "HOLDOUT_ACCESS_COUNT": payload["one_shot_execution"]["holdout_access_count"],
            }
        ),
        flush=True,
    )
    print(verdict, flush=True)
    return verdict


def main() -> int:
    try:
        execute()
        return 0
    except Exception as exc:  # fail closed and preserve the terminal verdict
        reason = f"{type(exc).__name__}:{exc}"
        write_blocked(reason)
        print(reason, flush=True)
        print("RECOVERY_HOLDOUT_BLOCKED", flush=True)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
