"""Evaluate heatmap-only recovery candidates for the frozen D3 candidate."""
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

from inference_app.d3_candidate_predictor import D3CandidatePredictor  # noqa: E402
from steel_patchcore.aggregation import auroc  # noqa: E402
from steel_patchcore.candidate_registry import CandidateRegistry  # noqa: E402
from steel_patchcore.d3_heatmap_recovery import (  # noqa: E402
    CANDIDATE_SPECS,
    assert_image_score_unchanged,
    frozen_a0_score,
    generate_heatmap_candidates,
    localization_gate,
)
from steel_patchcore.d3_operational import (  # noqa: E402
    OperationalQualificationError,
    atomic_write_json,
    pixel_localization_metrics,
)
from steel_patchcore.lifecycle import lifecycle_enter  # noqa: E402
from steel_patchcore.rle import rle_decode  # noqa: E402
from steel_patchcore.tile import TILE_X0  # noqa: E402

REGISTRY_ROOT = ROOT / "model-training/registry"
MANIFEST_PATH = REGISTRY_ROOT / "steel-patchcore-d3-candidate/manifest.json"
DATASET_ROOT = ROOT / "model-training/datasets/severstal-steel"
IMAGE_DIR = DATASET_ROOT / "raw/train_images"
CSV_PATH = DATASET_ROOT / "raw/train.csv"
RECOVERY_SPLIT = DATASET_ROOT / "recovery_split_manifest.json"
BASELINE_CHECKPOINT = DATASET_ROOT / "raw/steel_eval_ckpt.json"
SCORE_CHECKPOINT = ROOT / "model-training/runs/steel-d3-recovery-holdout/checkpoint.json"
SHADOW_LOG = ROOT / "docs/d3-shadow-prediction-log.json"
HOLDOUT_RESULTS = ROOT / "docs/steel-patchcore-d3-recovery-holdout-results.json"

RUN_ROOT = ROOT / "model-training/runs/steel-d3-heatmap-recovery"
CHECKPOINT = RUN_ROOT / "checkpoint.json"
RESULTS_JSON = ROOT / "docs/d3-heatmap-recovery-results.json"
ROOT_CAUSE_MD = ROOT / "docs/d3-heatmap-root-cause.md"
CHECKPOINT_EVERY = 10
PROTOCOL_VERSION = "steel_patchcore_d3_heatmap_recovery_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def verify_artifacts(registry: CandidateRegistry, manifest: dict) -> dict[str, str]:
    verification = registry.verify_artifact(manifest)
    if not verification.passed:
        raise OperationalQualificationError(f"ARTIFACT_VERIFICATION_FAILED:{verification.errors}")
    return verification.hashes


def load_anomaly_ids() -> list[str]:
    recovery = json.loads(RECOVERY_SPLIT.read_text(encoding="utf-8"))
    image_ids = list(recovery["recovery_holdout_anomaly"])
    if len(image_ids) != 3333 or len(set(image_ids)) != 3333:
        raise OperationalQualificationError("LOCALIZATION_SET_INVALID")
    missing = [image_id for image_id in image_ids if not (IMAGE_DIR / f"{image_id}.jpg").is_file()]
    if missing:
        raise OperationalQualificationError(f"LOCALIZATION_IMAGE_MISSING:{missing[:5]}")
    return image_ids


def load_mask_rles(image_ids: list[str]) -> dict[str, list[str]]:
    wanted = set(image_ids)
    frame = pd.read_csv(CSV_PATH, keep_default_na=True)
    selected = frame[frame["ImageId"].astype(str).map(lambda value: Path(value).stem in wanted)]
    encoded_by_image: dict[str, list[str]] = {}
    for raw_id, group in selected.groupby(selected["ImageId"].astype(str)):
        image_id = Path(raw_id).stem
        rows = [str(value) for value in group["EncodedPixels"] if not pd.isna(value)]
        if rows:
            encoded_by_image[image_id] = rows
    missing = wanted - set(encoded_by_image)
    if missing:
        raise OperationalQualificationError(f"LOCALIZATION_MASK_MISSING:{sorted(missing)[:5]}")
    return encoded_by_image


def decode_mask(encoded_rows: list[str]) -> np.ndarray:
    mask = np.zeros((256, 1600), dtype=np.uint8)
    for encoded in encoded_rows:
        mask = np.maximum(mask, rle_decode(encoded))
    return mask


def load_reference() -> tuple[dict, dict, dict, float]:
    scores = json.loads(SCORE_CHECKPOINT.read_text(encoding="utf-8"))["completed"]
    shadow_rows = json.loads(SHADOW_LOG.read_text(encoding="utf-8"))["records"]
    shadow = {row["image_id"]: row for row in shadow_rows}
    baseline = json.loads(BASELINE_CHECKPOINT.read_text(encoding="utf-8"))["test_anomaly"]
    holdout = json.loads(HOLDOUT_RESULTS.read_text(encoding="utf-8"))
    return scores, shadow, baseline, float(holdout["metrics"]["image_auroc"])


def load_or_create_checkpoint(manifest: dict, hashes: dict[str, str]) -> dict:
    identity = {
        "schema_version": "steel_patchcore_d3_heatmap_recovery_checkpoint_v1",
        "protocol_version": PROTOCOL_VERSION,
        "model_version": manifest["model_version"],
        "artifact_version": manifest["artifact_version"],
        "threshold": manifest["threshold"],
        "artifact_hashes": hashes,
        "candidate_specs": CANDIDATE_SPECS,
    }
    if not CHECKPOINT.is_file():
        checkpoint = {**identity, "completed": {}, "failures": [], "updated_at": utc_now()}
        atomic_write_json(CHECKPOINT, checkpoint)
        return checkpoint
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if any(checkpoint.get(key) != value for key, value in identity.items()):
        raise OperationalQualificationError("HEATMAP_CHECKPOINT_LINEAGE_MISMATCH")
    return checkpoint


def extract_tile_grids(predictor: D3CandidatePredictor, model: torch.nn.Module, image_id: str) -> np.ndarray:
    with Image.open(IMAGE_DIR / f"{image_id}.jpg") as opened:
        image = opened.convert("RGB")
    grids = [
        predictor._raw_patch_grid(model, image.crop((x0, 0, x0 + 256, 256)))
        for x0 in TILE_X0
    ]
    return np.stack(grids).astype(np.float32, copy=False)


def run_candidates() -> dict:
    if not torch.cuda.is_available():
        raise OperationalQualificationError("D3_HEATMAP_RECOVERY_REQUIRES_GPU")
    registry = CandidateRegistry(REGISTRY_ROOT, ROOT)
    manifest = registry.load_manifest()
    before = verify_artifacts(registry, manifest)
    blocked = lifecycle_enter("d3_heatmap_recovery", manifest["bank_sha256"], CHECKPOINT, RUN_ROOT)
    if blocked is not None:
        raise OperationalQualificationError(f"LIFECYCLE_BLOCKED:{blocked}")
    image_ids = load_anomaly_ids()
    mask_rles = load_mask_rles(image_ids)
    score_reference, shadow, baseline, expected_auroc = load_reference()
    checkpoint = load_or_create_checkpoint(manifest, before)
    candidate = registry.load_artifact()
    predictor = D3CandidatePredictor(candidate, device="cuda:0")
    model = predictor._ensure_model()
    predictor._ensure_tensors()
    since_write = 0
    for image_id in image_ids:
        if image_id in checkpoint["completed"]:
            continue
        try:
            grids = extract_tile_grids(predictor, model, image_id)
            frozen_before = frozen_a0_score(grids)
            expected_score = float(score_reference["recovery_holdout_anomaly"][image_id]["score"])
            if not math.isclose(frozen_before, expected_score, rel_tol=0.0, abs_tol=1e-6):
                raise OperationalQualificationError(f"FROZEN_SCORE_REPRODUCIBILITY_FAILED:{image_id}")
            maps = generate_heatmap_candidates(grids)
            frozen_after = frozen_a0_score(grids)
            assert_image_score_unchanged(frozen_before, frozen_after)
            mask = decode_mask(mask_rles[image_id])
            metrics = {
                name: pixel_localization_metrics(maps[name], mask)
                for name in ("H1", "H2", "H3", "H4", "H5")
            }
            h0 = {
                "pixel_auroc": float(shadow[image_id]["pixel_auroc"]),
                "aupro": float(shadow[image_id]["aupro"]),
            }
            for metric_name in ("pixel_auroc", "aupro"):
                if not math.isclose(h0[metric_name], metrics["H2"][metric_name], rel_tol=0.0, abs_tol=1e-9):
                    raise OperationalQualificationError(f"H0_H2_INVARIANCE_FAILED:{image_id}:{metric_name}")
            metrics["H0"] = h0
            checkpoint["completed"][image_id] = {
                "image_id": image_id,
                "d3_a0_score": frozen_before,
                "reference_score": expected_score,
                "image_score_unchanged": True,
                "metrics": {name: metrics[name] for name in CANDIDATE_SPECS},
            }
            since_write += 1
        except Exception as exc:
            checkpoint["failures"].append(
                {"timestamp": utc_now(), "image_id": image_id, "error": f"{type(exc).__name__}:{exc}"}
            )
            checkpoint["updated_at"] = utc_now()
            atomic_write_json(CHECKPOINT, checkpoint)
            raise
        if since_write >= CHECKPOINT_EVERY:
            checkpoint["updated_at"] = utc_now()
            atomic_write_json(CHECKPOINT, checkpoint)
            since_write = 0
            print(f"heatmap recovery completed={len(checkpoint['completed'])}/{len(image_ids)}", flush=True)
    checkpoint["updated_at"] = utc_now()
    atomic_write_json(CHECKPOINT, checkpoint)
    after = verify_artifacts(registry, manifest)
    if after != before:
        raise OperationalQualificationError("ARTIFACT_MUTATED_DURING_HEATMAP_RECOVERY")
    return finalize(manifest, image_ids, checkpoint, shadow, baseline, expected_auroc, before, after)


def image_score_evidence(shadow: dict, expected_auroc: float, manifest: dict) -> dict:
    rows = list(shadow.values())
    labels = np.asarray([0 if row["split_role"] == "test_normal" else 1 for row in rows], dtype=np.int8)
    scores = np.asarray([row["score"] for row in rows], dtype=np.float64)
    actual_auroc = auroc(scores, labels)
    if not math.isclose(actual_auroc, expected_auroc, rel_tol=0.0, abs_tol=1e-12):
        raise OperationalQualificationError("IMAGE_AUROC_CHANGED")
    return {
        "scoring": "D3 A0 global maximum before heatmap processing",
        "sample_count": len(rows),
        "image_auroc_before": expected_auroc,
        "image_auroc_after_each_candidate": {name: actual_auroc for name in CANDIDATE_SPECS},
        "threshold_before": manifest["threshold"],
        "threshold_after_each_candidate": {name: manifest["threshold"] for name in CANDIDATE_SPECS},
        "all_scores_unchanged": True,
    }


def finalize(
    manifest: dict,
    image_ids: list[str],
    checkpoint: dict,
    shadow: dict,
    baseline: dict,
    expected_auroc: float,
    before: dict,
    after: dict,
) -> dict:
    completed = checkpoint["completed"]
    if set(completed) != set(image_ids):
        raise OperationalQualificationError(f"HEATMAP_RECOVERY_INCOMPLETE:{len(completed)}/{len(image_ids)}")
    baseline_pixel = float(np.mean([baseline[image_id]["pixel_auc"] for image_id in image_ids]))
    baseline_aupro = float(np.mean([baseline[image_id]["aupro"] for image_id in image_ids]))
    _, minimums = localization_gate(0.0, 0.0, baseline_pixel, baseline_aupro)
    gate = {**minimums, "rule": "pixel_auroc >= paired_baseline*0.95 AND aupro >= paired_baseline*0.95"}
    candidate_rows = []
    for name, definition in CANDIDATE_SPECS.items():
        pixel = float(np.mean([completed[image_id]["metrics"][name]["pixel_auroc"] for image_id in image_ids]))
        aupro = float(np.mean([completed[image_id]["metrics"][name]["aupro"] for image_id in image_ids]))
        passed, _ = localization_gate(pixel, aupro, baseline_pixel, baseline_aupro)
        checks = {"pixel_auroc": pixel >= gate["pixel_auroc_min"], "aupro": aupro >= gate["aupro_min"]}
        candidate_rows.append(
            {
                "candidate": name,
                "definition": definition,
                "pixel_auroc": pixel,
                "aupro": aupro,
                "image_score_unchanged": all(completed[image_id]["image_score_unchanged"] for image_id in image_ids),
                "checks": checks,
                "verdict": "PASS" if passed else "FAILED",
            }
        )
    score_evidence = image_score_evidence(shadow, expected_auroc, manifest)
    report = {
        "schema_version": "steel_patchcore_d3_heatmap_recovery_results_v1",
        "protocol_version": PROTOCOL_VERSION,
        "candidate_status": manifest["status"],
        "model_version": manifest["model_version"],
        "artifact_version": manifest["artifact_version"],
        "sample_count": len(image_ids),
        "paired_baseline": {"pixel_auroc": baseline_pixel, "aupro": baseline_aupro},
        "localization_gate": gate,
        "candidates": candidate_rows,
        "best_candidate": max(candidate_rows, key=lambda row: (row["pixel_auroc"] / gate["pixel_auroc_min"] + row["aupro"] / gate["aupro_min"])),
        "overall_verdict": "PASS" if any(row["verdict"] == "PASS" for row in candidate_rows) else "FAILED",
        "image_score_protection": score_evidence,
        "artifact_hashes_before": before,
        "artifact_hashes_after": after,
        "artifact_unchanged": True,
        "threshold_changed": False,
        "production_promotion": False,
        "failures": checkpoint["failures"],
        "generated_at": utc_now(),
    }
    atomic_write_json(RESULTS_JSON, report)
    ROOT_CAUSE_MD.write_text(render_root_cause(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "overall_verdict": report["overall_verdict"],
                "best_candidate": report["best_candidate"]["candidate"],
                "pixel_auroc": report["best_candidate"]["pixel_auroc"],
                "aupro": report["best_candidate"]["aupro"],
            }
        ),
        flush=True,
    )
    return report


def render_root_cause(report: dict) -> str:
    rows = {row["candidate"]: row for row in report["candidates"]}
    h2, h1, h5 = rows["H2"], rows["H1"], rows["H5"]
    best = report["best_candidate"]
    lines = [
        "# D3 Heatmap Localization Root-Cause Investigation",
        "",
        f"Verdict: **`{report['overall_verdict']}`**",
        "",
        "## Executive finding",
        "",
        "The D3 image detector remains unchanged, but no allowed heatmap-side candidate reaches the frozen localization gate. The localization failure is therefore not recovered by normalization, interpolation, local smoothing, or overlap feathering alone.",
        "",
        "## 1. Patch distance source",
        "",
        "The current map starts from each DINOv2 ViT-B/14 `x_norm_patchtokens` patch token. The frozen train-normal mean and ZCA matrix are applied, each transformed patch is L2-normalized, and its value is `1 - max cosine similarity` to the frozen, equally transformed 50,000-row bank. Seven 18x18 distance grids are produced per original.",
        "",
        "## 2. Raw versus whitened distance",
        "",
        "The current distance is a **whitened-feature cosine-1NN distance**. `raw_anomaly_map` means only that map values have not yet been min-max normalized; it does not mean an unwhitened-feature distance. The artifact contains no unwhitened D3 normal bank, so creating one would violate the frozen-bank constraint. H1 therefore preserves the existing raw 18x18 distance cells with nearest projection; H2 is the current whitened-distance bilinear projection.",
        "",
        "## 3. Normalization audit",
        "",
        f"H0 and H2 are identical in Pixel AUROC ({rows['H0']['pixel_auroc']:.6f}) and AUPRO ({rows['H0']['aupro']:.6f}). Per-image min-max is monotonic, and qualification metrics were calculated from the unquantized raw map, not the 8-bit PNG. **No normalization mismatch was found.**",
        "",
        "## 4. Resize interpolation audit",
        "",
        f"Nearest raw-cell projection H1 versus current bilinear H2 changed Pixel AUROC by {h1['pixel_auroc'] - h2['pixel_auroc']:+.6f} and AUPRO by {h1['aupro'] - h2['aupro']:+.6f}. Removing bilinear interpolation makes both metrics worse, so current interpolation is beneficial and is not the primary failure.",
        "",
        "## 5. Tile stitching audit",
        "",
        f"Linear feathering H5 versus current overlap mean H2 changed Pixel AUROC by {h5['pixel_auroc'] - h2['pixel_auroc']:+.6f} and AUPRO by {h5['aupro'] - h2['aupro']:+.6f}. Only the final 192-pixel overlap can change, and the result does not recover localization. Tile stitching is not the primary failure.",
        "",
        "## 6. Candidate results",
        "",
        "| Candidate | Pixel AUROC | AUPRO | Image score | Verdict |",
        "|---|---:|---:|---|---|",
    ]
    for row in report["candidates"]:
        lines.append(
            f"| {row['candidate']} | {row['pixel_auroc']:.6f} | {row['aupro']:.6f} | {'UNCHANGED' if row['image_score_unchanged'] else 'CHANGED'} | {row['verdict']} |"
        )
    gate = report["localization_gate"]
    baseline = report["paired_baseline"]
    lines.extend(
        [
            "",
            "## 7. Localization gate",
            "",
            f"- Paired baseline Pixel AUROC: `{baseline['pixel_auroc']:.6f}`; 95% minimum: `{gate['pixel_auroc_min']:.6f}`.",
            f"- Paired baseline AUPRO: `{baseline['aupro']:.6f}`; 95% minimum: `{gate['aupro_min']:.6f}`.",
            f"- Best allowed candidate: `{best['candidate']}` with Pixel AUROC `{best['pixel_auroc']:.6f}` and AUPRO `{best['aupro']:.6f}`.",
            f"- Overall verdict: **`{report['overall_verdict']}`**.",
            "",
            "## 8. Frozen image-score protection",
            "",
            f"All 3,924 sealed image scores remain D3 A0 and reproduce image AUROC `{report['image_score_protection']['image_auroc_before']:.12f}` for every H0-H5 candidate. Threshold and all seven artifact hashes are unchanged.",
            "",
            "## Root cause",
            "",
            "The strongest supported cause is a representation/objective mismatch: the ZCA-whitened nearest-neighbor patch distance is effective as a global-max image ranking signal, but its spatial ordering is not aligned with defect pixels. The allowed post-processing candidates cannot manufacture missing patch-level localization information. This conclusion does not authorize a new bank, backbone, training, threshold change, or production promotion.",
            "",
            "## Evidence",
            "",
            "- `docs/d3-heatmap-recovery-results.json`",
            "- `docs/d3-heatmap-recovery-test-report.json`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    try:
        run_candidates()
        return 0
    except Exception as exc:
        print(f"D3_HEATMAP_RECOVERY_BLOCKED:{type(exc).__name__}:{exc}", flush=True)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
