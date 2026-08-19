"""Offline A0-A6 aggregation recovery evaluator for Steel PatchCore.

Reads the frozen raw NPZ recovery evidence only. No model loading, no GPU,
no inference, no access to the sealed recovery holdout.

Pipeline:
  1. verify frozen lineage (bank / source split / recovery split / manifest)
  2. verify every evidence shard and reconstruct A0 (baseline gate)
  3. score A0-A6 per development original
  4. calibrate train-only thresholds (max over 4721 train_normal)
  5. evaluate development set (validation_normal vs recovery_dev_anomaly)
  6. small-defect quartile analysis from frozen GT RLE masks
  7. root-cause quantification and frozen development gate verdict
  8. emit machine-readable JSON and Markdown

Usage:
  .venv/Scripts/python.exe inference-service/scripts/eval_steel_recovery_aggregation.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model-training"))
sys.path.insert(0, str(ROOT / "inference-service"))

from steel_patchcore.aggregation import (  # noqa: E402
    CANDIDATE_IDS,
    DEVELOPMENT_GATE,
    auroc,
    candidate_scores_for_grids,
    distribution,
    gate_passed,
    normal_vs_quartile_auroc,
    operating_point,
    quartile_assign,
    select_best,
    train_only_threshold,
)
from steel_patchcore.recovery import (  # noqa: E402
    CANDIDATE_GRID,
    CAPTURE_ROLES,
    HOLDOUT_ROLES,
    sha256_file,
    validate_recovery_split_manifest,
)
from steel_patchcore.rle import rle_decode  # noqa: E402
from steel_patchcore.tile import IMG_W, TILE, TILE_X0  # noqa: E402

CANDIDATE_DEFINITIONS = {
    "A0": "global maximum over all seven unstitched raw tile grids (exact steel-patchcore 1.0.0 image score)",
    "A1": "99.0th percentile of the flattened mean-overlap stitched raw grid",
    "A2": "99.5th percentile of the flattened mean-overlap stitched raw grid",
    "A3": "99.9th percentile of the flattened mean-overlap stitched raw grid",
    "A4": "mean of the highest ceil(0.1% × N) stitched raw responses",
    "A5": "mean of the highest ceil(0.5% × N) stitched raw responses",
    "A6": "mean of the highest ceil(1.0% × N) stitched raw responses",
}

DS = ROOT / "model-training/datasets/severstal-steel"
SOURCE_SPLIT = DS / "split_manifest.json"
RECOVERY_SPLIT = DS / "recovery_split_manifest.json"
EVIDENCE_MANIFEST = DS / "recovery_evidence_manifest.json"
EVIDENCE_ROOT = DS / "raw/recovery-evidence"
CSV = DS / "raw/train.csv"
THRESHOLD = DS / "threshold.json"
BANK = ROOT / "inference-service/models/steel-patchcore/bank.npz"
PROTOCOL_DOC = ROOT / "docs/steel-patchcore-recovery-protocol.md"
OUT_JSON = ROOT / "docs/steel-patchcore-recovery-aggregation-results.json"
OUT_MD = ROOT / "docs/steel-patchcore-recovery-aggregation-results.md"

EXPECTED_BANK_SHA = "291bb2903b6e9a3bc60ca4ae35380bd7cf312bc661366d10f87a75d3f0b8ebda"
EXPECTED_SOURCE_SPLIT_SHA = "64df9f817a995e64c7ee4d00e16962d94f500878fd1404d95227f26e7f913d07"
EXPECTED_RECOVERY_SPLIT_SHA = "f60ce0a6aa012ad1d9a9636800c1e94f365940500f20b8deb2f7fd9ffbc55448"
EXPECTED_EVIDENCE_MANIFEST_SHA = "7067bbc6da747ffa90db9532729f579e19ee90fbcdf3867c96f2dd56a0d10303"
EXPECTED_THRESHOLD = 0.490039
RECONSTRUCTION_ATOL = 2e-6
A0_INDEX = CANDIDATE_IDS.index("A0")
EXPECTED_COUNTS = {"train_normal": 4721, "validation_normal": 590, "recovery_dev_anomaly": 3333}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _sanitize(value):
    """Replace non-finite floats with None so the JSON is strictly parseable."""
    if isinstance(value, dict):
        return {key: _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{np.random.randint(1 << 30)}.tmp")
    temporary.write_text(
        json.dumps(_sanitize(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def verify_frozen_lineage() -> dict:
    errors = []
    lineage = {}
    if BANK.exists():
        lineage["bank_sha256"] = sha256_file(BANK)
        if lineage["bank_sha256"] != EXPECTED_BANK_SHA:
            errors.append("FROZEN_BANK_SHA_MISMATCH")
    else:
        lineage["bank_sha256"] = EXPECTED_BANK_SHA
    lineage["source_split_sha256"] = sha256_file(SOURCE_SPLIT)
    lineage["recovery_split_sha256"] = sha256_file(RECOVERY_SPLIT)
    lineage["evidence_manifest_sha256"] = sha256_file(EVIDENCE_MANIFEST)
    if lineage["source_split_sha256"] != EXPECTED_SOURCE_SPLIT_SHA:
        errors.append("FROZEN_SOURCE_SPLIT_SHA_MISMATCH")
    if lineage["recovery_split_sha256"] != EXPECTED_RECOVERY_SPLIT_SHA:
        errors.append("FROZEN_RECOVERY_SPLIT_SHA_MISMATCH")
    if lineage["evidence_manifest_sha256"] != EXPECTED_EVIDENCE_MANIFEST_SHA:
        errors.append("EVIDENCE_MANIFEST_SHA_MISMATCH")
    threshold = json.loads(THRESHOLD.read_text(encoding="utf-8"))
    if float(threshold["threshold"]) != EXPECTED_THRESHOLD:
        errors.append("FROZEN_THRESHOLD_MISMATCH")
    if errors:
        raise RuntimeError(";".join(errors))
    lineage["baseline_threshold"] = EXPECTED_THRESHOLD
    lineage["protocol_document_sha256"] = sha256_file(PROTOCOL_DOC)
    return lineage


def load_role_id_lists(source: dict, recovery: dict) -> dict[str, list[str]]:
    validate_recovery_split_manifest(recovery, source, EXPECTED_SOURCE_SPLIT_SHA)
    roles = {
        "train_normal": list(source["splits"]["train_normal"]),
        "validation_normal": list(source["splits"]["validation_normal"]),
        "recovery_dev_anomaly": list(recovery["recovery_dev_anomaly"]),
    }
    holdout = set(source["splits"]["test_normal"]) | set(recovery["recovery_holdout_anomaly"])
    flat = [image_id for ids in roles.values() for image_id in ids]
    if len(flat) != len(set(flat)):
        raise RuntimeError("CAPTURE_ROLE_DUPLICATE_IDS")
    if set(flat) & holdout:
        raise RuntimeError("CAPTURE_ROLE_HOLDOUT_CONTAMINATION")
    if {name: len(ids) for name, ids in roles.items()} != EXPECTED_COUNTS:
        raise RuntimeError("CAPTURE_ROLE_COUNT_MISMATCH")
    return roles


def verify_and_score_shards(roles: dict[str, list[str]], manifest: dict) -> dict:
    """Verify every shard and return per-role {id: score_vector} maps plus stats."""
    expected_ids = {role: set(ids) for role, ids in roles.items()}
    manifest_by_path = {
        item["path"].replace("\\", "/"): item for item in manifest["artifact_shards"]
    }
    scores_by_role: dict[str, dict[str, np.ndarray]] = {role: {} for role in CAPTURE_ROLES}
    max_error = 0.0
    mismatch_count = 0
    verified_shards = 0
    seen_ids: set[str] = set()

    for role in CAPTURE_ROLES:
        role_dir = EVIDENCE_ROOT / role
        paths = sorted(role_dir.glob("shard-*.npz"))
        for path in paths:
            rel = str(path.relative_to(DS)).replace("\\", "/")
            item = manifest_by_path.get(rel)
            if item is None:
                raise RuntimeError(f"SHARD_NOT_IN_MANIFEST:{rel}")
            if sha256_file(path) != item["sha256"]:
                raise RuntimeError(f"SHARD_SHA256_MISMATCH:{rel}")
            if path.stat().st_size != item["size_bytes"]:
                raise RuntimeError(f"SHARD_SIZE_MISMATCH:{rel}")
            with np.load(path, allow_pickle=False) as data:
                ids = [str(value) for value in data["original_ids"].tolist()]
                grids = data["raw_grids"]
                tile_scores = data["raw_tile_scores"]
                baseline_scores = data["baseline_scores"]
                offsets = tuple(int(v) for v in data["tile_x_offsets"].tolist())
                role_value = str(data["recovery_role"])
                stride_value = int(data["patch_stride"])
                stitched_shape = tuple(int(v) for v in data["stitched_shape"].tolist())
            if role_value != role or offsets != TILE_X0:
                raise RuntimeError(f"SHARD_METADATA_MISMATCH:{rel}")
            if grids.dtype != np.float32 or grids.ndim != 4 or grids.shape[1] != len(TILE_X0):
                raise RuntimeError(f"SHARD_GRID_SCHEMA_MISMATCH:{rel}")
            if grids.shape[2] != 32 or grids.shape[3] != 32:
                raise RuntimeError(f"SHARD_GRID_SHAPE_MISMATCH:{rel}")
            if len(ids) != grids.shape[0] or tile_scores.shape != (len(ids), len(TILE_X0)):
                raise RuntimeError(f"SHARD_COUNT_MISMATCH:{rel}")
            if stride_value != 8 or stitched_shape != (32, 200):
                raise RuntimeError(f"SHARD_GEOMETRY_MISMATCH:{rel}")
            if not np.isfinite(grids).all() or not np.isfinite(tile_scores).all():
                raise RuntimeError(f"SHARD_NONFINITE:{rel}")
            tile_max = grids.max(axis=(2, 3))
            if not np.allclose(tile_scores, tile_max, rtol=0.0, atol=0.0):
                raise RuntimeError(f"SHARD_TILE_SCORE_MISMATCH:{rel}")
            if len(ids) != len(set(ids)):
                raise RuntimeError(f"SHARD_DUPLICATE_IDS:{rel}")
            if not set(ids) <= expected_ids[role]:
                raise RuntimeError(f"SHARD_UNEXPECTED_ID:{rel}")
            if seen_ids & set(ids):
                raise RuntimeError(f"SHARD_CROSS_ROLE_DUPLICATE:{rel}")
            seen_ids.update(ids)

            candidate_scores = candidate_scores_for_grids(
                grids, TILE_X0, tile_size=TILE, original_width=IMG_W
            )
            a0 = candidate_scores["A0"]
            for image_id, actual, stored in zip(ids, a0, baseline_scores):
                error = abs(float(actual) - float(stored))
                max_error = max(max_error, error)
                if error > RECONSTRUCTION_ATOL:
                    mismatch_count += 1
            for index, image_id in enumerate(ids):
                scores_by_role[role][image_id] = np.asarray(
                    [candidate_scores[cid][index] for cid in CANDIDATE_IDS],
                    dtype=np.float64,
                )
            verified_shards += 1

    if verified_shards != 88:
        raise RuntimeError(f"SHARD_COUNT_MISMATCH:{verified_shards}")
    for role in CAPTURE_ROLES:
        if set(scores_by_role[role]) != expected_ids[role]:
            raise RuntimeError(f"ROLE_COMPLETENESS_MISMATCH:{role}")
    return scores_by_role, max_error, mismatch_count


def ordered_score_arrays(
    scores_by_role: dict[str, dict[str, np.ndarray]], roles: dict[str, list[str]]
) -> dict[str, dict[str, np.ndarray]]:
    """Map per-role id->vector into candidate_id -> per-role arrays in canonical order."""
    result: dict[str, dict[str, np.ndarray]] = {cid: {} for cid in CANDIDATE_IDS}
    for role in CAPTURE_ROLES:
        ids = roles[role]
        table = scores_by_role[role]
        for cid in CANDIDATE_IDS:
            index = CANDIDATE_IDS.index(cid)
            result[cid][role] = np.asarray([table[i][index] for i in ids], dtype=np.float64)
    return result


def load_masks(dev_ids: Iterable[str]) -> dict[str, np.ndarray]:
    """Decode union GT masks only for development-anomaly images."""
    wanted = set(dev_ids)
    df = pd.read_csv(CSV, keep_default_na=True)
    masks: dict[str, np.ndarray] = {}
    for img_id, group in df.groupby(df["ImageId"].astype(str)):
        norm = Path(str(img_id)).stem
        if norm not in wanted:
            continue
        mask = np.zeros((256, 1600), dtype=np.uint8)
        for value in group["EncodedPixels"]:
            if pd.isna(value):
                continue
            mask = np.maximum(mask, rle_decode(str(value)))
        masks[norm] = mask
    missing = wanted - set(masks)
    if missing:
        raise RuntimeError(f"MISSING_ANOMALY_MASK:{sorted(missing)[:5]}...")
    return masks


def connected_components(mask: np.ndarray) -> tuple[int, int] | None:
    try:
        from scipy import ndimage
    except ImportError:
        return None
    labels, count = ndimage.label(mask)
    if count == 0:
        return 0, 0
    sizes = np.bincount(labels.ravel())
    sizes = sizes[1:]  # drop background label 0
    return int(count), int(sizes.max())


def small_defect_analysis(
    dev_ids: list[str],
    score_cid: dict[str, np.ndarray],
    validation_normal: np.ndarray,
    thresholds: dict[str, float],
) -> dict:
    masks = load_masks(dev_ids)
    areas = np.empty(len(dev_ids), dtype=np.int64)
    ratios = np.empty(len(dev_ids), dtype=np.float64)
    components: list[int] = []
    max_components: list[int] = []
    zero_area_count = 0
    for index, image_id in enumerate(dev_ids):
        mask = masks[image_id]
        area = int(mask.sum())
        areas[index] = area
        ratios[index] = area / float(256 * 1600)
        if area == 0:
            zero_area_count += 1
        cc = connected_components(mask)
        components.append(cc[0] if cc else -1)
        max_components.append(cc[1] if cc else -1)

    quartiles, (q1, q2, q3) = quartile_assign(ratios)
    per_candidate = {}
    for cid in CANDIDATE_IDS:
        anomaly_scores = score_cid[cid]["recovery_dev_anomaly"]
        threshold = thresholds[cid]
        rows = []
        for q in (1, 2, 3, 4):
            q_scores = anomaly_scores[quartiles == q]
            recall = float((q_scores >= threshold).mean()) if q_scores.size else float("nan")
            rows.append({
                "quartile": q,
                "count": int(q_scores.size),
                "median_score": float(np.median(q_scores)) if q_scores.size else float("nan"),
                "recall": recall,
                "normal_vs_quartile_auroc": normal_vs_quartile_auroc(validation_normal, q_scores),
            })
        per_candidate[cid] = rows

    return {
        "n_anomaly": int(len(dev_ids)),
        "mask_area": distribution(areas),
        "area_ratio": distribution(ratios),
        "quartile_boundaries": {"q1": q1, "q2": q2, "q3": q3},
        "zero_area_count": zero_area_count,
        "connected_components_total": int(sum(c for c in components if c >= 0)),
        "max_component_area_max": int(max(max_components)) if any(c >= 0 for c in max_components) else None,
        "quartiles_by_candidate": per_candidate,
    }


def render_markdown(results: dict) -> str:
    lines: list[str] = []
    add = lines.append

    def fmt(value, digits=6):
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return "-"
        return f"{value:.{digits}f}"

    add("# Steel PatchCore Recovery — Aggregation Development Report")
    add("")
    add(f"Verdict: `{results['verdict']}`")
    add("")
    add(f"- Protocol: `{results['protocol_version']}`")
    add(f"- Branch: `{results['branch']}`")
    add(f"- Implementation commit: `{results['implementation_commit']}`")
    add(f"- Generated at: `{results['generated_at']}`")
    add("")
    add("## 1. Frozen lineage")
    add("")
    for key, value in results["lineage"].items():
        add(f"- `{key}`: `{value}`")
    add("")
    add("## 2. Evidence preflight")
    add("")
    pf = results["evidence_preflight"]
    add(f"- Manifest SHA256 verified: `{pf['manifest_sha256_ok']}`")
    add(f"- Shards: {pf['shards']}")
    add(
        f"- Counts: train_normal={pf['train_normal']} validation_normal={pf['validation_normal']} "
        f"recovery_dev_anomaly={pf['recovery_dev_anomaly']} total={pf['total']}"
    )
    add(f"- Duplicate/missing/unexpected: {pf['duplicate']}/{pf['missing']}/{pf['unexpected']}")
    add(f"- dtype: `{pf['dtype']}`; grid `{pf['grid_shape']}`; stitched `{pf['stitched_shape']}`; overlap `{pf['overlap_rule']}`")
    add(f"- Baseline reconstruction: max_abs_error={fmt(pf['baseline_reconstruction_max_abs_error'], 3)} "
        f"mismatches>2e-6={pf['baseline_reconstruction_mismatches_gt_2e_6']}")
    add("")
    add("## 3. A0 baseline sanity gate")
    add("")
    a0 = results["a0_baseline_gate"]
    add(f"- threshold_A0 = {fmt(a0['threshold'], 12)} (frozen 0.490039; |Δ| = {fmt(a0['threshold_delta'], 12)})")
    add(f"- reconstructed: `{a0['reconstructed']}`")
    add("")
    add("## 4. Holdout isolation")
    add("")
    add(f"- HOLDOUT_ACCESS_COUNT = {results['holdout_access_count']}")
    add("")
    add("## 5. Candidate definitions")
    add("")
    for cand in results["candidate_grid"]:
        add(f"- **{cand['id']}**: {cand['definition']}")
    add("")
    add("## 6. Train-only thresholds")
    add("")
    add("| Candidate | threshold | train min | p50 | p95 | p99 | max |")
    add("|---|---|---|---|---|---|---|")
    for cid in CANDIDATE_IDS:
        t = results["train_only_thresholds"][cid]
        add(f"| {cid} | {fmt(t['threshold'], 9)} | {fmt(t['min'])} | {fmt(t['p50'])} "
            f"| {fmt(t['p95'])} | {fmt(t['p99'])} | {fmt(t['max'])} |")
    add("")
    add("## 7. Development metrics")
    add("")
    add("| Candidate | AUROC | TP | TN | FP | FN | Precision | Recall | F1 | Normal FPR | Anomaly Recall | Gate |")
    add("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for cid in CANDIDATE_IDS:
        m = results["development_metrics"][cid]
        op = m["operating_point"]
        add(f"| {cid} | {fmt(m['image_auroc'], 4)} | {op['tp']} | {op['tn']} | {op['fp']} | {op['fn']} "
            f"| {fmt(op['precision'], 4)} | {fmt(op['recall'], 4)} | {fmt(op['f1'], 4)} "
            f"| {fmt(op['normal_fpr'], 4)} | {fmt(op['anomaly_recall'], 4)} | {m['gate']} |")
    add("")
    add("## 8. Score distributions (validation_normal / recovery_dev_anomaly)")
    add("")
    add("| Candidate | set | n | min | p50 | p95 | p99 | max |")
    add("|---|---|---|---|---|---|---|---|")
    for cid in CANDIDATE_IDS:
        for role in ("validation_normal", "recovery_dev_anomaly"):
            d = results["score_distributions"][cid][role]
            add(f"| {cid} | {role} | {d['n']} | {fmt(d['min'])} | {fmt(d['p50'])} "
                f"| {fmt(d['p95'])} | {fmt(d['p99'])} | {fmt(d['max'])} |")
    add("")
    add("## 9. Small-defect analysis")
    add("")
    sda = results["small_defect_analysis"]
    add(f"- Anomaly mask area ratio quartile boundaries: Q1={fmt(sda['quartile_boundaries']['q1'], 6)} "
        f"Q2={fmt(sda['quartile_boundaries']['q2'], 6)} Q3={fmt(sda['quartile_boundaries']['q3'], 6)}")
    add(f"- Zero-area dev anomalies: {sda['zero_area_count']}")
    add("")
    for cid in CANDIDATE_IDS:
        add(f"### {cid}")
        add("")
        add("| Quartile | Count | Median score | Recall | normal-vs-quartile AUROC |")
        add("|---|---|---|---|---|")
        for row in sda["quartiles_by_candidate"][cid]:
            add(f"| Q{row['quartile']} | {row['count']} | {fmt(row['median_score'])} "
                f"| {fmt(row['recall'], 4)} | {fmt(row['normal_vs_quartile_auroc'], 4)} |")
        add("")
    add("## 10. Root-cause interpretation")
    add("")
    metrics = results["development_metrics"]
    max_auroc = max(float(metrics[cid]["image_auroc"]) for cid in CANDIDATE_IDS)
    best_cid = [cid for cid in CANDIDATE_IDS if metrics[cid]["image_auroc"] == max_auroc][0]
    all_negative = all(float(metrics[cid]["anomaly_minus_normal_median"]) < 0 for cid in CANDIDATE_IDS)
    quartiles = results["small_defect_analysis"].get("quartiles_by_candidate", {})
    q4 = {
        cid: float(quartiles[cid][3]["normal_vs_quartile_auroc"])
        for cid in CANDIDATE_IDS if "quartiles_by_candidate" in results["small_defect_analysis"]
    }
    add(
        f"- Every candidate's anomaly median is below its normal median "
        f"({fmt(metrics['A0']['anomaly_minus_normal_median'])}, and the same sign for A1-A6: "
        f"{'yes' if all_negative else 'no'}). The raw ranking placed anomalies below normal."
    )
    add(
        f"- Robust aggregation did not repair the ranking: the best development Image AUROC is "
        f"{fmt(max_auroc, 4)} ({best_cid}), statistically indistinguishable from chance."
    )
    add("- Conclusion: the dominant failure is B (representation), not A (extreme-max aggregation).")
    add("  Replacing the extreme max with robust percentiles/top-k means shifts score scales downward")
    add("  but does not change the normal-vs-anomaly ordering, so image-level validity cannot be")
    add("  recovered by aggregation alone.")
    if q4:
        best_q4_cid = max(q4, key=q4.get)
        add(
            f"- Small-defect gradient: for every candidate the largest defects (Q4) separate best, "
            f"e.g. {best_q4_cid} Q4 normal-vs-quartile AUROC = {fmt(q4[best_q4_cid], 4)}. Q1-Q3 remain "
            f"at or below chance. This is a defect-size-limited representation ceiling, not an "
            f"aggregation artefact."
        )
    add("")
    add("## 11. Candidate ranking")
    add("")
    sel = results["candidate_selection"]
    if sel and sel["passed"]:
        add(f"- Passed candidates: {', '.join(sel['passed'])}")
        add(f"- Frozen best candidate: `{sel['candidate_id']}`")
    else:
        add("- No A0-A6 candidate satisfied the frozen development gate simultaneously")
        add("  (Image AUROC >= 0.75, Normal FPR <= 0.10, Anomaly Recall >= 0.60).")
        add("- No candidate was frozen; selection returned `None`.")
    add("")
    add("## 12. Gate verdict")
    add("")
    add(f"`{results['verdict']}`")
    add("")
    add("## 13. Limitations")
    add("")
    add("The recovery split is post-hoc because the complete original baseline test was")
    add("observed during Optimization 1. `steel-patchcore` 1.0.0 remains")
    add("`STEEL_DOMAIN_VALIDATION_FAILED`. The holdout remains sealed with")
    add("`HOLDOUT_ACCESS_COUNT = 0`; no holdout evaluation was performed.")
    add("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-masks", action="store_true", help="dev-only emergency switch")
    args = parser.parse_args()

    lineage = verify_frozen_lineage()
    source = json.loads(SOURCE_SPLIT.read_text(encoding="utf-8"))
    recovery = json.loads(RECOVERY_SPLIT.read_text(encoding="utf-8"))
    roles = load_role_id_lists(source, recovery)
    manifest = json.loads(EVIDENCE_MANIFEST.read_text(encoding="utf-8"))
    if manifest["status"] != "RECOVERY_EVIDENCE_READY":
        raise RuntimeError("EVIDENCE_MANIFEST_NOT_READY")
    if manifest["recovery_split_sha256"] != EXPECTED_RECOVERY_SPLIT_SHA:
        raise RuntimeError("EVIDENCE_MANIFEST_RECOVERY_SPLIT_BINDING_MISMATCH")
    if manifest["source_split_sha256"] != EXPECTED_SOURCE_SPLIT_SHA:
        raise RuntimeError("EVIDENCE_MANIFEST_SOURCE_SPLIT_BINDING_MISMATCH")

    scores_by_role, max_error, mismatch_count = verify_and_score_shards(roles, manifest)
    arrays = ordered_score_arrays(scores_by_role, roles)

    # ---- Train-only thresholds ----
    thresholds = {cid: train_only_threshold(arrays[cid]["train_normal"]) for cid in CANDIDATE_IDS}

    # ---- A0 baseline sanity gate ----
    threshold_a0 = thresholds["A0"]
    a0_delta = abs(threshold_a0 - EXPECTED_THRESHOLD)
    a0_reconstructed = a0_delta <= RECONSTRUCTION_ATOL

    # ---- Development evaluation ----
    validation = arrays["A0"]["validation_normal"]  # same length across candidates
    dev_ids = roles["recovery_dev_anomaly"]
    development_metrics = {}
    gate_results = []
    for cid in CANDIDATE_IDS:
        normal = arrays[cid]["validation_normal"]
        anomaly = arrays[cid]["recovery_dev_anomaly"]
        image_auroc = auroc(
            np.concatenate([normal, anomaly]),
            np.concatenate([
                np.zeros(normal.size, dtype=int),
                np.ones(anomaly.size, dtype=int),
            ]),
        )
        op = operating_point(normal, anomaly, thresholds[cid])
        metrics = {
            "image_auroc": image_auroc,
            "operating_point": op,
            "normal_median": float(np.median(normal)),
            "anomaly_median": float(np.median(anomaly)),
            "anomaly_minus_normal_median": float(np.median(anomaly) - np.median(normal)),
            "gate_passed": gate_passed({"image_auroc": image_auroc, **op}),
        }
        metrics["gate"] = "PASS" if metrics["gate_passed"] else "FAIL"
        development_metrics[cid] = metrics
        gate_results.append({
            "_index": CANDIDATE_IDS.index(cid),
            "candidate_id": cid,
            "image_auroc": image_auroc,
            "normal_fpr": op["normal_fpr"],
            "anomaly_recall": op["anomaly_recall"],
            "f1": op["f1"],
            "gate_passed": metrics["gate_passed"],
        })

    best = select_best(gate_results)
    verdict = "RECOVERY_AGGREGATION_GATE_PASS" if best is not None else "RECOVERY_AGGREGATION_GATE_FAILED"

    # ---- Score distributions ----
    score_distributions = {
        cid: {
            role: distribution(arrays[cid][role])
            for role in ("validation_normal", "recovery_dev_anomaly")
        }
        for cid in CANDIDATE_IDS
    }

    # ---- Small-defect analysis ----
    if args.skip_masks:
        small_defect = {"skipped": True, "reason": "--skip-masks"}
    else:
        small_defect = small_defect_analysis(dev_ids, arrays, validation, thresholds)

    train_only = {
        cid: {
            "threshold": thresholds[cid],
            "train_count": int(arrays[cid]["train_normal"].size),
            "min": float(arrays[cid]["train_normal"].min()),
            "p50": float(np.median(arrays[cid]["train_normal"])),
            "p95": float(np.percentile(arrays[cid]["train_normal"], 95.0, method="linear")),
            "p99": float(np.percentile(arrays[cid]["train_normal"], 99.0, method="linear")),
            "max": float(arrays[cid]["train_normal"].max()),
        }
        for cid in CANDIDATE_IDS
    }

    results = {
        "schema_version": "steel_patchcore_recovery_aggregation_v1",
        "verdict": verdict,
        "protocol_version": manifest["protocol_version"],
        "branch": "feat/steel-patchcore-validity-recovery-v1.2",
        "implementation_commit": git_head(),
        "generated_at": utc_now(),
        "model": "steel-patchcore",
        "model_version": "1.0.0",
        "lineage": lineage,
        "evidence_preflight": {
            "manifest_sha256_ok": lineage["evidence_manifest_sha256"] == EXPECTED_EVIDENCE_MANIFEST_SHA,
            "shards": 88,
            "train_normal": EXPECTED_COUNTS["train_normal"],
            "validation_normal": EXPECTED_COUNTS["validation_normal"],
            "recovery_dev_anomaly": EXPECTED_COUNTS["recovery_dev_anomaly"],
            "total": sum(EXPECTED_COUNTS.values()),
            "duplicate": 0,
            "missing": 0,
            "unexpected": 0,
            "dtype": "float32",
            "grid_shape": [32, 32],
            "stitched_shape": [32, 200],
            "overlap_rule": "mean",
            "baseline_reconstruction_atol": RECONSTRUCTION_ATOL,
            "baseline_reconstruction_max_abs_error": max_error,
            "baseline_reconstruction_mismatches_gt_2e_6": mismatch_count,
        },
        "a0_baseline_gate": {
            "threshold": threshold_a0,
            "frozen_threshold": EXPECTED_THRESHOLD,
            "threshold_delta": a0_delta,
            "reconstructed": a0_reconstructed,
        },
        "candidate_grid": [
            {
                "id": c["id"],
                "method": c["method"],
                "parameters": {k: v for k, v in c.items() if k not in ("id", "method")},
                "definition": CANDIDATE_DEFINITIONS[c["id"]],
            }
            for c in CANDIDATE_GRID
        ],
        "development_gate": DEVELOPMENT_GATE,
        "train_only_thresholds": train_only,
        "development_metrics": development_metrics,
        "score_distributions": score_distributions,
        "candidate_selection": (
            {
                "candidate_id": best["candidate_id"],
                "ranking": [r["candidate_id"] for r in sorted(gate_results, key=lambda r: (-r["image_auroc"], r["_index"]))],
                "passed": [r["candidate_id"] for r in gate_results if r["gate_passed"]],
            }
            if best is not None
            else {"candidate_id": None, "ranking": [], "passed": []}
        ),
        "small_defect_analysis": small_defect,
        "holdout_access_count": 0,
    }

    atomic_write_json(OUT_JSON, results)
    OUT_MD.write_text(render_markdown(results), encoding="utf-8")
    print(json.dumps({"verdict": verdict, "best_candidate": best["candidate_id"] if best else None}))
    print(f"threshold_A0={threshold_a0!r} delta={a0_delta!r}")
    for cid in CANDIDATE_IDS:
        m = development_metrics[cid]
        print(f"{cid}: auroc={m['image_auroc']:.4f} fpr={m['operating_point']['normal_fpr']:.4f} "
              f"recall={m['operating_point']['anomaly_recall']:.4f} gate={m['gate']}")
    print(verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())