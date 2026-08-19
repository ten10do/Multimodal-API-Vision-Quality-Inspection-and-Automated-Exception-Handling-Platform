"""Offline (no-GPU) representation diagnostics over frozen raw evidence.

Consumes the frozen 8644 raw evidence shards, GT masks, canonical splits, and
bank metadata. It does NOT run inference and does NOT touch the holdout. It
quantifies raw nearest-bank distance distributions, tile-position response,
memory-bank coverage, and defect-size correlates — before any GPU experiment.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model-training"))
sys.path.insert(0, str(ROOT / "inference-service"))

from steel_patchcore.aggregation import distribution  # noqa: E402
from steel_patchcore.recovery import (  # noqa: E402
    CAPTURE_ROLES,
    sha256_file,
    validate_recovery_split_manifest,
)
from steel_patchcore.rle import rle_decode  # noqa: E402
from steel_patchcore.tile import TILE_X0  # noqa: E402

DS = ROOT / "model-training/datasets/severstal-steel"
SOURCE_SPLIT = DS / "split_manifest.json"
RECOVERY_SPLIT = DS / "recovery_split_manifest.json"
EVIDENCE_MANIFEST = DS / "recovery_evidence_manifest.json"
EVIDENCE_ROOT = DS / "raw/recovery-evidence"
CSV = DS / "raw/train.csv"
OUT_JSON = ROOT / "docs/steel-patchcore-representation-diagnostics.json"
OUT_MD = ROOT / "docs/steel-patchcore-representation-diagnostics.md"

EXPECTED_SOURCE_SPLIT_SHA = "64df9f817a995e64c7ee4d00e16962d94f500878fd1404d95227f26e7f913d07"
EXPECTED_RECOVERY_SPLIT_SHA = "f60ce0a6aa012ad1d9a9636800c1e94f365940500f20b8deb2f7fd9ffbc55448"
EXPECTED_EVIDENCE_MANIFEST_SHA = "7067bbc6da747ffa90db9532729f579e19ee90fbcdf3867c96f2dd56a0d10303"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sanitize(value):
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(_sanitize(payload), open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2, allow_nan=False)


def spearman(x: np.ndarray, y: np.ndarray) -> dict:
    from scipy.stats import spearmanr

    rho, p = spearmanr(x, y)
    return {"rho": float(rho) if np.isfinite(rho) else None, "p": float(p) if np.isfinite(p) else None}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-masks", action="store_true")
    args = parser.parse_args()

    source = json.loads(SOURCE_SPLIT.read_text(encoding="utf-8"))
    recovery = json.loads(RECOVERY_SPLIT.read_text(encoding="utf-8"))
    validate_recovery_split_manifest(recovery, source, EXPECTED_SOURCE_SPLIT_SHA)
    manifest = json.loads(EVIDENCE_MANIFEST.read_text(encoding="utf-8"))
    if sha256_file(EVIDENCE_MANIFEST) != EXPECTED_EVIDENCE_MANIFEST_SHA:
        raise RuntimeError("EVIDENCE_MANIFEST_SHA_MISMATCH")
    if sha256_file(SOURCE_SPLIT) != EXPECTED_SOURCE_SPLIT_SHA:
        raise RuntimeError("SOURCE_SPLIT_SHA_MISMATCH")
    if sha256_file(RECOVERY_SPLIT) != EXPECTED_RECOVERY_SPLIT_SHA:
        raise RuntimeError("RECOVERY_SPLIT_SHA_MISMATCH")

    role_ids = {
        "train_normal": list(source["splits"]["train_normal"]),
        "validation_normal": list(source["splits"]["validation_normal"]),
        "recovery_dev_anomaly": list(recovery["recovery_dev_anomaly"]),
    }

    # per-role accumulators
    a0: dict[str, list[float]] = {r: [] for r in CAPTURE_ROLES}
    patch_mean: dict[str, list[float]] = {r: [] for r in CAPTURE_ROLES}
    patch_p99: dict[str, list[float]] = {r: [] for r in CAPTURE_ROLES}
    tile_max: dict[str, list[np.ndarray]] = {r: [] for r in CAPTURE_ROLES}
    argmax_hist: dict[str, np.ndarray] = {r: np.zeros(7, dtype=np.int64) for r in CAPTURE_ROLES}

    for role in CAPTURE_ROLES:
        expected = set(role_ids[role])
        seen: set[str] = set()
        for path in sorted((EVIDENCE_ROOT / role).glob("shard-*.npz")):
            with np.load(path, allow_pickle=False) as data:
                ids = [str(v) for v in data["original_ids"].tolist()]
                grids = data["raw_grids"]
                tmax = data["raw_tile_scores"]
            for index, image_id in enumerate(ids):
                g = grids[index].astype(np.float64)
                saw_max = tmax[index]
                a0[role].append(float(g.max()))
                patch_mean[role].append(float(g.mean()))
                patch_p99[role].append(float(np.percentile(g, 99.0, method="linear")))
                tile_max[role].append(saw_max.astype(np.float64))
                argmax_hist[role][int(np.argmax(saw_max))] += 1
                if image_id in seen or image_id not in expected:
                    raise RuntimeError(f"SHARD_ID_INVARIANT:{role}:{image_id}")
                seen.add(image_id)
        if len(seen) != len(expected):
            raise RuntimeError(f"ROLE_COUNT_MISMATCH:{role}:{len(seen)}")

    roles_out = {}
    for role in CAPTURE_ROLES:
        tmax = np.stack(tile_max[role]) if tile_max[role] else np.zeros((0, 7))
        roles_out[role] = {
            "n": len(a0[role]),
            "a0": distribution(np.asarray(a0[role])),
            "patch_mean": distribution(np.asarray(patch_mean[role])),
            "patch_p99": distribution(np.asarray(patch_p99[role])),
            "median_tile_max": [float(v) for v in np.median(tmax, axis=0)] if len(tmax) else [],
            "argmax_tile_histogram": [int(v) for v in argmax_hist[role]],
        }

    # ---- coverage audit ----
    train_pm = np.asarray(patch_mean["train_normal"])
    val_pm = np.asarray(patch_mean["validation_normal"])
    train_a0 = np.asarray(a0["train_normal"])
    val_a0 = np.asarray(a0["validation_normal"])
    coverage = {
        "train_normal_median_patch_mean": float(np.median(train_pm)),
        "validation_normal_median_patch_mean": float(np.median(val_pm)),
        "validation_minus_train_median_patch_mean": float(np.median(val_pm) - np.median(train_pm)),
        "validation_p95_patch_mean": float(np.percentile(val_pm, 95.0)),
        "train_p95_patch_mean": float(np.percentile(train_pm, 95.0)),
        "validation_top5_a0_min": float(np.sort(val_a0)[-5:].min()) if len(val_a0) else None,
        "train_a0_max": float(train_a0.max()) if len(train_a0) else None,
        "median_tile_max_train": roles_out["train_normal"]["median_tile_max"],
        "median_tile_max_validation": roles_out["validation_normal"]["median_tile_max"],
    }

    # ---- defect-area correlates ----
    correlates = None
    if not args.skip_masks:
        dev_ids = role_ids["recovery_dev_anomaly"]
        wanted = set(dev_ids)
        df = pd.read_csv(CSV, keep_default_na=True)
        area_ratio: dict[str, float] = {}
        max_component: dict[str, int] = {}
        for img_id, group in df.groupby(df["ImageId"].astype(str)):
            norm = Path(str(img_id)).stem
            if norm not in wanted:
                continue
            mask = np.zeros((256, 1600), dtype=np.uint8)
            for value in group["EncodedPixels"]:
                if pd.isna(value):
                    continue
                mask = np.maximum(mask, rle_decode(str(value)))
            area_ratio[norm] = float(mask.sum()) / (256 * 1600)
            try:
                from scipy import ndimage
                labels, count = ndimage.label(mask)
                sizes = np.bincount(labels.ravel())[1:] if count else np.asarray([0])
                max_component[norm] = int(sizes.max()) if sizes.size else 0
            except ImportError:
                max_component[norm] = -1

        dev_ar = np.asarray([area_ratio[i] for i in dev_ids], dtype=np.float64)
        dev_a0 = np.asarray(a0["recovery_dev_anomaly"], dtype=np.float64)
        dev_mean = np.asarray(patch_mean["recovery_dev_anomaly"], dtype=np.float64)
        mc = np.asarray([max_component[i] for i in dev_ids], dtype=np.float64)
        correlates = {
            "area_ratio_vs_a0": spearman(dev_ar, dev_a0),
            "area_ratio_vs_patch_mean": spearman(dev_ar, dev_mean),
            "max_component_vs_a0": spearman(mc[mc >= 0], dev_a0[mc >= 0]) if (mc >= 0).any() else None,
            "quartile_a0_median": None,
        }
        q1, q2, q3 = (float(v) for v in np.quantile(dev_ar, [0.25, 0.5, 0.75], method="linear"))
        q = np.empty(len(dev_ar), dtype=np.int8)
        q[dev_ar < q1] = 1
        q[(dev_ar >= q1) & (dev_ar < q2)] = 2
        q[(dev_ar >= q2) & (dev_ar < q3)] = 3
        q[dev_ar >= q3] = 4
        q_median = {int(i): float(np.median(dev_a0[q == i])) for i in (1, 2, 3, 4) if (q == i).any()}
        correlates["quartile_a0_median"] = q_median
        correlates["anomaly_quartile_boundaries"] = {"q1": q1, "q2": q2, "q3": q3}

    # ---- median ordering ----
    median_delta = {
        "anomaly_minus_validation_normal_a0_median": float(
            np.median(a0["recovery_dev_anomaly"]) - np.median(a0["validation_normal"])
        ),
        "train_minus_validation_normal_patch_mean_median": float(np.median(train_pm) - np.median(val_pm)),
    }

    output = {
        "schema_version": "steel_patchcore_representation_diagnostics_v1",
        "generated_at": utc_now(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "lineage": {
            "source_split_sha256": EXPECTED_SOURCE_SPLIT_SHA,
            "recovery_split_sha256": EXPECTED_RECOVERY_SPLIT_SHA,
            "evidence_manifest_sha256": EXPECTED_EVIDENCE_MANIFEST_SHA,
        },
        "roles": roles_out,
        "median_delta": median_delta,
        "coverage_audit": coverage,
        "defect_correlates": correlates,
        "holdout_access_count": 0,
    }

    atomic_write_json(OUT_JSON, output)
    _render_markdown(output, OUT_MD)
    print(json.dumps({
        "anomaly_minus_normal_a0_median": median_delta["anomaly_minus_validation_normal_a0_median"],
        "val_vs_train_patch_mean_gap": coverage["validation_minus_train_median_patch_mean"],
        "area_ratio_vs_a0_rho": correlates["area_ratio_vs_a0"]["rho"] if correlates else None,
    }))
    print("REPRESENTATION_DIAGNOSTICS_DONE")
    return 0


def _render_markdown(output: dict, path: Path) -> None:
    lines: list[str] = []

    def add(s=""):
        lines.append(s)

    def fmt(v, d=6):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "-"
        return f"{v:.{d}f}"

    add("# Steel PatchCore — Representation Diagnostics (offline)")
    add("")
    add(f"- Schema: `{output['schema_version']}`")
    add(f"- Commit: `{output['commit']}`")
    add(f"- Holdout access: `{output['holdout_access_count']}`")
    add("")
    add("## 1. Raw nearest-bank distance distributions")
    add("")
    add("| Role | n | A0 median | A0 p99 | A0 max | patch-mean median | patch-mean p99 | patch-p99 median |")
    add("|---|---|---|---|---|---|---|---|")
    for role in ("train_normal", "validation_normal", "recovery_dev_anomaly"):
        r = output["roles"][role]
        a = r["a0"]; pm = r["patch_mean"]; pp = r["patch_p99"]
        add(f"| {role} | {r['n']} | {fmt(a['p50'])} | {fmt(a['p99'])} | {fmt(a['max'])} | {fmt(pm['p50'])} | {fmt(pm['p99'])} | {fmt(pp['p50'])} |")
    add("")
    add("## 2. Median ordering")
    add("")
    md = output["median_delta"]
    add(f"- anomaly − validation-normal A0 median = {fmt(md['anomaly_minus_validation_normal_a0_median'])}")
    add("")
    add("## 3. Memory-bank coverage audit")
    add("")
    c = output["coverage_audit"]
    add(f"- train-normal median patch-mean distance = {fmt(c['train_normal_median_patch_mean'])}")
    add(f"- validation-normal median patch-mean distance = {fmt(c['validation_normal_median_patch_mean'])}")
    add(f"- validation − train gap = {fmt(c['validation_minus_train_median_patch_mean'])}")
    add(f"- validation p95 patch-mean = {fmt(c['validation_p95_patch_mean'])}")
    add("")
    add("Per-tile median raw max (tiles 0..6):")
    add("")
    add(f"- train: {', '.join(fmt(v) for v in c['median_tile_max_train'])}")
    add(f"- validation: {', '.join(fmt(v) for v in c['median_tile_max_validation'])}")
    add("")
    add("## 4. Defect-size correlates")
    add("")
    dc = output["defect_correlates"]
    if dc:
        add(f"- area ratio vs A0 Spearman rho = {dc['area_ratio_vs_a0']['rho']}")
        add(f"- max component area vs A0 Spearman rho = {dc['max_component_vs_a0']['rho'] if dc['max_component_vs_a0'] else None}")
        if dc.get("quartile_a0_median"):
            add(f"- quartile A0 medians: {{Q1: {fmt(dc['quartile_a0_median'].get(1))}, Q2: {fmt(dc['quartile_a0_median'].get(2))}, Q3: {fmt(dc['quartile_a0_median'].get(3))}, Q4: {fmt(dc['quartile_a0_median'].get(4))}}}")
    else:
        add("- skipped (no mask analysis)")
    add("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())