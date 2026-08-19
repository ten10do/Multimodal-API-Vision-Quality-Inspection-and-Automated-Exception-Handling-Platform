"""Offline spatial-scale geometry audit + defect-to-grid quantification.

No GPU, no inference, no holdout. Computes, from frozen GT masks of the
3333 recovery_dev_anomaly originals, the geometry mapping between defects and
the current feature grids (layer2 32x32, layer3 16x16 -> bilinear 32x32).
Outputs docs/steel-patchcore-spatial-scale-diagnostics.md (+ .json).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys_path = str(ROOT / "model-training")
import sys  # noqa: E402

sys.path.insert(0, sys_path)

from steel_patchcore.recovery import sha256_file  # noqa: E402
from steel_patchcore.rle import rle_decode  # noqa: E402

DS = ROOT / "model-training/datasets/severstal-steel"
CSV = DS / "raw/train.csv"
RECOVERY_SPLIT = DS / "recovery_split_manifest.json"
SUBSET_MANIFEST = DS / "representation_diagnostic_manifest.json"
OUT_MD = ROOT / "docs/steel-patchcore-spatial-scale-diagnostics.md"
OUT_JSON = ROOT / "docs/steel-patchcore-spatial-scale-diagnostics.json"

EXPECTED_RECOVERY_SPLIT_SHA = "f60ce0a6aa012ad1d9a9636800c1e94f365940500f20b8deb2f7fd9ffbc55448"

# Grid geometry (from the frozen 256x256 tile path).
GRID = {
    "layer2": {"shape": [32, 32], "stride_px": 8.0, "cell_px": 8.0},
    "layer3_native": {"shape": [16, 16], "stride_px": 16.0, "cell_px": 16.0},
    "layer3_upsampled_used": {
        "shape": [32, 32],
        "stride_px": 8.0,
        "note": "grid sample spacing 8px but source info from 16x16 native cells (bilinear interpolation)",
    },
}


def _quartile(q1: float, q2: float, q3: float, ratio: float) -> int:
    if ratio < q1:
        return 1
    if ratio < q2:
        return 2
    if ratio < q3:
        return 3
    return 4


def main() -> int:
    if sha256_file(RECOVERY_SPLIT) != EXPECTED_RECOVERY_SPLIT_SHA:
        raise RuntimeError("FROZEN_RECOVERY_SPLIT_SHA_MISMATCH")

    from scipy import ndimage

    recovery = json.loads(RECOVERY_SPLIT.read_text(encoding="utf-8"))
    dev_ids = list(recovery["recovery_dev_anomaly"])
    wanted = set(dev_ids)
    assert not (wanted & set(recovery.get("recovery_holdout_anomaly", [])))  # sanity

    manifest = json.loads(SUBSET_MANIFEST.read_text(encoding="utf-8"))
    q1 = float(manifest["anomaly_quartile_boundaries"]["q1"])
    q2 = float(manifest["anomaly_quartile_boundaries"]["q2"])
    q3 = float(manifest["anomaly_quartile_boundaries"]["q3"])

    df = pd.read_csv(CSV, keep_default_na=True)
    rows: list[dict] = []
    for img_id, group in df.groupby(df["ImageId"].astype(str)):
        norm = Path(str(img_id)).stem
        if norm not in wanted:
            continue
        mask = np.zeros((256, 1600), dtype=np.uint8)
        for value in group["EncodedPixels"]:
            if pd.isna(value):
                continue
            mask = np.maximum(mask, rle_decode(str(value)))
        area_px = int(mask.sum())
        ratio = area_px / (256 * 1600)

        ys, xs = np.nonzero(mask)
        union_w = int(xs.max() - xs.min() + 1)
        union_h = int(ys.max() - ys.min() + 1)

        labeled, n_comp = ndimage.label(mask)
        comp_slices = ndimage.find_objects(labeled)
        comp_areas = np.array([int((mask[s] > 0).sum()) for s in comp_slices], dtype=np.int64)
        max_idx = int(comp_areas.argmax())
        ms = comp_slices[max_idx]
        max_w = int(ms[1].stop - ms[1].start)
        max_h = int(ms[0].stop - ms[0].start)
        max_area = int(comp_areas[max_idx])
        max_side = max(max_w, max_h)

        rows.append({
            "image_id": norm,
            "area_px": area_px,
            "area_ratio": ratio,
            "quartile": _quartile(q1, q2, q3, ratio),
            "union_bbox_w_px": union_w,
            "union_bbox_h_px": union_h,
            "num_components": int(n_comp),
            "max_component_area_px": max_area,
            "max_component_bbox_w_px": max_w,
            "max_component_bbox_h_px": max_h,
            "max_component_bbox_side_px": max_side,
        })
    missing = wanted - {r["image_id"] for r in rows}
    if missing:
        raise RuntimeError(f"MISSING_ANOMALY_MASK:{sorted(missing)[:5]}")

    ratios = np.array([r["area_ratio"] for r in rows])
    side = np.array([r["max_component_bbox_side_px"] for r in rows], dtype=np.float64)
    wpx = np.array([r["max_component_bbox_w_px"] for r in rows], dtype=np.float64)
    hpx = np.array([r["max_component_bbox_h_px"] for r in rows], dtype=np.float64)
    qarr = np.array([r["quartile"] for r in rows], dtype=int)

    def _median(arr: np.ndarray) -> float:
        return float(np.median(arr))

    summary = {"n": len(rows), "quartile": {}, "grid_geometry": GRID}
    for q in (1, 2, 3, 4):
        q_sel = qarr == q
        summary["quartile"][str(q)] = {
            "count": int(q_sel.sum()),
            "median_max_component_bbox_w_px": _median(wpx[q_sel]),
            "median_max_component_bbox_h_px": _median(hpx[q_sel]),
            "median_max_component_bbox_side_px": _median(side[q_sel]),
            "median_bbox_w_layer2_cells": _median(wpx[q_sel] / 8.0),
            "median_bbox_h_layer2_cells": _median(hpx[q_sel] / 8.0),
            "median_bbox_w_layer3_cells": _median(wpx[q_sel] / 16.0),
            "median_bbox_h_layer3_cells": _median(hpx[q_sel] / 16.0),
        }
    summary["overall"] = {
        "median_max_component_bbox_w_px": _median(wpx),
        "median_max_component_bbox_h_px": _median(hpx),
        "median_max_component_bbox_side_px": _median(side),
        "median_bbox_w_layer2_cells": _median(wpx / 8.0),
        "median_bbox_h_layer2_cells": _median(hpx / 8.0),
    }

    # fraction of defects whose largest-component bbox fits within ~1 / 2x2 / 4x4
    # layer2 cells (1 layer2 cell = 8px side). Report overall + per quartile.
    def _frac_le(thr: float, sel=None) -> tuple[float, int, int]:
        s = side if sel is None else side[sel]
        return float((s <= thr).mean()), int((s <= thr).sum()), int(s.size)

    thresholds = {"<=1_cell(8px)": 8.0, "<=2x2_cells(16px)": 16.0, "<=4x4_cells(32px)": 32.0}
    frac = {"overall": {}}
    for name, thr in thresholds.items():
        f, c, n = _frac_le(thr)
        frac["overall"][name] = {"fraction": f, "count": c, "total": n}
    for q in (1, 2, 3, 4):
        sel = qarr == q
        frac[str(q)] = {}
        for name, thr in thresholds.items():
            f, c, n = _frac_le(thr, sel)
            frac[str(q)][name] = {"fraction": f, "count": c, "total": n}
    summary["fraction_largest_component_bbox_within"] = frac

    OUT_MD.write_text(_render_md(summary, manifest), encoding="utf-8")
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"n": summary["n"], "overall_median_side_px": summary["overall"]["median_max_component_bbox_side_px"],
                      "q1_median_side_px": summary["quartile"]["1"]["median_max_component_bbox_side_px"],
                      "q4_median_side_px": summary["quartile"]["4"]["median_max_component_bbox_side_px"],
                      "le_1cell_frac": frac["overall"]["<=1_cell(8px)"]["fraction"]}, indent=2))
    print("SPATIAL_SCALE_DIAGNOSTICS_WRITTEN")
    return 0


def _render_md(s: dict, manifest: dict) -> str:
    L: list[str] = []
    A = L.append
    A("# Steel PatchCore — Spatial Scale Diagnostics (offline)")
    A("")
    A(f"- Schema: `steel_patchcore_spatial_scale_diagnostics_v1`")
    A(f"- Dev anomalies analyzed: {s['n']} (frozen recovery_dev_anomaly)")
    A(f"- Quartile boundaries (area ratio) reused from frozen manifest: "
      f"Q1={manifest['anomaly_quartile_boundaries']['q1']}, "
      f"Q2={manifest['anomaly_quartile_boundaries']['q2']}, "
      f"Q3={manifest['anomaly_quartile_boundaries']['q3']}")
    A("- Holdout access: 0")
    A("")
    A("## 1. Feature-grid geometry (nominal, from frozen 256x256 tile path)")
    A("")
    A("| Grid | shape | stride (px) | nominal cell footprint |")
    A("|---|---|---|---|")
    A("| layer2 | 32x32 | 8 | 8x8 px |")
    A("| layer3 (native) | 16x16 | 16 | 16x16 px |")
    A("| layer3 (as used: bilinear upsample) | 32x32 | 8 (grid) | 16x16 px info source, interpolated |")
    A("")
    A("These are **feature-grid stride / nominal footprint**, not theoretical or")
    A("effective receptive field. Effective RF is a model property that is not")
    A("computable from tiling code alone and is intentionally NOT claimed here.")
    A("")
    A("## 2. Defect-to-grid (largest connected component bbox)")
    A("")
    A("| Quartile | count | median bbox W (px) | median bbox H (px) | median side (px) | median W (l2 cells) | median H (l2 cells) | median W (l3 cells) | median H (l3 cells) |")
    A("|---|---|---|---|---|---|---|---|---|")
    for q in ("1", "2", "3", "4"):
        d = s["quartile"][q]
        A(f"| Q{q} | {d['count']} | {d['median_max_component_bbox_w_px']:.1f} | {d['median_max_component_bbox_h_px']:.1f} | {d['median_max_component_bbox_side_px']:.1f} | {d['median_bbox_w_layer2_cells']:.2f} | {d['median_bbox_h_layer2_cells']:.2f} | {d['median_bbox_w_layer3_cells']:.2f} | {d['median_bbox_h_layer3_cells']:.2f} |")
    o = s["overall"]
    A(f"| overall | {s['n']} | {o['median_max_component_bbox_w_px']:.1f} | {o['median_max_component_bbox_h_px']:.1f} | {o['median_max_component_bbox_side_px']:.1f} | {o['median_bbox_w_layer2_cells']:.2f} | {o['median_bbox_h_layer2_cells']:.2f} | - | - |")
    A("")
    A("## 3. Fraction of defects whose largest-component bbox fits within N cells")
    A("")
    A("(1 layer2 cell = 8px side; 2x2 = 16px; 4x4 = 32px)")
    A("")
    A("| Group | <=1 cell (8px) | <=2x2 cells (16px) | <=4x4 cells (32px) |")
    A("|---|---|---|---|")
    for key in ("overall", "1", "2", "3", "4"):
        d = s["fraction_largest_component_bbox_within"][key]
        A(f"| {key} | {d['<=1_cell(8px)']['fraction']:.3f} | {d['<=2x2_cells(16px)']['fraction']:.3f} | {d['<=4x4_cells(32px)']['fraction']:.3f} |")
    A("")
    A("## 4. Interpretation guardrails")
    A("")
    A("- This is a **geometry overlap analysis**, not a proof of feature receptive field.")
    A("- It does NOT assert that small defects are 'sub-resolution'; it quantifies")
    A("  how defect bbox sizes map onto the current grids.")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())