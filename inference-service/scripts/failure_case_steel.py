"""Checkpoint-only failure analysis for the steel PatchCore evaluation.

The completed evaluator checkpoint is the evidence source.  This script never
loads the model and never recomputes an original image.

Usage:
  python inference-service/scripts/failure_case_steel.py
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "model-training/datasets/severstal-steel"
CSV = DS / "raw/train.csv"
SPLIT = DS / "split_manifest.json"
CHECKPOINT = DS / "raw/steel_eval_ckpt.json"
METRICS = ROOT / "docs/steel-patchcore-eval/metrics.json"
OUT_MD = ROOT / "docs/steel-patchcore-failure-analysis.md"


def _mask_areas() -> dict[str, int]:
    areas: dict[str, int] = defaultdict(int)
    with open(CSV, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            image_id = Path(row["ImageId"]).stem
            rle = row.get("EncodedPixels") or ""
            values = [int(value) for value in rle.split()]
            areas[image_id] += sum(values[1::2])
    return areas


def _representative(
    rows: dict[str, dict], *, predicted: int, highest: bool
) -> tuple[str, dict] | None:
    matches = [(image_id, row) for image_id, row in rows.items() if int(row["pred"]) == predicted]
    if not matches:
        return None
    return sorted(matches, key=lambda item: float(item[1]["score"]), reverse=highest)[0]


def main() -> int:
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    splits = json.loads(SPLIT.read_text(encoding="utf-8"))["splits"]

    for split_name, expected_count in (("test_normal", 591), ("test_anomaly", 6666)):
        actual = checkpoint[split_name]
        if len(actual) != expected_count or set(actual) != set(splits[split_name]):
            raise RuntimeError(f"FAILURE_ANALYSIS_CHECKPOINT_MISMATCH: {split_name}")

    anomaly_rows = checkpoint["test_anomaly"]
    for image_id, row in anomaly_rows.items():
        if not all(key in row for key in ("score", "pred", "pixel_auc", "aupro")):
            raise RuntimeError(f"FAILURE_ANALYSIS_PIXEL_EVIDENCE_MISSING: {image_id}")

    normal_rows = checkpoint["test_normal"]
    cases = {
        "TP (highest score)": _representative(anomaly_rows, predicted=1, highest=True),
        "TN (lowest score)": _representative(normal_rows, predicted=0, highest=False),
        "FP (highest score)": _representative(normal_rows, predicted=1, highest=True),
        "FN (highest score)": _representative(anomaly_rows, predicted=0, highest=True),
        "FN (lowest score)": _representative(anomaly_rows, predicted=0, highest=False),
    }

    areas_by_id = _mask_areas()
    anomaly_ids = list(anomaly_rows)
    scores = np.asarray([float(anomaly_rows[image_id]["score"]) for image_id in anomaly_ids])
    pixel_aucs = np.asarray([float(anomaly_rows[image_id]["pixel_auc"]) for image_id in anomaly_ids])
    aupros = np.asarray([float(anomaly_rows[image_id]["aupro"]) for image_id in anomaly_ids])
    areas = np.asarray([areas_by_id[image_id] for image_id in anomaly_ids])
    area_quartiles = np.quantile(areas, [0.0, 0.25, 0.5, 0.75, 1.0])
    quartile_score_means = []
    for index in range(4):
        upper = areas <= area_quartiles[index + 1] if index == 3 else areas < area_quartiles[index + 1]
        selected = (areas >= area_quartiles[index]) & upper
        quartile_score_means.append(float(scores[selected].mean()))

    op = metrics["operating_point"]
    distributions = metrics["score_distributions"]
    normal_distribution = distributions["test_normal"]
    anomaly_distribution = distributions["test_anomaly"]
    lines = [
        "# Steel-domain PatchCore Failure Analysis",
        "",
        "Scope: frozen `steel-patchcore` 1.0.0 on the formal Severstal test split. "
        "All values below come from the completed checkpoint; no original image was re-evaluated.",
        "",
        "## Operating point",
        "",
        f"- Threshold: {metrics['threshold']}",
        f"- Formal population: {metrics['formal_test']['test_normal']} test normal + "
        f"{metrics['formal_test']['test_anomaly']} test anomaly = {metrics['formal_test']['total']}",
        f"- TP {op['tp']} / TN {op['tn']} / FP {op['fp']} / FN {op['fn']}",
        f"- Precision {op['precision']} / Recall {op['recall']} / F1 {op['f1']}",
        f"- Normal FPR {op['normal_fpr']} / Anomaly Recall {op['anomaly_recall']}",
        f"- Image AUROC {metrics['image_auroc']}",
        "",
        "## Representative checkpoint cases",
        "",
        "| outcome | image ID | image score | pixel AUROC | AUPRO |",
        "|---|---|---:|---:|---:|",
    ]
    for outcome, item in cases.items():
        if item is None:
            lines.append(f"| {outcome} | none | - | - | - |")
            continue
        image_id, row = item
        pixel_auc = f"{float(row['pixel_auc']):.6f}" if "pixel_auc" in row else "n/a"
        aupro = f"{float(row['aupro']):.6f}" if "aupro" in row else "n/a"
        lines.append(f"| {outcome} | `{image_id}` | {float(row['score']):.6f} | {pixel_auc} | {aupro} |")

    lines.extend(
        [
            "",
            "## Quantitative failure analysis",
            "",
            f"- Score separation is inverted/overlapped: anomaly median {anomaly_distribution['p50']} "
            f"is below test-normal median {normal_distribution['p50']}; anomaly maximum "
            f"{anomaly_distribution['max']} is below the frozen threshold, while test-normal maximum "
            f"{normal_distribution['max']} exceeds it.",
            "- The operating point therefore detects zero of 6,666 anomalies and raises one false alarm. "
            "This is an image-level representation/calibration failure, not a sample-count artifact.",
            f"- Localization retains signal: mean per-image Pixel AUROC is "
            f"{metrics['pixel_auroc_mean_per_image']} and mean per-image AUPRO is "
            f"{metrics['aup_pro_mean_per_image']}. Image score has only weak Pearson correlation with "
            f"per-image Pixel AUROC ({np.corrcoef(scores, pixel_aucs)[0, 1]:.4f}) and AUPRO "
            f"({np.corrcoef(scores, aupros)[0, 1]:.4f}).",
            f"- Defect mask area ranges from {int(areas.min())} to {int(areas.max())} pixels "
            f"(median {int(np.median(areas))}). Mean image scores by increasing area quartile are "
            + ", ".join(f"{value:.6f}" for value in quartile_score_means)
            + ". Smaller defects receive lower scores on average, although every area quartile is missed.",
            "",
            "## Qualitative axes and limits",
            "",
            "- Small/low-contrast defects: the area relationship supports a scale limitation; direct contrast "
            "was not stored and is not inferred here.",
            "- Steel texture and illumination: the severe normal/anomaly score overlap is compatible with "
            "domain-feature sensitivity, but this checkpoint alone cannot attribute individual errors to either cause.",
            "- Tile edges and stitch overlap: the frozen seven-tile/mean-overlap protocol was preserved. "
            "No heatmaps were persisted, so edge-specific error frequency cannot be measured without prohibited re-inference.",
            "- Annotation ambiguity: no label-quality adjudication was performed; the audit does not relabel source masks.",
            "- PatchCore limitation: frozen ImageNet features retain useful local ranking but max-over-tiles image "
            "aggregation does not produce domain-separating scores on this split.",
            "- Threshold limitation: max(train-normal) is correctly bound to the frozen bank, but all anomaly scores "
            "fall below it. Changing it now would violate the frozen evaluation and would not repair AUROC 0.4817.",
            "",
            "## Metric semantics",
            "",
            "Image AUROC is pooled over the 7,257 formal test originals. Pixel AUROC and AUPRO are means of "
            "per-anomaly-image metrics; they are not pooled-pixel estimates. Validation normals are diagnostic only "
            "and do not enter formal image or confusion metrics.",
            "",
            "## Verdict",
            "",
            "`STEEL_DOMAIN_VALIDATION_FAILED`",
            "",
            "The baseline is not eligible for MLOps CANDIDATE registration. Production remains untouched.",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT_MD}")
    print("FAILURE_CASE_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
