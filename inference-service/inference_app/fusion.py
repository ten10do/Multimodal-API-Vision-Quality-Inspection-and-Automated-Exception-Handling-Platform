"""Vision Fusion Layer (Phase 6, 6E).

A pure classifier that combines YOLO detections with the PatchCore anomaly
verdict into one of four objective classes:

    NORMAL_CANDIDATE           detections == 0 and anomaly normal
    KNOWN_DEFECT               detections > 0 and anomaly normal
    UNKNOWN_ANOMALY            detections == 0 and anomaly anomalous
    KNOWN_DEFECT_WITH_ANOMALY  detections > 0 and anomaly anomalous

The fusion class is NOT a quality result. It is consumed by the Quality Rule
Engine, which decides PASS / REVIEW / FAIL.
"""

from __future__ import annotations

from typing import Literal

FusionClass = Literal["NORMAL_CANDIDATE", "KNOWN_DEFECT", "UNKNOWN_ANOMALY", "KNOWN_DEFECT_WITH_ANOMALY"]

# When PatchCore is unavailable, the anomaly verdict is treated as "no
# anomaly evidence" and the fusion falls back to the YOLO-only view.


def fuse(detection_count: int, is_anomalous: bool | None) -> FusionClass:
    """Map YOLO + PatchCore evidence to a fusion class.

    is_anomalous=None means PatchCore was unavailable (load failure, timeout);
    the anomaly channel contributes no evidence.
    """
    has_known_defect = detection_count > 0
    anomaly_evidence = bool(is_anomalous)
    if not has_known_defect and not anomaly_evidence:
        return "NORMAL_CANDIDATE"
    if has_known_defect and not anomaly_evidence:
        return "KNOWN_DEFECT"
    if not has_known_defect and anomaly_evidence:
        return "UNKNOWN_ANOMALY"
    return "KNOWN_DEFECT_WITH_ANOMALY"


__all__ = ["FusionClass", "fuse"]
