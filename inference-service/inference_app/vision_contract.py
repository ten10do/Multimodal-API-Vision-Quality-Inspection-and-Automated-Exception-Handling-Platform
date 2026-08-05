"""Re-export of the shared vision contract package.

The canonical schema lives in packages/vision-contract (installed as the
`vision_contract` distribution). This module keeps the previous import path
(`app.vision_contract`) working.
"""

from vision_contract import (  # noqa: F401
    AnomalyRegion,
    AnomalyResult,
    Detection,
    InferenceResult,
    NEU_DET_CLASSES,
    VisionResult,
    utc_now_iso,
)

__all__ = [
    "AnomalyRegion",
    "AnomalyResult",
    "Detection",
    "InferenceResult",
    "NEU_DET_CLASSES",
    "VisionResult",
    "utc_now_iso",
]
