"""Re-export of the shared vision contract package.

The canonical schema lives in packages/vision-contract (installed as the
`vision_contract` distribution). This module keeps the previous import path
(`app.vision_contract`) working.
"""

from vision_contract import Detection, InferenceResult, NEU_DET_CLASSES, utc_now_iso  # noqa: F401

__all__ = ["Detection", "InferenceResult", "NEU_DET_CLASSES", "utc_now_iso"]
