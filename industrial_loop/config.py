"""Frozen lineage + policy constants for the industrial closed-loop layer.

The values below are READ-ONLY references into the frozen D3 release
(``steel-patchcore-d3-release@1.3.0``). They mirror the committed release
artifacts (``model-training/steel_patchcore/d3_release_package.py`` and the
dual candidate registry) and are never tuned by this layer. The decision
engine refuses to auto-judge with any other threshold (fail-close).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --- frozen D3 release lineage (read-only mirrors, see docs/release/) --------
RELEASE_ID = "steel-patchcore-d3-release@1.3.0"
MODEL_VERSION = "1.3.0-candidate.1"
FROZEN_THRESHOLD = 0.8471092581748962

EVENT_SCHEMA_VERSION = "industrial_loop_event_v1"

# --- runtime artifacts (gitignored) ------------------------------------------
RUNTIME_ROOT = Path(
    os.environ.get("INDUSTRIAL_LOOP_RUNTIME", str(PROJECT_ROOT / "runs" / "industrial-loop"))
)


@dataclass(frozen=True)
class DecisionPolicy:
    """Peripheral decision policy.

    ``reject_threshold`` is the frozen D3 image threshold; it is never changed
    here. ``hold_margin_ratio`` optionally routes borderline sub-threshold
    products to human review instead of auto-PASS (strictly more conservative;
    0.0 disables the band so decisions mirror the frozen threshold exactly).
    """

    reject_threshold: float = FROZEN_THRESHOLD
    expected_model_version: str = MODEL_VERSION
    hold_margin_ratio: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 < self.reject_threshold < 10.0:
            raise ValueError("reject_threshold out of plausible range")
        if self.hold_margin_ratio < 0.0 or self.hold_margin_ratio >= 1.0:
            raise ValueError("hold_margin_ratio must be in [0, 1)")
