"""Phase 6 integration fixtures: NORMAL_CANDIDATE and anomaly-channel
unavailable behaviour, which the current cross-domain demo data cannot
produce naturally (YOLO falsely detects every bottle image; PatchCore flags
almost every NEU image).

These are explicit integration fixtures, NOT benchmark-distribution claims.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from vision_contract import AnomalyResult, VisionResult, utc_now_iso

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_inspection_api import SAMPLE_JPG, StubInference  # noqa: E402


def _vision(detections: int = 0, fusion: str | None = None, anomaly: bool | None = False) -> VisionResult:
    """Build a VisionResult; anomaly=None simulates the PatchCore channel
    being unavailable (bank missing / timeout / load failure)."""
    anomaly_result = None
    if anomaly is not None:
        anomaly_result = AnomalyResult(
            model_name="patchcore-wrn50-2",
            model_version="mvtec-bottle-baseline",
            anomaly_score=0.15 if not anomaly else 0.5,
            threshold=0.2,
            is_anomalous=anomaly,
            latency_ms=5.0,
        )
    if fusion is None:
        if detections == 0 and not (anomaly or False):
            fusion = "NORMAL_CANDIDATE"
        elif detections > 0 and not (anomaly or False):
            fusion = "KNOWN_DEFECT"
        elif detections == 0 and anomaly:
            fusion = "UNKNOWN_ANOMALY"
        else:
            fusion = "KNOWN_DEFECT_WITH_ANOMALY"
    return VisionResult(
        inspection_id="insp-fixture",
        model_name="yolov8s",
        model_version="phase1-baseline",
        image_width=200,
        image_height=200,
        detections=[],
        anomaly=anomaly_result,
        fusion_class=fusion,
        latency_yolo_ms=10.0,
        latency_anomaly_ms=5.0 if anomaly_result else 0.0,
        latency_fusion_ms=0.1,
        inference_latency_ms=15.1,
        device="cuda:0",
        timestamp=utc_now_iso(),
    )


@pytest.mark.asyncio
async def test_normal_candidate_fixture_persists(client, db_session, stub_infer):
    """NORMAL_CANDIDATE (no YOLO defect + normal) -> PASS, fusion persisted."""
    from sqlalchemy import select

    from app.models import Inspection

    stub_infer(StubInference(result=_vision(anomaly=False)))
    resp = await client.post(
        "/api/v1/inspections",
        files={"file": ("x.jpg", SAMPLE_JPG, "image/jpeg")},
        data={"product_id": "P-NC-1", "production_line": "line-a", "station": "qc-01"},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["quality_result"] == "PASS"
    assert body["fusion_class"] == "NORMAL_CANDIDATE"
    assert body["is_anomalous"] is False
    row = (await db_session.execute(select(Inspection).where(Inspection.inspection_id == body["inspection_id"]))).scalar_one()
    assert row.fusion_class == "NORMAL_CANDIDATE"


@pytest.mark.asyncio
async def test_anomaly_channel_unavailable_falls_back_to_yolo(client, db_session, stub_infer):
    """PatchCore unavailable (anomaly=None) -> fusion falls back to the
    YOLO-only view: detections>0 -> KNOWN_DEFECT (no false REVIEW)."""
    stub_infer(StubInference(result=_vision(anomaly=None, detections=0, fusion="NORMAL_CANDIDATE")))
    resp = await client.post(
        "/api/v1/inspections",
        files={"file": ("x.jpg", SAMPLE_JPG, "image/jpeg")},
        data={"product_id": "P-NA-1", "production_line": "line-a", "station": "qc-01"},
    )
    assert resp.status_code == 201
    assert resp.json()["quality_result"] == "PASS"
    assert resp.json()["fusion_class"] == "NORMAL_CANDIDATE"
    assert resp.json()["is_anomalous"] is None
    # the quality decision must not depend on a dead anomaly channel
    assert resp.json()["quality_result"] == "PASS"
