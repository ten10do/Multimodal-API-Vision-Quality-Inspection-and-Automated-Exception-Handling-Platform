"""Contract-level tests for the standardized vision inference schema.

Enforces the Phase 1C decision: the vision layer carries objective geometry
only. severity, quality_result, PASS / REVIEW / FAIL are rejected outright.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from inference_app.vision_contract import Detection, InferenceResult, utc_now_iso

OK_DETECTION = dict(
    class_id=0,
    class_name="crazing",
    confidence=0.87,
    bbox_xyxy=(10.0, 12.0, 40.0, 55.0),
    bbox_normalized=(0.05, 0.06, 0.2, 0.275),
    defect_area_px=1050.0,
    defect_area_ratio=0.02625,
)


def ok_result(**overrides) -> dict:
    base = dict(
        inspection_id="insp-test-1",
        model_name="yolov8s",
        model_version="phase1-baseline",
        image_width=200,
        image_height=200,
        detections=[],
        inference_latency_ms=8.3,
        device="cuda:0",
        timestamp=utc_now_iso(),
    )
    base.update(overrides)
    return base


def test_result_without_detections_is_valid():
    result = InferenceResult(**ok_result())
    assert result.detections == []
    assert result.has_detections is False


def test_result_with_single_detection_is_valid():
    result = InferenceResult(**ok_result(detections=[Detection(**OK_DETECTION)]))
    assert result.has_detections is True
    assert result.detections[0].defect_area_px == pytest.approx(1050.0)


def test_result_with_multiple_detections_is_valid():
    dets = [Detection(**OK_DETECTION), Detection(**{**OK_DETECTION, "class_id": 5, "class_name": "scratches"})]
    result = InferenceResult(**ok_result(detections=dets))
    assert len(result.detections) == 2


def test_severity_rejected_in_detection():
    with pytest.raises(ValidationError):
        Detection(**{**OK_DETECTION, "severity": "high"})


def test_quality_verdict_rejected_in_result():
    with pytest.raises(ValidationError):
        InferenceResult(**ok_result(quality_result="PASS"))


def test_confidence_out_of_range_rejected():
    with pytest.raises(ValidationError):
        Detection(**{**OK_DETECTION, "confidence": 1.4})


def test_negative_confidence_rejected():
    with pytest.raises(ValidationError):
        Detection(**{**OK_DETECTION, "confidence": -0.1})


def test_area_ratio_out_of_range_rejected():
    with pytest.raises(ValidationError):
        Detection(**{**OK_DETECTION, "defect_area_ratio": 1.5})


def test_inverted_bbox_rejected():
    with pytest.raises(ValidationError):
        Detection(**{**OK_DETECTION, "bbox_xyxy": (40.0, 55.0, 10.0, 12.0)})


def test_normalized_bbox_out_of_range_rejected():
    with pytest.raises(ValidationError):
        Detection(**{**OK_DETECTION, "bbox_normalized": (1.5, 0.0, 0.2, 0.2)})


def test_invalid_timestamp_rejected():
    with pytest.raises(ValidationError):
        InferenceResult(**ok_result(timestamp="not-a-date"))


def test_contract_has_no_quality_fields():
    assert "severity" not in InferenceResult.model_fields
    assert "quality_result" not in InferenceResult.model_fields
    assert "severity" not in Detection.model_fields
