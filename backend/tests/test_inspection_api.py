from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from vision_contract import Detection, InferenceResult, utc_now_iso

from app.enums import QualityResult, Severity
from app.inference.client import (
    InferenceConnectionError,
    InferenceContractError,
    InferenceHTTPError,
    InferenceTimeoutError,
)
from app.models import Inspection, Product, QualityRule

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REAL_IMAGE = PROJECT_ROOT / "model-training/datasets/neu-det-yolo/test/images/crazing_101.jpg"
SAMPLE_JPG = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9"


def contract(*, detections: list[dict] | None = None) -> dict:
    return InferenceResult(
        inspection_id="insp-fake",
        model_name="yolov8s",
        model_version="phase1-baseline",
        image_width=200,
        image_height=200,
        detections=[Detection(**d) for d in (detections or [])],
        inference_latency_ms=11.2,
        device="cuda:0",
        timestamp=utc_now_iso(),
    )


def detection(class_name: str, confidence: float, area_ratio: float) -> dict:
    return {
        "class_id": 0,
        "class_name": class_name,
        "confidence": confidence,
        "bbox_xyxy": (10.0, 10.0, 50.0, 50.0),
        "bbox_normalized": (0.05, 0.05, 0.25, 0.25),
        "defect_area_px": 1600.0,
        "defect_area_ratio": area_ratio,
    }


class StubInference:
    def __init__(self, *, result: InferenceResult | None = None, error: Exception | None = None):
        self.result = result
        self.error = error

    async def infer(self, image_bytes: bytes, filename: str = "image.jpg", request_id: str | None = None):
        if self.error is not None:
            raise self.error
        return self.result


async def count_inspections(session: AsyncSession) -> int:
    result = await session.execute(select(func.count()).select_from(Inspection))
    return result.scalar_one()


async def test_create_inspection_pass_path(client, db_session, stub_infer):
    stub_infer(StubInference(result=contract()))
    resp = await client.post(
        "/api/v1/inspections",
        files={"file": ("clean.png", SAMPLE_JPG, "image/png")},
        data={"product_id": "NEU-0001", "production_line": "line-a", "station": "qc-01"},
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["quality_result"] == "PASS"
    assert body["status"] == "completed"
    assert body["defects"] == []
    assert await count_inspections(db_session) == 1


async def test_create_inspection_fail_path_with_seeded_rule(client, db_session, stub_infer):
    db_session.add(
        QualityRule(
            defect_type="crazing",
            min_confidence=0.5,
            max_area_ratio=1.0,
            action=QualityResult.FAIL,
            severity=Severity.HIGH,
            priority=1,
        )
    )
    await db_session.commit()
    stub_infer(StubInference(result=contract(detections=[detection("crazing", 0.9, 0.1)])))

    resp = await client.post(
        "/api/v1/inspections",
        files={"file": ("crazing.png", SAMPLE_JPG, "image/png")},
        data={"product_id": "NEU-0002"},
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["quality_result"] == "FAIL"
    assert body["severity"] == "high"
    assert len(body["defects"]) == 1
    assert body["defects"][0]["class_name"] == "crazing"


async def test_fetch_inspection_and_product_history(client, db_session, stub_infer):
    stub_infer(StubInference(result=contract(detections=[detection("scratches", 0.8, 0.05)])))
    r1 = await client.post(
        "/api/v1/inspections",
        files={"file": ("a.png", SAMPLE_JPG, "image/png")},
        data={"product_id": "NEU-0003"},
    )
    r2 = await client.post(
        "/api/v1/inspections",
        files={"file": ("b.png", SAMPLE_JPG, "image/png")},
        data={"product_id": "NEU-0003"},
    )
    assert r1.status_code == r2.status_code == 201
    insp1, insp2 = r1.json()["inspection_id"], r2.json()["inspection_id"]

    fetched = await client.get(f"/api/v1/inspections/{insp1}")
    assert fetched.status_code == 200
    assert fetched.json()["product_id"] == "NEU-0003"

    history = await client.get("/api/v1/products/NEU-0003/inspections")
    assert history.status_code == 200
    assert [i["inspection_id"] for i in history.json()] == [insp1, insp2]


async def test_inference_timeout_maps_to_504(client, db_session, stub_infer):
    stub_infer(StubInference(error=InferenceTimeoutError("timed out")))
    resp = await client.post(
        "/api/v1/inspections",
        files={"file": ("a.png", SAMPLE_JPG, "image/png")},
        data={"product_id": "NEU-0004"},
    )
    assert resp.status_code == 504
    assert resp.json()["error"]["code"] == "inference_failed"
    row = await db_session.execute(select(Inspection))
    assert row.scalar_one().status.value == "failed"


async def test_inference_http_500_maps_to_502(client, db_session, stub_infer):
    stub_infer(StubInference(error=InferenceHTTPError(500, "boom")))
    resp = await client.post(
        "/api/v1/inspections",
        files={"file": ("a.png", SAMPLE_JPG, "image/png")},
        data={"product_id": "NEU-0005"},
    )
    assert resp.status_code == 502


async def test_invalid_contract_maps_to_502(client, db_session, stub_infer):
    stub_infer(StubInference(error=InferenceContractError("bad schema")))
    resp = await client.post(
        "/api/v1/inspections",
        files={"file": ("a.png", SAMPLE_JPG, "image/png")},
        data={"product_id": "NEU-0006"},
    )
    assert resp.status_code == 502


async def test_connection_error_maps_to_504(client, db_session, stub_infer):
    stub_infer(StubInference(error=InferenceConnectionError("unreachable")))
    resp = await client.post(
        "/api/v1/inspections",
        files={"file": ("a.png", SAMPLE_JPG, "image/png")},
        data={"product_id": "NEU-0007"},
    )
    assert resp.status_code == 504


async def test_invalid_image_rejected_422(client, db_session, stub_infer):
    stub_infer(StubInference(result=contract()))
    resp = await client.post(
        "/api/v1/inspections",
        files={"file": ("fake.png", b"this is not an image", "image/png")},
        data={"product_id": "NEU-0008"},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "invalid_image"
    assert await count_inspections(db_session) == 0


async def test_idempotency_key_returns_same_inspection(client, db_session, stub_infer):
    stub_infer(StubInference(result=contract()))
    payload = {
        "files": {"file": ("a.png", SAMPLE_JPG, "image/png")},
        "data": {"product_id": "NEU-0009", "idempotency_key": "key-1"},
    }
    first = await client.post("/api/v1/inspections", **payload)
    second = await client.post("/api/v1/inspections", **payload)
    assert first.status_code == 201
    assert second.status_code == 200
    assert first.json()["inspection_id"] == second.json()["inspection_id"]
    assert await count_inspections(db_session) == 1


async def test_db_rollback_on_commit_failure(client, db_session, stub_infer, monkeypatch):
    stub_infer(StubInference(result=contract()))

    original_commit = db_session.commit

    async def failing_commit():
        raise SQLAlchemyError("simulated db failure")

    monkeypatch.setattr(db_session, "commit", failing_commit)
    resp = await client.post(
        "/api/v1/inspections",
        files={"file": ("a.png", SAMPLE_JPG, "image/png")},
        data={"product_id": "NEU-0010"},
    )
    assert resp.status_code == 500
    assert resp.json()["error"]["code"] == "db_write_failed"
    assert await count_inspections(db_session) == 0
    monkeypatch.setattr(db_session, "commit", original_commit)
