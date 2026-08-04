"""Mock-inference end-to-end test: backend + real PostgreSQL + stubbed inference.

Marked ``integration``. Unlike test_e2e.py this does NOT require a running
inference service or a GPU: the inference client is stubbed, so the whole
plumbing (HTTP API -> service -> rule engine -> PostgreSQL -> traceability) is
verified against the real test database with no model dependency.

Requires the test database (industrialvision_test by default) to be migrated.
"""

from __future__ import annotations

import os
import subprocess
import sys
import uuid
from pathlib import Path

import httpx
import pytest
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from vision_contract import Detection, InferenceResult, utc_now_iso

pytestmark = pytest.mark.integration

BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REAL_IMAGE = PROJECT_ROOT / "model-training/datasets/neu-det-yolo/test/images/crazing_101.jpg"

DB_URL = os.environ.get("IVQC_DATABASE_URL", "postgresql+asyncpg://vision_qc:vision_qc@127.0.0.1:5433/industrialvision_test")


async def _db_up(url: str) -> bool:
    try:
        engine = create_async_engine(url)
        async with engine.connect() as conn:
            await conn.execute(__import__("sqlalchemy").text("SELECT 1"))
        await engine.dispose()
        return True
    except Exception:
        return False


def _fake_contract(*, detections: list[dict]) -> InferenceResult:
    return InferenceResult(
        inspection_id="insp-fake",
        model_name="yolov8s",
        model_version="phase1-baseline",
        image_width=200,
        image_height=200,
        detections=[Detection(**d) for d in detections],
        inference_latency_ms=11.0,
        device="cpu",
        timestamp=utc_now_iso(),
    )


class StubInference:
    def __init__(self, result: InferenceResult):
        self.result = result

    async def infer(self, image_bytes: bytes, filename: str = "image.jpg", request_id: str | None = None):
        return self.result


@pytest.fixture(scope="module")
def db_ready():
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    try:
        from prepare_test_db import prepare_test_db

        prepare_test_db()
    except SystemExit as exc:
        pytest.fail(f"test database provisioning failed: {exc}")
    return {**os.environ, "IVQC_DATABASE_URL": DB_URL}


async def test_mock_e2e_no_gpu_required(db_ready):
    engine = create_async_engine(DB_URL)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    from app.api.inspections import get_inspection_service
    from app.database import get_session
    from app.main import create_app
    from app.models import Defect, Inspection, Product, QualityRule
    from app.services.inspection_service import InspectionService

    async with factory() as session:
        session.add(
            QualityRule(
                defect_type="crazing",
                min_confidence=0.3,
                max_area_ratio=1.0,
                action=__import__("app.enums", fromlist=["QualityResult"]).QualityResult.FAIL,
                severity=__import__("app.enums", fromlist=["Severity"]).Severity.HIGH,
                priority=1,
                rule_version=999,
            )
        )
        await session.commit()

    app = create_app()

    async def override_get_session():
        async with factory() as s:
            yield s

    def override_service():
        return InspectionService(
            inference_client=StubInference(
                _fake_contract(
                    detections=[
                        {
                            "class_id": 0,
                            "class_name": "crazing",
                            "confidence": 0.9,
                            "bbox_xyxy": (10.0, 10.0, 50.0, 50.0),
                            "bbox_normalized": (0.05, 0.05, 0.25, 0.25),
                            "defect_area_px": 1600.0,
                            "defect_area_ratio": 0.04,
                        }
                    ]
                )
            )
        )

    app.dependency_overrides[get_session] = override_get_session
    app.dependency_overrides[get_inspection_service] = override_service

    product_id = f"E2E-MOCK-{uuid.uuid4().hex[:8]}"
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/inspections",
            files={"file": ("crazing.jpg", REAL_IMAGE.read_bytes(), "image/jpeg")},
            data={"product_id": product_id, "production_line": "line-a", "station": "qc-01"},
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["quality_result"] == "FAIL"
        assert body["severity"] == "high"
        assert len(body["defects"]) == 1

        inspection_id = body["inspection_id"]
        fetched = await client.get(f"/api/v1/inspections/{inspection_id}")
        assert fetched.status_code == 200
        history = await client.get(f"/api/v1/products/{product_id}/inspections")
        assert history.status_code == 200
        assert [i["inspection_id"] for i in history.json()] == [inspection_id]

    async with factory() as cleanup:
        product_row = (
            await cleanup.execute(select(Product).where(Product.product_id == product_id))
        ).scalar_one_or_none()
        if product_row is not None:
            inspection_ids = list(
                (await cleanup.execute(select(Inspection.id).where(Inspection.product_id == product_row.id))).scalars()
            )
            if inspection_ids:
                await cleanup.execute(delete(Defect).where(Defect.inspection_id.in_(inspection_ids)))
                await cleanup.execute(delete(Inspection).where(Inspection.id.in_(inspection_ids)))
            await cleanup.execute(delete(Product).where(Product.id == product_row.id))
        await cleanup.execute(delete(QualityRule).where(QualityRule.rule_version == 999))
        await cleanup.commit()

    await engine.dispose()
