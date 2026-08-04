"""Real end-to-end test: backend -> inference service (HTTP, real model) -> PostgreSQL.

Marked ``integration``. Requires:
- the inference service running (default http://127.0.0.1:8100)
- PostgreSQL reachable (default postgresql+asyncpg://vision_qc:vision_qc@127.0.0.1:5432/vision_qc)

Running Alembic against an empty schema is part of this test, which doubles as
the Phase 2 gate check "migration from empty DB builds all tables".
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

pytestmark = pytest.mark.integration

BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REAL_IMAGE = PROJECT_ROOT / "model-training/datasets/neu-det-yolo/test/images/crazing_101.jpg"

DB_URL = os.environ.get("IVQC_DATABASE_URL", "postgresql+asyncpg://vision_qc:vision_qc@127.0.0.1:5432/vision_qc")
INF_URL = os.environ.get("IVQC_INFERENCE_SERVICE_URL", "http://127.0.0.1:8100")


async def _http_up(url: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            return (await client.get(f"{url}/health")).status_code == 200
    except Exception:
        return False


async def _db_up(url: str) -> bool:
    try:
        engine = create_async_engine(url)
        async with engine.connect() as conn:
            await conn.execute(__import__("sqlalchemy").text("SELECT 1"))
        await engine.dispose()
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def services_ready():
    if not asyncio.run(_http_up(INF_URL)):
        pytest.skip("inference service not reachable, run it first")
    if not asyncio.run(_db_up(DB_URL)):
        pytest.skip("postgres not reachable, start docker compose postgres first")
    if not REAL_IMAGE.exists():
        pytest.skip("real test image missing")
    env = {**os.environ, "IVQC_DATABASE_URL": DB_URL, "IVQC_INFERENCE_SERVICE_URL": INF_URL}
    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=BACKEND_ROOT,
        env=env,
        check=True,
        capture_output=True,
    )
    return env


async def test_end_to_end_inspection_flow(services_ready):
    engine = create_async_engine(DB_URL)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        from app.api.inspections import get_inspection_service
        from app.database import get_session
        from app.enums import QualityResult, Severity
        from app.inference.client import InferenceClient
        from app.main import create_app
        from app.models import Defect, Inspection, Product, QualityRule
        from app.services.inspection_service import InspectionService

        for rule in (
            QualityRule(defect_type="scratches", min_confidence=0.6, max_area_ratio=0.3, action=QualityResult.PASS, severity=Severity.LOW, priority=10),
            QualityRule(defect_type="crazing", min_confidence=0.3, max_area_ratio=1.0, action=QualityResult.REVIEW, severity=Severity.MEDIUM, priority=20),
            QualityRule(defect_type="*", min_confidence=0.95, max_area_ratio=0.1, action=QualityResult.REVIEW, severity=Severity.LOW, priority=30),
        ):
            session.add(rule)
        await session.commit()

        app = create_app()

        async def override_get_session():
            async with factory() as s:
                yield s

        def override_service():
            return InspectionService(inference_client=InferenceClient(base_url=INF_URL))

        app.dependency_overrides[get_session] = override_get_session
        app.dependency_overrides[get_inspection_service] = override_service

        product_id = "E2E-PROD-0001"
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            image = REAL_IMAGE.read_bytes()
            resp = await client.post(
                "/api/v1/inspections",
                files={"file": ("crazing_101.jpg", image, "image/jpeg")},
                data={"product_id": product_id, "production_line": "line-a", "station": "qc-01"},
                timeout=60,
            )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["product_id"] == product_id
        assert body["status"] == "completed"
        assert body["quality_result"] in {"PASS", "REVIEW", "FAIL"}
        assert body["model_version"] == "phase1-baseline"

        inspection_id = body["inspection_id"]
        fetched = await client.get(f"/api/v1/inspections/{inspection_id}")
        assert fetched.status_code == 200
        detail = fetched.json()
        assert detail["quality_result"] == body["quality_result"]

        history = await client.get(f"/api/v1/products/{product_id}/inspections")
        assert history.status_code == 200
        assert [i["inspection_id"] for i in history.json()] == [inspection_id]

        async with factory() as cleanup:
            inspection_ids = [
                i.id for i in (await cleanup.execute(__import__("sqlalchemy").select(Inspection))).scalars()
                if i.product.product_id == product_id
            ]
            if inspection_ids:
                await cleanup.execute(delete(Defect).where(Defect.inspection_id.in_(inspection_ids)))
                await cleanup.execute(delete(Inspection).where(Inspection.id.in_(inspection_ids)))
            await cleanup.execute(delete(Product).where(Product.product_id == product_id))
            await cleanup.commit()

    await engine.dispose()
