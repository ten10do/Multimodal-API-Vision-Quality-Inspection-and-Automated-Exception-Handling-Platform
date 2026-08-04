from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.database import get_session  # noqa: E402
from app.main import create_app  # noqa: E402
from app.models import Base  # noqa: E402


@pytest_asyncio.fixture
async def db_session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        yield session
    await engine.dispose()


@pytest_asyncio.fixture
async def app():
    application = create_app()
    yield application
    application.dependency_overrides.clear()


@pytest_asyncio.fixture
async def client(app, db_session):
    async def override_get_session():
        yield db_session

    app.dependency_overrides[get_session] = override_get_session
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def stub_infer(app):
    """Install a stub inference client through dependency overrides."""

    def _install(stub):
        from app.api.inspections import get_inspection_service
        from app.services.inspection_service import InspectionService

        app.dependency_overrides[get_inspection_service] = lambda: InspectionService(inference_client=stub)
        return stub

    return _install
