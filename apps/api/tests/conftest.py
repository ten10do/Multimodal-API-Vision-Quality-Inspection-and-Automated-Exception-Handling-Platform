import os
from collections.abc import AsyncIterator
from pathlib import Path

import pytest_asyncio
from httpx import ASGITransport, AsyncClient

TEST_ROOT = Path(__file__).parent / ".runtime"
TEST_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("AI_MODE", "mock")
os.environ.setdefault("CELERY_TASK_ALWAYS_EAGER", "true")
os.environ.setdefault("DATABASE_URL", f"sqlite+aiosqlite:///{TEST_ROOT / 'test.db'}")
os.environ.setdefault("UPLOAD_DIR", str(TEST_ROOT / "uploads"))

from app.db import Base, engine  # noqa: E402
from app.main import app  # noqa: E402


@pytest_asyncio.fixture(autouse=True)
async def reset_database() -> AsyncIterator[None]:
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)
        await connection.run_sync(Base.metadata.create_all)
    yield


@pytest_asyncio.fixture
async def client() -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as test_client:
        yield test_client
