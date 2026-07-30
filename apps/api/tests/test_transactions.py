from uuid import uuid4

import pytest
from sqlalchemy import select

from app.db import SessionLocal
from app.enums import InspectionStatus
from app.models import Inspection


async def test_database_transaction_failure_rolls_back() -> None:
    inspection_id = uuid4()
    with pytest.raises(RuntimeError):
        async with SessionLocal() as session, session.begin():
            session.add(
                Inspection(
                    id=inspection_id,
                    idempotency_key="rollback-transaction",
                    product_code="ROLLBACK",
                    batch_code="TEST",
                    original_filename="part.png",
                    stored_filename="rollback.png",
                    content_type="image/png",
                    file_sha256="0" * 64,
                    status=InspectionStatus.QUEUED,
                )
            )
            await session.flush()
            raise RuntimeError("synthetic transaction failure")

    async with SessionLocal() as session:
        assert (
            await session.scalar(select(Inspection).where(Inspection.id == inspection_id)) is None
        )
