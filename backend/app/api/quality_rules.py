from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_session
from ..models import QualityRule
from ..schemas import RuleCreate, RuleOut, RuleUpdate

router = APIRouter(prefix="/api/v1/quality-rules", tags=["quality-rules"])


@router.get("", response_model=list[RuleOut])
async def list_rules(
    enabled_only: bool = True,
    session: AsyncSession = Depends(get_session),
) -> list[QualityRule]:
    stmt = select(QualityRule).order_by(QualityRule.priority, QualityRule.defect_type)
    if enabled_only:
        stmt = stmt.where(QualityRule.enabled.is_(True))
    result = await session.execute(stmt)
    return list(result.scalars())


@router.post("", response_model=RuleOut, status_code=201)
async def create_rule(rule: RuleCreate, session: AsyncSession = Depends(get_session)) -> QualityRule:
    row = QualityRule(**rule.model_dump())
    session.add(row)
    await session.commit()
    await session.refresh(row)
    return row


@router.patch("/{rule_id}", response_model=RuleOut)
async def update_rule(rule_id: uuid.UUID, patch: RuleUpdate, session: AsyncSession = Depends(get_session)) -> QualityRule:
    result = await session.execute(select(QualityRule).where(QualityRule.id == rule_id))
    row = result.scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail=_err("not_found", "rule not found"))
    for field, value in patch.model_dump(exclude_unset=True).items():
        setattr(row, field, value)
    await session.commit()
    await session.refresh(row)
    return row


def _err(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message, "request_id": uuid.uuid4().hex[:12]}}
