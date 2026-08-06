"""Model Registry API (Phase 8).

POST   /api/v1/models                    register a model
GET    /api/v1/models?status=&model_type= list
GET    /api/v1/models/production/{name}  current production for a model
GET    /api/v1/models/{id}               detail
POST   /api/v1/models/{id}/gate          dry-run promotion gate
POST   /api/v1/models/{id}/promote       run the gate; promote if passed
POST   /api/v1/models/{id}/archive
POST   /api/v1/models/rollback           switch production back to a version
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_session
from ..mlops.promotion_gate import evaluate
from ..services.registry_service import RegistryError, RegistryService, get_registry_service

router = APIRouter(prefix="/api/v1/models", tags=["models"])


def _err(code: str, message: str) -> dict:
    return {"code": code, "message": message}


class RegisterIn(BaseModel):
    model_name: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    model_type: str = Field(min_length=1)  # yolo | patchcore
    artifact_uri: str | None = None
    artifact_sha256: str | None = None
    dataset_version: str | None = None
    training_run_id: str | None = None
    metrics: dict | None = None
    domain_validated: bool = False
    notes: str | None = None


class PromoteIn(BaseModel):
    required_domain: str = "steel"
    thresholds: dict | None = None


class RollbackIn(BaseModel):
    model_name: str
    model_version: str


def _out(m) -> dict:
    return {
        "id": str(m.id),
        "model_name": m.model_name,
        "model_version": m.model_version,
        "model_type": m.model_type,
        "artifact_uri": m.artifact_uri,
        "artifact_sha256": m.artifact_sha256,
        "dataset_version": m.dataset_version,
        "training_run_id": m.training_run_id,
        "status": m.status,
        "metrics": m.metadata_json or {},
        "domain_validated": m.domain_validated,
        "promoted_at": m.promoted_at.isoformat() if m.promoted_at else None,
        "notes": m.notes,
        "created_at": m.created_at.isoformat() if m.created_at else None,
    }


@router.post("")
async def register(body: RegisterIn, session: AsyncSession = Depends(get_session)) -> dict:
    svc = get_registry_service()
    try:
        m = await svc.register(
            session, model_name=body.model_name, model_version=body.model_version,
            model_type=body.model_type, artifact_uri=body.artifact_uri,
            artifact_sha256=body.artifact_sha256, dataset_version=body.dataset_version,
            training_run_id=body.training_run_id, metrics=body.metrics,
            domain_validated=body.domain_validated, notes=body.notes,
        )
        await session.commit()
    except RegistryError as exc:
        raise HTTPException(status_code=409, detail=_err(exc.code, exc.message)) from exc
    return _out(m)


@router.get("")
async def list_models(status: str | None = None, model_type: str | None = None,
                      session: AsyncSession = Depends(get_session)) -> list[dict]:
    svc = get_registry_service()
    return [_out(m) for m in await svc.list(session, status=status, model_type=model_type)]


@router.get("/production/{model_name}")
async def production(model_name: str, session: AsyncSession = Depends(get_session)) -> dict:
    m = await get_registry_service().get_production(session, model_name)
    if m is None:
        raise HTTPException(status_code=404, detail=_err("no_production", f"no production model for {model_name}"))
    return _out(m)


@router.get("/{entry_id}")
async def detail(entry_id: uuid.UUID, session: AsyncSession = Depends(get_session)) -> dict:
    m = await get_registry_service().get(session, entry_id)
    if m is None:
        raise HTTPException(status_code=404, detail=_err("not_found", "model not found"))
    return _out(m)


@router.post("/{entry_id}/gate")
async def gate(entry_id: uuid.UUID, body: PromoteIn, session: AsyncSession = Depends(get_session)) -> dict:
    m = await get_registry_service().get(session, entry_id)
    if m is None:
        raise HTTPException(status_code=404, detail=_err("not_found", "model not found"))
    g = evaluate(
        m.model_type, metrics=m.metadata_json or {}, thresholds=body.thresholds,
        domain_validated=m.domain_validated, required_domain=body.required_domain,
    )
    return {"model": f"{m.model_name}@{m.model_version}", "gate": g.to_dict()}


@router.post("/{entry_id}/promote")
async def promote(entry_id: uuid.UUID, body: PromoteIn, session: AsyncSession = Depends(get_session)) -> dict:
    svc = get_registry_service()
    m = await svc.get(session, entry_id)
    if m is None:
        raise HTTPException(status_code=404, detail=_err("not_found", "model not found"))
    g = evaluate(
        m.model_type, metrics=m.metadata_json or {}, thresholds=body.thresholds,
        domain_validated=m.domain_validated, required_domain=body.required_domain,
    )
    try:
        m = await svc.promote(session, m, gate=g, required_domain=body.required_domain)
        await session.commit()
    except RegistryError as exc:
        raise HTTPException(status_code=422, detail=_err(exc.code, exc.message)) from exc
    return {**_out(m), "gate": g.to_dict()}


@router.post("/{entry_id}/archive")
async def archive(entry_id: uuid.UUID, session: AsyncSession = Depends(get_session)) -> dict:
    svc = get_registry_service()
    m = await svc.get(session, entry_id)
    if m is None:
        raise HTTPException(status_code=404, detail=_err("not_found", "model not found"))
    await svc.archive(session, m)
    await session.commit()
    return _out(m)


@router.post("/rollback")
async def rollback(body: RollbackIn, session: AsyncSession = Depends(get_session)) -> dict:
    svc = get_registry_service()
    try:
        m = await svc.rollback(session, body.model_name, body.model_version)
        await session.commit()
    except RegistryError as exc:
        raise HTTPException(status_code=404, detail=_err(exc.code, exc.message)) from exc
    return _out(m)


@router.delete("/{entry_id}")
async def delete(entry_id: uuid.UUID, session: AsyncSession = Depends(get_session)) -> dict:
    """Hard-delete a registry row (used by the E2E / test fixtures; the
    production lifecycle path uses archive/promote instead)."""
    from sqlalchemy import delete as sa_delete

    from ..models import ModelRegistry

    m = await get_registry_service().get(session, entry_id)
    if m is None:
        raise HTTPException(status_code=404, detail=_err("not_found", "model not found"))
    await session.execute(sa_delete(ModelRegistry).where(ModelRegistry.id == entry_id))
    await session.commit()
    return {"deleted": entry_id}
