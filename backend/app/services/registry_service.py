"""Model Registry service (8A / 8C).

Lifecycle: CANDIDATE -> STAGING -> PRODUCTION -> ARCHIVED. At most one
PRODUCTION row per model_name (enforced by a partial unique index). A model
can only be promoted when the promotion gate passes AND the domain is
validated; domain mismatch (e.g. MVTec PatchCore promoted to a steel
production model) is rejected. Rollback switches the registry pointer, it
never rebuilds the system (8L).
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ..mlops.promotion_gate import GateResult, evaluate
from ..models import ModelRegistry

logger = logging.getLogger(__name__)


class RegistryError(Exception):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


class RegistryService:
    async def register(
        self,
        session: AsyncSession,
        *,
        model_name: str,
        model_version: str,
        model_type: str,
        artifact_uri: str | None = None,
        artifact_sha256: str | None = None,
        dataset_version: str | None = None,
        training_run_id: str | None = None,
        metrics: dict | None = None,
        domain_validated: bool = False,
        notes: str | None = None,
    ) -> ModelRegistry:
        """Register a new model entry. Duplicate (name, version) is rejected."""
        dup = await session.execute(
            select(ModelRegistry).where(
                ModelRegistry.model_name == model_name,
                ModelRegistry.model_version == model_version,
            )
        )
        if dup.scalar_one_or_none() is not None:
            raise RegistryError("duplicate_version", f"{model_name}@{model_version} already registered")

        entry = ModelRegistry(
            model_name=model_name,
            model_version=model_version,
            model_type=model_type,
            artifact_uri=artifact_uri,
            artifact_sha256=artifact_sha256,
            dataset_version=dataset_version,
            training_run_id=training_run_id,
            status="CANDIDATE",
            metadata_json=metrics,
            domain_validated=bool(domain_validated),
            notes=notes,
        )
        session.add(entry)
        await session.flush()
        return entry

    async def list(self, session: AsyncSession, *, status: str | None = None,
                  model_type: str | None = None) -> list[ModelRegistry]:
        stmt = select(ModelRegistry).order_by(ModelRegistry.created_at.desc())
        if status:
            stmt = stmt.where(ModelRegistry.status == status)
        if model_type:
            stmt = stmt.where(ModelRegistry.model_type == model_type)
        return list((await session.execute(stmt)).scalars().all())

    async def get(self, session: AsyncSession, entry_id: uuid.UUID) -> ModelRegistry | None:
        return await session.get(ModelRegistry, entry_id)

    async def get_by_version(self, session: AsyncSession, model_name: str, model_version: str) -> ModelRegistry | None:
        r = await session.execute(
            select(ModelRegistry).where(
                ModelRegistry.model_name == model_name,
                ModelRegistry.model_version == model_version,
            )
        )
        return r.scalar_one_or_none()

    async def get_production(self, session: AsyncSession, model_name: str) -> ModelRegistry | None:
        r = await session.execute(
            select(ModelRegistry).where(
                ModelRegistry.model_name == model_name,
                ModelRegistry.status == "PRODUCTION",
            )
        )
        return r.scalar_one_or_none()

    async def _to_staging(self, session: AsyncSession, entry: ModelRegistry) -> None:
        if entry.status not in ("CANDIDATE", "STAGING", "PRODUCTION"):
            raise RegistryError("bad_state", f"cannot stage a model in {entry.status}")
        entry.status = "STAGING"

    async def promote(
        self,
        session: AsyncSession,
        entry: ModelRegistry,
        *,
        gate: GateResult,
        required_domain: str,
    ) -> ModelRegistry:
        """Promote a candidate to PRODUCTION. The gate decides; this method
        only applies the decision. If the gate failed, raise (8F boundary:
        promotion is NOT a manual click)."""
        if not gate.passed:
            raise RegistryError("promotion_gate_failed", "; ".join(gate.blocked or ["gate failed"]))
        if entry.status == "ARCHIVED":
            raise RegistryError("bad_state", "cannot promote an ARCHIVED model")
        if not entry.domain_validated:
            raise RegistryError("domain_mismatch", f"model not validated for domain {required_domain}")

        now = datetime.now(timezone.utc)
        # demote the current production (unique constraint would otherwise fire)
        await session.execute(
            update(ModelRegistry)
            .where(ModelRegistry.model_name == entry.model_name, ModelRegistry.status == "PRODUCTION")
            .values(status="ARCHIVED")
        )
        entry.status = "PRODUCTION"
        entry.promoted_at = now
        await session.flush()
        return entry

    async def rollback(
        self,
        session: AsyncSession,
        model_name: str,
        model_version: str,
    ) -> ModelRegistry:
        """Rollback: switch the production pointer back to `model_version`
        without rebuilding anything (8L). The previous production is archived."""
        target = await self.get_by_version(session, model_name, model_version)
        if target is None:
            raise RegistryError("not_found", f"{model_name}@{model_version} not in registry")
        if target.status == "ARCHIVED":
            target.status = "STAGING"
        now = datetime.now(timezone.utc)
        await session.execute(
            update(ModelRegistry)
            .where(ModelRegistry.model_name == model_name, ModelRegistry.status == "PRODUCTION")
            .values(status="ARCHIVED")
        )
        target.status = "PRODUCTION"
        target.promoted_at = now
        await session.flush()
        return target

    async def archive(self, session: AsyncSession, entry: ModelRegistry) -> ModelRegistry:
        entry.status = "ARCHIVED"
        await session.flush()
        return entry


def get_registry_service() -> RegistryService:
    return RegistryService()
