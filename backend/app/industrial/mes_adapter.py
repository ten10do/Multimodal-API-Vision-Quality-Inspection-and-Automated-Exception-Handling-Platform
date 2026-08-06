"""MES Adapter (Phase 7).

All MES HTTP traffic lives here (no scattered requests in business code).
Handles success / timeout / 4xx / 5xx / retry and duplicate submission.
Submissions are idempotent keyed by (inspection_id, kind); retries reuse the
same key so the MES never double-records.
"""

from __future__ import annotations

import logging
import time
from typing import Literal

import httpx

logger = logging.getLogger(__name__)

MesSyncStatus = Literal["PENDING", "SYNCED", "FAILED"]


class MesError(Exception):
    """Base MES failure."""


class MesUnreachable(MesError):
    pass


class MesRejected(MesError):
    pass


class MesAdapter:
    def __init__(self, base_url: str, timeout_seconds: float = 2.0, max_retries: int = 2) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout_seconds
        self.max_retries = max_retries

    async def _post(self, path: str, payload: dict) -> tuple[bool, float, int]:
        """POST with bounded retry. Returns (ok, latency_ms, attempts)."""
        attempts = 0
        for attempt in range(self.max_retries + 1):
            attempts += 1
            started = time.perf_counter()
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(f"{self.base_url}{path}", json=payload)
                latency = (time.perf_counter() - started) * 1000.0
                if resp.status_code == 200:
                    return True, round(latency, 2), attempts
                if 400 <= resp.status_code < 500:
                    # 4xx is not retryable (bad payload), except 408/429
                    if resp.status_code not in (408, 429):
                        raise MesRejected(f"mes http {resp.status_code}: {resp.text[:120]}")
            except httpx.TimeoutException:
                latency = (time.perf_counter() - started) * 1000.0
                logger.warning("mes timeout attempt=%d", attempt + 1)
                continue
            except httpx.HTTPError as exc:
                latency = (time.perf_counter() - started) * 1000.0
                logger.warning("mes unreachable attempt=%d: %s", attempt + 1, exc)
                continue
        raise MesUnreachable(f"mes unreachable after {attempts} attempts")

    async def post_inspection_result(
        self, *, inspection_id: str, product_id: str, batch_id: str | None, line: str, station: str,
        ai_result: str | None, model_version: str | None, rule_version: int | None,
        industrial_state: str | None, timestamp: str,
    ) -> tuple[bool, float, int]:
        ok, latency, attempts = await self._post(
            "/v1/inspection-results",
            {
                "inspection_id": inspection_id,
                "product_id": product_id,
                "batch_id": batch_id,
                "line": line,
                "station": station,
                "ai_result": ai_result,
                "model_version": model_version,
                "rule_version": rule_version,
                "industrial_state": industrial_state,
                "timestamp": timestamp,
            },
        )
        return ok, latency, attempts

    async def post_final_quality_result(
        self, *, inspection_id: str, product_id: str, batch_id: str | None, final_result: str,
        reviewed_by: str | None, industrial_state: str | None, timestamp: str,
    ) -> tuple[bool, float, int]:
        ok, latency, attempts = await self._post(
            "/v1/final-quality-results",
            {
                "inspection_id": inspection_id,
                "product_id": product_id,
                "batch_id": batch_id,
                "final_result": final_result,
                "reviewed_by": reviewed_by,
                "industrial_state": industrial_state,
                "timestamp": timestamp,
            },
        )
        return ok, latency, attempts


def get_mes_adapter() -> MesAdapter:
    from ..config import get_settings

    settings = get_settings()
    return MesAdapter(
        base_url=settings.mes_url,
        timeout_seconds=settings.mes_timeout_seconds,
        max_retries=settings.mes_max_retries,
    )
