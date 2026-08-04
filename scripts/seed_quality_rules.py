"""Seed default quality rules (idempotent).

Run from repo root with the venv:
    .venv/Scripts/python.exe scripts/seed_quality_rules.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from sqlalchemy import select  # noqa: E402
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.database import SessionLocal  # noqa: E402
from app.enums import QualityResult, Severity  # noqa: E402
from app.models import QualityRule  # noqa: E402

DEFAULT_RULES: list[dict] = [
    # critical classes -> FAIL
    {"defect_type": "inclusion", "min_confidence": 0.7, "max_area_ratio": 1.0, "action": QualityResult.FAIL, "severity": Severity.HIGH, "priority": 5},
    {"defect_type": "patches", "min_confidence": 0.7, "max_area_ratio": 1.0, "action": QualityResult.FAIL, "severity": Severity.HIGH, "priority": 6},
    {"defect_type": "pitted_surface", "min_confidence": 0.7, "max_area_ratio": 1.0, "action": QualityResult.FAIL, "severity": Severity.HIGH, "priority": 7},
    # allowed scratch below area threshold -> PASS, above -> REVIEW
    {"defect_type": "scratches", "min_confidence": 0.6, "max_area_ratio": 0.3, "action": QualityResult.PASS, "severity": Severity.LOW, "priority": 10},
    {"defect_type": "scratches", "min_confidence": 0.6, "max_area_ratio": 1.0, "action": QualityResult.REVIEW, "severity": Severity.MEDIUM, "priority": 20},
    # weak / hard classes -> REVIEW
    {"defect_type": "crazing", "min_confidence": 0.3, "max_area_ratio": 1.0, "action": QualityResult.REVIEW, "severity": Severity.MEDIUM, "priority": 30},
    {"defect_type": "rolled-in_scale", "min_confidence": 0.7, "max_area_ratio": 1.0, "action": QualityResult.REVIEW, "severity": Severity.MEDIUM, "priority": 25},
    # unknown class with very high confidence -> REVIEW (not auto-FAIL)
    {"defect_type": "*", "min_confidence": 0.95, "max_area_ratio": 1.0, "action": QualityResult.REVIEW, "severity": Severity.LOW, "priority": 100},
]


async def main() -> None:
    async with SessionLocal() as session:
        # Idempotent upsert keyed on the unique business key
        # (defect_type, priority, rule_version) from migration 0002.
        inserted = 0
        for spec in DEFAULT_RULES:
            values = {**spec, "rule_version": 1, "enabled": True}
            stmt = pg_insert(QualityRule).values(**values)
            stmt = stmt.on_conflict_do_nothing(
                index_elements=["defect_type", "priority", "rule_version"]
            )
            result = await session.execute(stmt)
            inserted += result.rowcount or 0
        await session.commit()
        print(f"seeded {inserted} new rules (existing kept, total unaffected)")


if __name__ == "__main__":
    asyncio.run(main())
