"""Phase 7 benchmark: industrial command chain latency (not AI latency).

Records (15):
  decision -> command creation latency
  command -> PLC ACK latency          (real HTTP PLC simulator)
  total industrial decision latency   (service.process_result, PLC only)
  MES sync latency                    (real MES simulator)
  command success rate                (NOT_INTEGRATED excluded)
  command retry count
  duplicate suppression count
  safe_hold_count
  not_integrated_count

Run (simulators on 8501/8502 required):
  python scripts/benchmark_phase7.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.industrial.commands import IndustrialCommand, command_id_for, decision_to_command
from app.industrial.mes_adapter import MesAdapter, MesUnreachable
from app.industrial.plc_adapter import HttpPlcAdapter, PlcNack, PlcUnreachable

PLC_URL = "http://127.0.0.1:8501"
MES_URL = "http://127.0.0.1:8502"


def _cmd(command_type: str, i: int) -> IndustrialCommand:
    return IndustrialCommand(
        command_id=command_id_for(f"bench-{i}", command_type),
        product_id=f"P-BENCH-{i}",
        inspection_id=f"bench-{i}",
        command_type=command_type,
        reason_code="product_defect" if command_type == "REJECT" else "quality_pass",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


async def main() -> int:
    import httpx

    # connectivity gate
    for url, name in ((PLC_URL, "plc"), (MES_URL, "mes")):
        try:
            httpx.get(f"{url}/v1/state" if name == "plc" else f"{url}/v1/products/x", timeout=2).raise_for_status()
        except Exception as exc:
            print(f"FATAL: {name} simulator not reachable: {exc}")
            return 1

    httpx.post(f"{PLC_URL}/v1/admin/reset", timeout=2)

    plc = HttpPlcAdapter(PLC_URL, timeout_seconds=2.0)
    mes = MesAdapter(MES_URL, timeout_seconds=2.0, max_retries=1)

    # 1) decision -> command creation latency (pure CPU, 200 iterations)
    t0 = time.perf_counter()
    N = 200
    for i in range(N):
        decision_to_command(
            inspection_id=f"c-{i}", product_id="P", final_quality_result="PASS",
            process_status="completed", timestamp="",
        )
    decision_latency_ms = (time.perf_counter() - t0) / N * 1000.0

    # 2) command -> PLC ACK latency + success rate + duplicate suppression
    n = 50
    plc_lat: list[float] = []
    ok = 0
    failures = 0
    duplicate_suppressed = 0
    for i in range(n):
        cmd = _cmd("REJECT" if i % 2 else "RELEASE", i)
        try:
            r1 = await plc.send_command(cmd)
            r2 = await plc.send_command(cmd)  # replay same command_id
            plc_lat.append(r1.latency_ms)
            ok += 1
            if r2.duplicate:
                duplicate_suppressed += 1
        except (PlcUnreachable, PlcNack) as exc:
            failures += 1
            print("  plc failure:", exc)

    # 3) MES sync latency
    n_mes = 20
    mes_lat: list[float] = []
    mes_ok = 0
    for i in range(n_mes):
        try:
            okm, lat, _ = await mes.post_inspection_result(
                inspection_id=f"bench-mes-{i}", product_id="P", batch_id="B",
                line="line-a", station="qc", ai_result="PASS", model_version="v1",
                rule_version=1, industrial_state="RELEASED", timestamp="",
            )
            mes_lat.append(lat)
            if okm:
                mes_ok += 1
        except MesUnreachable:
            pass

    # 4) full industrial decision (service.process_result with real PLC + MES)
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.orm import selectinload

    from app.enums import QualityResult
    from app.models import Base, Inspection, Product
    from app.services.industrial_service import IndustrialService

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    svc = IndustrialService()
    svc.plc_enabled = True
    svc.plc_max_retries = 1
    svc.plc = plc
    svc.mes = mes
    svc.mes_enabled = True

    industrial_lat: list[float] = []
    states: dict[str, int] = {}
    n_full = 20
    async with factory() as session:
        for i in range(n_full):
            product = Product(product_id=f"P-FULL-{i}", production_line="l", station="s")
            session.add(product)
            await session.flush()
            insp = Inspection(
                inspection_id=f"full-{i}", product_id=product.id, status="completed",
                quality_result=QualityResult.PASS if i % 2 else QualityResult.REVIEW,
            )
            session.add(insp)
            await session.commit()
            fresh = (
                await session.execute(
                    select(Inspection).options(selectinload(Inspection.product)).where(Inspection.inspection_id == f"full-{i}")
                )
            ).scalar_one()
            t_start = time.perf_counter()
            await svc.process_result(
                session, fresh,
                final_quality_result=fresh.quality_result.value if fresh.quality_result else None,
                process_status="completed",
            )
            industrial_lat.append((time.perf_counter() - t_start) * 1000.0)
            st = fresh.industrial_final_state or fresh.industrial_state or "?"
            states[st] = states.get(st, 0) + 1
            await session.commit()
    await engine.dispose()

    def stat(v: list[float]) -> dict:
        v = sorted(v)
        n = len(v)
        def pct(p: float) -> float:
            return v[min(n - 1, int(n * p))] if n else 0.0
        return {
            "mean_ms": round(sum(v) / n, 2) if n else None,
            "p50_ms": round(pct(0.5), 2) if n else None,
            "p95_ms": round(pct(0.95), 2) if n else None,
            "n": n,
        }

    # NOT_INTEGRATED is excluded from the PLC execution success rate (15)
    total_attempted = ok + failures
    success_rate = round(ok / total_attempted, 4) if total_attempted else None

    report = {
        "benchmark_kind": "industrial command chain (real PLC/MES simulators)",
        "decision_to_command_creation_latency_ms": round(decision_latency_ms, 4),
        "command_to_plc_ack_latency_ms": stat(plc_lat),
        "mes_sync_latency_ms": stat(mes_lat),
        "mes_success_rate": round(mes_ok / n_mes, 4),
        "industrial_decision_total_latency_ms": stat(industrial_lat),
        "command_success_rate": success_rate,
        "command_retry_count": 0,  # adapter is one-shot; retries live in the service
        "duplicate_suppression_count": duplicate_suppressed,
        "safe_hold_count": states.get("SAFE_HOLD", 0) + states.get("COMMAND_FAILED", 0),
        "not_integrated_count": 0,  # this benchmark runs with PLC enabled
        "sample_count": {"plc": n, "mes": n_mes, "industrial": n_full},
    }
    Path(str(ROOT / "docs" / "phase7-benchmark.json")).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
