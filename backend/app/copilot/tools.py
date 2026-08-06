"""Copilot tool registry (9D): a fixed allowlist of read-only analytics
tools. Every tool has an explicit input schema, a time window, a result cap,
a timeout and error handling. There is deliberately NO arbitrary-SQL tool:
the LLM only ever sees these controlled schemas and can never issue writes
(9A / 9D).
"""

from __future__ import annotations

import asyncio
import statistics
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..enums import QualityResult
from ..models import Defect, Inspection, PlcEvent, Product, ReviewDecision, ReviewTask
from .llm import ToolCall

DEFAULT_WINDOW_DAYS = 1
MAX_RESULTS_DEFAULT = 50
TOOL_TIMEOUT_SECONDS = 8.0


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _resolve_window(args: dict, default_days: int = DEFAULT_WINDOW_DAYS) -> tuple[datetime, datetime, str]:
    """Explicit time window: current = [now - window, now], or caller-provided
    from/to. The label is part of every result so the answer always states the
    window (9E)."""
    now = _utcnow()
    to = _parse_iso(args.get("time_to")) or now
    fr = _parse_iso(args.get("time_from")) or (to - timedelta(days=default_days))
    label = f"{fr.isoformat()} → {to.isoformat()}"
    return fr, to, label


def _cap(items: list[Any], args: dict, default: int = MAX_RESULTS_DEFAULT) -> list[Any]:
    return items[: max(1, int(args.get("max_results", default)))]


def _out(window: str, **extra: Any) -> dict:
    return {"time_window": window, **extra}


class CopilotTool:
    def __init__(
        self,
        *,
        name: str,
        description: str,
        parameters: dict,
        handler: Callable[[AsyncSession, dict], Awaitable[dict]],
        timeout: float = TOOL_TIMEOUT_SECONDS,
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters
        self.handler = handler
        self.timeout = timeout

    def schema(self) -> dict:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


class ToolError(Exception):
    pass


# ---------------------------------------------------------------------------
# tool implementations (read-only; queries are predefined, LLM never writes)
# ---------------------------------------------------------------------------

async def _quality_summary(session: AsyncSession, args: dict) -> dict:
    fr, to, win = _resolve_window(args)
    stmt = select(Inspection).where(Inspection.created_at >= fr, Inspection.created_at <= to)
    if args.get("line"):
        stmt = stmt.join(Inspection.product).where(Product.production_line == args["line"])
    rows = list((await session.execute(stmt)).scalars().all())
    completed = [i for i in rows if i.status == "completed"]
    failed = [i for i in rows if i.status == "failed"]
    passed = [i for i in completed if i.quality_result == QualityResult.PASS]
    failed_q = [i for i in completed if i.quality_result == QualityResult.FAIL]
    review = [i for i in completed if i.quality_result == QualityResult.REVIEW]
    return _out(
        win,
        inspected=len(rows),
        completed=len(completed),
        pass_count=len(passed),
        fail_count=len(failed_q),
        review_count=len(review),
        system_failed=len(failed),
        yield_rate=round(len(passed) / max(1, len(completed)), 4),
        review_rate=round(len(review) / max(1, len(completed)), 4),
    )


async def _yield_trend(session: AsyncSession, args: dict) -> dict:
    days = max(1, min(30, int(args.get("days", 7))))
    fr = _utcnow() - timedelta(days=days)
    stmt = select(Inspection).where(Inspection.created_at >= fr)
    if args.get("line"):
        stmt = stmt.join(Inspection.product).where(Product.production_line == args["line"])
    rows = list((await session.execute(stmt)).scalars().all())
    per_day: dict[str, list[Inspection]] = {}
    for i in rows:
        day = i.created_at.date().isoformat()
        per_day.setdefault(day, []).append(i)
    trend = []
    for day in sorted(per_day):
        comp = [i for i in per_day[day] if i.status == "completed"]
        passed = len([i for i in comp if i.quality_result == QualityResult.PASS])
        trend.append(
            {
                "date": day,
                "inspected": len(per_day[day]),
                "yield_rate": round(passed / max(1, len(comp)), 4),
                "pass": passed,
            }
        )
    return _out(f"{fr.isoformat()} → {_utcnow().isoformat()}", points=_cap(trend, args, 60), days=days)


async def _defect_distribution(session: AsyncSession, args: dict) -> dict:
    fr, to, win = _resolve_window(args)
    stmt = (
        select(Defect.class_name, func.count())
        .join(Inspection, Inspection.id == Defect.inspection_id)
        .where(Inspection.created_at >= fr, Inspection.created_at <= to)
        .group_by(Defect.class_name)
    )
    if args.get("line"):
        stmt = stmt.join(Inspection.product).where(Product.production_line == args["line"])
    counts = dict((await session.execute(stmt)).all())
    total = sum(counts.values()) or 1
    dist = {k: {"count": v, "share": round(v / total, 4)} for k, v in sorted(counts.items(), key=lambda x: -x[1])}
    return _out(win, total_defects=sum(counts.values()), distribution=_cap([{"defect": k, **v} for k, v in dist.items()], args, 30))


async def _defect_trend(session: AsyncSession, args: dict) -> dict:
    days = max(1, min(30, int(args.get("days", 7))))
    defect = args.get("defect_type")
    fr = _utcnow() - timedelta(days=days)
    stmt = (
        select(Inspection.created_at, Defect.class_name)
        .join(Defect, Defect.inspection_id == Inspection.id)
        .where(Inspection.created_at >= fr)
    )
    if defect:
        stmt = stmt.where(Defect.class_name == defect)
    rows = list((await session.execute(stmt)).all())
    per_day: dict[str, int] = {}
    for created, cls in rows:
        key = created.date().isoformat() if isinstance(created, datetime) else str(created)
        per_day[key] = per_day.get(key, 0) + 1
    series = [{"date": d, "count": per_day[d]} for d in sorted(per_day)]
    return _out(
        f"{fr.isoformat()} → {_utcnow().isoformat()}",
        defect_type=defect or "all",
        series=_cap(series, args, 60),
        days=days,
    )


async def _compare_lines(session: AsyncSession, args: dict) -> dict:
    fr, to, win = _resolve_window(args)
    rows = (
        await session.execute(
            select(Inspection, Product)
            .join(Product, Product.id == Inspection.product_id)
            .where(Inspection.created_at >= fr, Inspection.created_at <= to)
        )
    ).all()
    by_line: dict[str, list[Inspection]] = {}
    for insp, prod in rows:
        by_line.setdefault(prod.production_line, []).append(insp)
    out = []
    for line, items in by_line.items():
        comp = [i for i in items if i.status == "completed"]
        passed = len([i for i in comp if i.quality_result == QualityResult.PASS])
        review = len([i for i in comp if i.quality_result == QualityResult.REVIEW])
        out.append(
            {
                "line": line,
                "inspected": len(items),
                "yield_rate": round(passed / max(1, len(comp)), 4),
                "review_rate": round(review / max(1, len(comp)), 4),
                "pass": passed,
            }
        )
    return _out(win, lines=_cap(sorted(out, key=lambda x: -x["inspected"]), args, 20))


async def _batch_quality(session: AsyncSession, args: dict) -> dict:
    batch = args.get("batch_id")
    if not batch:
        raise ToolError("batch_id is required")
    stmt = select(Inspection).where(Inspection.batch_id == batch)
    rows = list((await session.execute(stmt)).scalars().all())
    if not rows:
        raise ToolError(f"no inspections found for batch {batch}")
    comp = [i for i in rows if i.status == "completed"]
    passed = len([i for i in comp if i.quality_result == QualityResult.PASS])
    defects: dict[str, int] = {}
    for i in rows:
        for d in i.defects:
            defects[d.class_name] = defects.get(d.class_name, 0) + 1
    lines = {p.production_line for p in (r.product for r in rows if r.product)}
    stations = {p.station for p in (r.product for r in rows if r.product)}
    models = {r.model_version for r in rows if r.model_version}
    rules = {r.rule_version for r in rows if r.rule_version is not None}
    anomaly = [r.anomaly_score for r in rows if r.anomaly_score is not None]
    return _out(
        "batch-scope (all time)",
        batch_id=batch,
        inspected=len(rows),
        yield_rate=round(passed / max(1, len(comp)), 4),
        defect_mix=dict(sorted(defects.items(), key=lambda x: -x[1])),
        lines=sorted(lines), stations=sorted(stations),
        model_versions=sorted(models), rule_versions=sorted(rules),
        anomaly_scores=[round(a, 4) for a in anomaly[:20]],
        review_count=len([i for i in comp if i.quality_result == QualityResult.REVIEW]),
    )


async def _inspection_detail(session: AsyncSession, args: dict) -> dict:
    iid = args.get("inspection_id")
    if not iid:
        raise ToolError("inspection_id is required")
    stmt = (
        select(Inspection)
        .options(
            selectinload(Inspection.product),
            selectinload(Inspection.defects),
        )
        .where(Inspection.inspection_id == iid)
    )
    insp = (await session.execute(stmt)).scalar_one_or_none()
    if insp is None:
        raise ToolError(f"inspection {iid} not found")
    prod = insp.product
    # review task is not a backref on Inspection; resolve it explicitly
    task = (
        await session.execute(select(ReviewTask).where(ReviewTask.inspection_id == insp.id))
    ).scalar_one_or_none()
    decision = (
        await session.execute(select(ReviewDecision).where(ReviewDecision.review_task_id == task.id))
    ).scalar_one_or_none() if task else None
    events = list(
        (await session.execute(select(PlcEvent).where(PlcEvent.inspection_id == iid).order_by(PlcEvent.created_at))).scalars().all()
    )
    return _out(
        "point-in-time",
        inspection_id=iid,
        product_id=prod.product_id if prod else None,
        production_line=prod.production_line if prod else None,
        station=prod.station if prod else None,
        batch_id=insp.batch_id,
        status=insp.status.value if hasattr(insp.status, "value") else str(insp.status),
        quality_result=insp.quality_result.value if insp.quality_result else None,
        final_quality_result=insp.final_quality_result.value if insp.final_quality_result else None,
        model_name=insp.model_name,
        model_version=insp.model_version,
        deployment_version=insp.deployment_version,
        rule_version=insp.rule_version,
        anomaly_score=round(insp.anomaly_score, 4) if insp.anomaly_score is not None else None,
        inference_latency_ms=insp.inference_latency_ms,
        defects=[{"class": d.class_name, "confidence": round(float(d.confidence), 4), "bbox_xyxy": d.bbox_xyxy} for d in insp.defects],
        human_review={
            "status": task.status.value if task else None,
            "human_decision": decision.human_decision.value if decision else None,
            "human_label": decision.human_label if decision else None,
            "reason": decision.reason if decision else None,
        } if (task or decision) else None,
        desired_command=insp.desired_command,
        execution_status=insp.execution_status,
        industrial_final_state=insp.industrial_final_state,
        plc_adapter_type=insp.plc_adapter_type,
        plc_reason_code=insp.plc_reason_code,
        mes_sync_status=insp.mes_sync_status,
        plc_events=[
            {
                "command": e.command,
                "execution_status": e.execution_status,
                "industrial_state": e.industrial_state,
                "adapter_type": e.adapter_type,
                "status": e.status,
                "created_at": e.created_at.isoformat() if e.created_at else None,
            }
            for e in events
        ],
    )


async def _product_history(session: AsyncSession, args: dict) -> dict:
    pid = args.get("product_id")
    if not pid:
        raise ToolError("product_id is required")
    stmt = select(Inspection).join(Inspection.product).where(Product.product_id == pid).order_by(Inspection.created_at.desc())
    rows = list((await session.execute(stmt)).scalars().all())
    if not rows:
        raise ToolError(f"no inspections found for product {pid}")
    return _out(
        "product-scope (all time)",
        product_id=pid,
        inspections=_cap(
            [
                {
                    "inspection_id": i.inspection_id,
                    "created_at": i.created_at.isoformat() if i.created_at else None,
                    "status": i.status.value if hasattr(i.status, "value") else str(i.status),
                    "quality_result": i.quality_result.value if i.quality_result else None,
                    "final_quality_result": i.final_quality_result.value if i.final_quality_result else None,
                    "industrial_final_state": i.industrial_final_state,
                }
                for i in rows
            ],
            args,
            50,
        ),
    )


async def _review_metrics(session: AsyncSession, args: dict) -> dict:
    fr, to, win = _resolve_window(args, default_days=7)
    stmt = (
        select(ReviewDecision, Inspection)
        .join(Inspection, Inspection.id == ReviewDecision.inspection_id)
        .where(ReviewDecision.created_at >= fr, ReviewDecision.created_at <= to)
    )
    if args.get("model_version"):
        stmt = stmt.where(Inspection.model_version == args["model_version"])
    rows = (await session.execute(stmt)).all()
    resolved = len(rows)
    confirm = sum(1 for d, _ in rows if d.human_decision.value == "CONFIRM_DEFECT")
    pass_ov = sum(1 for d, _ in rows if d.human_decision.value == "PASS")
    corrected = sum(1 for d, _ in rows if d.human_decision.value in ("CORRECT_DEFECT", "OTHER_DEFECT"))
    agree = 0
    for d, _ in rows:
        if d.human_decision.value != "CONFIRM_DEFECT":
            continue
        snapshot = d.ai_defects_snapshot or []
        top = snapshot[0].get("class_name") if snapshot else None
        if d.human_label and top and d.human_label == top:
            agree += 1
    return _out(
        win,
        resolved=resolved,
        defect_confirmation_rate=round(confirm / max(1, resolved), 4),
        ai_human_label_agreement_rate=round(agree / max(1, resolved), 4),
        pass_override_rate=round(pass_ov / max(1, resolved), 4),
        corrected_label_rate=round(corrected / max(1, resolved), 4),
    )


async def _review_backlog(session: AsyncSession, args: dict) -> dict:
    stmt = select(ReviewTask).where(ReviewTask.status == "PENDING").order_by(ReviewTask.created_at.asc())
    tasks = list((await session.execute(stmt)).scalars().all())
    return _out(
        "now",
        backlog_count=len(tasks),
        oldest_first=_cap(
            [
                {
                    "review_task_id": str(t.id),
                    "inspection_id": str(t.inspection_id),
                    "production_line": t.production_line,
                    "station": t.station,
                    "batch_id": t.batch_id,
                    "created_at": t.created_at.isoformat() if t.created_at else None,
                }
                for t in tasks
            ],
            args,
            30,
        ),
    )


async def _model_metrics(session: AsyncSession, args: dict) -> dict:
    fr, to, win = _resolve_window(args, default_days=7)
    stmt = select(Inspection).where(Inspection.created_at >= fr, Inspection.created_at <= to)
    if args.get("model_version"):
        stmt = stmt.where(Inspection.model_version == args["model_version"])
    rows = list((await session.execute(stmt)).scalars().all())
    comp = [i for i in rows if i.status == "completed"]
    failed = [i for i in rows if i.status == "failed"]
    lat = [i.inference_latency_ms for i in comp if i.inference_latency_ms is not None]
    lat_sorted = sorted(lat)
    p95 = lat_sorted[int(len(lat_sorted) * 0.95) - 1] if lat_sorted else None
    review = [i for i in comp if i.quality_result == QualityResult.REVIEW]
    return _out(
        win,
        model_version=args.get("model_version"),
        inference_count=len(comp),
        error_count=len(failed),
        error_rate=round(len(failed) / max(1, len(comp) + len(failed)), 4),
        inference_latency_avg_ms=round(statistics.fmean(lat), 2) if lat else None,
        inference_latency_p95_ms=round(p95, 2) if p95 else None,
        review_rate=round(len(review) / max(1, len(comp)), 4),
    )


async def _drift_status(session: AsyncSession, args: dict) -> dict:
    from ..mlops.drift import classify_psi, defect_distribution_delta, psi, review_rate_delta

    baseline_days = max(1, int(args.get("baseline_days", 7)))
    now = _utcnow()
    baseline_start = now - timedelta(days=baseline_days * 2)
    cutoff = now - timedelta(days=baseline_days)

    async def fetch(fr, to):
        stmt = (
            select(Inspection)
            .options(selectinload(Inspection.defects))
            .where(Inspection.created_at >= fr, Inspection.created_at <= to)
            .limit(200)
        )
        if args.get("model_version"):
            stmt = stmt.where(Inspection.model_version == args["model_version"])
        return list((await session.execute(stmt)).scalars().all())

    baseline = await fetch(baseline_start, cutoff)
    current = await fetch(cutoff, now)

    def confs(items):
        return [float(i.defects[0].confidence) for i in items if i.defects]

    b_conf, c_conf = confs(baseline), confs(current)
    psi_val = psi(b_conf, c_conf) if b_conf and c_conf else 0.0
    level = classify_psi(psi_val) if b_conf and c_conf else "NORMAL"

    b_rate = len([i for i in baseline if i.status == "completed" and i.quality_result == QualityResult.REVIEW]) / max(1, len(baseline))
    c_rate = len([i for i in current if i.status == "completed" and i.quality_result == QualityResult.REVIEW]) / max(1, len(current))
    return _out(
        f"{baseline_start.isoformat()} → {now.isoformat()}",
        model_version=args.get("model_version"),
        baseline_window={"from": baseline_start.isoformat(), "to": cutoff.isoformat(), "n": len(baseline)},
        current_window={"from": cutoff.isoformat(), "to": now.isoformat(), "n": len(current)},
        confidence_psi=round(psi_val, 4),
        confidence_drift_level=level,
        review_rate_baseline=round(b_rate, 4),
        review_rate_current=round(c_rate, 4),
        review_rate_level=review_rate_delta(b_rate, c_rate),
        note="data drift only; quality degradation requires human-review ground truth",
    )


async def _industrial_events(session: AsyncSession, args: dict) -> dict:
    fr, to, win = _resolve_window(args, default_days=7)
    stmt = select(PlcEvent).where(PlcEvent.created_at >= fr, PlcEvent.created_at <= to).order_by(PlcEvent.created_at.desc())
    if args.get("inspection_id"):
        stmt = stmt.where(PlcEvent.inspection_id == args["inspection_id"])
    events = list((await session.execute(stmt)).scalars().all())
    return _out(
        win,
        event_count=len(events),
        events=_cap(
            [
                {
                    "inspection_id": e.inspection_id,
                    "command": e.command,
                    "execution_status": e.execution_status,
                    "industrial_state": e.industrial_state,
                    "adapter_type": e.adapter_type,
                    "reason_code": e.reason_code,
                    "created_at": e.created_at.isoformat() if e.created_at else None,
                }
                for e in events
            ],
            args,
            50,
        ),
    )


async def _plc_fault_summary(session: AsyncSession, args: dict) -> dict:
    fr, to, win = _resolve_window(args, default_days=7)
    rows = (
        await session.execute(
            select(PlcEvent.status, PlcEvent.industrial_state, func.count())
            .where(PlcEvent.created_at >= fr, PlcEvent.created_at <= to)
            .group_by(PlcEvent.status, PlcEvent.industrial_state)
        )
    ).all()
    summary: dict[str, int] = {}
    for status, state, cnt in rows:
        key = state or status or "unknown"
        summary[key] = summary.get(key, 0) + cnt
    return _out(win, fault_summary=summary)


async def _mes_sync_summary(session: AsyncSession, args: dict) -> dict:
    fr, to, win = _resolve_window(args, default_days=7)
    rows = (
        await session.execute(
            select(Inspection.mes_sync_status, func.count())
            .where(Inspection.created_at >= fr, Inspection.created_at <= to)
            .group_by(Inspection.mes_sync_status)
        )
    ).all()
    return _out(win, mes_sync={k or "none": v for k, v in rows})


# ---------------------------------------------------------------------------
# registry (the ONLY surface the LLM can call)
# ---------------------------------------------------------------------------

_TIME_PARAMS = {
    "time_from": {"type": "string", "description": "ISO-8601 start of the window (optional)"},
    "time_to": {"type": "string", "description": "ISO-8601 end of the window (optional)"},
}
_COMMON_PROPS = {**_TIME_PARAMS, "max_results": {"type": "integer", "description": "cap on returned rows (default 50)"}}


def _schema(required: list[str], props: dict) -> dict:
    return {
        "type": "object",
        "properties": {**props, **_COMMON_PROPS},
        "required": required,
    }


TOOLS: list[CopilotTool] = [
    CopilotTool(
        name="get_quality_summary",
        description="Overall quality summary (inspected/completed/pass/fail/review/yield/system_failed) over a time window, optionally per line.",
        parameters=_schema([], {"line": {"type": "string", "description": "production line filter"}}),
        handler=_quality_summary,
    ),
    CopilotTool(
        name="get_yield_trend",
        description="Daily yield trend over the last N days (default 7), optionally per line.",
        parameters=_schema([], {"days": {"type": "integer"}, "line": {"type": "string"}}),
        handler=_yield_trend,
    ),
    CopilotTool(
        name="get_defect_distribution",
        description="Defect class distribution (counts + shares) over a time window, optionally per line.",
        parameters=_schema([], {"line": {"type": "string"}}),
        handler=_defect_distribution,
    ),
    CopilotTool(
        name="get_defect_trend",
        description="Daily occurrence count of a defect type (or all) over the last N days.",
        parameters=_schema([], {"days": {"type": "integer"}, "defect_type": {"type": "string"}}),
        handler=_defect_trend,
    ),
    CopilotTool(
        name="compare_production_lines",
        description="Per-production-line yield/review comparison over a time window.",
        parameters=_schema([], {}),
        handler=_compare_lines,
    ),
    CopilotTool(
        name="get_batch_quality",
        description="Batch analysis: yield, defect mix, lines/stations, model versions, rules, anomaly scores.",
        parameters=_schema(["batch_id"], {"batch_id": {"type": "string"}}),
        handler=_batch_quality,
    ),
    CopilotTool(
        name="get_inspection_detail",
        description="Full read-only trace of a single inspection: product, defects, model version, rule, human review, PLC command/ACK/state.",
        parameters=_schema(["inspection_id"], {"inspection_id": {"type": "string"}}),
        handler=_inspection_detail,
    ),
    CopilotTool(
        name="get_product_history",
        description="All inspections of one product (product_id) with outcomes.",
        parameters=_schema(["product_id"], {"product_id": {"type": "string"}}),
        handler=_product_history,
    ),
    CopilotTool(
        name="get_review_metrics",
        description="Human review ground-truth metrics (confirmation/agreement/pass-override/corrected) over a window, optionally per model version.",
        parameters=_schema([], {"model_version": {"type": "string"}}),
        handler=_review_metrics,
    ),
    CopilotTool(
        name="get_review_backlog",
        description="Current pending human review backlog (count + oldest tasks).",
        parameters=_schema([], {}),
        handler=_review_backlog,
    ),
    CopilotTool(
        name="get_model_metrics",
        description="Production model metrics: inference/error counts, latency avg+p95, review rate over a window.",
        parameters=_schema([], {"model_version": {"type": "string"}}),
        handler=_model_metrics,
    ),
    CopilotTool(
        name="get_drift_status",
        description="Data drift between baseline and current windows (PSI/deltas). Returns NORMAL/WARNING/CRITICAL. NOT a claim of model accuracy loss.",
        parameters=_schema([], {"baseline_days": {"type": "integer"}, "model_version": {"type": "string"}}),
        handler=_drift_status,
    ),
    CopilotTool(
        name="get_industrial_events",
        description="PLC command events (command/execution_status/industrial_state) over a window, optionally per inspection.",
        parameters=_schema([], {"inspection_id": {"type": "string"}}),
        handler=_industrial_events,
    ),
    CopilotTool(
        name="get_plc_fault_summary",
        description="PLC fault summary: counts of NACK/COMMAND_FAILED/SAFE_HOLD/TIMEOUT states over a window.",
        parameters=_schema([], {}),
        handler=_plc_fault_summary,
    ),
    CopilotTool(
        name="get_mes_sync_summary",
        description="MES sync status distribution (SYNCED/FAILED/PENDING) over a window.",
        parameters=_schema([], {}),
        handler=_mes_sync_summary,
    ),
]

_REGISTRY: dict[str, CopilotTool] = {t.name: t for t in TOOLS}


class ToolRegistry:
    """Fixed allowlist. No arbitrary tool, no write tools (9A/9D)."""

    @property
    def names(self) -> list[str]:
        return list(_REGISTRY)

    def schemas(self) -> list[dict]:
        return [t.schema() for t in TOOLS]

    def has(self, name: str) -> bool:
        return name in _REGISTRY

    async def call(self, session: AsyncSession, call: ToolCall) -> dict:
        tool = _REGISTRY.get(call.name)
        if tool is None:
            return {"error": f"unknown tool {call.name}", "tool": call.name}
        started = time.perf_counter()
        try:
            result = await asyncio.wait_for(tool.handler(session, call.arguments), timeout=tool.timeout)
            result = dict(result)
            result["tool"] = tool.name
            result["latency_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
            return result
        except asyncio.TimeoutError:
            return {"error": f"tool {call.name} timed out after {tool.timeout}s", "tool": call.name}
        except ToolError as exc:
            return {"error": str(exc), "tool": call.name}
        except Exception as exc:  # noqa: BLE001 - tools must never crash the loop
            return {"error": f"{type(exc).__name__}: {exc}", "tool": call.name}


registry = ToolRegistry()
