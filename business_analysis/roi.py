"""Deterministic ROI calculations; monetary inputs use one consistent currency."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RoiInputs:
    inspectors: int
    annual_cost_per_inspector: float
    automation_rate: float
    annual_ai_operating_cost: float
    initial_investment: float
    baseline_escape_rate: float
    ai_escape_rate: float
    annual_units: int
    escape_cost_per_unit: float


def calculate_roi(inputs: RoiInputs) -> dict[str, float | None]:
    if inputs.inspectors < 0 or inputs.annual_units < 0:
        raise ValueError("counts must be non-negative")
    if any(value < 0 for value in (
        inputs.annual_cost_per_inspector,
        inputs.annual_ai_operating_cost,
        inputs.initial_investment,
        inputs.escape_cost_per_unit,
    )):
        raise ValueError("costs must be non-negative")
    if not 0 <= inputs.automation_rate <= 1:
        raise ValueError("automation_rate must be between 0 and 1")
    if not 0 <= inputs.baseline_escape_rate <= 1 or not 0 <= inputs.ai_escape_rate <= 1:
        raise ValueError("escape rates must be between 0 and 1")

    labor_baseline = inputs.inspectors * inputs.annual_cost_per_inspector
    labor_saving = labor_baseline * inputs.automation_rate
    escape_avoidance = (
        max(0.0, inputs.baseline_escape_rate - inputs.ai_escape_rate)
        * inputs.annual_units
        * inputs.escape_cost_per_unit
    )
    annual_net_benefit = labor_saving + escape_avoidance - inputs.annual_ai_operating_cost
    roi = annual_net_benefit / inputs.initial_investment if inputs.initial_investment else None
    payback_years = inputs.initial_investment / annual_net_benefit if annual_net_benefit > 0 else None
    detection_improvement = inputs.baseline_escape_rate - inputs.ai_escape_rate
    return {
        "labor_baseline": labor_baseline,
        "labor_saving": labor_saving,
        "escape_avoidance": escape_avoidance,
        "annual_net_benefit": annual_net_benefit,
        "roi": roi,
        "payback_years": payback_years,
        "detection_improvement": detection_improvement,
    }
