"""Drift scenarios: normal production, brightness shift, material change.

Scenario A (normal production, 10000 frames)  -> drift NORMAL
Scenario B (illumination/brightness shift)    -> drift WARNING (production continues + alert)
Scenario C (material change / embedding shift)-> drift CRITICAL -> HOLD, cannot PASS

Run: python -m monitoring.drift.scenarios --frames-a 10000
Report: runs/industrial-loop/drift_simulation_report.json (gitignored).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from industrial_loop.decision_service import D3InferenceResult, DecisionEngine
from industrial_loop.events import Decision
from industrial_loop.plc_adapter import InMemoryPlc, PlcCommand
from monitoring.drift.collector import FeatureDriftCollector
from monitoring.drift.detector import DriftDetector, DriftState, DriftThresholds

DIM = 768
BASELINE_SAMPLES = 2000


def _make_collector(seed: int, *, window_size: int = 4096) -> tuple[FeatureDriftCollector, np.random.Generator]:
    rng = np.random.default_rng(seed)
    collector = FeatureDriftCollector(dim=DIM, window_size=window_size)
    collector.set_baseline(rng.standard_normal((BASELINE_SAMPLES, DIM)))
    return collector, rng


def _stream(
    collector: FeatureDriftCollector,
    detector: DriftDetector,
    rng: np.random.Generator,
    frames: int,
    *,
    chunk: int,
    shift: float = 0.0,
    scale: float = 1.0,
) -> list[str]:
    """Stream `frames` embeddings in chunks, evaluating after each chunk."""
    states: list[str] = []
    emitted = 0
    while emitted < frames:
        size = min(chunk, frames - emitted)
        batch = rng.standard_normal((size, DIM)) * scale + shift
        collector.extend(batch)
        report = detector.evaluate(collector)
        states.append(report.state.value)
        emitted += size
    return states


def run_scenario_a(frames: int = 10000, seed: int = 42, chunk: int = 500) -> dict:
    """Normal production: embeddings from the unchanged distribution."""
    collector, rng = _make_collector(seed)
    detector = DriftDetector(DriftThresholds())
    states = _stream(collector, detector, rng, frames, chunk=chunk)
    final = detector.latest()
    assert final is not None
    return {
        "scenario": "A_normal_production",
        "frames": frames,
        "final_state": final.state.value,
        "expected_state": "NORMAL",
        "pass": final.state is DriftState.NORMAL,
        "states_seen": sorted(set(states)),
        "psi_mean": round(final.psi_mean, 6),
        "cosine_shift": round(final.cosine_shift, 6),
        "mean_distance": round(final.mean_distance, 6),
    }


def run_scenario_b(frames: int = 4000, seed: int = 43, chunk: int = 500, shift: float = 0.40) -> dict:
    """Illumination change: a common-mode brightness shift on every dimension.

    Calibrated so the aggregate PSI lands inside the WARNING band (0.10-0.25;
    empirical decile PSI ~ delta^2): production continues with an alert.
    """
    collector, rng = _make_collector(seed)
    detector = DriftDetector(DriftThresholds())
    states = _stream(collector, detector, rng, frames, chunk=chunk, shift=shift)
    final = detector.latest()
    assert final is not None
    return {
        "scenario": "B_brightness_shift",
        "frames": frames,
        "shift_sigma": shift,
        "final_state": final.state.value,
        "expected_state": "WARNING",
        "pass": final.state is DriftState.WARNING,
        "states_seen": sorted(set(states)),
        "psi_mean": round(final.psi_mean, 6),
        "mean_distance": round(final.mean_distance, 6),
        "alerts": final.alerts,
    }


def run_scenario_c(frames: int = 4000, seed: int = 44, chunk: int = 500,
                   shift: float = 1.5, scale: float = 1.3) -> dict:
    """Material change: large embedding mean shift + variance inflation.

    CRITICAL drift must fail production closed: subsequent inspections HOLD
    with reason DATA_DISTRIBUTION_SHIFT and the PLC receives stop_signal -
    and no PASS can be produced after the CRITICAL transition.
    """
    collector, rng = _make_collector(seed)
    detector = DriftDetector(DriftThresholds())
    engine = DecisionEngine()
    plc = InMemoryPlc()
    plc.start()

    critical_at: int | None = None
    decisions_after: list[str] = []
    reasons_after: set[str] = set()
    emitted = 0
    product_index = 0
    while emitted < frames:
        size = min(chunk, frames - emitted)
        batch = rng.standard_normal((size, DIM)) * scale + shift
        collector.extend(batch)
        report = detector.evaluate(collector)
        emitted += size
        if report.state is DriftState.CRITICAL and critical_at is None:
            critical_at = emitted
        # one product decision per chunk after the baseline exists
        product_index += 1
        if report.state is DriftState.CRITICAL:
            result = D3InferenceResult.failure(
                f"data_distribution_shift:psi={report.psi_mean:.4f}",
                kind="data_distribution_shift",
            )
        else:
            # pre-critical the line runs normally (synthetic healthy inference)
            result = D3InferenceResult(
                ok=True,
                model_version="1.3.0-candidate.1",
                artifact_version="steel-patchcore-d3-release@1.3.0",
                image_score=0.80,
                pixel_score=0.05,
                threshold=0.8471092581748962,
            )
        event = engine.decide(result, product_id=f"P{product_index:04d}",
                              batch_id="B-DRIFT", camera_id="steel-camera-01")
        if critical_at is not None and emitted >= critical_at:
            decisions_after.append(event.decision.value)
            reasons_after.add(event.reason_code.value)
        plc.apply(PlcCommand(command_id=f"cmd-{event.id}", event_id=event.id,
                             product_id=event.product_id, decision=event.decision))

    final = detector.latest()
    assert final is not None and critical_at is not None
    holds = [d for d in decisions_after if d == "HOLD"]
    fail_closed = (
        bool(holds)
        and reasons_after == {"DATA_DISTRIBUTION_SHIFT"}
        and "PASS" not in decisions_after
        and plc.counters["stop_signals"] >= 1
    )
    return {
        "scenario": "C_material_change",
        "frames": frames,
        "shift_sigma": shift,
        "scale": scale,
        "critical_at_frame": critical_at,
        "final_state": final.state.value,
        "expected_state": "CRITICAL",
        "pass": final.state is DriftState.CRITICAL and fail_closed,
        "decisions_after_critical": {
            "counts": {d: decisions_after.count(d) for d in sorted(set(decisions_after))},
            "reasons": sorted(reasons_after),
            "all_hold_no_pass": "PASS" not in decisions_after and bool(holds),
        },
        "plc_stop_signals": plc.counters["stop_signals"],
        "fail_closed": fail_closed,
        "psi_mean": round(final.psi_mean, 6),
        "mean_distance": round(final.mean_distance, 6),
    }


def run_all(frames_a: int = 10000, frames_bc: int = 4000, seed: int = 42) -> dict:
    a = run_scenario_a(frames=frames_a, seed=seed)
    b = run_scenario_b(frames=frames_bc, seed=seed + 1)
    c = run_scenario_c(frames=frames_bc, seed=seed + 2)
    return {
        "schema_version": "industrial_drift_simulation_report_v1",
        "dim": DIM,
        "baseline_samples": BASELINE_SAMPLES,
        "scenarios": [a, b, c],
        "all_pass": a["pass"] and b["pass"] and c["pass"],
    }


def main() -> int:  # pragma: no cover - CLI entry
    parser = argparse.ArgumentParser(description="Industrial drift monitoring simulation")
    parser.add_argument("--frames-a", type=int, default=10000)
    parser.add_argument("--frames-bc", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    report = run_all(frames_a=args.frames_a, frames_bc=args.frames_bc, seed=args.seed)
    out = Path(args.out) if args.out else Path("runs/industrial-loop/drift_simulation_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for scenario in report["scenarios"]:
        print(
            f"{scenario['scenario']}: state={scenario['final_state']} "
            f"expected={scenario['expected_state']} pass={scenario['pass']}"
        )
    print(f"all_pass={report['all_pass']} report written: {out}")
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
