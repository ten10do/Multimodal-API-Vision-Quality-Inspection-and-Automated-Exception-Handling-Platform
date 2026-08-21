"""End-to-end: camera -> edge runtime -> D3 -> decision -> PLC (drift-aware)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from industrial_loop.camera.virtual_file_camera import write_placeholder_png
from industrial_loop.decision_service import D3InferenceResult, DecisionEngine
from industrial_loop.events import Decision, ReasonCode
from industrial_loop.factory_simulator import SyntheticD3Backend
from industrial_loop.plc_adapter import InMemoryPlc, PlcCommand, plc_status_for
from industrial_runtime.config import EdgeConfig
from industrial_runtime.resource_monitor import ResourceMonitor
from industrial_runtime.runtime_manager import EdgeRuntimeManager, RuntimeState, ServiceSpec
from monitoring.drift.detector import DriftDetector, DriftState, DriftThresholds

FROZEN_THRESHOLD = 0.8471092581718962  # placeholder, replaced by exact value below
FROZEN_THRESHOLD = 0.8471092581748962


class DriftAwareLine:
    """Minimal production line wiring every layer the task requires."""

    def __init__(self, dataset: Path, *, products: int = 24, seed: int = 7, dim: int = 32,
                 window_size: int = 64):
        self.config = EdgeConfig(device="cpu")
        self.monitor = ResourceMonitor(interval_seconds=1.0)
        self.runtime = EdgeRuntimeManager(self.config, monitor=self.monitor)
        self.plc = InMemoryPlc()
        self.engine = DecisionEngine()
        self.backend = SyntheticD3Backend(seed=seed)
        self.camera = self._build_camera(dataset, seed)
        self.rng = np.random.default_rng(seed + 5)
        self.dim = dim
        self.detector = DriftDetector(DriftThresholds(
            cosine_warning=100.0, cosine_critical=200.0,  # PSI + distance drive the verdict
        ))
        self.collector = self._build_collector(seed, window_size)
        self.products = products
        self.defects = np.random.default_rng(seed).random(products) < 0.08
        self.events: list[object] = []
        self._register_services()

    def _build_camera(self, dataset: Path, seed: int):
        from industrial_loop.camera.virtual_file_camera import VirtualFileCamera

        return VirtualFileCamera(dataset, camera_id="steel-camera-01", seed=seed)

    def _build_collector(self, seed: int, window_size: int):
        from monitoring.drift.collector import FeatureDriftCollector

        collector = FeatureDriftCollector(dim=self.dim, window_size=window_size)
        collector.set_baseline(self.rng.standard_normal((300, self.dim)))
        # warm production window so drift evaluations are valid from frame 1
        collector.extend(self.rng.standard_normal((max(40, window_size), self.dim)))
        return collector

    def _register_services(self) -> None:
        self.runtime.register(ServiceSpec(
            name="camera",
            start=self.camera.connect,
            stop=self.camera.disconnect,
            health=lambda: self.camera.health.is_online,
        ))
        self.runtime.register(ServiceSpec(
            name="decision_engine",
            health=lambda: self.engine.policy.reject_threshold == FROZEN_THRESHOLD,
        ))
        self.runtime.register(ServiceSpec(
            name="plc_link",
            health=lambda: self.plc.state.value != "STOP",
        ))

    def embedding_for(self, index: int, frame) -> np.ndarray:
        """Simulated DINO embedding for the drift layer (peripheral only)."""
        vec = self.rng.standard_normal(self.dim)
        return vec

    def run(self, *, drift_shift: float = 0.0) -> list:
        from industrial_loop.camera.camera_trigger import CameraTriggerService

        trigger_service = CameraTriggerService(plc=self.plc)
        status = self.runtime.start()
        assert status["state"] in ("RUNNING", "DEGRADED")
        events = []
        with self.camera:
            for index in range(self.products):
                health = self.runtime.health_check()
                assert health["overall"] in ("healthy", "degraded")
                if self.runtime.state is not RuntimeState.RUNNING:
                    break  # fail-safe: degraded runtime stops production
                product_id = f"P{index + 1:04d}"
                trigger = trigger_service.request_trigger(self.camera)
                frame = self.camera.capture()
                # drift layer (peripheral): embedding from the captured frame index
                emb = self.embedding_for(index, frame) + drift_shift
                self.collector.add_sample(emb)
                report = self.detector.evaluate(self.collector)
                if report.state is DriftState.CRITICAL:
                    result = D3InferenceResult.failure(
                        f"data_distribution_shift:psi={report.psi_mean:.4f}",
                        kind="data_distribution_shift",
                    )
                else:
                    result, _ = self.backend.infer(
                        frame, defect_injected=bool(self.defects[index]), product_id=product_id
                    )
                self.monitor.record_request(result.latency_ms, error=not result.ok)
                event = self.engine.decide(
                    result, product_id=product_id, batch_id="B-E2E",
                    camera_id=frame.camera_id, trace_id=f"trc-{index:06d}",
                )
                plc_result = self.plc.apply(PlcCommand(
                    command_id=f"cmd-{event.id}", event_id=event.id,
                    product_id=event.product_id, decision=event.decision,
                ))
                event = event.with_updates(plc_status=plc_status_for(plc_result))
                if plc_result.state_after.value == "STOP":
                    self.plc.reset()
                events.append(event)
        self.runtime.stop()
        self.events = events
        return events


@pytest.fixture()
def dataset(tmp_path: Path) -> Path:
    for index in range(4):
        write_placeholder_png(tmp_path / f"s{index}.png", width=160, height=40)
    return tmp_path


class TestEdgeDriftE2E:
    def test_full_chain_camera_runtime_d3_decision_plc(self, dataset):
        line = DriftAwareLine(dataset, products=24, seed=7)
        events = line.run()
        assert len(events) == 24
        assert line.runtime.state is RuntimeState.STOPPED
        assert all(e.camera_id == "steel-camera-01" for e in events)
        assert {e.plc_status.value for e in events} <= {"ACK_RUNNING", "ACK_REJECT_SIGNAL", "ACK_STOP_SIGNAL"}
        assert line.monitor.latest().request_count >= 20

    def test_runtime_must_be_running_before_production(self, dataset):
        line = DriftAwareLine(dataset, products=4, seed=8)
        # a manager that refuses to reach RUNNING blocks the whole line
        line.runtime.register(ServiceSpec(
            name="broken_gate", start=lambda: (_ for _ in ()).throw(RuntimeError("no device")),
        ))
        events = line.run()
        assert events == []  # fail-safe: degraded runtime => zero production

    def test_drift_critical_midrun_holds_rest_of_line(self, dataset):
        line = DriftAwareLine(dataset, products=60, seed=9, window_size=32)
        original = line.embedding_for
        reset_done = {"done": False}

        def shifted(index: int, frame):
            # after frame 10 the "material" changes: strong common-mode shift;
            # the collector tracks the current regime (window reset at change)
            if index >= 10 and not reset_done["done"]:
                line.collector.reset_window()
                reset_done["done"] = True
            return original(index, frame) + (2.0 if index >= 10 else 0.0)

        line.embedding_for = shifted  # type: ignore[method-assign]
        events = line.run()
        after_critical = [e for e in events if int(e.product_id[1:]) >= 45]
        assert after_critical, "expected products after the drift transition"
        assert all(e.decision is Decision.HOLD for e in after_critical)
        assert all(e.reason_code is ReasonCode.DATA_DISTRIBUTION_SHIFT for e in after_critical)
        assert not any(e.decision is Decision.PASS for e in after_critical)

    def test_drift_warning_keeps_production_running(self, dataset):
        line = DriftAwareLine(dataset, products=40, seed=10, window_size=32)
        original = line.embedding_for
        reset_done = {"done": False}

        def mild(index: int, frame):
            if index >= 5 and not reset_done["done"]:
                line.collector.reset_window()
                reset_done["done"] = True
            return original(index, frame) + (0.40 if index >= 5 else 0.0)

        line.embedding_for = mild  # type: ignore[method-assign]
        events = line.run()
        assert len(events) == 40
        # WARNING must not force HOLDs: passes/rejects still occur post-shift
        post = [e for e in events if int(e.product_id[1:]) >= 20]
        assert any(e.decision in (Decision.PASS, Decision.REJECT) for e in post)
        assert line.detector.latest() is not None
        assert line.detector.latest().state is DriftState.WARNING

    def test_plc_receives_stop_signal_on_critical_drift(self, dataset):
        line = DriftAwareLine(dataset, products=60, seed=11, window_size=32)
        original = line.embedding_for
        reset_done = {"done": False}

        def shifted(index: int, frame):
            if index >= 10 and not reset_done["done"]:
                line.collector.reset_window()
                reset_done["done"] = True
            return original(index, frame) + (2.0 if index >= 10 else 0.0)

        line.embedding_for = shifted  # type: ignore[method-assign]
        events = line.run()
        stop_events = [e for e in events if e.plc_status.value == "ACK_STOP_SIGNAL"]
        assert stop_events, "CRITICAL drift must reach the PLC as stop_signal"
        assert all(
            e.reason_code is ReasonCode.DATA_DISTRIBUTION_SHIFT for e in stop_events
        )