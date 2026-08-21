"""Industrial drift monitoring tests (PSI / cosine / thresholds / fail-closed)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from industrial_loop.dashboard import LoopStore, create_app
from industrial_loop.decision_service import D3InferenceResult, DecisionEngine
from industrial_loop.events import Decision, ReasonCode
from industrial_runtime.config import EdgeConfig
from monitoring.drift.collector import BaselineStats, FeatureDriftCollector
from monitoring.drift.detector import DriftDetector, DriftState, DriftThresholds
from monitoring.drift.metrics import (
    cosine_distribution_shift,
    embedding_mean_distance,
    psi_1d,
    psi_embedding,
    psi_embedding_from_stats,
    psi_from_stats,
)
from monitoring.drift.scenarios import (
    run_scenario_a,
    run_scenario_b,
    run_scenario_c,
)

DIM = 64


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _collector(seed: int = 1, *, n: int = 500, dim: int = DIM, window: int = 2048) -> FeatureDriftCollector:
    collector = FeatureDriftCollector(dim=dim, window_size=window)
    collector.set_baseline(_rng(seed).standard_normal((n, dim)))
    return collector


# --- PSI -----------------------------------------------------------------------

class TestPsi:
    def test_identical_distributions_near_zero(self):
        rng = _rng(1)
        sample = rng.standard_normal(5000)
        assert psi_1d(sample, sample) == pytest.approx(0.0, abs=1e-9)
        other = rng.standard_normal(5000)
        assert psi_1d(sample, other) < 0.02

    def test_psi_increases_with_shift(self):
        rng = _rng(2)
        expected = rng.standard_normal(5000)
        scores = [
            psi_1d(expected, expected + shift) for shift in (0.0, 0.3, 0.6, 1.2)
        ]
        assert scores == sorted(scores)
        assert scores[0] < 0.02

    def test_psi_small_shift_magnitude_tracks_delta_squared(self):
        rng = _rng(3)
        expected = rng.standard_normal(20000)
        value = psi_1d(expected, expected + 0.3)
        assert value == pytest.approx(0.3**2, rel=0.35)  # empirical decile PSI ~ delta^2

    def test_psi_handles_disjoint_support_via_eps_clipping(self):
        rng = _rng(4)
        expected = rng.standard_normal(2000)
        far = expected + 50.0
        value = psi_1d(expected, far)
        assert np.isfinite(value) and value > 5.0

    def test_psi_from_stats_matches_empirical_direction(self):
        value = psi_from_stats(0.0, 1.0, 0.3, 1.0)
        assert 0.0 < value < 0.25
        assert psi_from_stats(0.0, 1.0, 0.0, 1.0) == pytest.approx(0.0, abs=1e-12)

    def test_psi_embedding_aggregates_dimensions(self):
        rng = _rng(5)
        base = rng.standard_normal((800, 6))
        shifted = base + np.array([0.0, 0.0, 0.0, 2.0, 2.0, 2.0])
        result = psi_embedding(base, shifted)
        assert result["max"] > result["mean"] > 0.1
        assert len(result["per_dim"]) == 6

    def test_psi_embedding_from_stats_vectorized_consistent(self):
        rng = _rng(6)
        base = rng.standard_normal((900, 5))
        cur = rng.standard_normal((900, 5)) + 0.5
        b = BaselineStats.from_samples(base)
        c = BaselineStats.from_samples(cur)
        analytic = psi_embedding_from_stats(b.mean, b.std, c.mean, c.std)
        empirical = psi_embedding(base, cur)
        assert analytic["mean"] == pytest.approx(empirical["mean"], rel=0.6, abs=0.05)

    def test_psi_rejects_bad_std(self):
        with pytest.raises(ValueError):
            psi_from_stats(0.0, 0.0, 0.0, 1.0)


# --- cosine shift + mean distance -------------------------------------------------

class TestCosineAndDistance:
    def test_cosine_shift_zero_for_same_distribution(self):
        rng = _rng(7)
        base = rng.standard_normal((500, DIM)) + np.ones(DIM) * 3.0
        cur = rng.standard_normal((500, DIM)) + np.ones(DIM) * 3.0
        assert cosine_distribution_shift(base, cur) < 0.02

    def test_cosine_shift_detects_concentration_change(self):
        rng = _rng(8)
        base = rng.standard_normal((800, DIM)) + np.ones(DIM) * 3.0
        collapsed = rng.standard_normal((800, DIM)) * 0.02 + np.ones(DIM) * 3.0
        assert cosine_distribution_shift(base, collapsed) > 0.05

    def test_cosine_shift_requires_nonzero_baseline_mean(self):
        degenerate = np.zeros((10, 4))  # all-zero embeddings -> zero mean vector
        with pytest.raises(ValueError):
            cosine_distribution_shift(degenerate, _rng(10).standard_normal((10, 4)))

    def test_mean_distance_zero_for_identical_stats(self):
        stats = BaselineStats.from_samples(_rng(11).standard_normal((400, DIM)))
        assert embedding_mean_distance(stats.mean, stats.std, stats.mean) == pytest.approx(0.0, abs=1e-12)

    def test_mean_distance_rms_standardized(self):
        mean0 = np.zeros(DIM)
        std1 = np.ones(DIM)
        shifted = np.full(DIM, 0.5)
        assert embedding_mean_distance(mean0, std1, shifted) == pytest.approx(0.5)
        mixed = np.linspace(0.0, 1.0, DIM)
        assert embedding_mean_distance(mean0, std1, mixed) == pytest.approx(float(np.sqrt(np.mean(mixed**2))))

    def test_mean_distance_guards_zero_variance(self):
        mean0 = np.zeros(DIM)
        zero_std = np.zeros(DIM)
        assert embedding_mean_distance(mean0, zero_std, np.zeros(DIM)) == pytest.approx(0.0)


# --- collector -------------------------------------------------------------------

class TestFeatureDriftCollector:
    def test_baseline_stats_mean_variance_count(self):
        samples = _rng(12).standard_normal((500, 8)) * 2.0 + 1.0
        stats = BaselineStats.from_samples(samples)
        assert stats.count == 500 and stats.dim == 8
        assert np.allclose(stats.mean, samples.mean(axis=0))
        assert np.allclose(stats.variance, samples.var(axis=0))

    def test_baseline_requires_minimum_samples(self):
        collector = FeatureDriftCollector(dim=4, min_baseline_samples=100)
        with pytest.raises(ValueError):
            collector.set_baseline(_rng(13).standard_normal((99, 4)))

    def test_baseline_dimension_enforced(self):
        collector = FeatureDriftCollector(dim=DIM)
        with pytest.raises(ValueError):
            collector.set_baseline(_rng(14).standard_normal((300, DIM + 1)))

    def test_window_rolls_and_keeps_recent(self):
        collector = FeatureDriftCollector(dim=4, window_size=10)
        collector.set_baseline(_rng(15).standard_normal((200, 4)))
        collector.extend(_rng(16).standard_normal((25, 4)))
        assert collector.current_window().shape == (10, 4)
        assert collector.total_seen == 25

    def test_current_stats_matches_manual(self):
        collector = FeatureDriftCollector(dim=4, window_size=100)
        collector.set_baseline(_rng(17).standard_normal((200, 4)))
        batch = _rng(18).standard_normal((40, 4)) + 0.5
        collector.extend(batch)
        stats = collector.current_stats()
        assert stats is not None and stats.count == 40
        assert np.allclose(stats.mean, batch.mean(axis=0))

    def test_nonfinite_embeddings_rejected(self):
        collector = FeatureDriftCollector(dim=4, window_size=10)
        collector.set_baseline(_rng(19).standard_normal((200, 4)))
        bad = _rng(20).standard_normal((5, 4))
        bad[0, 0] = np.nan
        with pytest.raises(ValueError):
            collector.extend(bad)


# --- detector bands + fail-safe ---------------------------------------------------

class TestDriftDetector:
    def _evaluate_with_shift(self, shift: float, *, thresholds: DriftThresholds | None = None,
                             scale: float = 1.0) -> DriftState:
        collector = _collector(seed=21)
        detector = DriftDetector(thresholds or DriftThresholds())
        collector.extend(_rng(22).standard_normal((400, DIM)) * scale + shift)
        return detector.evaluate(collector)

    def test_normal_band(self):
        report = self._evaluate_with_shift(0.0)
        assert report.state is DriftState.NORMAL
        assert report.psi_mean < 0.10 and all(v == "NORMAL" for v in report.checks.values())

    def test_warning_band_psi(self):
        thresholds = DriftThresholds(cosine_warning=10.0, mean_dist_warning=10.0,
                                     cosine_critical=20.0, mean_dist_critical=20.0)
        report = self._evaluate_with_shift(0.40, thresholds=thresholds)
        assert report.state is DriftState.WARNING
        assert 0.10 <= report.psi_mean < 0.25
        assert report.checks["psi"] == "WARNING"

    def test_critical_band_psi(self):
        thresholds = DriftThresholds(cosine_warning=10.0, mean_dist_warning=10.0,
                                     cosine_critical=20.0, mean_dist_critical=20.0)
        report = self._evaluate_with_shift(1.5, thresholds=thresholds)
        assert report.state is DriftState.CRITICAL
        assert report.psi_mean >= 0.25

    def test_detector_uses_worst_check(self):
        # psi normal but mean distance critical -> overall CRITICAL
        thresholds = DriftThresholds(psi_warning=100.0, psi_critical=200.0,
                                     cosine_warning=100.0, cosine_critical=200.0,
                                     mean_dist_warning=0.10, mean_dist_critical=0.40)
        report = self._evaluate_with_shift(0.55, thresholds=thresholds)
        assert report.state is DriftState.CRITICAL
        assert report.checks["mean_distance"] == "CRITICAL"

    def test_custom_thresholds_change_classification(self):
        strict = DriftThresholds(psi_warning=0.001, psi_critical=0.002,
                                 cosine_warning=100.0, cosine_critical=200.0,
                                 mean_dist_warning=100.0, mean_dist_critical=200.0)
        report = self._evaluate_with_shift(0.0, thresholds=strict)
        # sampling noise (~0.009 psi) exceeds the strict critical band
        assert report.state is DriftState.CRITICAL

    def test_thresholds_from_edge_config(self):
        config = EdgeConfig.load()
        thresholds = DriftThresholds.from_config(config)
        assert thresholds.psi_critical == config.psi_critical

    def test_report_fields_complete(self):
        report = self._evaluate_with_shift(0.0)
        payload = report.as_dict()
        for field in ("state", "psi_mean", "psi_max", "cosine_shift", "mean_distance",
                      "checks", "n_baseline", "n_current", "timestamp", "alerts"):
            assert field in payload

    def test_insufficient_data_is_normal_not_blocking(self):
        collector = _collector(seed=23)
        detector = DriftDetector()
        collector.extend(_rng(24).standard_normal((10, DIM)))
        report = detector.evaluate(collector)
        assert report.sufficient_data is False
        assert report.state is DriftState.NORMAL

    def test_history_and_latest(self):
        collector = _collector(seed=25)
        detector = DriftDetector()
        collector.extend(_rng(26).standard_normal((300, DIM)))
        detector.evaluate(collector)
        detector.evaluate(collector)
        assert len(detector.history()) == 2
        assert detector.latest() is not None
        detector.reset_history()
        assert detector.latest() is None


# --- fail-safe decision bridge -----------------------------------------------------

class TestFailSafeIntegration:
    def test_data_distribution_shift_reason_allowed_for_hold(self):
        event = DecisionEngine().decide(
            D3InferenceResult.failure("drift psi=0.4", kind="data_distribution_shift"),
            product_id="P", batch_id="B", camera_id="C",
        )
        assert event.decision is Decision.HOLD
        assert event.reason_code is ReasonCode.DATA_DISTRIBUTION_SHIFT

    def test_default_kind_preserves_ai_system_failure(self):
        event = DecisionEngine().decide(
            D3InferenceResult.failure("gpu stream error"),
            product_id="P", batch_id="B", camera_id="C",
        )
        assert event.reason_code is ReasonCode.AI_SYSTEM_FAILURE

    def test_critical_drift_cannot_pass(self):
        engine = DecisionEngine()
        for psi in (0.25, 0.5, 1.0):
            event = engine.decide(
                D3InferenceResult.failure(f"drift psi={psi}", kind="data_distribution_shift"),
                product_id="P", batch_id="B", camera_id="C",
            )
            assert event.decision is not Decision.PASS
            assert event.decision is Decision.HOLD

    def test_warning_drift_continues_production(self):
        # WARNING semantics: healthy inference result still judged normally
        engine = DecisionEngine()
        event = engine.decide(
            D3InferenceResult(
                ok=True, model_version="1.3.0-candidate.1", artifact_version="rel",
                image_score=0.80, pixel_score=0.05, threshold=0.8471092581748962,
            ),
            product_id="P", batch_id="B", camera_id="C",
        )
        assert event.decision is Decision.PASS


# --- dashboard endpoints -----------------------------------------------------------

class TestDashboardDriftRuntimeEndpoints:
    def _app(self):
        store = LoopStore()
        config = EdgeConfig(device="cpu")
        from industrial_runtime.resource_monitor import ResourceMonitor
        from industrial_runtime.runtime_manager import EdgeRuntimeManager, ServiceSpec

        monitor = ResourceMonitor(interval_seconds=1.0)
        manager = EdgeRuntimeManager(config, monitor=monitor)
        manager.register(ServiceSpec(name="svc", health=lambda: True))
        manager.start()
        detector = DriftDetector()
        collector = _collector(seed=27)
        collector.extend(_rng(28).standard_normal((300, DIM)))
        detector.evaluate(collector)
        return create_app(store, runtime_manager=manager, drift_detector=detector)

    def test_runtime_status_endpoint(self):
        client = TestClient(self._app())
        payload = client.get("/api/runtime/status").json()
        assert payload["available"] is True
        assert payload["runtime"]["state"] == "RUNNING"
        assert payload["health"]["metrics"]["timestamp"]
        history = client.get("/api/runtime/history").json()
        assert len(history) >= 1

    def test_drift_status_endpoint(self):
        client = TestClient(self._app())
        payload = client.get("/api/drift/status").json()
        assert payload["available"] is True
        assert payload["state"] in {"NORMAL", "WARNING", "CRITICAL"}
        assert "psi_critical" in payload["thresholds"]
        history = client.get("/api/drift/history").json()
        assert len(history) == 1

    def test_endpoints_absent_without_layers(self):
        client = TestClient(create_app(LoopStore()))
        assert client.get("/api/runtime/status").json() == {"available": False}
        assert client.get("/api/drift/status").json() == {"available": False}
        assert client.get("/api/drift/history").json() == []

    def test_runtime_page_served(self):
        client = TestClient(self._app())
        page = client.get("/").text
        assert 'id="page-runtime"' in page and "#/drift" in page


# --- scenarios (small fast versions; full-scale in the simulation run) -------------

class TestScenarios:
    def test_scenario_a_normal(self):
        result = run_scenario_a(frames=1500, seed=42, chunk=500)
        assert result["pass"] is True and result["final_state"] == "NORMAL"

    def test_scenario_b_brightness_warning(self):
        result = run_scenario_b(frames=1500, seed=43, chunk=500)
        assert result["pass"] is True and result["final_state"] == "WARNING"

    def test_scenario_c_material_critical_holds(self):
        result = run_scenario_c(frames=1500, seed=44, chunk=500)
        assert result["pass"] is True and result["final_state"] == "CRITICAL"
        assert result["fail_closed"] is True
        assert result["decisions_after_critical"]["reasons"] == ["DATA_DISTRIBUTION_SHIFT"]