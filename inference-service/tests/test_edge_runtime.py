"""Industrial edge runtime tests (lifecycle / restart / health / monitoring / config)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from industrial_runtime.config import EdgeConfig, load_edge_config
from industrial_runtime.resource_monitor import ResourceMonitor, RuntimeMetrics
from industrial_runtime.runtime_manager import (
    EdgeRuntimeManager,
    RuntimeState,
    ServiceSpec,
)
from industrial_runtime.service import create_app as create_edge_app


# --- configuration -------------------------------------------------------------

class TestEdgeConfig:
    def test_packaged_default_config_loads(self):
        config = load_edge_config()
        assert config.device == "cuda"
        assert config.batch_size == 1
        assert config.timeout_ms == 3000
        assert config.log_level == "INFO"
        assert config.monitoring_interval_seconds == 5.0

    def test_drift_thresholds_in_config(self):
        config = load_edge_config()
        assert config.psi_warning == 0.10 and config.psi_critical == 0.25
        assert 0 < config.cosine_warning < config.cosine_critical
        assert 0 < config.mean_dist_warning < config.mean_dist_critical

    def test_custom_yaml_override(self, tmp_path):
        cfg_file = tmp_path / "edge.yaml"
        cfg_file.write_text(
            "runtime:\n  device: cpu\n  batch_size: 4\n  timeout_ms: 1500\n"
            "logging:\n  level: DEBUG\nmonitoring:\n  interval_seconds: 2\n",
            encoding="utf-8",
        )
        config = EdgeConfig.load(cfg_file)
        assert config.device == "cpu" and config.batch_size == 4
        assert config.timeout_ms == 1500 and config.log_level == "DEBUG"
        assert config.monitoring_interval_seconds == 2.0

    def test_partial_yaml_fills_defaults(self, tmp_path):
        cfg_file = tmp_path / "partial.yaml"
        cfg_file.write_text("runtime:\n  batch_size: 8\n", encoding="utf-8")
        config = EdgeConfig.load(cfg_file)
        assert config.batch_size == 8
        assert config.device == "cuda" and config.timeout_ms == 3000

    def test_env_variable_override(self, tmp_path, monkeypatch):
        cfg_file = tmp_path / "env.yaml"
        cfg_file.write_text("runtime:\n  timeout_ms: 999\n", encoding="utf-8")
        monkeypatch.setenv("INDUSTRIAL_EDGE_CONFIG", str(cfg_file))
        assert EdgeConfig.load().timeout_ms == 999

    @pytest.mark.parametrize(
        "payload",
        [
            {"runtime": {"batch_size": 0}},
            {"runtime": {"timeout_ms": -1}},
            {"runtime": {"device": "tpu"}},
            {"logging": {"level": "VERBOSE"}},
            {"monitoring": {"interval_seconds": 0}},
            {"drift": {"psi_warning": 0.3, "psi_critical": 0.1}},
        ],
    )
    def test_invalid_config_rejected(self, payload):
        with pytest.raises(ValueError):
            EdgeConfig.from_mapping(payload)

    def test_summary_roundtrip_fields(self):
        summary = load_edge_config().summary()
        assert {"device", "batch_size", "timeout_ms", "log_level"} <= set(summary)
        assert "psi_critical" in summary["drift_thresholds"]


# --- lifecycle -------------------------------------------------------------------

def _spec(name, calls, *, healthy=True, start_fails=False):
    def start():
        if start_fails:
            raise RuntimeError("boom")
        calls.append(f"{name}:start")

    return ServiceSpec(
        name=name,
        start=start,
        stop=lambda: calls.append(f"{name}:stop"),
        health=lambda: healthy,
    )


class TestEdgeRuntimeLifecycle:
    def test_initial_state_is_init(self):
        manager = EdgeRuntimeManager(EdgeConfig())
        assert manager.state is RuntimeState.INIT

    def test_start_transitions_to_running(self):
        manager = EdgeRuntimeManager(EdgeConfig())
        manager.register(ServiceSpec(name="svc", health=lambda: True))
        manager.start()
        assert manager.state is RuntimeState.RUNNING

    def test_start_runs_services_in_order(self):
        calls: list[str] = []
        manager = EdgeRuntimeManager(EdgeConfig())
        manager.register(_spec("camera", calls))
        manager.register(_spec("inference", calls))
        manager.start()
        assert calls == ["camera:start", "inference:start"]

    def test_stop_reverses_and_marks_stopped(self):
        calls: list[str] = []
        manager = EdgeRuntimeManager(EdgeConfig())
        manager.register(_spec("camera", calls))
        manager.register(_spec("plc", calls))
        manager.start()
        manager.stop()
        assert manager.state is RuntimeState.STOPPED
        assert calls == ["camera:start", "plc:start", "plc:stop", "camera:stop"]

    def test_stop_is_idempotent(self):
        manager = EdgeRuntimeManager(EdgeConfig())
        manager.register(ServiceSpec(name="svc"))
        manager.start()
        first = manager.stop()
        second = manager.stop()
        assert first["state"] == second["state"] == "STOPPED"

    def test_start_is_idempotent_when_running(self):
        calls: list[str] = []
        manager = EdgeRuntimeManager(EdgeConfig())
        manager.register(_spec("svc", calls))
        manager.start()
        manager.start()
        assert calls.count("svc:start") == 1

    def test_restart_returns_to_running(self):
        calls: list[str] = []
        manager = EdgeRuntimeManager(EdgeConfig())
        manager.register(_spec("svc", calls))
        manager.start()
        status = manager.restart()
        assert status["state"] == "RUNNING"
        assert calls == ["svc:start", "svc:stop", "svc:start"]

    def test_restart_from_stopped_works(self):
        manager = EdgeRuntimeManager(EdgeConfig())
        manager.register(ServiceSpec(name="svc", health=lambda: True))
        manager.stop()  # from INIT -> STOPPED is allowed (idempotent no-op path)
        manager.restart()
        assert manager.state is RuntimeState.RUNNING

    def test_failed_service_start_degrades_runtime(self):
        calls: list[str] = []
        manager = EdgeRuntimeManager(EdgeConfig())
        manager.register(_spec("broken", calls, start_fails=True))
        manager.register(ServiceSpec(name="ok", health=lambda: True))
        manager.start()
        assert manager.state is RuntimeState.DEGRADED
        status = manager.get_status()
        assert status["services"]["broken"]["started"] is False
        assert "boom" in status["services"]["broken"]["last_error"]

    def test_register_after_start_rejected(self):
        manager = EdgeRuntimeManager(EdgeConfig())
        manager.register(ServiceSpec(name="a"))
        manager.start()
        with pytest.raises(RuntimeError):
            manager.register(ServiceSpec(name="b"))

    def test_duplicate_service_names_rejected(self):
        manager = EdgeRuntimeManager(EdgeConfig())
        manager.register(ServiceSpec(name="dup"))
        with pytest.raises(ValueError):
            manager.register(ServiceSpec(name="dup"))


# --- health checks -----------------------------------------------------------------

class TestEdgeHealthCheck:
    def test_all_healthy_reports_healthy(self):
        manager = EdgeRuntimeManager(EdgeConfig())
        manager.register(ServiceSpec(name="a", health=lambda: True))
        manager.register(ServiceSpec(name="b", health=lambda: True))
        manager.start()
        report = manager.health_check()
        assert report["overall"] == "healthy"
        assert report["services"] == {"a": "healthy", "b": "healthy"}

    def test_unhealthy_probe_degrades(self):
        flag = {"ok": True}
        manager = EdgeRuntimeManager(EdgeConfig())
        manager.register(ServiceSpec(name="flaky", health=lambda: flag["ok"]))
        manager.start()
        assert manager.state is RuntimeState.RUNNING
        flag["ok"] = False
        report = manager.health_check()
        assert report["overall"] == "degraded"
        assert manager.state is RuntimeState.DEGRADED

    def test_health_probe_exception_counts_as_unhealthy(self):
        manager = EdgeRuntimeManager(EdgeConfig())
        manager.register(ServiceSpec(name="exploding", health=lambda: (_ for _ in ()).throw(RuntimeError("x"))))
        manager.start()
        report = manager.health_check()
        assert report["overall"] == "degraded"

    def test_mark_degraded_external_hook(self):
        manager = EdgeRuntimeManager(EdgeConfig())
        manager.register(ServiceSpec(name="svc", health=lambda: True))
        manager.start()
        manager.mark_degraded("drift_alert")
        assert manager.state is RuntimeState.DEGRADED
        assert manager.health_check()["degraded_reasons"] == ["drift_alert"]

    def test_health_check_samples_metrics(self):
        monitor = ResourceMonitor(interval_seconds=1.0)
        manager = EdgeRuntimeManager(EdgeConfig(), monitor=monitor)
        manager.register(ServiceSpec(name="svc", health=lambda: True))
        manager.start()
        report = manager.health_check()
        assert report["metrics"]["timestamp"]
        assert monitor.latest() is not None

    def test_get_status_fields(self):
        manager = EdgeRuntimeManager(EdgeConfig())
        manager.register(ServiceSpec(name="svc", health=lambda: True))
        manager.start()
        status = manager.get_status()
        assert {"state", "services", "uptime_s", "config", "monitor"} <= set(status)
        assert status["config"]["device"] == "cuda"


# --- resource monitoring ---------------------------------------------------------

class TestResourceMonitor:
    def test_metrics_fields_complete(self):
        monitor = ResourceMonitor(interval_seconds=1.0)
        metrics = monitor.sample()
        assert isinstance(metrics, RuntimeMetrics)
        payload = metrics.as_dict()
        for field in (
            "timestamp", "cpu_percent", "memory_mb", "gpu_memory_mb",
            "latency_ms", "request_count", "error_count",
        ):
            assert field in payload

    def test_request_accounting(self):
        monitor = ResourceMonitor(interval_seconds=1.0)
        monitor.record_request(10.0)
        monitor.record_request(20.0)
        monitor.record_request(error=True)
        metrics = monitor.sample()
        assert metrics.request_count == 3
        assert metrics.error_count == 1
        assert metrics.latency_ms == pytest.approx(15.0)

    def test_latency_none_without_requests(self):
        monitor = ResourceMonitor(interval_seconds=1.0)
        assert monitor.sample().latency_ms is None

    def test_throughput_positive_after_requests(self):
        monitor = ResourceMonitor(interval_seconds=1.0, window_seconds=60.0)
        for _ in range(50):
            monitor.record_request(5.0)
        metrics = monitor.sample()
        assert metrics.requests_per_second > 0.0

    def test_history_capped(self):
        monitor = ResourceMonitor(interval_seconds=1.0, max_history=5)
        for _ in range(9):
            monitor.sample()
        assert len(monitor.history()) == 5

    def test_gpu_memory_none_without_cuda(self):
        import torch

        monitor = ResourceMonitor(interval_seconds=1.0)
        metrics = monitor.sample()
        if not torch.cuda.is_available():
            assert metrics.gpu_memory_mb is None
        else:
            assert metrics.gpu_memory_mb >= 0.0

    def test_snapshot_shape(self):
        monitor = ResourceMonitor(interval_seconds=1.0)
        monitor.sample()
        snapshot = monitor.snapshot()
        assert snapshot["history_size"] == 1
        assert snapshot["latest"]["request_count"] == 0


# --- container readiness ----------------------------------------------------------

class TestContainerReadiness:
    def test_dockerfile_exists_with_required_directives(self):
        dockerfile = ROOT / "docker/edge-runtime/Dockerfile"
        assert dockerfile.exists()
        content = dockerfile.read_text(encoding="utf-8")
        assert "HEALTHCHECK" in content
        assert 'VOLUME ["/models"]' in content  # read-only artifact mount point
        assert "CMD" in content

    def test_healthcheck_script_probes_endpoint(self):
        import importlib.util

        script = ROOT / "docker/edge-runtime/healthcheck.py"
        spec = importlib.util.spec_from_file_location("edge_healthcheck", script)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        ok, detail = module.probe("http://127.0.0.1:1/health", timeout_s=0.2)
        assert ok is False and detail.startswith("probe_failed")

    def test_edge_service_health_endpoints(self):
        from fastapi.testclient import TestClient

        app = create_edge_app(EdgeConfig(device="cpu"))
        client = TestClient(app)
        started = client.post("/admin/start").json()
        assert started["state"] in ("RUNNING", "DEGRADED")
        health = client.get("/health").json()
        assert health["overall"] in ("healthy", "degraded")
        ready = client.get("/ready").json()
        assert ready["status"] == "ready"
        client.post("/admin/stop")
        assert client.get("/ready").json()["status"] == "not_ready"

    def test_edge_service_restart_endpoint(self):
        from fastapi.testclient import TestClient

        app = create_edge_app(EdgeConfig(device="cpu"))
        client = TestClient(app)
        client.post("/admin/start")
        restarted = client.post("/admin/restart").json()
        assert restarted["state"] in ("RUNNING", "DEGRADED")