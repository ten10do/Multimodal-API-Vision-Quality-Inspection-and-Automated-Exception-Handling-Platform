"""Minimal edge service entry point for the container deployment simulation.

Exposes the health endpoints the Docker healthcheck probes. The D3 artifacts
are expected as READ-ONLY volume mounts (never baked into the image); this
service only reports runtime status - inference itself stays in the unchanged
inference service.
"""
from __future__ import annotations

import argparse

from fastapi import FastAPI

from .config import EdgeConfig
from .resource_monitor import ResourceMonitor
from .runtime_manager import EdgeRuntimeManager, ServiceSpec


def create_app(config: EdgeConfig | None = None) -> FastAPI:
    cfg = config or EdgeConfig.load()
    monitor = ResourceMonitor(interval_seconds=cfg.monitoring_interval_seconds)
    manager = EdgeRuntimeManager(cfg, monitor=monitor)
    # Core edge services; the inference client is provided by the deployment
    # (IVQC_D3_CANDIDATE_MANIFEST points at the read-only artifact mount).
    manager.register(ServiceSpec(name="config", health=lambda: True))
    manager.register(ServiceSpec(name="resource_monitor", health=lambda: monitor.latest() is not None))
    app = FastAPI(title="IndustrialVision-QC Edge Runtime", version="1.0.0")

    @app.get("/health")
    async def health() -> dict:
        return manager.health_check()

    @app.get("/ready")
    async def ready() -> dict:
        status = manager.get_status()
        return {
            "status": "ready" if manager.state.value in ("RUNNING", "DEGRADED") else "not_ready",
            "state": manager.state.value,
            "device": cfg.device,
        }

    @app.post("/admin/start")
    async def start() -> dict:
        return manager.start()

    @app.post("/admin/stop")
    async def stop() -> dict:
        return manager.stop()

    @app.post("/admin/restart")
    async def restart() -> dict:
        return manager.restart()

    @app.get("/status")
    async def status() -> dict:
        return manager.get_status()

    app.state.edge_manager = manager
    return app


def main() -> int:  # pragma: no cover - container entry point
    parser = argparse.ArgumentParser(description="Industrial edge runtime service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    import uvicorn

    config = EdgeConfig.load()
    app = create_app(config)
    app.state.edge_manager.start()  # bring the runtime up with the server
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
