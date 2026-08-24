# Edge Runtime

## Runtime model

`EdgeRuntimeManager` coordinates injected camera, inference, decision, and link services through:

```text
INIT → STARTING → READY → RUNNING
                         ↘ DEGRADED
RUNNING / DEGRADED → STOPPED
```

Services start in registration order and stop in reverse order. Start, stop, and restart are idempotent. A failed service or health probe degrades the runtime and records the reason.

## Configuration

`industrial_runtime/edge_config.yaml` defines device, batch size, timeout, logging, monitoring interval, and drift bands. Invalid device names, non-positive sizes, or inconsistent warning/critical thresholds are rejected at load time.

## Resource monitoring

The bounded telemetry history records:

- process CPU and memory;
- GPU memory when CUDA is available;
- request count and error count;
- inference latency and requests per second;
- service health and last error.

Monitoring probes degrade gracefully and cannot crash the runtime they observe.

## Container boundary

The edge Docker image contains runtime code and a standard-library health probe. Model artifacts are mounted read-only at runtime. Health endpoints expose `/health`, `/ready`, `/status`, and controlled lifecycle operations.

## Current boundary

The Docker setup simulates industrial-PC packaging. Target hardware qualification, driver lifecycle, redundant power/network, watchdog integration, and long-duration site soak remain deployment activities.

## Trace

[Detailed edge design](../industrial-edge-runtime-design.md) · [`industrial_runtime/`](../../industrial_runtime/) · [`docker/edge-runtime/`](../../docker/edge-runtime/)
