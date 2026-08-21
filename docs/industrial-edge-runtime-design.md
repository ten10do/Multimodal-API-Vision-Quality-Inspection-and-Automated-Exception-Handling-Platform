# Industrial Edge Runtime Design (`industrial_runtime/`)

Status: **peripheral edge-deployment simulation layer** around the frozen D3 release
(`steel-patchcore-d3-release@1.3.0`). It never touches the D3 model, weights, whitening
artifact, bank, threshold, feature extractor, production candidate manifest or any
existing evaluation metric.

## 1. 工业边缘部署架构 (Edge deployment architecture)

```
+----------------+     +---------------------+     +--------------------+
| Camera Adapter | --> |   Edge Runtime      | --> | D3 Inference       |
| (virtual today)|     | EdgeRuntimeManager  |     | Service (unchanged)|
+----------------+     | + ResourceMonitor   |     +---------+----------+
                       | + EdgeConfig (yaml) |               |
                       +---------------------+               v
                                  |                  +------------------+
                                  v                  | Decision Service |
                          Health / Metrics           +--------+---------+
                                  |                           |
                          +-------+--------+          +------+------+
                          | /runtime page  |          | PLC  | MES  |
                          +----------------+          +-------------+
```

Package layout:

| File | Responsibility |
|---|---|
| `industrial_runtime/config.py` | `EdgeConfig` — config-driven settings loaded from YAML |
| `industrial_runtime/edge_config.yaml` | the packaged default configuration |
| `industrial_runtime/runtime_manager.py` | `EdgeRuntimeManager` lifecycle + `ServiceSpec` |
| `industrial_runtime/resource_monitor.py` | `ResourceMonitor` + `RuntimeMetrics` |
| `industrial_runtime/service.py` | FastAPI health/status endpoints for container mode |
| `docker/edge-runtime/Dockerfile` | industrial-PC container simulation (slim image) |
| `docker/edge-runtime/healthcheck.py` | Docker `HEALTHCHECK` probe (stdlib HTTP) |

## 2. Runtime lifecycle

States: `INIT -> STARTING -> READY -> RUNNING (or DEGRADED) -> STOPPED`.

```python
manager = EdgeRuntimeManager(EdgeConfig.load(), monitor=ResourceMonitor(...))
manager.register(ServiceSpec(name="camera", start=..., stop=..., health=...))
manager.register(ServiceSpec(name="decision_engine", health=...))
manager.register(ServiceSpec(name="plc_link", health=...))
manager.start(); manager.health_check(); manager.restart(); manager.stop()
```

* Services are injected as callables (`ServiceSpec`), so the manager is transport-agnostic.
* `start()` brings services up in registration order; a failing service **degrades** the
  runtime instead of crashing it (`DEGRADED` = running with alerts).
* `health_check()` probes every service, samples one `RuntimeMetrics`, and moves a live
  runtime between RUNNING/DEGRADED accordingly.
* `stop()` shuts down in reverse order; `stop()`/`start()` are idempotent; `restart()`
  composes both. `mark_degraded(reason)` is the external fail-safe hook.

## 3. Resource monitoring

`RuntimeMetrics` fields: `timestamp`, `cpu_percent`, `memory_mb`, `gpu_memory_mb`,
`latency_ms`, `request_count`, `error_count` (+ derived `requests_per_second`).

* CPU/memory via `psutil` (process-scoped); GPU memory via CUDA only when a device is
  actually present (CPU-only edge units report `null`); every probe degrades gracefully -
  monitoring must never crash the runtime.
* Latency/throughput come from `record_request(latency_ms, error=...)` calls made by the
  serving path, aggregated over a rolling window.
* The bounded history feeds `/api/runtime/history` on the dashboard.

## 4. Configuration (config-driven, nothing hardcoded)

`industrial_runtime/edge_config.yaml`:

```yaml
runtime:    {device: cuda, batch_size: 1, timeout_ms: 3000}
logging:    {level: INFO}
monitoring: {interval_seconds: 5}
drift:      {psi_warning: 0.10, psi_critical: 0.25, cosine_warning: 0.05,
             cosine_critical: 0.20, mean_dist_warning: 0.30, mean_dist_critical: 1.00,
             min_baseline_samples: 200}
```

Override order: explicit path argument > `INDUSTRIAL_EDGE_CONFIG` env var > packaged
default. Invalid values (bad device/log level, non-positive sizes, warning>=critical)
are rejected at load time.

## 5. Container readiness simulation

The Dockerfile models an industrial PC deployment without building a CUDA image:

* slim `python:3.11-slim` base; runtime deps only;
* D3 artifacts are **never baked in** - they mount read-only at `/models`
  (`IVQC_D3_CANDIDATE_MANIFEST=/models/.../manifest.json`);
* `HEALTHCHECK` runs `healthcheck.py`, which probes `/health` and exits non-zero unless
  the runtime reports healthy/degraded;
* `python -m industrial_runtime.service` exposes `/health`, `/ready`,
  `/admin/start|stop|restart`, `/status`.

## 6. Failure handling

| condition | behavior |
|---|---|
| service fails to start | runtime DEGRADED, error recorded per service |
| health probe false/raises | runtime DEGRADED (production continues + alert) |
| external fail-safe (`mark_degraded`) | DEGRADED with reason (e.g. drift alert) |
| stop/restart during any state | idempotent, reverse-order shutdown |

The e2e test proves the fail-close chain: a manager that cannot reach RUNNING blocks the
line entirely (zero products inspected).

## 7. Tests

`inference-service/tests/test_edge_runtime.py` — 25 tests: config loading/override/
validation, lifecycle transitions and idempotency, ordered start / reverse stop, degraded
paths, health probes, status fields, resource metrics/accounting/history cap, GPU-null on
CPU, Dockerfile directives, healthcheck probe behavior, service endpoints.
