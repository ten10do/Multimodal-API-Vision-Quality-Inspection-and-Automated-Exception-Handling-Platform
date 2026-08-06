# Benchmark Summary

All numbers are from **real runs** on this machine (Windows + RTX 5060 8 GB,
CPU fallback noted). Latencies are grouped by component and are **never mixed**
across benchmarks. Raw JSON: `docs/phase6-benchmark.json`,
`docs/phase7-benchmark.json`, `docs/phase8-e2e.json`,
`docs/copilot-eval.json`, `docs/phase3-benchmark.md`.

## YOLO (known-defect detection)

| Metric | Value | Notes |
|---|---|---|
| Dataset | NEU-DET steel surface (6 defect classes) | train split |
| Precision | 0.81 | baseline run (Phase 1) |
| Recall | 0.78 | baseline run |
| mAP50 | 0.82 | |
| mAP50-95 | 0.51 | |
| Per-class AP50 | crazing 0.80 · inclusion 0.85 · patches 0.79 | representative |
| Latency (RTX 5060, 640 letterbox) | mean 15.4 ms · p50 13.3 ms · p95 21.5 ms | `docs/phase6-benchmark.json` |

## PatchCore (unknown-anomaly detection)

| Metric | Value | Notes |
|---|---|---|
| Dataset / domain | **MVTec AD bottle** (benchmark only) | steel-domain accuracy **NOT validated** |
| Image AUROC | 1.000 | on MVTec bottle |
| Pixel AUROC | 0.986 | |
| AUPRO | 0.955 | |
| Latency (224×224) | mean 755 ms · p95 791 ms | host CPU/GPU per Phase 6 run |
| Promotion to steel | **Blocked** (`steel_domain_validated=false`) | domain gate |

## Pipeline (frame → inference → rule → DB → WS)

| Metric | Value | Notes |
|---|---|---|
| Throughput | 4.85 inspections/s | Phase 3 benchmark, GPU |
| E2E P50 / P95 | 47.4 ms / 73.8 ms | after httpx pooled-client fix |
| E2E before fix | avg 561.7 ms / P95 822 ms | **~8.8× improvement** from client reuse |
| Queue behavior | bounded queue + worker; backpressure | `simulator/run_pipeline.py` |

## Industrial (PLC / MES, real simulators)

| Metric | Value |
|---|---|
| Decision→command creation | 0.002 ms |
| Command→PLC ACK | mean 239.8 ms · p50 256.7 ms · p95 276.3 ms |
| MES sync | mean 257.3 ms · p95 271.1 ms |
| Industrial decision total (PLC+MES) | mean 844.8 ms · p95 951.6 ms |
| Command success rate | 1.0 (excl. NOT_INTEGRATED) |
| Duplicate suppression (idempotency) | 50/50 |
| Fault injection (6 scenarios) | all pass — REJECT/HELD/RELEASED/SAFE_HOLD/COMMAND_FAILED/MES-FAILED |

## Copilot (deterministic eval, 46 fixed cases — FakeLlmProvider)

| Metric | Value |
|---|---|
| Tool selection accuracy | 1.0 |
| Numeric grounding accuracy | 1.0 |
| Required fact coverage | 1.0 |
| **Unsupported critical numeric claim rate** | **0.0** (target 0) |
| Forbidden (causal) claim rate | 0.0 |
| Tool error recovery rate | 1.0 |
| Avg tool calls | 1.96 |
| Total latency P50 / P95 | 25.1 ms / 69.7 ms |
| Tokens (in / out) | 5824 / 5072 |

Real-LLM provider latency is **not reported** here — that gate is
`REAL_LLM_GATE_NOT_RUN` (no external endpoint on this machine).

## MLflow / Registry evidence

- YOLO baseline run `8299fc50` — artifact SHA-256 matches the deployment
  manifest (`9c9409aa…`).
- PatchCore baseline run `c5bb57a6` — SHA-256 matches (`79375393…`).
- Phase 8 E2E: register → promote v1 → inspection (deployment 2026.08.1) →
  metrics (inference 609, error 0.0%, p95 570 ms) → promote v2 → rollback →
  1.0.0 restored.
