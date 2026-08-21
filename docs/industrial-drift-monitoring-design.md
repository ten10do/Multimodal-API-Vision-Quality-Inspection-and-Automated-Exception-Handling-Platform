# Industrial Drift Monitoring Design (`monitoring/drift/`)

Status: **peripheral data-drift monitoring layer**. It consumes only DINO feature
embeddings and never modifies the D3 model, weights, whitening artifact, bank, threshold,
feature extractor or any evaluation result.

## 1. Drift principle

Industrial image distributions shift (illumination drift, material lots, lens contamination,
camera aging). The layer watches the *feature space* the frozen D3 model operates in:

```
camera frame -> DINO feature embedding -> FeatureDriftCollector -> DriftDetector
    NORMAL   -> continue production
    WARNING  -> continue production + alert
    CRITICAL -> HOLD (reason DATA_DISTRIBUTION_SHIFT)  [fail-safe, never a model change]
```

The baseline is frozen reference statistics built once from normal production:

* `mean embedding` (per dimension)
* `variance` (per dimension)
* `sample count`

plus a capped raw-sample window for distribution-shaped metrics.

## 2. Metrics

| metric | definition | interpretation |
|---|---|---|
| **PSI** (headline) | Population Stability Index, decile bins; per embedding dimension, aggregated as the mean across dimensions. Also implemented analytically from normal-distribution statistics so the detector can run from summary stats alone | ~0 identical; empirical magnitude ≈ (shift in σ)² for a pure location shift |
| **PSI max** | worst single dimension | reported for transparency, excluded from the verdict (a max over hundreds of noisy estimates is dimension-count-dependent) |
| **Cosine distribution shift** | \|coherence(reference) − coherence(current)\| where coherence = mean cos(x, μ). The baseline is split in halves (μ fitted on one half) to remove the in-sample bias | detects concentration/shape change that mean-shift metrics miss |
| **Embedding mean distance** | RMS per-dimension standardized mean shift `sqrt(mean(((μ_cur−μ_base)/σ_base)²))` | ≈ shift magnitude in σ units; resampling noise stays ~1/√n |

## 3. Detector and thresholds (configurable)

`DriftDetector` evaluates the current window against the frozen baseline and takes the
**worst** of three checks (psi, cosine_shift, mean_distance). All bands come from
`edge_config.yaml` (`drift:` section) via `DriftThresholds.from_config`:

| check | NORMAL | WARNING | CRITICAL (defaults) |
|---|---|---|---|
| PSI | < 0.10 | 0.10 – 0.25 | ≥ 0.25 |
| cosine shift | < 0.05 | 0.05 – 0.20 | ≥ 0.20 |
| mean distance | < 0.30 | 0.30 – 1.00 | ≥ 1.00 |

Fewer than 30 current samples => `sufficient_data=false`, state NORMAL (missing data never
fabricates an alert). Every evaluation appends a `DriftReport` (state, all metric values,
per-check bands, alerts, sample counts, timestamp) to a bounded history for the dashboard.

## 4. Alert strategy & production handling (fail-safe)

| drift state | production action |
|---|---|
| NORMAL | continue |
| WARNING | **continue production + alert** (dashboard + report alerts) |
| CRITICAL | **HOLD** — subsequent inspections are bridged as inference failures with `kind="data_distribution_shift"`, which the (unchanged) decision engine maps to `HOLD · DATA_DISTRIBUTION_SHIFT`; the PLC receives `stop_signal` |

Hard rules:

* CRITICAL drift can never produce a PASS (verified by tests and the scenario simulation).
* The layer cannot and does not retrain, recalibrate, tune the threshold or touch any
  artifact — recovery from CRITICAL is a human/operational decision.
* WARNING is visibility only: scoring and decisions continue unchanged.

## 5. Dashboard integration

`/drift` page (hash route) shows the live drift state, PSI mean/max, cosine shift,
embedding distance, the configured thresholds, and the evaluation history table.
`/api/drift/status` + `/api/drift/history` back it; `/runtime` mirrors the edge runtime.

## 6. Simulation results (full-scale, seed 42)

`python -m monitoring.drift.scenarios --frames-a 10000` (report:
`runs/industrial-loop/drift_simulation_report.json`):

| scenario | frames | injected change | final state | psi_mean | mean distance | verdict |
|---|---|---|---|---|---|---|
| A normal production | 10000 | none | **NORMAL** | 0.0013 | 0.028 | pass |
| B brightness shift | 4000 | +0.40σ all dims | **WARNING** | 0.155 | 0.402 | pass (production continues + alert) |
| C material change | 4000 | +1.5σ, ×1.3 variance | **CRITICAL** | 1.436 | 1.501 | pass (HOLD ×8, reasons = DATA_DISTRIBUTION_SHIFT, 8 PLC stop_signals, zero PASS after critical) |

## 7. Tests

`inference-service/tests/test_drift_monitoring.py` — 34 tests: PSI (identity, monotonicity,
magnitude law, eps clipping, stats-vs-empirical, per-dim aggregation), cosine shift and
mean distance semantics, collector baseline/window validation, detector bands (normal /
warning / critical / worst-check / custom thresholds / insufficient data / history),
fail-safe decision bridge (DATA_DISTRIBUTION_SHIFT hold, default unchanged, critical
cannot pass, warning continues), dashboard endpoints, and fast scenario versions.
`test_edge_drift_e2e.py` adds the camera -> edge runtime -> D3 -> decision -> PLC chain
with mid-run CRITICAL drift failing the line closed.
