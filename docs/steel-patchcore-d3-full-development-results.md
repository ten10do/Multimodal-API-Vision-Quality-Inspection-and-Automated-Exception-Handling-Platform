# Steel PatchCore — D3 Full-Development Confirmation Results

Verdict: **`D3_FULL_DEVELOPMENT_CONFIRMED`**

- Small-defect full-dev signal: **`TRUE`** (Q1 0.7372 ≥ 0.65)
- Canonical PatchCore status: `CANONICAL_PATCHCORE_REFERENCE_BLOCKED`
- Holdout access count: **0**

## Diagnostic vs Full (confirmation)

| | Image AUROC | Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|---|
| Diagnostic D3 (1000/300/1000) | 0.8208 | 0.7341 | 0.7959 | 0.8324 | 0.9209 |
| **Full D3 (4721/590/3333)** | **0.8362** | 0.7372 | 0.8087 | 0.8676 | 0.9312 |

- Δ (diagnostic − full) = **−0.0154** (full slightly higher → no regression; `regression_flagged = false`).
- Primary Gate: full AUROC 0.8362 ≥ 0.75 **and** anomaly median (0.8173) > normal median (0.7946) → **pass**.

## Frozen D3 method (unchanged from diagnostic, now re-fit at full scale)

- Backbone: DINOv2 ViT-B/14 (`dinov2_vitb14`), weights SHA256 `0b8b82f8…fad8c73`, raw `x_norm_patchtokens` (CLS/register excluded), 18×18=324 tokens/tile (256→252 bilinear).
- Adaptation: train-normal ZCA whitening → per-patch L2 → cosine 1-NN → A0 global max.
- Whitening re-fit on **all 4721 train normals** (streaming Chan-style float64, 10,707,228 tokens,
  never materialized): ε = 2.029e-6, eigenvalues 2.03e-6 … 1.28e2, condition number 6.29e7;
  artifact SHA256 `c8d9d2ed…7cae8c3a`.
- Pre-L2 numerical sanity (deterministic uniform stride sample, 41 originals / 92,988 tokens):
  whitened cov diag p50 = 0.982, off-diag |·| max = 0.181, non-finite = 0 → **healthy**.
- Bank: reservoir 50k / seed 42 over whitened+L2 train tokens; SHA256 `40fe4333…3cbbda`;
  candidate patches 10,707,228.
- Threshold: **0.8471** = max of all 4721 train-normal image scores (train-only).

## Full development metrics (590 normal + 3333 anomaly)

- Image AUROC = **0.8362**; anomaly median − normal median = **+0.0227** (correct ordering).
- Normal (590): min 0.6799 / p50 0.7946 / p95 0.8220 / p99 0.8322 / max 0.8452.
- Anomaly (3333): min 0.7396 / p50 0.8173 / p95 0.8365 / p99 0.8429 / max 0.8503.
- Train calibration distribution (4721): min 0.6525 / p50 0.7848 / p95 0.8164 / p99 0.8287 / max 0.8471.
- Operating point @ 0.8471 (report-only; threshold is intentionally conservative via max-train rule):
  TP=7, FP=0, TN=590, FN=3326; precision 1.0000, recall 0.0021, F1 0.0042, Normal FPR 0.0000.

## Quartile results (area-ratio quartiles of the 3333 dev anomalies, boundaries unchanged)

| Quartile | Count | normal vs quartile AUROC |
|---|---|---|
| Q1 | 833 | 0.7372 |
| Q2 | 833 | 0.8087 |
| Q3 | 833 | 0.8676 |
| Q4 | 834 | 0.9312 |

## Interpretation

Train-normal ZCA whitening of frozen DINOv2 ViT-B/14 patch tokens **reproduces at full
development scale** — the full-split AUROC (0.8362) meets the frozen confirmation gate
(AUROC ≥ 0.75, anomaly median > normal median) and is slightly higher than the diagnostic
estimate (0.8208). Ranking/ordering validity is confirmed; the near-zero recall at the
max-train threshold is expected (see §17 of the protocol: operating-point calibration is
not this phase's objective), so AUROC + score ordering are the primary evidence.

## Prior-phase status (unchanged)

- `STEEL_DOMAIN_VALIDATION_FAILED`, `RECOVERY_AGGREGATION_GATE_FAILED`,
  `REPRESENTATION_BASE_FEATURE_GATE_FAILED`, `SPATIAL_REPRESENTATION_GATE_FAILED`,
  `CANONICAL_PATCHCORE_REFERENCE_BLOCKED`, `DOMAIN_REPRESENTATION_GATE_FAILED`,
  `DOMAIN_REPRESENTATION_CAPACITY_GATE_FAILED`, and diagnostic
  `STEEL_DOMAIN_ADAPTATION_SIGNAL_FOUND` all remain frozen.

Holdout access count: **0** (test_normal and recovery_holdout_anomaly never accessed).
This is development confirmation, **not** steel-domain validation; one-shot recovery holdout
is the next (separately authorized) step and was not run.