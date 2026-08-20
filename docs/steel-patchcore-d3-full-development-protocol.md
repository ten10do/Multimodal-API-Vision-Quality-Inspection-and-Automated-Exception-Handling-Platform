# Steel PatchCore — D3 Full-Development Confirmation Protocol

## Purpose
Not a candidate search. Answer one question with D3 fully frozen: does the strong
diagnostic recovery signal reproduce when the train/development scale is expanded
from the diagnostic subset to the complete authorized development split?

## Frozen history (never overwritten)
- `STEEL_DOMAIN_VALIDATION_FAILED` (steel-patchcore 1.0.0)
- `RECOVERY_AGGREGATION_GATE_FAILED`
- `REPRESENTATION_BASE_FEATURE_GATE_FAILED`
- `SPATIAL_REPRESENTATION_GATE_FAILED`
- `CANONICAL_PATCHCORE_REFERENCE_BLOCKED`
- `DOMAIN_REPRESENTATION_GATE_FAILED`
- `DOMAIN_REPRESENTATION_CAPACITY_GATE_FAILED`
- Diagnostic D3 = `STEEL_DOMAIN_ADAPTATION_SIGNAL_FOUND` (diagnostic signal ≠ domain validation)

## D3 method identity (frozen at diagnostic commit `85d1457`)
- Backbone: DINOv2 ViT-B/14 (`dinov2_vitb14`), weights SHA256 `0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73`.
- Extraction: `forward_features(x)["x_norm_patchtokens"]` (CLS/register excluded), 256→252 bilinear, 18×18=324 tokens/tile, 768-d.
- Adaptation: train-normal ZCA covariance whitening → per-patch L2 → cosine 1-NN → A0 global max.
- Whitening: `Σ_reg = Σ + εI`, `ε = 1e-6·trace(Σ)/d` (frozen rule, NOT searched), 768→768, no PCA/truncation.
- Statistics: streaming Chan-style float64; full token matrix never materialized.
- Bank: reservoir Algorithm R, budget 50000, seed 42.
- Tiling: 7 tiles x0 ∈ {0,256,512,768,1024,1280,1344}.
- Threshold: `max(train-normal image scores)`, train-only.

## Authorized development data boundaries
- train_normal = 4721 (full), validation_normal = 590 (full), recovery_dev_anomaly = 3333 (full).
- Sealed (forbidden, `HOLDOUT_ACCESS_COUNT = 0`): test_normal 591, recovery_holdout_anomaly 3333.

## Stages (checkpointed, resumable per-original)
- A: streaming whitening statistics over all 4721 train normals → `stats.npz` (every 50 images).
- B: fit ZCA → `whitening.npz` + manifest; pre-L2 numerical sanity on a deterministic, uniformly
  strided sample (stride 118, ≈41 originals) → `sanity.json`.
- C: full 50k reservoir bank over whitened+L2 train tokens (resume via `bank_progress.npz` + RNG state).
- D: calibration scores for all 4721 train normals → `scores.json`.
- E: scores for 590 validation normal + 3333 dev anomaly → `scores.json`.
- Evaluate + quartiles + confirmation gate → `results.json` / `results.md`.

## Numerical gate (frozen, non-retroactive)
- All finite; regularized eigenvalues > 0; deterministic; no catastrophic explosion.
- Pre-L2 whitened sanity reports: max/mean abs mean, cov diag min/p50/p95/max,
  off-diagonal abs mean/p95/p99/max, non-finite count, eigenvalue min/max/condition number.
- Blocker (non-finite / non-positive eigenvalue / explosion) → `D3_FULL_DEV_NUMERICAL_BLOCKED` STOP.

## Confirmation gate (frozen before run)
- Primary: full-dev Image AUROC ≥ 0.75 **AND** anomaly median > normal median
  → `D3_FULL_DEVELOPMENT_CONFIRMED`, else `D3_FULL_DEVELOPMENT_FAILED`.
- Infinite-precision AUROC is not a goal: Recall/F1 are reported but not gated (threshold is
  intentionally conservative), because this phase validates representation ranking, not operating-point calibration.
- Secondary (report only): full-dev Q1 AUROC ≥ 0.65 → `SMALL_DEFECT_FULL_DEV_SIGNAL = TRUE`.
- Regression signal (report): diagnostic AUROC (0.8208) − full AUROC. A ≥0.05 drop flags
  `DEVELOPMENT_SCALE_REGRESSION` but does NOT move the gate.

## Quartile evaluation
Reuse the frozen area-ratio quartile definition (`np.quantile` at 0.25/0.50/0.75 over the
3333 dev-anomaly area ratios). Quartile boundaries are not redefined.

## Holdout isolation (fail-closed)
The runner rejects any test_normal or recovery_holdout_anomaly ID at membership level.

## Git
Branch `feat/steel-patchcore-validity-recovery-v1.2`; no `git add .`; no merge main; no
MLOps candidate; no production promotion. Whitening matrix / bank / checkpoints / weights are
runtime-only and Git-ignored.