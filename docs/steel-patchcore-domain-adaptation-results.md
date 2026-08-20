# Steel PatchCore — Steel-Domain Adaptation Results (D0 WRN / D1 S / D2 B / D3 B+ZCA)

Verdict: **`STEEL_DOMAIN_ADAPTATION_SIGNAL_FOUND`**

Adaptation Gate (frozen): D3 Image AUROC ≥ 0.75 **AND** D3 − D2 ≥ +0.05.

| Reference | Image AUROC | Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|---|
| **D0** — WRN S2 (frozen) | 0.6029 | 0.4790 | 0.5305 | 0.6145 | 0.7876 |
| **D1** — DINOv2 ViT-S/14 (frozen) | 0.6699 | 0.5843 | 0.6086 | 0.6607 | 0.8261 |
| **D2** — DINOv2 ViT-B/14 (frozen) | 0.6938 | 0.6043 | 0.6318 | 0.7035 | 0.8358 |
| **D3** — DINOv2 ViT-B/14 + train-normal ZCA | **0.8208** | 0.7341 | 0.7959 | 0.8324 | 0.9209 |

- Δ vs D2 = **+0.1270** (≥ +0.05) and D3 AUROC 0.8208 ≥ 0.75 → **gate passed**.
- Strong signal (≥ 0.80): **true**.
- Small-defect adaptation signal (D3 Q1 − D2 Q1 ≥ +0.05): **true** (+0.1298; Q1 vs D0 +0.2551).

## D3 candidate (frozen)

- Backbone unchanged: **DINOv2 ViT-B/14** (`dinov2_vitb14`, embed 768, patch 14, weights
  SHA256 `0b8b82f8…fad8c73`), raw `x_norm_patchtokens` (CLS/register excluded), 18×18=324
  tokens/tile (256→252 bilinear).
- Adaptation: **train-normal-only ZCA covariance whitening** — `x → μ_train`, then
  `Σ_train^(−1/2)(x−μ)` where `Σ_reg = Σ + εI`, `ε = 1e-6·trace(Σ)/d` (frozen rule,
  not a searched hyperparameter), 768→768 (no PCA dim reduction).
- Statistics: streaming Chan-style mean/covariance in float64 over 2,268,000 train-normal
  tokens (never materialized in RAM). Whitening artifact SHA256 `e4b211f9…32bdc5e28`;
  condition number 6.10e7; eigenvalues 2.03e-6 … 1.24e2.
- Numerical sanity (45,360-token train sample): whitened cov diag p50 = 0.997, off-diag
  |·| max = 0.172, max |mean| = 0.169 — all finite; numerical gate **healthy**.
  (The frozen ε rule yields a partial whitening — near-degenerate directions retain residual
  variance/mean — but the transform is numerically stable under the documented bounds.)
- Bank: reservoir 50k/seed 42 over whitened+L2-normalized tokens; bank SHA256 `a64c8a53…3e3a5933`.
- Distance/metrics: per-patch L2 + cosine 1-NN; 7 tiles A0 max; threshold = max(train-normal
  scores) = 0.8270 (train-only, diagnostic).

## Diagnostic metrics (train-only operating point)

- Image AUROC = 0.8208; anomaly-median − normal-median = +0.0207.
- Normal (300): min 0.6734 / p50 0.7971 / p95 0.8269 / p99 0.8362 / max 0.8439.
- Anomaly (1000): min 0.7630 / p50 0.8178 / p95 0.8383 / p99 0.8450 / max 0.8546.
- Operating point (threshold 0.8270): TP=241, FP=14, TN=286, FN=759; precision 0.9451,
  recall 0.2410, F1 0.3841, Normal FPR 0.0467, Anomaly Recall 0.2410.

## Interpretation

Train-normal ZCA whitening of frozen DINOv2 ViT-B/14 patch tokens is the strongest development
representation observed so far (0.8208 image AUROC, vs 0.6938 unwhitened B/14 and 0.6029 WRN),
recovering both overall and small-defect (Q1) separability. It is a **domain-representation
metric-geometry adaptation**, not fine-tuning: the backbone and the anomaly-detection algorithm
(memory bank / cosine 1-NN / A0) are unchanged, and the whitening statistics use train-normal
data only.

## Prior-phase status (unchanged)

- D1 (`DOMAIN_REPRESENTATION_GATE_FAILED`, `SMALL_DEFECT_SIGNAL = TRUE`) and
  D2 (`DOMAIN_REPRESENTATION_CAPACITY_GATE_FAILED`) remain frozen historical results.
- Canonical PatchCore cross-check remains **`CANONICAL_PATCHCORE_REFERENCE_BLOCKED`** (not FAILED).

Holdout access count: **0** (test_normal and recovery_holdout_anomaly never accessed).
Full-development confirmation and holdout evaluation are deferred to the next authorization.