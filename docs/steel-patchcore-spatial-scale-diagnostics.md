# Steel PatchCore — Spatial Scale Diagnostics (offline)

- Schema: `steel_patchcore_spatial_scale_diagnostics_v1`
- Dev anomalies analyzed: 3333 (frozen recovery_dev_anomaly)
- Quartile boundaries (area ratio) reused from frozen manifest: Q1=0.010888671875, Q2=0.02656494140625, Q3=0.07214111328125
- Holdout access: 0

## 1. Feature-grid geometry (nominal, from frozen 256x256 tile path)

| Grid | shape | stride (px) | nominal cell footprint |
|---|---|---|---|
| layer2 | 32x32 | 8 | 8x8 px |
| layer3 (native) | 16x16 | 16 | 16x16 px |
| layer3 (as used: bilinear upsample) | 32x32 | 8 (grid) | 16x16 px info source, interpolated |

These are **feature-grid stride / nominal footprint**, not theoretical or
effective receptive field. Effective RF is a model property that is not
computable from tiling code alone and is intentionally NOT claimed here.

## 2. Defect-to-grid (largest connected component bbox)

| Quartile | count | median bbox W (px) | median bbox H (px) | median side (px) | median W (l2 cells) | median H (l2 cells) | median W (l3 cells) | median H (l3 cells) |
|---|---|---|---|---|---|---|---|---|
| Q1 | 833 | 29.0 | 100.0 | 102.0 | 3.62 | 12.50 | 1.81 | 6.25 |
| Q2 | 833 | 35.0 | 216.0 | 218.0 | 4.38 | 27.00 | 2.19 | 13.50 |
| Q3 | 833 | 55.0 | 254.0 | 255.0 | 6.88 | 31.75 | 3.44 | 15.88 |
| Q4 | 834 | 314.0 | 239.0 | 314.0 | 39.25 | 29.88 | 19.62 | 14.94 |
| overall | 3333 | 49.0 | 201.0 | 252.0 | 6.12 | 25.12 | - | - |

## 3. Fraction of defects whose largest-component bbox fits within N cells

(1 layer2 cell = 8px side; 2x2 = 16px; 4x4 = 32px)

| Group | <=1 cell (8px) | <=2x2 cells (16px) | <=4x4 cells (32px) |
|---|---|---|---|
| overall | 0.000 | 0.000 | 0.006 |
| 1 | 0.001 | 0.001 | 0.023 |
| 2 | 0.000 | 0.000 | 0.000 |
| 3 | 0.000 | 0.000 | 0.000 |
| 4 | 0.000 | 0.000 | 0.000 |

## 4. Interpretation guardrails

- This is a **geometry overlap analysis**, not a proof of feature receptive field.
- It does NOT assert that small defects are 'sub-resolution'; it quantifies
  how defect bbox sizes map onto the current grids.
