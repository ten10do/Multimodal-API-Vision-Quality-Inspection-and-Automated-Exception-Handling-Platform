# D3 Heatmap Localization Root-Cause Investigation

Verdict: **`FAILED`**

## Executive finding

The D3 image detector remains unchanged, but no allowed heatmap-side candidate reaches the frozen localization gate. The localization failure is therefore not recovered by normalization, interpolation, local smoothing, or overlap feathering alone.

## 1. Patch distance source

The current map starts from each DINOv2 ViT-B/14 `x_norm_patchtokens` patch token. The frozen train-normal mean and ZCA matrix are applied, each transformed patch is L2-normalized, and its value is `1 - max cosine similarity` to the frozen, equally transformed 50,000-row bank. Seven 18x18 distance grids are produced per original.

## 2. Raw versus whitened distance

The current distance is a **whitened-feature cosine-1NN distance**. `raw_anomaly_map` means only that map values have not yet been min-max normalized; it does not mean an unwhitened-feature distance. The artifact contains no unwhitened D3 normal bank, so creating one would violate the frozen-bank constraint. H1 therefore preserves the existing raw 18x18 distance cells with nearest projection; H2 is the current whitened-distance bilinear projection.

## 3. Normalization audit

H0 and H2 are identical in Pixel AUROC (0.600212) and AUPRO (0.265291). Per-image min-max is monotonic, and qualification metrics were calculated from the unquantized raw map, not the 8-bit PNG. **No normalization mismatch was found.**

## 4. Resize interpolation audit

Nearest raw-cell projection H1 versus current bilinear H2 changed Pixel AUROC by -0.019891 and AUPRO by -0.026275. Removing bilinear interpolation makes both metrics worse, so current interpolation is beneficial and is not the primary failure.

## 5. Tile stitching audit

Linear feathering H5 versus current overlap mean H2 changed Pixel AUROC by -0.001425 and AUPRO by -0.001877. Only the final 192-pixel overlap can change, and the result does not recover localization. Tile stitching is not the primary failure.

## 6. Candidate results

| Candidate | Pixel AUROC | AUPRO | Image score | Verdict |
|---|---:|---:|---|---|
| H0 | 0.600212 | 0.265291 | UNCHANGED | FAILED |
| H1 | 0.580320 | 0.239016 | UNCHANGED | FAILED |
| H2 | 0.600212 | 0.265291 | UNCHANGED | FAILED |
| H3 | 0.652970 | 0.329387 | UNCHANGED | FAILED |
| H4 | 0.656437 | 0.336327 | UNCHANGED | FAILED |
| H5 | 0.598787 | 0.263414 | UNCHANGED | FAILED |

## 7. Localization gate

- Paired baseline Pixel AUROC: `0.834361`; 95% minimum: `0.792643`.
- Paired baseline AUPRO: `0.587579`; 95% minimum: `0.558200`.
- Best allowed candidate: `H4` with Pixel AUROC `0.656437` and AUPRO `0.336327`.
- Overall verdict: **`FAILED`**.

## 8. Frozen image-score protection

All 3,924 sealed image scores remain D3 A0 and reproduce image AUROC `0.817907171428` for every H0-H5 candidate. Threshold and all seven artifact hashes are unchanged.

## Root cause

The strongest supported cause is a representation/objective mismatch: the ZCA-whitened nearest-neighbor patch distance is effective as a global-max image ranking signal, but its spatial ordering is not aligned with defect pixels. The allowed post-processing candidates cannot manufacture missing patch-level localization information. This conclusion does not authorize a new bank, backbone, training, threshold change, or production promotion.

## Evidence

- `docs/d3-heatmap-recovery-results.json`
- `docs/d3-heatmap-recovery-test-report.json`
