# D3 Localization-Aware Representation Investigation

Verdict: **`PASS`**

## 1. Frozen baseline audit

D3 remains DINOv2-B/14 final patch tokens → frozen train-normal ZCA → per-patch L2 → cosine-1NN → A0 global maximum. Its candidate artifact, threshold, and sealed image scores are unchanged.

- Image AUROC: `0.817907171428`
- Threshold: `0.8471092581748962`

## 2. Representation experiments

| Candidate | Standalone image AUROC | Δ vs D3 | Mean abs score Δ | Pixel AUROC | AUPRO | Standalone gate | Dual gate |
|---|---:|---:|---:|---:|---:|---|---|
| R-L1 | 0.558232 | -0.259675 | 0.489103 | 0.890777 | 0.724124 | FAILED | PASS |
| R-L2 | 0.666612 | -0.151295 | 0.343071 | 0.900884 | 0.748056 | FAILED | PASS |
| R-L3 | 0.661694 | -0.156214 | 0.452400 | 0.924139 | 0.799398 | FAILED | PASS |
| R-L4 | 0.649724 | -0.168183 | 0.425478 | 0.914756 | 0.774088 | FAILED | PASS |

## 3. Dual-objective design

The image branch is always the frozen D3 A0 score. Each representation is evaluated as an independent pixel branch. Standalone representation image AUROC is diagnostic only and never replaces D3 scoring.

- Separation supported: `True`
- Passing pixel branches: `['R-L1', 'R-L2', 'R-L3', 'R-L4']`
- Preferred pixel branch by localization metrics: `R-L3`
- Every representation fails the standalone three-metric gate because its own image AUROC is below 0.75; every dual configuration passes when the immutable D3 image branch is retained.

## 4. Localization gate

PASS requires Pixel AUROC ≥ 0.75, AUPRO ≥ 0.50, and immutable D3 image AUROC ≥ 0.75.

Overall verdict: **`PASS`**.

## 5. Isolation

R-L1/R-L2 banks are runtime-only experimental evidence. R-L4 reuses a previously frozen experimental DINOv2-S bank. None is registered as, copied into, or substituted for the D3 candidate artifact. No fine-tuning, supervised training, threshold tuning, or production promotion occurred.

## Evidence

- `docs/d3-localization-representation-results.json`
- `docs/d3-localization-representation-test-report.json`
- Runtime checkpoints and experimental banks remain under ignored `model-training/runs/steel-d3-localization-representation/`.
