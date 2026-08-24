# Anomaly Detection

## Problem definition

Steel inspection must detect abnormal surfaces without assuming every defect appearance is represented in supervised labels. The anomaly channel therefore learns normal feature geometry and scores deviations, while the known-defect YOLO path remains a separate capability.

## Baseline

The first steel PatchCore baseline used frozen ImageNet WideResNet-50-2 `layer2 + layer3` features, a 50,000-row normal memory bank, cosine 1-NN, seven tiles, and maximum patch distance as the image score.

Formal steel-domain evaluation showed:

| Metric | Baseline result |
|---|---:|
| Image AUROC | `0.4817` |
| TP / FP / TN / FN | `0 / 1 / 590 / 6666` |
| Anomaly recall at frozen threshold | `0.0` |
| Mean per-image pixel AUROC | `0.8319` |
| Mean per-image AUPRO | `0.5838` |

The local map retained useful signal, but the global image ordering was inverted and overlapped. A threshold change could not repair AUROC below random ordering.

## D3 image branch

The recovered image branch freezes:

- `dinov2_vitb14` weights and final normalized patch tokens;
- train-normal-only ZCA mean and whitening matrix;
- per-patch L2 normalization;
- cosine 1-NN against a 50,000-row seed-42 bank;
- seven 256×256 tiles and A0 global maximum;
- threshold `0.8471092581748962`.

Sealed recovery evidence reports image AUROC `0.817907171428`, 95% bootstrap CI `[0.7967992294, 0.8377211833]`, and correct median ordering.

## Localization branch

Heatmap post-processing could not recover the original D3 localization objective. The final candidate therefore separates objectives:

- R-L1: block-7 tokens at 252 input resolution;
- R-L2: final tokens at 448 input resolution;
- R-L3: equal mean of resized R-L1/R-L2 raw distance maps, followed by overlap-mean stitching.

R-L3 achieves pixel AUROC `0.924139385743` and AUPRO `0.799398106991`. It never contributes to image score or threshold.

## Validation discipline

- Adaptation statistics use train-normal samples only.
- Full-development confirmation precedes one-shot sealed holdout access.
- Every manifest and artifact is bound by SHA-256.
- Candidate loading rejects missing, mismatched, malformed, non-finite, or mutated artifacts.
- Production promotion is a separate governance decision; evaluation does not imply deployment.

## Evidence

- [Baseline failure analysis](../steel-patchcore-failure-analysis.md)
- [D3 system design](../steel-patchcore-d3-system-design.md)
- [Dual-branch protocol](../d3-dual-branch-protocol.md)
- [Dual-branch evaluation](../dual-branch-evaluation-report.md)
