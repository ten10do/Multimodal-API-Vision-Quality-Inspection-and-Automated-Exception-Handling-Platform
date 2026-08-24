# D3 Domain Adaptation

## Experimental ladder

Each stage changed one representation assumption while holding the anomaly algorithm, tiling, bank budget, seed, distance, and aggregation policy controlled.

| Reference | Representation | Image AUROC | Q1 AUROC |
|---|---|---:|---:|
| D0 | WideResNet-50-2 steel baseline | `0.6029` | `0.4790` |
| D1 | DINOv2 ViT-S/14 | `0.6699` | `0.5843` |
| D2 | DINOv2 ViT-B/14 | `0.6938` | `0.6043` |
| D3 | DINOv2 ViT-B/14 + train-normal ZCA | `0.8208` | `0.7341` |

D3 improved `+0.1270` over unwhitened ViT-B/14 and `+0.2179` over the controlled WRN reference. The full-development confirmation reached image AUROC `0.8362`; the sealed holdout reached `0.8179071714`.

## Why DINOv2

The baseline failed because its feature geometry did not separate steel normals and anomalies at image level. DINOv2 patch tokens provided a stronger frozen representation for texture and structure without supervised steel fine-tuning. ViT-B/14 also improved the small-defect quartile over ViT-S/14.

This was an evidence-led choice, not a general claim that transformers always outperform convolutional features. The controlled D0–D2 comparison is the basis.

## Why ZCA

Even DINOv2-B remained only moderately separated. ZCA estimates train-normal mean and covariance, regularizes the covariance, and transforms directions into a metric space aligned with normal steel texture statistics. After per-patch L2 normalization, cosine 1-NN becomes less dominated by high-variance normal directions.

ZCA is not backbone fine-tuning:

- backbone weights remain frozen;
- statistics are computed from train normals only;
- anomaly labels and holdout images are excluded;
- the transform, regularization rule, bank seed, and downstream distance are frozen artifacts.

## Leakage and reproducibility controls

1. Canonical source and recovery split manifests are hash-bound.
2. Train-normal statistics are streamed in float64 without materializing all tokens.
3. The 50,000-row bank uses deterministic reservoir sampling with seed 42.
4. The holdout access count is audited and holdout evaluation is one-shot.
5. Full precision hashes and the threshold are stored in the candidate manifest.

## What the result establishes

The result establishes rank-based anomaly separation for the frozen dataset and holdout. It does not establish universal site generalization, an optimal operating threshold, or production authorization. The conservative threshold has low anomaly recall and is reported honestly as a separate operating-point limitation.

## Evidence

- [Domain-adaptation results](../steel-patchcore-domain-adaptation-results.md)
- [Full-development results](../steel-patchcore-d3-full-development-results.md)
- [Recovery-holdout results](../steel-patchcore-d3-recovery-holdout-results.md)
- [D3 system design](../steel-patchcore-d3-system-design.md)
