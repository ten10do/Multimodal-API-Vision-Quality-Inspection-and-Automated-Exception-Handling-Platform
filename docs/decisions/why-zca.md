# Decision: Why ZCA

## Context

DINOv2-B/14 improved steel anomaly ordering but remained at image AUROC `0.6938`. Normal steel texture contains high-variance feature directions that distort cosine nearest-neighbor geometry.

## Decision

Apply a frozen ZCA transform estimated only from train-normal DINOv2-B/14 patch tokens, then per-patch L2 normalization and cosine 1-NN.

## Evidence

- D2 unwhitened ViT-B/14 image AUROC: `0.6938`.
- D3 train-normal ZCA image AUROC: `0.8208` (`+0.1270`).
- D3 Q1 AUROC: `0.7341`, up `+0.1298` over D2.
- Full-development confirmation: `0.8362`.
- Sealed recovery holdout: `0.8179071714`.

## Why this is controlled domain adaptation

ZCA changes metric geometry, not backbone parameters. The covariance regularization rule is frozen, anomaly labels are excluded, holdout access is audited, and the resulting mean/matrix is hashed as an immutable artifact.

## Trade-offs

- The covariance is ill-conditioned and requires explicit numerical regularization and finite checks.
- The transform is dataset-specific and must not be silently recomputed at a new site.
- It improves rank separation but does not make the conservative operating threshold optimal.

## Trace

[D3 domain adaptation](../ai/d3-domain-adaptation.md) · [Full-development protocol](../steel-patchcore-d3-full-development-protocol.md)
