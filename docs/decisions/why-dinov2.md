# Decision: Why DINOv2

## Context

The WideResNet PatchCore baseline retained local mask-ranking signal but failed image-level separation on steel. The decision required a stronger frozen representation without supervised fine-tuning or holdout leakage.

## Decision

Use DINOv2 ViT-B/14 patch tokens as the D3 backbone representation.

## Evidence

- Controlled WRN reference D0 image AUROC: `0.6029`.
- Frozen DINOv2 ViT-S/14 D1: `0.6699`.
- Frozen DINOv2 ViT-B/14 D2: `0.6938`.
- ViT-B/14 also improved Q1 small-defect AUROC from `0.5843` to `0.6043`.

The improvement was useful but insufficient by itself, leading to the separate ZCA decision.

## Trade-offs

- Higher embedding dimension and compute cost than ViT-S/14.
- Artifact size and runtime qualification become more important at the edge.
- Frozen pretrained features still require site-specific validation; no universal generalization is assumed.

## Rejected alternatives

- Threshold adjustment: cannot repair AUROC ordering.
- Continue only with WRN layer combinations: controlled gates remained below requirements.
- Fine-tuning: outside the frozen, normal-only adaptation boundary.

## Trace

[Domain-adaptation results](../steel-patchcore-domain-adaptation-results.md) · [Representation investigation](../steel-patchcore-representation-investigation.md)
