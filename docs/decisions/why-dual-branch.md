# Decision: Why a Dual Branch

## Context

D3-ZCA achieved image-level validity, but its original patch-distance heatmap failed the localization gate. Post-processing candidates could not recover the missing spatial ordering.

Localization-aware representations then produced strong pixel metrics but weaker standalone image AUROC:

| Branch | Image AUROC | Pixel AUROC | AUPRO |
|---|---:|---:|---:|
| D3 image branch | `0.817907` | not selected for final localization | not selected |
| R-L3 localization representation | `0.661694` | `0.924139` | `0.799398` |

## Decision

Keep D3-ZCA A0 as the only image-score branch and use R-L3 as an independent localization branch.

## Rationale

- Image discrimination and spatial localization are different objectives.
- R-L3 can improve review evidence without changing the frozen decision score.
- Integration is testable through an exact invariant: zero D3 score mismatches.
- Separate artifact hashes and manifests make each branch auditable.

## Rejected alternatives

- Replace D3 image score with R-L3: standalone image AUROC is below the gate.
- Normalize or smooth the original heatmap more aggressively: allowed H0–H5 candidates all failed.
- Jointly tune threshold and localization: violates the frozen candidate boundary and confounds objectives.

## Trace

[Localization investigation](../d3-localization-representation-investigation.md) · [Dual-branch protocol](../d3-dual-branch-protocol.md) · [Evaluation report](../dual-branch-evaluation-report.md)
