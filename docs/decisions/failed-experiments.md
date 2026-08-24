# Failed Experiments

Negative results are retained as engineering evidence. A failed gate is not rewritten as success, and a new candidate is not registered until the stated objective passes.

| Experiment family | Result | What it ruled out |
|---|---|---|
| Steel WRN PatchCore baseline | Image AUROC `0.4817`, zero anomaly recall | The original ImageNet feature geometry was not valid for steel image ranking |
| Aggregation recovery | Gate failed | Top-k/generalized aggregation alone could not repair feature ordering |
| WRN feature-layer variants | Gates failed | Simple layer selection was insufficient |
| Spatial scale/context variants | Gates failed | Resolution/context changes alone did not recover validity |
| DINOv2-S and unwhitened DINOv2-B | Improved but below adaptation gate | Backbone capacity alone was insufficient |
| Original D3 heatmap H0–H5 | Best pixel AUROC `0.656437`, AUPRO `0.336327` | Normalization, interpolation, smoothing, and stitching were not the root fix |
| R-L1/R-L2/R-L3/R-L4 as standalone image scorers | All image AUROC below `0.75` | Localization features should not replace the D3 image branch |

## Decision rules preserved

- Do not lower a threshold to make a failed representation appear valid.
- Do not use holdout data for candidate search.
- Do not interpret a larger bank or backbone as evidence without a controlled gate.
- Do not allow a localization improvement to change the image score.
- Do not silently skip an unavailable integration gate.

## Source evidence

- [Steel baseline failure](../steel-patchcore-failure-analysis.md)
- [Representation investigation](../steel-patchcore-representation-investigation.md)
- [Heatmap root cause](../d3-heatmap-root-cause.md)
- [Localization representation results](../d3-localization-representation-investigation.md)
