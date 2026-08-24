# Representation Investigation

## Investigation sequence

The recovery work treated each explanation as a falsifiable hypothesis rather than immediately changing the model:

1. Audit the actual feature extraction and aggregation code.
2. Recompute metrics from frozen evidence before re-inference.
3. Test aggregation-only alternatives.
4. Test scale, context, feature layer, and bank-coverage hypotheses under fixed controls.
5. Compare frozen DINOv2 capacities.
6. Test train-normal metric adaptation.
7. Separate image ranking from pixel localization when one representation cannot optimize both.

## Findings

| Investigation | Finding | Consequence |
|---|---|---|
| Threshold audit | Threshold was correctly bound to the bank; anomaly ordering was wrong | Do not tune threshold to hide representation failure |
| Aggregation recovery | Max/top-k/generalized-mean changes did not recover the gate | Keep failure status; investigate features |
| WRN layer/scale/context | Some local signal, insufficient image validity | No production candidate |
| Bank coverage | Low sampling fraction was a risk, not sufficient root cause | Avoid assuming a larger bank would solve ordering |
| DINOv2 S/B | Frozen representations improved overall and Q1 ordering | Continue controlled capacity comparison |
| DINOv2-B + ZCA | Strong overall and small-defect signal | Confirm at full scale and sealed holdout |
| D3 heatmap post-processing | Normalization, interpolation, smoothing, and stitching did not recover localization | Do not manufacture localization with post-processing |
| Localization-aware features | R-L3 passed pixel metrics but not standalone image AUROC | Use dual-objective architecture |

## Design outcome

The final design preserves D3-ZCA A0 as the immutable image branch and adds R-L3 only as a localization branch. This is a deliberate multi-objective decomposition:

- image ranking needs globally discriminative anomaly evidence;
- pixel localization needs spatially aligned multi-scale features;
- forcing one score representation to optimize both degraded one objective;
- branch isolation makes regressions measurable: image-score mismatch count must remain zero.

## Negative results as evidence

Failed experiments remain documented because they constrain future changes. They show why threshold tuning, extra smoothing, interpolation changes, or a larger-looking architecture are not automatically valid fixes.

## Evidence trail

- [Failure analysis](../steel-patchcore-failure-analysis.md)
- [Representation investigation](../steel-patchcore-representation-investigation.md)
- [Heatmap root cause](../d3-heatmap-root-cause.md)
- [Localization investigation](../d3-localization-representation-investigation.md)
- [Failed experiments summary](../decisions/failed-experiments.md)
