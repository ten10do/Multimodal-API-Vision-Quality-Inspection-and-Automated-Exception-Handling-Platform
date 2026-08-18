# Steel-domain PatchCore Failure Analysis

Scope: frozen `steel-patchcore` 1.0.0 on the formal Severstal test split. All values below come from the completed checkpoint; no original image was re-evaluated.

## Operating point

- Threshold: 0.490039
- Formal population: 591 test normal + 6666 test anomaly = 7257
- TP 0 / TN 590 / FP 1 / FN 6666
- Precision 0.0 / Recall 0.0 / F1 0.0
- Normal FPR 0.0017 / Anomaly Recall 0.0
- Image AUROC 0.4817

## Representative checkpoint cases

| outcome | image ID | image score | pixel AUROC | AUPRO |
|---|---|---:|---:|---:|
| TP (highest score) | none | - | - | - |
| TN (lowest score) | `3e1a0403b` | 0.251204 | n/a | n/a |
| FP (highest score) | `385f291af` | 0.490773 | n/a | n/a |
| FN (highest score) | `a4f1180ba` | 0.477310 | 0.919063 | 0.792464 |
| FN (lowest score) | `e86b2f3e2` | 0.255288 | 0.942334 | 0.808067 |

## Quantitative failure analysis

- Score separation is inverted/overlapped: anomaly median 0.358937 is below test-normal median 0.368732; anomaly maximum 0.47731 is below the frozen threshold, while test-normal maximum 0.490773 exceeds it.
- The operating point therefore detects zero of 6,666 anomalies and raises one false alarm. This is an image-level representation/calibration failure, not a sample-count artifact.
- Localization retains signal: mean per-image Pixel AUROC is 0.8319 and mean per-image AUPRO is 0.5838. Image score has only weak Pearson correlation with per-image Pixel AUROC (0.0996) and AUPRO (0.1266).
- Defect mask area ranges from 115 to 368240 pixels (median 10934). Mean image scores by increasing area quartile are 0.347143, 0.349639, 0.359009, 0.384548. Smaller defects receive lower scores on average, although every area quartile is missed.

## Qualitative axes and limits

- Small/low-contrast defects: the area relationship supports a scale limitation; direct contrast was not stored and is not inferred here.
- Steel texture and illumination: the severe normal/anomaly score overlap is compatible with domain-feature sensitivity, but this checkpoint alone cannot attribute individual errors to either cause.
- Tile edges and stitch overlap: the frozen seven-tile/mean-overlap protocol was preserved. No heatmaps were persisted, so edge-specific error frequency cannot be measured without prohibited re-inference.
- Annotation ambiguity: no label-quality adjudication was performed; the audit does not relabel source masks.
- PatchCore limitation: frozen ImageNet features retain useful local ranking but max-over-tiles image aggregation does not produce domain-separating scores on this split.
- Threshold limitation: max(train-normal) is correctly bound to the frozen bank, but all anomaly scores fall below it. Changing it now would violate the frozen evaluation and would not repair AUROC 0.4817.

## Metric semantics

Image AUROC is pooled over the 7,257 formal test originals. Pixel AUROC and AUPRO are means of per-anomaly-image metrics; they are not pooled-pixel estimates. Validation normals are diagnostic only and do not enter formal image or confusion metrics.

## Verdict

`STEEL_DOMAIN_VALIDATION_FAILED`

The baseline is not eligible for MLOps CANDIDATE registration. Production remains untouched.
