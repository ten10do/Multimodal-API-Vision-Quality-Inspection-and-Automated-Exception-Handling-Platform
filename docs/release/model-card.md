# D3 1.3 Model Card

## Identity and status

- Model: `steel-patchcore-d3-candidate@1.3.0-candidate.1`
- Release package: `steel-patchcore-d3-release@1.3.0`
- Status: `RELEASE_CANDIDATE_PACKAGE`; not production
- Frozen threshold: `0.8471092581748962`
- Encoder: DINOv2-B/14

## Intended use

Candidate-only anomaly inspection of 1600×256 steel-strip images from the validated input protocol. The image score supports anomaly triage. The heatmap supports localization and human review. The system must retain the decision layer and fail-safe PLC/MES mapping.

It is not authorized for autonomous production deployment, threshold modification, online learning, retraining, or use on an unqualified camera/domain.

## Branches and metrics

| Branch | Representation | Metric | Result |
|---|---|---:|---:|
| Image | D3-ZCA, cosine 1-NN, A0 | Image AUROC | 0.817907171428 |
| Localization | R-L3 multi-scale fusion | Pixel AUROC | 0.924139385743 |
| Localization | R-L3 multi-scale fusion | AUPRO | 0.799398106991 |

The image score mismatch count during dual-branch evaluation was zero. The image gate and threshold are immutable in this release.

## Evaluation lineage

- Dual-branch evaluation: PASS.
- Production readiness: `PRODUCTION_CANDIDATE_QUALIFIED`.
- Factory acceptance: `FACTORY_ACCEPTANCE_PASS`.
- The earlier 1.2 operational heatmap failure remains historical evidence. It was addressed by the independently evaluated R-L3 localization branch; the failed report itself is not represented as a PASS.

## Limitations and risks

- Image AUROC is below perfect separation; ambiguous anomalies require human review.
- Qualification uses the sealed Severstal-derived protocol and does not establish performance on a new mill, camera, illumination system or material grade.
- FAT used measured-latency replay for the virtual eight-hour workload, not an eight-hour wall-clock production soak.
- Confidence is a threshold-margin indicator, not a calibrated probability.
- Distribution shift is not proof of quality degradation without ground-truth review.
