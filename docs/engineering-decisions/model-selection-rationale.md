# Model Selection Rationale

This document explains the model-family choices for steel anomaly inspection. It does not claim a controlled YOLO-versus-D3 benchmark where none exists, and it does not generalize beyond the committed evaluation protocol.

## Why Not Use YOLO as the Primary Anomaly Detector?

### Observed

The target problem includes unknown or weakly specified surface anomalies, while a YOLO detector is organized around supervised, predefined classes and localized annotations. The repository contains known-defect detection and model-registry examples, but the committed steel recovery reports do not contain a same-data, same-objective, controlled YOLO-versus-D3 experiment.

### Decision

Do not claim that D3 numerically outperforms YOLO. Select anomaly detection as the primary unknown-defect path because it can learn a normal reference and flag departures without requiring every future defect class to be enumerated in advance. Retain supervised detection as a complementary approach when stable defect classes and trustworthy annotations exist.

### Future

If a site has sufficient class-balanced boxes or masks, evaluate supervised detection and anomaly detection under one predeclared business protocol: known-defect coverage, unknown-defect behavior, localization, false-negative cost, labeling cost, latency, and maintenance burden.

Source for system responsibility boundaries: [anomaly detection](../ai/anomaly-detection.md) and [system architecture](../architecture/system-architecture.md).

## Why PatchCore Was the Baseline

### Observed

PatchCore provides a transparent normal-memory-bank baseline with nearest-neighbor patch evidence and no supervised defect training. On the formal steel split, the initial WideResNet PatchCore result failed image ranking: image AUROC `0.4817` and anomaly recall `0.0`. At the same checkpoint, mean per-anomaly-image Pixel AUROC was `0.8319` and mean AUPRO was `0.5838`, demonstrating useful local structure despite the invalid image score.

### Decision

Use PatchCore as an investigative baseline and preserve its memory-bank/cosine-neighbor structure while diagnosing the feature representation. Do not promote the failed baseline, lower its threshold, or treat local metrics as proof of image-level validity.

### Future

Future baselines may compare coreset selection, alternative distances, or other anomaly families, but only through isolated protocols that retain the failed baseline as a fixed reference.

Source: [steel PatchCore failure analysis](../steel-patchcore-failure-analysis.md).

## Why DINOv2?

### Observed

Under the controlled domain-representation comparison, D0 WideResNet S2, D1 DINOv2 ViT-S/14, and D2 DINOv2 ViT-B/14 achieved image AUROC `0.6029`, `0.6699`, and `0.6938`. Q1 AUROC increased from D0 `0.4790` to D1 `0.5843` and D2 `0.6043`. DINOv2 improved the representation, but unwhitened D2 still remained below the `0.75` adaptation gate.

### Decision

Choose frozen DINOv2 ViT-B/14 patch tokens as the representation foundation. The larger frozen encoder gave the best tested unadapted ordering and small-defect signal without introducing supervised fine-tuning or holdout leakage.

### Future

New encoder families or smaller edge variants require controlled accuracy, memory, latency, and site-generalization evidence. DINOv2-B remains frozen for this candidate; it is not silently exchanged at deployment time.

Source: [domain-adaptation results](../steel-patchcore-domain-adaptation-results.md).

## Why ZCA Whitening?

### Observed

Unwhitened DINOv2-B reached image AUROC `0.6938`. Train-normal ZCA raised diagnostic D3 to `0.8208`, an improvement of `+0.1270`, and raised Q1 to `0.7341`. Full-development confirmation reached `0.8362`, while the sealed recovery holdout reached `0.8179071714`.

ZCA changes metric geometry rather than backbone weights. The committed method estimates regularized mean/covariance from train-normal patch tokens, applies whitening, then per-patch L2 normalization and cosine 1-NN.

### Decision

Select ZCA because it produced the decisive controlled improvement while preserving a frozen backbone and normal-only adaptation boundary. Store whitening statistics as an immutable, hashed artifact tied to the dataset and candidate manifest.

### Future

Do not recompute ZCA automatically from live traffic. A new site or material may justify a new normal-only transform, but that work must produce a new dataset identity, artifact, candidate, sealed evaluation, and rollback target.

Sources: [domain-adaptation results](../steel-patchcore-domain-adaptation-results.md), [full-development confirmation](../steel-patchcore-d3-full-development-results.md), [sealed holdout report](../steel-patchcore-d3-recovery-holdout-results.md), and [machine-readable holdout evidence](../steel-patchcore-d3-recovery-holdout-results.json).

## Selection Summary

| Question | Observed evidence | Decision |
|---|---|---|
| Primary unknown-defect method | No controlled YOLO-versus-D3 result exists; the task includes anomalies outside fixed class definitions | Use anomaly detection for the unknown-defect path; keep supervised detection complementary |
| Initial anomaly baseline | PatchCore image AUROC `0.4817`, but local Pixel AUROC `0.8319` | Keep the transparent memory-bank baseline for diagnosis, not promotion |
| Representation | D0/D1/D2 image AUROC `0.6029/0.6699/0.6938` | Use frozen DINOv2-B patch tokens |
| Domain adaptation | D3 diagnostic/full/holdout image AUROC `0.8208/0.8362/0.8179071714` | Use train-normal ZCA as an immutable candidate artifact |
