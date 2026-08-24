# AI Model Questions

## Why did the original PatchCore baseline fail?

Its steel image scores were inverted/overlapped: image AUROC `0.4817`, anomaly median below normal median, and all 6,666 anomalies below the frozen threshold. Local pixel ranking remained useful, so the root problem was image-level representation and aggregation geometry rather than absence of all anomaly signal.

Trace: [Failure analysis](../steel-patchcore-failure-analysis.md)

## Why could threshold tuning not fix it?

Thresholds choose an operating point on a score ordering; they cannot repair AUROC below random or make anomaly scores rank above normal scores. Changing the threshold would hide, not solve, representation failure.

## Why DINOv2 instead of fine-tuning WideResNet?

The controlled objective was to test frozen representation quality without supervised anomaly learning or holdout leakage. DINOv2-S/B improved the ordering, and ViT-B improved small-defect Q1. Fine-tuning was outside the experiment boundary and would add label and generalization risks.

Trace: [Why DINOv2](../decisions/why-dinov2.md)

## What exactly does ZCA adapt?

It estimates train-normal feature mean/covariance and transforms the metric space before L2 normalization and cosine 1-NN. It does not update backbone weights. Its mean, matrix, regularization rule, and downstream bank are frozen and hash-bound.

Trace: [Why ZCA](../decisions/why-zca.md)

## How was holdout leakage prevented?

Canonical manifests separate train, development, test-normal, and sealed recovery anomaly roles. Adaptation uses train normals; development chooses the method; full-development confirms it; one-shot holdout evaluation occurs after freeze with access count and hashes recorded.

Trace: [Recovery-holdout protocol](../steel-patchcore-d3-recovery-holdout-protocol.md)

## Why does D3 have low recall at its threshold despite good AUROC?

AUROC measures ranking across thresholds. The frozen operating point is the maximum train-normal score, intentionally conservative and therefore high. The project reports both rank validity and confusion metrics instead of treating them as interchangeable.

Trace: [Full-development results](../steel-patchcore-d3-full-development-results.md)

## Why was localization separated from image scoring?

D3-ZCA was strong for global anomaly ordering but weak for defect-pixel alignment. R-L3 produced strong localization but weaker image AUROC. A dual branch preserves the strongest representation for each objective and enforces zero image-score change.

Trace: [Why dual branch](../decisions/why-dual-branch.md)

## What would invalidate the current model evidence?

Artifact/hash mismatch, role-confused or duplicated evaluation records, non-finite values, a changed threshold, holdout reuse for candidate search, or material site distribution shift outside the qualified evidence.

## What is still required before a physical deployment?

Site-shift validation, actual camera/process qualification, operating-point acceptance, throughput/latency on target hardware, SAT, continuous monitoring evidence, and explicit production authorization.

Trace: [Production readiness](../d3-production-readiness-report.md) · [Maturity report](../industrial-platform-maturity-report.md)
