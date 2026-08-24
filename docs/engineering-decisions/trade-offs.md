# Engineering Trade-offs

Industrial AI decisions optimize a system, not a single metric. The trade-offs below use committed evidence and preserve the distinction between measured behavior, current engineering policy, and future work.

## Accuracy vs Stability

### Observed

The sealed D3 image AUROC is `0.817907171428`, and dual-branch integration produced `0` image-score mismatches. The candidate performance report's 1,000-image cumulative run recorded latency p50/p95/p99 of `140.1992/145.2795/149.4903 ms`, `0` errors, and unchanged before/after artifact hashes. This is a local candidate benchmark, not a plant service-level guarantee.

### Decision

Prefer a reproducible frozen candidate with explicit limits over continuous changes that may improve a development score while invalidating runtime identity, latency evidence, or rollback. Accuracy improvements require a new version; stability is not maintained by editing the current artifact in place.

### Future

Profile on qualified site hardware and evaluate new candidates against accuracy, tail latency, memory, thermal behavior, endurance, and rollback together. A faster or more accurate candidate must still reproduce its artifact and decision evidence.

Sources: [dual-branch evaluation](../dual-branch-evaluation-report.md) and [candidate performance report](../d3-performance-report.json).

## Automation vs Human Review

### Observed

The FAT feedback evidence records three supported feedback modes—operator review, false positive, and false negative—and keeps `automatic_retraining=false`. Drift WARNING continues with alerting; CRITICAL maps to HOLD and human investigation rather than automatic recalibration.

### Decision

Automate deterministic acquisition, inference, trace creation, PLC/MES idempotency, and alerting. Keep uncertain disposition, false-alarm correction, missed-defect escalation, recovery, promotion, and retraining under explicit human authority. Human decisions append evidence instead of erasing the model observation.

### Future

Reduce review load only after measuring review agreement, queue delay, defect cost, and site risk under a controlled policy. Feedback may seed an offline candidate but never mutates the running model.

Sources: [FAT human-feedback report](../d3-fat-human-feedback-report.json), [human-review workflow](../d3-human-review-workflow.md), and [drift monitoring](../industrial/drift-monitoring.md).

## Model Complexity vs Deployment Cost

### Observed

DINOv2 ViT-S/14 and ViT-B/14 achieved image AUROC `0.6699` and `0.6938`; the larger encoder improved ordering but still required ZCA to reach D3 `0.8208`. The final R-L3 branch improved Pixel AUROC/AUPRO to `0.924139/0.799398`, while its standalone image AUROC was `0.661694`. The dual design therefore adds a localization representation without replacing the valid image branch.

### Decision

Accept additional encoder, whitening, bank, and localization complexity only where measured evidence supports a distinct objective. Keep the branches isolated so a localization improvement does not expand the image-decision risk surface. Deployment must verify exact artifacts and measure resource cost rather than assuming architectural gains are free.

### Future

Explore compression, shared feature computation, batching, smaller backbones, or reduced banks only as new candidates with measured image validity, localization, latency, memory, and rollback evidence. Do not optimize deployment cost by mutating D3 in place.

Sources: [domain-adaptation results](../steel-patchcore-domain-adaptation-results.md) and [localization investigation](../d3-localization-representation-investigation.md).

## False Positive vs False Negative

### Observed

The original baseline operating point produced normal FPR `0.0017` and anomaly recall `0.0`. After representation recovery, full-development D3 image AUROC reached `0.8362`, but the conservative max-train threshold `0.8471` produced FP `0`, TP `7`, FN `3,326`, and recall `0.0021` on 590 normal and 3,333 anomaly development images. Ranking validity and operating-point utility are therefore separate questions.

### Decision

Do not lower or tune the frozen threshold after seeing evaluation outcomes. Preserve the recorded operating point, use HOLD/human review for uncertainty, and report the recall limitation explicitly. In an industrial setting, false-positive cost, missed-defect cost, line-stop cost, and review capacity must be approved together rather than hidden behind AUROC.

### Future

A future threshold-calibration candidate may optimize an approved cost or recall/FPR objective using authorized development data and then validate once on a sealed set. It must create a new version and cannot revise the evidence of the current frozen release.

Sources: [baseline failure analysis](../steel-patchcore-failure-analysis.md) and [D3 full-development confirmation](../steel-patchcore-d3-full-development-results.md).

## Trade-off Policy

1. Never exchange auditability for an unversioned metric improvement.
2. Never exchange human authority for silent online adaptation.
3. Never exchange deployment cost for an unmeasured loss of image or pixel validity.
4. Never optimize false alarms without explicitly measuring missed-defect consequences.
5. Treat every changed model, artifact, threshold, representation, or policy as a new governed candidate.
