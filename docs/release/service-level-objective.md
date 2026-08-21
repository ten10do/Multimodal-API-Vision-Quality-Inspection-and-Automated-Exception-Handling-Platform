# D3 Release Service Level Objective

These are proposed approval SLOs for `steel-patchcore-d3-release@1.3.0`, not evidence of an existing production SLA.

## Availability and correctness

- Monthly inference API availability target: **99.5%**, excluding approved maintenance.
- Successful-request target: **≥99.5%** for valid in-protocol images.
- Artifact integrity target: **100%** verified loads; any mismatch blocks readiness.
- Decision safety target: **100%** of uncertain/system-failure states map to `HOLD`, never implicit `RELEASE`.

## Latency

For a warmed, healthy qualified GPU and one 1600×256 image:

- D3 dual-branch inference p50: **≤600 ms**.
- D3 dual-branch inference p95: **≤750 ms**.
- D3 dual-branch inference p99: **≤1000 ms**.
- Request timeout: **2000 ms**; a timeout fails closed and enters recovery/review.
- Queue latency p95 under the FAT load profile: **≤100 ms**.

Qualification reference: the 240-request stability profile measured p50 533.44 ms, p95 546.67 ms, and p99 562.13 ms. FAT injected failures produced longer recovery-tail latency and are reported separately.

## Rollback target

- Detection-to-safe-hold target: **≤1 minute** after a blocking integrity or readiness failure.
- Candidate rollback RTO: **≤15 minutes** from authorization.
- Rollback verification target: **≤5 minutes** for manifest/hash validation and smoke inference.
- Recovery point: no loss of prediction, operator feedback, command, or audit records acknowledged before rollback.

## Monitoring and alerting

Monitor request count, error rate, p50/p95/p99 latency, queue depth, timeout count, GPU utilization/memory, CPU memory, model/artifact versions and hashes, input statistics, feature distribution and score distribution.

- Immediate critical alert: artifact mismatch, missing artifact, model-load failure, non-finite output, health/readiness failure.
- Warning: p95 latency above 750 ms for 5 minutes, error rate above 0.5% for 5 minutes, or detected distribution shift.
- Critical: p99 above 2000 ms, artifact integrity failure, or error rate above 2% for 5 minutes.
- Drift alerts create investigation/human review only. They cannot retrain, tune thresholds, or promote a model automatically.
