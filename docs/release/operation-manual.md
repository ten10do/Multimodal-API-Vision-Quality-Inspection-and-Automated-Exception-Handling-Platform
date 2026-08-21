# D3 Release Operation Manual

## Startup checklist

1. Confirm release and candidate versions.
2. Verify release-manifest and artifact hashes.
3. Confirm the frozen threshold is `0.8471092581748962`.
4. Confirm GPU availability and successful model load.
5. Run a known smoke request and verify heatmap dimensions are 256×1600.
6. Keep the candidate in candidate-only state until a separate human production authorization exists.

## Per-request records

Record timestamp, trace ID, image ID, latency, image score, decision, heatmap path, model version and artifact version. Never overwrite the raw model evidence after human review.

## Decision handling

- `PASS`: eligible for `RELEASE` only after the complete decision pipeline succeeds.
- `FAIL`: confirmed product defect; map to `REJECT`.
- `REVIEW_REQUIRED`: map to `HOLD` until human resolution.
- Timeout, missing file, load failure, unknown state or hash mismatch: fail closed to `HOLD`.

## Monitoring

Monitor request count, p50/p95/p99 latency, error rate, GPU memory, artifact hash, score distribution, input statistics and the localization feature probe. A distribution shift raises a warning and review task; it never triggers threshold tuning or retraining.

## Human feedback

Feedback records must retain trace ID, image ID, operator, timestamp, prediction snapshot, feedback type (`operator_review`, `false_positive`, `false_negative`) and annotation. Feedback is evidence for later governed analysis, not an online training signal.

## Shutdown

Stop accepting new requests, drain the queue, retain logs, and close the runtime. Do not delete artifacts or monitoring evidence during shutdown.
