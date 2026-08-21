# Steel PatchCore D3 Operational Qualification Protocol

## Scope and immutable identity

This protocol qualifies `steel-patchcore-d3-candidate@1.2.0-candidate.1`, artifact version `d3-full-development-9b1ea19`, as a **candidate only**. The registered manifest, DINOv2-B weights, whitening transform, 50,000-row bank, A0 score, cosine 1-NN distance, seven-tile geometry, and threshold `0.8471092581748962` are read-only inputs.

Training, fine-tuning, backbone search, threshold calibration, artifact replacement, and production promotion are outside scope.

## Frozen execution plan

1. Performance uses cumulative prefixes of one deterministic, interleaved sealed-image sequence at 1, 10, 100, and 1,000 requests. It records p50/p95/p99 inference latency, process RSS, GPU memory, and GPU utilization. Model/artifact loading is outside the timed request interval.
2. Shadow validation reads all 591 sealed test-normal and 3,333 sealed recovery-holdout anomaly originals. Every result records timestamp, image ID, model version, artifact version, score, and a derived heatmap path. A checkpoint permits crash-safe resume. Input images and artifacts are never written.
3. Reproducibility compares every shadow score with the previously sealed D3 holdout score at absolute tolerance `1e-6`.
4. Heatmap evaluation uses the full-resolution raw stitched map. Pixel AUROC and AUPRO match the existing baseline evaluator: metrics are calculated per anomalous image, AUPRO integrates foreground recall through background FPR 0.3, and the report is their mean over 3,333 images.
5. Heatmap acceptance is paired by image ID against the existing steel PatchCore checkpoint. Both candidate mean pixel AUROC and candidate mean AUPRO must be greater than or equal to their paired baseline values. Failure is terminal evidence for this task and does not authorize tuning.
6. Monitoring records request count, latency, error rate, GPU memory, and artifact hashes. Artifact mismatch, missing files, or model-load failure latch readiness closed.
7. Rollback maintains hash-verified `active_candidate` and `previous_candidate` slots. Activation atomically shifts active to previous. Rollback re-verifies the previous manifest and artifacts before swapping. The state cannot represent production or automatic production upgrade.

## Outputs

- `docs/d3-performance-report.json`
- `docs/d3-shadow-prediction-log.json`
- `docs/heatmap-validation-report.json`
- `docs/d3-monitoring-report.json`
- `docs/d3-rollback-report.json`
- `docs/d3-regression-test-report.json`
- `docs/steel-patchcore-d3-operational-qualification-report.json`
- `docs/steel-patchcore-d3-operational-qualification-report.md`

Derived heatmap PNGs and the resumable shadow checkpoint remain under the ignored `model-training/runs/steel-d3-operational-qualification/` directory. They are operational evidence, not model artifacts, and are not committed.

## Stop rule

After reports, regression suites, and the precise commit are complete, stop with the candidate still in `CANDIDATE ONLY` state and wait for explicit human authorization.
