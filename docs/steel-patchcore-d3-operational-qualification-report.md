# Steel PatchCore D3 Operational Qualification

Verdict: **`OPERATIONAL_QUALIFICATION_FAILED`**

Candidate: `steel-patchcore-d3-candidate@1.2.0-candidate.1` (`CANDIDATE ONLY`).

## Qualification checks

| Check | Result |
|---|---|
| performance_schema | PASS |
| shadow_complete | PASS |
| inference_reproducibility | PASS |
| heatmap_acceptance | FAIL |
| monitoring_fail_closed | PASS |
| rollback_mechanism | PASS |
| regression_tests | PASS |
| candidate_only | PASS |

## Operational evidence

| Evidence | Result |
|---|---:|
| 1,000-image latency p50 / p95 / p99 | 140.199 / 145.280 / 149.490 ms |
| 1,000-image GPU memory peak | 2048.0 MB |
| Shadow predictions / errors | 3924 / 0 |
| D3 pixel AUROC / paired baseline | 0.600212 / 0.834361 |
| D3 AUPRO / paired baseline | 0.265291 / 0.587579 |

## Guardrails

- Artifact and threshold were unchanged.
- No training, backbone search, fine-tuning, or production promotion was performed.
- Rollback is candidate-only and hash-verifies both activation and rollback. No earlier candidate manifest exists, so the live previous slot remains empty.
- A failed heatmap gate is terminal evidence; it does not authorize tuning.

## Evidence

- `docs/d3-performance-report.json`
- `docs/d3-shadow-prediction-log.json`
- `docs/heatmap-validation-report.json`
- `docs/d3-monitoring-report.json`
- `docs/d3-rollback-report.json`
- `docs/d3-regression-test-report.json`
