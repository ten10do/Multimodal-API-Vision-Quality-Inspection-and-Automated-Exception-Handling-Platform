# Steel PatchCore D3 Recovery Holdout Results

Verdict: **`RECOVERY_HOLDOUT_PASS`**

## 1. Handoff Audit

- Branch: `feat/steel-patchcore-validity-recovery-v1.2`
- Pre-holdout HEAD: `99b0bbbcc98e2355f16e390a7e0027daf8ea34e2`
- Working tree, worker, GPU-worker, and lifecycle-lock audit passed before implementation.

## 2. Pre-Holdout Freeze

- Protocol: `d3_recovery_holdout_protocol_v1`
- Freeze commit: `99b0bbbcc98e2355f16e390a7e0027daf8ea34e2`
- Evaluator, protocol, and CPU tests were frozen before any holdout image was opened.

## 3. Frozen D3 Lineage

- `baseline_bank_sha256`: `291bb2903b6e9a3bc60ca4ae35380bd7cf312bc661366d10f87a75d3f0b8ebda`
- `source_split_sha256`: `64df9f817a995e64c7ee4d00e16962d94f500878fd1404d95227f26e7f913d07`
- `recovery_split_sha256`: `f60ce0a6aa012ad1d9a9636800c1e94f365940500f20b8deb2f7fd9ffbc55448`
- `evidence_manifest_sha256`: `7067bbc6da747ffa90db9532729f579e19ee90fbcdf3867c96f2dd56a0d10303`
- `dino_weights_sha256`: `0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73`
- `whitening_sha256`: `c8d9d2ed39fb7ba6d0013a27beba81e8d7b70c66da0e38b7d19e15ea7cae8c3a`
- `d3_bank_sha256`: `40fe43331885422c8a32364a48fc403b766f807f69faafee775a2eb2403cbbda`
- `quartile_manifest_sha256`: `8d39aec04c35ac7a7edc931c6dfb3494c6ada6a0a6e13e77b89bc894f9919075`
- `d3_results_sha256`: `1d511ccd20a6c007f6f0b298de8e7b5bd649ff6c229a183b60c06dda1bbca35c`

## 4. Holdout Dataset

- test_normal: 591 unique originals
- recovery_holdout_anomaly: 3333 unique originals
- development intersection: 0
- HOLDOUT_ACCESS_COUNT: 3924

## 5. One-shot Execution

- DINOv2 ViT-B/14 raw x_norm_patchtokens; CLS/register excluded; 18x18x768 per tile.
- Frozen full-development ZCA, 50,000-row seed-42 bank, seven frozen tiles, cosine 1-NN, A0 global max.
- Checkpoint resume counts at launch: `{'test_normal': 0, 'recovery_holdout_anomaly': 0}`
- Each original has exactly one checkpoint row bound to its split role.

## 6. Metrics

- Image AUROC: **0.817907**
- Normal distribution: `{'n': 591, 'min': 0.6666797399520874, 'p50': 0.7956956028938293, 'p95': 0.8252429962158203, 'p99': 0.8330505907535553, 'max': 0.8457531929016113}`
- Anomaly distribution: `{'n': 3333, 'min': 0.7347317934036255, 'p50': 0.817256510257721, 'p95': 0.836722218990326, 'p99': 0.8437078547477722, 'max': 0.8539726138114929}`
- Anomaly median - normal median: **0.021561**

## 7. Bootstrap CI

- Stratified bootstrap: seed=42, iterations=2000
- Median AUROC: 0.818093
- 95% percentile CI: [0.796799, 0.837721]

## 8. Threshold Diagnostics

- Frozen full-development threshold (loaded, not recalibrated): `0.8471092581748962`
- TP=10 TN=591 FP=0 FN=3323
- Precision=1.000000 Recall=0.003000 F1=0.005983
- Normal FPR=0.000000 Anomaly Recall=0.003000
- Confusion metrics are report-only and do not participate in the gate.

## 9. Q1-Q4

Frozen development area boundaries: `{'q1': 0.010888671875, 'q2': 0.02656494140625, 'q3': 0.07214111328125}`

| Quartile | Count | Normal-vs-quartile AUROC |
|---|---:|---:|
| Q1 | 814 | 0.729817 |
| Q2 | 843 | 0.781463 |
| Q3 | 862 | 0.837311 |
| Q4 | 814 | 0.923192 |

## 10. Gate Verdict

- Gate: AUROC >= 0.75 AND anomaly median > normal median.
- Verdict: **`RECOVERY_HOLDOUT_PASS`**

## 11. Tests

- Pre-holdout command: `.venv/Scripts/python.exe -m pytest inference-service/tests/test_steel_d3_recovery_holdout.py -q`
- Coverage includes membership/isolation, fail-closed lineage and split behavior, threshold-only loading, no recalibration, checkpoint resume and duplicate rejection, A0 and metrics, frozen quartiles, deterministic bootstrap, and artifact immutability.

## 12. Git

- Branch: `feat/steel-patchcore-validity-recovery-v1.2`
- Freeze commit: `99b0bbbcc98e2355f16e390a7e0027daf8ea34e2`
- Results are committed separately with precise staging; main is not merged.

## 13. Limitations

- This is a single sealed holdout evaluation, not a production estimate across sites or acquisition shifts.
- The threshold confusion metrics are diagnostic only and can be poor even when the rank-based gate passes.
- No post-holdout tuning, recalibration, candidate search, or alternative-model evaluation was performed.
