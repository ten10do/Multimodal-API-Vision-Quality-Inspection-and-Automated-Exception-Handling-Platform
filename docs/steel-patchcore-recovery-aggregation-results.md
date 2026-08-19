# Steel PatchCore Recovery — Aggregation Development Report

Verdict: `RECOVERY_AGGREGATION_GATE_FAILED`

- Protocol: `recovery_protocol_v1`
- Branch: `feat/steel-patchcore-validity-recovery-v1.2`
- Implementation commit: `faac08cf1767b72604a4a7249651d42708dc8546`
- Generated at: `2026-08-19T09:05:50.715034Z`

## 1. Frozen lineage

- `bank_sha256`: `291bb2903b6e9a3bc60ca4ae35380bd7cf312bc661366d10f87a75d3f0b8ebda`
- `source_split_sha256`: `64df9f817a995e64c7ee4d00e16962d94f500878fd1404d95227f26e7f913d07`
- `recovery_split_sha256`: `f60ce0a6aa012ad1d9a9636800c1e94f365940500f20b8deb2f7fd9ffbc55448`
- `evidence_manifest_sha256`: `7067bbc6da747ffa90db9532729f579e19ee90fbcdf3867c96f2dd56a0d10303`
- `baseline_threshold`: `0.490039`
- `protocol_document_sha256`: `139c14ac0083b1ecb0afcb4270b15fcb3e73bb59c351b0836201b0f4542ea7b1`

## 2. Evidence preflight

- Manifest SHA256 verified: `True`
- Shards: 88
- Counts: train_normal=4721 validation_normal=590 recovery_dev_anomaly=3333 total=8644
- Duplicate/missing/unexpected: 0/0/0
- dtype: `float32`; grid `[32, 32]`; stitched `[32, 200]`; overlap `mean`
- Baseline reconstruction: max_abs_error=0.000 mismatches>2e-6=0

## 3. A0 baseline sanity gate

- threshold_A0 = 0.490038812160 (frozen 0.490039; |Δ| = 0.000000187840)
- reconstructed: `True`

## 4. Holdout isolation

- HOLDOUT_ACCESS_COUNT = 0

## 5. Candidate definitions

- **A0**: global maximum over all seven unstitched raw tile grids (exact steel-patchcore 1.0.0 image score)
- **A1**: 99.0th percentile of the flattened mean-overlap stitched raw grid
- **A2**: 99.5th percentile of the flattened mean-overlap stitched raw grid
- **A3**: 99.9th percentile of the flattened mean-overlap stitched raw grid
- **A4**: mean of the highest ceil(0.1% × N) stitched raw responses
- **A5**: mean of the highest ceil(0.5% × N) stitched raw responses
- **A6**: mean of the highest ceil(1.0% × N) stitched raw responses

## 6. Train-only thresholds

| Candidate | threshold | train min | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| A0 | 0.490038812 | 0.245600 | 0.364893 | 0.433795 | 0.453317 | 0.490039 |
| A1 | 0.422482371 | 0.206066 | 0.291365 | 0.374138 | 0.393747 | 0.422482 |
| A2 | 0.431709230 | 0.212282 | 0.306940 | 0.385839 | 0.404108 | 0.431709 |
| A3 | 0.448658526 | 0.227227 | 0.334669 | 0.407543 | 0.424075 | 0.448659 |
| A4 | 0.455577280 | 0.233679 | 0.346788 | 0.417263 | 0.433842 | 0.455577 |
| A5 | 0.442614928 | 0.222057 | 0.324677 | 0.399604 | 0.415131 | 0.442615 |
| A6 | 0.434695907 | 0.215671 | 0.311810 | 0.389403 | 0.406158 | 0.434696 |

## 7. Development metrics

| Candidate | AUROC | TP | TN | FP | FN | Precision | Recall | F1 | Normal FPR | Anomaly Recall | Gate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 | 0.4943 | 0 | 590 | 0 | 3333 | - | 0.0000 | 0.0000 | 0.0000 | 0.0000 | FAIL |
| A1 | 0.4884 | 0 | 590 | 0 | 3333 | - | 0.0000 | 0.0000 | 0.0000 | 0.0000 | FAIL |
| A2 | 0.4941 | 0 | 590 | 0 | 3333 | - | 0.0000 | 0.0000 | 0.0000 | 0.0000 | FAIL |
| A3 | 0.5007 | 0 | 590 | 0 | 3333 | - | 0.0000 | 0.0000 | 0.0000 | 0.0000 | FAIL |
| A4 | 0.4988 | 0 | 590 | 0 | 3333 | - | 0.0000 | 0.0000 | 0.0000 | 0.0000 | FAIL |
| A5 | 0.4978 | 0 | 590 | 0 | 3333 | - | 0.0000 | 0.0000 | 0.0000 | 0.0000 | FAIL |
| A6 | 0.4946 | 0 | 590 | 0 | 3333 | - | 0.0000 | 0.0000 | 0.0000 | 0.0000 | FAIL |

## 8. Score distributions (validation_normal / recovery_dev_anomaly)

| Candidate | set | n | min | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|---|
| A0 | validation_normal | 590 | 0.239378 | 0.365912 | 0.430511 | 0.451142 | 0.462758 |
| A0 | recovery_dev_anomaly | 3333 | 0.259972 | 0.359619 | 0.422501 | 0.440421 | 0.477310 |
| A1 | validation_normal | 590 | 0.205288 | 0.293450 | 0.375273 | 0.399294 | 0.414550 |
| A1 | recovery_dev_anomaly | 3333 | 0.218540 | 0.284242 | 0.363230 | 0.375200 | 0.396008 |
| A2 | validation_normal | 590 | 0.211623 | 0.308830 | 0.386532 | 0.408126 | 0.422184 |
| A2 | recovery_dev_anomaly | 3333 | 0.229561 | 0.299287 | 0.375189 | 0.387100 | 0.406413 |
| A3 | validation_normal | 590 | 0.226127 | 0.336866 | 0.408551 | 0.423661 | 0.438589 |
| A3 | recovery_dev_anomaly | 3333 | 0.248306 | 0.328366 | 0.395398 | 0.409904 | 0.429629 |
| A4 | validation_normal | 590 | 0.233832 | 0.347934 | 0.417098 | 0.432615 | 0.448649 |
| A4 | recovery_dev_anomaly | 3333 | 0.253128 | 0.340130 | 0.405070 | 0.420917 | 0.453391 |
| A5 | validation_normal | 590 | 0.220576 | 0.326702 | 0.399952 | 0.419639 | 0.431750 |
| A5 | recovery_dev_anomaly | 3333 | 0.242748 | 0.317248 | 0.388177 | 0.400946 | 0.419713 |
| A6 | validation_normal | 590 | 0.214360 | 0.312692 | 0.389978 | 0.411942 | 0.424709 |
| A6 | recovery_dev_anomaly | 3333 | 0.233398 | 0.304094 | 0.378193 | 0.390622 | 0.410874 |

## 9. Small-defect analysis

- Anomaly mask area ratio quartile boundaries: Q1=0.010889 Q2=0.026565 Q3=0.072141
- Zero-area dev anomalies: 0

### A0

| Quartile | Count | Median score | Recall | normal-vs-quartile AUROC |
|---|---|---|---|---|
| Q1 | 833 | 0.345825 | 0.0000 | 0.3954 |
| Q2 | 833 | 0.350853 | 0.0000 | 0.4237 |
| Q3 | 833 | 0.358869 | 0.0000 | 0.4892 |
| Q4 | 834 | 0.394201 | 0.0000 | 0.6685 |

### A1

| Quartile | Count | Median score | Recall | normal-vs-quartile AUROC |
|---|---|---|---|---|
| Q1 | 833 | 0.269348 | 0.0000 | 0.0352 |
| Q2 | 833 | 0.277658 | 0.0000 | 0.0675 |
| Q3 | 833 | 0.289655 | 0.0000 | 0.1267 |
| Q4 | 834 | 0.340035 | 0.0000 | 0.2918 |

### A2

| Quartile | Count | Median score | Recall | normal-vs-quartile AUROC |
|---|---|---|---|---|
| Q1 | 833 | 0.285088 | 0.0000 | 0.0811 |
| Q2 | 833 | 0.292211 | 0.0000 | 0.1184 |
| Q3 | 833 | 0.302583 | 0.0000 | 0.1830 |
| Q4 | 834 | 0.352257 | 0.0000 | 0.3619 |

### A3

| Quartile | Count | Median score | Recall | normal-vs-quartile AUROC |
|---|---|---|---|---|
| Q1 | 833 | 0.315803 | 0.0000 | 0.2134 |
| Q2 | 833 | 0.319358 | 0.0000 | 0.2447 |
| Q3 | 833 | 0.328817 | 0.0000 | 0.3148 |
| Q4 | 834 | 0.371871 | 0.0000 | 0.5086 |

### A4

| Quartile | Count | Median score | Recall | normal-vs-quartile AUROC |
|---|---|---|---|---|
| Q1 | 833 | 0.328517 | 0.0000 | 0.2792 |
| Q2 | 833 | 0.332286 | 0.0000 | 0.3101 |
| Q3 | 833 | 0.341086 | 0.0000 | 0.3804 |
| Q4 | 834 | 0.381068 | 0.0000 | 0.5752 |

### A5

| Quartile | Count | Median score | Recall | normal-vs-quartile AUROC |
|---|---|---|---|---|
| Q1 | 833 | 0.304500 | 0.0000 | 0.1575 |
| Q2 | 833 | 0.309036 | 0.0000 | 0.1924 |
| Q3 | 833 | 0.318954 | 0.0000 | 0.2618 |
| Q4 | 834 | 0.364544 | 0.0000 | 0.4536 |

### A6

| Quartile | Count | Median score | Recall | normal-vs-quartile AUROC |
|---|---|---|---|---|
| Q1 | 833 | 0.290842 | 0.0000 | 0.0979 |
| Q2 | 833 | 0.296717 | 0.0000 | 0.1354 |
| Q3 | 833 | 0.306642 | 0.0000 | 0.2018 |
| Q4 | 834 | 0.355255 | 0.0000 | 0.3852 |

## 10. Root-cause interpretation

- Every candidate's anomaly median is below its normal median (-0.006293, and the same sign for A1-A6: yes). The raw ranking placed anomalies below normal.
- Robust aggregation did not repair the ranking: the best development Image AUROC is 0.5007 (A3), statistically indistinguishable from chance.
- Conclusion: the dominant failure is B (representation), not A (extreme-max aggregation).
  Replacing the extreme max with robust percentiles/top-k means shifts score scales downward
  but does not change the normal-vs-anomaly ordering, so image-level validity cannot be
  recovered by aggregation alone.
- Small-defect gradient: for every candidate the largest defects (Q4) separate best, e.g. A0 Q4 normal-vs-quartile AUROC = 0.6685. Q1-Q3 remain at or below chance. This is a defect-size-limited representation ceiling, not an aggregation artefact.

## 11. Candidate ranking

- No A0-A6 candidate satisfied the frozen development gate simultaneously
  (Image AUROC >= 0.75, Normal FPR <= 0.10, Anomaly Recall >= 0.60).
- No candidate was frozen; selection returned `None`.

## 12. Gate verdict

`RECOVERY_AGGREGATION_GATE_FAILED`

## 13. Limitations

The recovery split is post-hoc because the complete original baseline test was
observed during Optimization 1. `steel-patchcore` 1.0.0 remains
`STEEL_DOMAIN_VALIDATION_FAILED`. The holdout remains sealed with
`HOLDOUT_ACCESS_COUNT = 0`; no holdout evaluation was performed.

