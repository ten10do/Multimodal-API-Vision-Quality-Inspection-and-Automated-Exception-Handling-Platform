# RECOVERY RAW EVIDENCE CAPTURE REPORT

Verdict: `RECOVERY_EVIDENCE_READY`

Capture completed at `2026-08-19T08:36:33.189184Z`. This report closes the
dev-only raw-evidence capture phase of Optimization 1.1. It does not authorize
or contain A0-A6 development evaluation, candidate ranking, candidate-specific
threshold selection, holdout evaluation, representation changes, or
Optimization 2 work.

## 1. Resume audit

- The handoff capture was first verified at `train_normal=2800/4721`, with 28
  complete shards. Only finalized, hashed shards represented completed work.
- After an interrupted monitoring session, the final continuation audited the
  actual checkpoint at `4721/4721`, `590/590`, and `2500/3333`, with 79
  complete shards and `last_updated=2026-08-19T07:38:21.271606Z`.
- At that continuation point, the lifecycle lock was absent, the Steel recovery
  Python worker count was zero, and no old Python GPU worker was present.
- `--verify-only` passed all 79 existing shards before the unique evaluator was
  restarted. The evaluator skipped all completed IDs and captured only the
  remaining 833 development-anomaly originals.
- The evaluator exited normally after writing the final 33-original tail shard,
  evidence manifest, and manifest SHA256 file. The lifecycle lock is absent and
  the Steel recovery Python worker count is now zero.

## 2. Frozen lineage

| Item | Frozen value | Result |
|---|---|---|
| Branch | `feat/steel-patchcore-validity-recovery-v1.2` | PASS |
| Extractor commit | `22333883150d19b80b91c5a1d2c1a5961012aca2` | PASS |
| Bank SHA256 | `291bb2903b6e9a3bc60ca4ae35380bd7cf312bc661366d10f87a75d3f0b8ebda` | PASS |
| Baseline threshold | `0.490039` | PASS |
| Source split SHA256 | `64df9f817a995e64c7ee4d00e16962d94f500878fd1404d95227f26e7f913d07` | PASS |
| Recovery split SHA256 | `f60ce0a6aa012ad1d9a9636800c1e94f365940500f20b8deb2f7fd9ffbc55448` | PASS |

The model, bank, threshold, source split, recovery split, backbone, layers,
tiling, float32 canonical format, and nearest-neighbour distance semantics were
not changed.

## 3. Capture counts

| Role | Complete | Expected |
|---|---:|---:|
| `train_normal` | 4,721 | 4,721 |
| `validation_normal` | 590 | 590 |
| `recovery_dev_anomaly` | 3,333 | 3,333 |
| **Total** | **8,644** | **8,644** |

There are 8,644 unique original IDs, zero duplicates, zero missing IDs, and
zero unexpected IDs. Each role exactly equals its frozen expected membership.
The 3,333 development-anomaly IDs and 3,333 recovery-holdout IDs are disjoint
and their union exactly equals the frozen 6,666 source anomaly IDs.

## 4. Shard integrity

- Shards: 88 total (`train_normal=48`, `validation_normal=6`,
  `recovery_dev_anomaly=34`).
- Runtime artifact size: 248,671,536 bytes.
- Evidence manifest SHA256:
  `7067bbc6da747ffa90db9532729f579e19ee90fbcdf3867c96f2dd56a0d10303`.
- Post-capture `--verify-only`: `RECOVERY_CAPTURE_VERIFIED`.
- Shard SHA256 mismatches: 0.
- Shard size mismatches: 0.
- Metadata/geometry failures: 0.
- Non-finite evidence failures: 0.

The raw NPZ shards, checkpoint, dataset, bank, virtual environments, caches,
and other large runtime artifacts remain outside Git.

## 5. Raw evidence semantics

- Canonical dtype: `float32`.
- Evidence per original: seven raw pre-normalization patch-distance grids.
- Tile-grid shape: `32x32`.
- Tile x offsets: `0, 256, 512, 768, 1024, 1280, 1344`.
- Patch stride: 8 pixels.
- Deterministically reconstructed stitched-grid shape: `32x200`.
- Stitched overlap rule: arithmetic mean.
- Raw tile scores exactly equal the maximum of their corresponding raw grids.
- A0 is the global maximum across the seven unstitched raw grids.

## 6. Baseline reconstruction

For all 8,644 originals, A0 reconstructed from the canonical raw grids was
compared with the preserved frozen evaluator/train score evidence at absolute
tolerance `2e-6` and zero relative tolerance:

| Statistic | Absolute error |
|---|---:|
| Maximum | `4.999465942345793e-7` |
| p99 | `4.900609588665005e-7` |
| Mean | `1.3527170566962582e-7` |
| Mismatch count `>2e-6` | **0** |

Within the canonical float32 shards, `max(raw_grids)` and the stored float32
baseline score agree exactly for every original: maximum, p99, and mean error
are all 0, with zero mismatches.

## 7. Normalization semantic verification

`RAW_NORMALIZATION_SEMANTICS_GATE = PASS`

The deterministic regression reconstructs the existing predictor pixel-map
path from a raw grid and verifies exact semantic agreement with the frozen
per-tile min-max normalization path. The raw capture point remains before that
normalization.

## 8. Candidate grid

A0-A6 definitions remain frozen exactly as documented in the recovery
protocol. Tests verify only their deterministic definitions on synthetic data.

- Development evaluation: **NOT RUN**.
- AUROC or operating-point ranking: **NOT RUN**.
- A1-A6 threshold calculation: **NOT RUN**.
- Best-candidate selection: **NOT RUN**.

## 9. Holdout isolation

- `test_normal` recovery inference access: **0**.
- `recovery_holdout_anomaly` inference access: **0**.
- `HOLDOUT_ACCESS_COUNT = 0`.

The capture-role construction excludes the union of both holdout roles, and
the final runtime manifest records `holdout_inference_count=0`.

## 10. Tests

The precise Steel recovery matrix covered capture/resume, shard integrity,
recovery split, holdout isolation, baseline reconstruction, raw stitching,
normalization semantics, tiling, immutability, and artifact lineage.

- Passed: 29.
- Failed: 0.
- Skipped: 1.
- Skip reason: MLOps registration has not been run. The skip is not counted as
  a pass and is expected because this phase forbids promotion/registration.

## 11. Git

- Branch retained: `feat/steel-patchcore-validity-recovery-v1.2`.
- Capture/extractor HEAD before this report:
  `22333883150d19b80b91c5a1d2c1a5961012aca2`.
- Runtime artifacts remain ignored and unstaged.
- Only this report and its small machine-readable summary are eligible for
  precise staging.
- No merge to `main` and no production promotion were performed.

## 12. Limitations and stop condition

The recovery split is post-hoc because the complete original baseline test was
observed during Optimization 1; it is not a pristine independent test. The
`steel-patchcore` 1.0.0 conclusion remains
`STEEL_DOMAIN_VALIDATION_FAILED`. This gate establishes only that complete,
reconstructible development-side raw evidence is ready for a separately
authorized next phase.

`RECOVERY_EVIDENCE_READY`

STOP: do not run A0-A6 development evaluation, candidate ranking, threshold
selection, representation investigation, holdout evaluation, or Optimization
2 without explicit authorization.
