# Steel PatchCore D3 One-Shot Recovery Holdout Protocol

Protocol version: `d3_recovery_holdout_protocol_v1`

This protocol is frozen, tested, and committed before any image in either sealed holdout role is opened. The commit containing this document, the evaluator, and its tests is the **PRE-HOLDOUT FREEZE COMMIT**. Its full Git SHA is recorded by the result artifact at execution time.

## Frozen D3 definition

- Backbone: DINOv2 ViT-B/14 (`dinov2_vitb14`).
- Feature: `forward_features(x)["x_norm_patchtokens"]`; CLS and register tokens are excluded.
- Geometry: each frozen 256x256 tile is bilinearly adapted to 252x252, producing an 18x18 patch grid with 768-dimensional embeddings.
- Adaptation: the existing full-development train-normal ZCA mean and 768x768 whitening matrix. No whitening statistic is recomputed.
- Normalization and distance: post-whitening per-patch L2 normalization and cosine 1-NN distance (`1 - max cosine similarity`).
- Bank: the existing full-development D3 bank, reservoir budget 50,000, Algorithm R, seed 42.
- Tiling: x offsets `{0, 256, 512, 768, 1024, 1280, 1344}`.
- Aggregation: A0 global maximum patch distance across all seven tiles.
- Threshold: full precision `0.8471092581748962`, loaded only from the committed full-development result JSON.
- No recalibration, threshold fitting, threshold tuning, candidate search, model change, bank change, or post-holdout adjustment is permitted.

## Artifact lineage

Evaluation fails closed with `RECOVERY_HOLDOUT_BLOCKED` if any file is missing or any full SHA256 differs.

| Artifact | Frozen SHA256 |
|---|---|
| baseline bank | `291bb2903b6e9a3bc60ca4ae35380bd7cf312bc661366d10f87a75d3f0b8ebda` |
| source split | `64df9f817a995e64c7ee4d00e16962d94f500878fd1404d95227f26e7f913d07` |
| recovery split | `f60ce0a6aa012ad1d9a9636800c1e94f365940500f20b8deb2f7fd9ffbc55448` |
| recovery evidence manifest | `7067bbc6da747ffa90db9532729f579e19ee90fbcdf3867c96f2dd56a0d10303` |
| DINOv2-B weights | `0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73` |
| full-development ZCA artifact | `c8d9d2ed39fb7ba6d0013a27beba81e8d7b70c66da0e38b7d19e15ea7cae8c3a` |
| full-development D3 bank | `40fe43331885422c8a32364a48fc403b766f807f69faafee775a2eb2403cbbda` |
| frozen quartile manifest | `8d39aec04c35ac7a7edc931c6dfb3494c6ada6a0a6e13e77b89bc894f9919075` |
| committed full-development result JSON | `1d511ccd20a6c007f6f0b298de8e7b5bd649ff6c229a183b60c06dda1bbca35c` |

The evaluator hashes all artifacts before unsealing and again after inference and metrics. Frozen artifacts are read-only inputs and must remain byte-identical.

## Holdout split semantics

The frozen manifests define exactly two evaluation roles:

- `test_normal`: 591 unique normal originals.
- `recovery_holdout_anomaly`: 3,333 unique anomaly originals.

The evaluator rejects duplicates within a role, intersection between holdout roles, intersection with train-normal, validation-normal, or recovery-development anomaly IDs, and any anomaly partition that does not exactly reconstruct the source `test_anomaly` set with the recovery-development role. Manifest inspection does not count as image access. `HOLDOUT_ACCESS_COUNT` is the number of unique originals with a final checkpointed score and must finish at 3,924.

No D0, D1, or D2 holdout evaluation is authorized.

## Metric semantics

- Image AUROC: binary AUROC over 591 normal scores (label 0) and 3,333 anomaly scores (label 1).
- Score distributions: for each class report `n`, `min`, `p50`, `p95`, `p99`, and `max` using linear percentiles.
- Median delta: anomaly median minus normal median.
- Frozen-threshold diagnostics: prediction is anomaly when `score >= threshold`; report TP, TN, FP, FN, precision, recall, F1, normal FPR, and anomaly recall.
- Threshold confusion metrics are report-only and do not participate in the gate.
- Bootstrap: class-stratified resampling with replacement, seed 42, 2,000 iterations; report median AUROC and the 2.5/97.5 percentile interval. Bootstrap is report-only.
- Quartiles: anomaly area ratios are assigned with the development-frozen boundaries Q1=`0.010888671875`, Q2=`0.02656494140625`, Q3=`0.07214111328125`. Boundaries are never recomputed on holdout. For each Q1-Q4, report count and normal-vs-quartile AUROC using all 591 normals.

## Gate definition

The only success gate is:

```text
Image AUROC >= 0.75
AND
anomaly median > normal median
```

The terminal verdict is exactly one of:

- `RECOVERY_HOLDOUT_PASS`
- `RECOVERY_HOLDOUT_FAILED`
- `RECOVERY_HOLDOUT_BLOCKED`

A completed failed gate stops immediately without tuning. A precondition, lineage, membership, checkpoint, non-finite score, or immutability failure is blocked.

## Checkpoint resume rules

The ignored run artifact `model-training/runs/steel-d3-recovery-holdout/checkpoint.json` stores:

- completed original IDs grouped by exact split role;
- one finite score and frozen-threshold prediction per original;
- complete artifact lineage;
- full-precision threshold;
- protocol and checkpoint schema versions;
- UTC update timestamp.

On resume, the evaluator validates the entire checkpoint before opening another image. Foreign lineage, a different threshold, an unknown ID, role confusion, duplicate IDs, invalid predictions, or malformed/non-finite scores block execution. Completed originals are skipped. The checkpoint is never deleted to restart the one-shot evaluation, and every original can have only one final result.

## Test and execution order

Before holdout access:

```powershell
.venv\Scripts\python.exe -m pytest inference-service\tests\test_steel_d3_recovery_holdout.py -q
git diff --check
```

After tests pass, precisely stage only the protocol, evaluator, pure module, tests, and required lifecycle registration, then create:

```text
test: freeze D3 one-shot recovery holdout protocol
```

Only after that commit may the GPU runner unseal the two roles. It generates the final Markdown and JSON result artifacts. Those results are precisely staged and committed separately. Main is not merged, and the workflow stops at the terminal verdict.
