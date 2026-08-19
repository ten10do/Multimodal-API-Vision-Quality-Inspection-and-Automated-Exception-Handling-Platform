# Steel PatchCore Representation Investigation Protocol v1

Protocol status: **frozen before any GPU experiment**

Protocol version: `representation_protocol_v1`

This is Optimization 1.1 (representation investigation), not Optimization 2.
It is a post-hoc recovery investigation after
`RECOVERY_AGGREGATION_GATE_FAILED`. `steel-patchcore` 1.0.0 remains
`STEEL_DOMAIN_VALIDATION_FAILED`.

## Immutable inputs (unchanged)

- Frozen bank SHA256 `291bb2903b6e9a3bc60ca4ae35380bd7cf312bc661366d10f87a75d3f0b8ebda`
- Baseline threshold `0.490039`
- Source split SHA256 `64df9f817a995e64c7ee4d00e16962d94f500878fd1404d95227f26e7f913d07`
- Recovery split SHA256 `f60ce0a6aa012ad1d9a9636800c1e94f365940500f20b8deb2f7fd9ffbc55448`
- Raw evidence manifest SHA256 `7067bbc6da747ffa90db9532729f579e19ee90fbcdf3867c96f2dd56a0d10303`
- Backbone `wide_resnet50_2 / IMAGENET1K_V1`; tiling `0,256,512,768,1024,1280,1344`

## Authorized development data only

`train_normal=4721`, `validation_normal=590`, `recovery_dev_anomaly=3333`.
Sealed holdout (`test_normal=591`, `recovery_holdout_anomaly=3333`) is forbidden
for inference, features, scores, thresholds, AUROC, heatmaps, distributions,
ranking, and model selection. `HOLDOUT_ACCESS_COUNT = 0` always.

## Diagnostic subset (deterministic, frozen before GPU)

| Field | Value |
|---|---|
| seed | 42 |
| train_normal_subset | 1000 |
| validation_normal_subset | 300 |
| dev_anomaly_subset | 1000 |
| anomaly stratification | defect-area quartile Q1/Q2/Q3/Q4, balanced (250/250/250/250) |

Determinism rules:

- `train_normal_subset` = first 1000 IDs of the source `train_normal` order.
- `validation_normal_subset` = first 300 IDs of the source `validation_normal` order.
- `recovery_dev_anomaly_subset` = for each area-ratio quartile (computed as in
  the aggregation stage), take 250 IDs in the canonical `recovery_dev_anomaly`
  order (fixed ordered subsample).
- No duplicates; every subset is a subset of an authorized development role.
- The manifest records role membership SHA-bound and a payload SHA256.

Artifact: `model-training/datasets/severstal-steel/representation_diagnostic_manifest.json`
plus `.sha256`. It is a small JSON and is eligible for Git; feature tensors and
experimental banks are not.

Quartile boundaries are derived once from the frozen recovery-dev anomaly GT
masks using NumPy `method="linear"` at 0.25/0.50/0.75 (identical to the
aggregation stage) and recorded in the manifest.

## Stage R candidates (feature layer) — first round only

Holding fixed: sampler = reservoir, budget = 50000, seed = 42, distance =
cosine 1-NN (`1 - max sim`), tiling = frozen 7 tiles, image aggregation = A0
global max.

| ID | Feature | Patch dim |
|---|---|---|
| R0 | layer2 + bilinear-upsampled layer3, per-patch L2 (current) | 1536 |
| R1 | layer2 only, per-patch L2 | 512 |
| R2 | bilinear-upsampled layer3 only, per-patch L2 | 1024 |

Each candidate builds its own experimental bank from the `train_normal_subset`
patches with the frozen reservoir sampler (seed 42), so the layer variable is
isolated. Experimental banks live under ignored runtime path
`runtime/recovery-representation/{R0,R1,R2}/` with manifest + SHA256 + lineage.

Threshold: `max(train_normal_subset A0 scores)` per candidate, **diagnostic
only** (never a production/recovery final threshold).

## Stage R evaluation

On `validation_normal_subset` + `dev_anomaly_subset` per R candidate report:

- Image AUROC, normal median, anomaly median, anomaly−normal median delta;
- diagnostic TP/TN/FP/FN, precision, recall, F1, normal FPR, anomaly recall;
- normal-vs-Q1/Q2/Q3/Q4 AUROC (per anomaly area quartile).

## Feature Layer Gate

| Condition | Threshold |
|---|---|
| Image AUROC vs R0 | ΔAUROC ≥ 0.10 |
| Image AUROC | ≥ 0.60 |

If R1 or R2 satisfies both → `FEATURE_LAYER_SIGNAL_FOUND` (freeze best, STOP).
If neither satisfies both → `FEATURE_LAYER_GATE_FAILED`.

## Stage N candidates (normalization/distance) — only if R gate failed

On the frozen R0 feature setup (current layers), holding bank sampler/budget/
seed/tiling/aggregation fixed:

| ID | Semantics |
|---|---|
| N0 | current: concat then per-patch L2, cosine distance (reference) |
| N1 | explicit per-patch L2 + cosine distance (code-equivalent control) |
| N2 | per-layer L2 normalization **before** concat, then cosine distance |

Audit note: the current implementation already applies per-patch L2 +
cosine, so N1 is expected to reproduce N0 and is kept as a control. No other
normalization variant is authorized in this round.

## Normalization Gate

| Condition | Threshold |
|---|---|
| Image AUROC | ≥ 0.60 |
| Image AUROC vs N0 | ΔAUROC ≥ 0.10 |

If satisfied → `NORMALIZATION_SIGNAL_FOUND` (freeze candidate, STOP).
Otherwise → `NORMALIZATION_GATE_FAILED`.

## Terminal verdicts

- Both R and N gates fail → `REPRESENTATION_BASE_FEATURE_GATE_FAILED` (STOP;
  do not change backbone, tiling, bank sampling, or patch scale without
  authorization).
- A candidate passes and is frozen → STOP; next round needs authorisation for
  full development confirmation.

## GPU / lifecycle safety

At most one representation experiment worker at a time. Each candidate writes a
small metrics/manifest artifact and releases the GPU. Jobs are checkpointable.
Existing 8644 raw evidence and the frozen evidence manifest are never
overwritten.