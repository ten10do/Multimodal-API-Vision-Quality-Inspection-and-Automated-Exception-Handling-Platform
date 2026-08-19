# Steel PatchCore — Spatial Scale & Local Context Protocol

`spatial_context_protocol_v1`

This protocol is frozen BEFORE any GPU result is observed. It defines the
Spatial Scale & Local Context Investigation (Optimization 1.1 continuation),
not Optimization 2. No speed tuning, no backbone change, no registry change,
no holdout access, no memory-bank sampling change.

## 1. Frozen lineage (unchanged)

- bank SHA256 `291bb2903b6e9a3bc60ca4ae35380bd7cf312bc661366d10f87a75d3f0b8ebda`
- baseline threshold `0.490039`
- source split SHA256 `64df9f817a995e64c7ee4d00e16962d94f500878fd1404d95227f26e7f913d07`
- recovery split SHA256 `f60ce0a6aa012ad1d9a9636800c1e94f365940500f20b8deb2f7fd9ffbc55448`
- evidence manifest SHA256 `7067bbc6da747ffa90db9532729f579e19ee90fbcdf3867c96f2dd56a0d10303`
- diagnostic subset manifest SHA256 `8d39aec04c35ac7a7edc931c6dfb3494c6ada6a0a6e13e77b89bc894f9919075`
- `steel-patchcore` 1.0.0 permanently `STEEL_DOMAIN_VALIDATION_FAILED`

## 2. Hypotheses under test (only these two)

- **H1 — LOCAL CONTEXT FAILURE**: the bare per-position embedding lacks local
  texture context, so steel texture variation is not stably separable from
  real defects.
- **H2 — PATCH SCALE FAILURE**: the spatial representation's response to
  defect size mismatches the actual defect spatial scale, so smaller/quieter
  defects (Q1/Q2) are substantially weaker than Q4.

## 3. Diagnostic manifest (REUSED, not resampled)

Reuse the frozen representation diagnostic subset: 1000 train_normal,
300 validation_normal, 1000 recovery_dev_anomaly (250 × Q1..Q4), seed 42.
Manifest SHA256 `8d39aec0…9919075`. Holdout access count = 0.

## 4. Frozen candidates — Stage S

| Candidate | Layer | Context | Dim | Semantics |
|---|---|---|---|---|
| S0 | layer3 (bilinear 32x32) | none | 1024 | reference == R2 (current) |
| S1 | layer3 (bilinear 32x32) | 3x3 avg | 1024 | 3x3 avg pool then per-patch L2, cosine 1-NN |
| S2 | layer3 (bilinear 32x32) | 5x5 avg | 1024 | 5x5 avg pool then per-patch L2, cosine 1-NN |

Only 3 candidates in the first round. No 7x7 / Gaussian / max-pool / concat /
multi-scale / attention. This round answers exactly one question: does adding
local context improve steel texture/defect separation?

### Context operator (fixed, testable)

`AveragePool2d(kernel=k, stride=1, padding=k//2, count_include_pad=False)` on
the frozen 32x32 feature map (same grid). Zero padding of `k//2` on each side;
the mean is over valid (non-padded) neighbors only. Channels unchanged.

- padding semantics: zero padding, valid-neighbor mean (`count_include_pad=False`)
- tensor shape: (1, 1024, 32, 32) -> (1, 1024, 32, 32)
- feature dimension: 1024 (unchanged)
- normalization order: context FIRST, then per-patch L2 (no raw+pooled concat)

## 5. Shared fixed semantics (S and P)

- distance: cosine 1-NN `1 - max(emb @ bank^T)`, no `num_neighbors`
- bank: reservoir Algorithm R, budget 50000, seed 42, from the frozen
  train_normal diagnostic subset (1000 images)
- if available candidate patches < 50000, use all and record
- tiling: frozen 7 tiles at x0 0/256/512/768/1024/1280/1344, 256x256 (no tile change)
- image score: A0 = global max over tiles of tile max (no aggregation change)
- threshold: `max(train-normal candidate image scores)` — diagnostic only

## 6. Stage S evaluation

On 300 validation_normal + 1000 stratified dev anomaly: Image AUROC, normal /
anomaly median + delta, TP/TN/FP/FN/Precision/Recall/F1/Normal-FPR/Anomaly-
Recall, and validation-normal vs Q1..Q4 AUROC.

## 7. Spatial Context Gate (frozen, not lowered retroactively)

S1/S2 must EACH satisfy: Image AUROC >= 0.65 AND delta vs S0 >= +0.10.
Q1/Q2 movement is inspected but does not override an overall failure.

- If any candidate passes: select the unique best by highest Image AUROC
  (ties within 0.01 broken by higher Q1+Q2 mean AUROC), freeze candidate +
  kernel + bank hash + manifest + metrics + diagnostic threshold +
  implementation commit, emit `SPATIAL_CONTEXT_SIGNAL_FOUND`, STOP.
- If none pass: emit `SPATIAL_CONTEXT_GATE_FAILED`, proceed to Stage P.

## 8. Frozen candidates — Stage P (patch-scale diagnostic)

Run only if the Spatial Context Gate FAILED. Still no 256x256 tiling change;
this is a controlled feature-map experiment.

| Candidate | Layer | Context | Dim | Notes |
|---|---|---|---|---|
| P0 | layer3 (bilinear 32x32) | none | 1024 | == S0 reference |
| P1 | layer2 (32x32) | 3x3 avg | 512 | higher grid + context; tests thin/small-defect response |

Raw layer2 alone failed before (R1), but P1 answers whether higher spatial
grid + local context helps small defects. No further candidates added.

## 9. Patch Scale Gate (frozen)

P1 must satisfy ALL: overall AUROC >= 0.60, delta vs R1-layer2 >= +0.10, and
Q1 AUROC delta vs R1-layer2 Q1 >= +0.10.

- Pass: emit `PATCH_SCALE_SIGNAL_FOUND`, freeze, STOP.
- Fail: emit `SPATIAL_REPRESENTATION_GATE_FAILED`, STOP.

## 10. Outcome mapping

- `SPATIAL_CONTEXT_SIGNAL_FOUND`
- `PATCH_SCALE_SIGNAL_FOUND`
- `SPATIAL_REPRESENTATION_GATE_FAILED`
- `SPATIAL_INVESTIGATION_BLOCKED` (only on a real blocker)

## 11. Runtime discipline

- experimental banks/features under `model-training/runs/steel-spatial-context/`
  (Git-ignored); separate subdir per candidate
- one GPU worker; lifecycle lock; per-candidate build/evaluate/persist/release;
  long tasks checkpointable
- never overwrite the baseline bank or the representation-phase experimental
  banks; baseline bank immutability re-verified by SHA256 each run
- STOP after any final verdict; do not auto-enter bank sampling / backbone /
  domain adaption / full development / holdout / Optimization 2