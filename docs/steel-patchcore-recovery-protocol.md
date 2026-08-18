# Steel PatchCore Recovery Protocol v1

Protocol status: **frozen before raw evidence capture**

Protocol version: `recovery_protocol_v1`

Optimization 1.1 is a post-hoc recovery experiment. The complete Severstal
baseline test was observed during Optimization 1, so the recovery holdout is
not a pristine independent test. The 1.0.0 conclusion remains
`STEEL_DOMAIN_VALIDATION_FAILED` regardless of later recovery results.

## Immutable inputs

- Model: `steel-patchcore` 1.0.0
- Bank SHA256: `291bb2903b6e9a3bc60ca4ae35380bd7cf312bc661366d10f87a75d3f0b8ebda`
- Source split SHA256: `64df9f817a995e64c7ee4d00e16962d94f500878fd1404d95227f26e7f913d07`
- Baseline threshold: `0.490039` (reference only; not reused for new candidates)
- Backbone: frozen Wide-ResNet-50-2 ImageNet-1K V1
- Layers: `layer2 + layer3`
- Feature dimension: 1536
- Feature normalization: frozen per-patch L2 normalization
- Distance: frozen cosine one-nearest-neighbour distance, `1 - max_similarity`
- Memory bank: frozen 50,000 rows
- Tiling offsets: `0, 256, 512, 768, 1024, 1280, 1344`
- Tile overlap rule: mean

This phase does not change the representation, feature normalization, bank,
reservoir, tiling, nearest-neighbour semantics, or model version.

## Deterministic recovery split

Seed: `42`

The ordered `test_anomaly` list in the frozen source split is permuted once with
`numpy.random.default_rng(42)`. The first 3,333 IDs become
`recovery_dev_anomaly`; the remaining 3,333 become
`recovery_holdout_anomaly`.

Development-side evidence capture:

| Role | Count | Use |
|---|---:|---|
| `train_normal` | 4,721 | Candidate-specific threshold calibration only |
| `validation_normal` | 590 | Development normal |
| `recovery_dev_anomaly` | 3,333 | Development anomaly |

Sealed recovery holdout:

| Role | Count |
|---|---:|
| `test_normal` | 591 |
| `recovery_holdout_anomaly` | 3,333 |

Dev and holdout IDs must be unique, disjoint, and together equal the frozen
6,666 anomaly IDs. No holdout inference is permitted during evidence capture or
candidate selection.

## Canonical raw evidence

Canonical dtype is `float32`. For each permitted original, capture:

- original ID and recovery role;
- seven tile indices and pixel x offsets;
- the actual raw nearest-neighbour patch-distance grid before any per-tile
  min-max normalization;
- raw tile score, defined as the maximum of that tile's raw grid; and
- lineage hashes and actual feature/grid geometry.

The grid shape and spatial stride must be derived from the actual embedding
output and asserted. With the current implementation it is expected—but not
hardcoded—to be 32×32 with stride 8. The deterministic mean-overlap stitched
raw representation is expected to be 32×200 and must be reconstructible from
the canonical seven grids plus offsets. Full 256×1600 float maps are not stored.

Artifacts are sharded every 100 originals beneath the ignored runtime path
`model-training/datasets/severstal-steel/raw/recovery-evidence/`. Every shard is
hashed and a durable checkpoint records completed IDs, completed shards,
current role, last update, and reconstruction error. Resume processes only IDs
not present in verified shards.

## Frozen aggregation-only candidate grid

No candidate may be added during the first-round evaluation.

| ID | Frozen definition |
|---|---|
| A0 | Exact 1.0.0 baseline: global maximum over all seven unstitched raw tile grids |
| A1 | 99.0th percentile of the flattened mean-overlap stitched raw grid |
| A2 | 99.5th percentile of the flattened mean-overlap stitched raw grid |
| A3 | 99.9th percentile of the flattened mean-overlap stitched raw grid |
| A4 | Mean of the highest `ceil(0.1% × N)` stitched raw responses |
| A5 | Mean of the highest `ceil(0.5% × N)` stitched raw responses |
| A6 | Mean of the highest `ceil(1.0% × N)` stitched raw responses |

Percentiles use NumPy's deterministic linear method. Top-percentage selection
uses at least one response and includes exactly the configured ceiling count.

A0 deliberately uses the unstitched grids because overlap-mean can change an
extreme response; this is necessary to reconstruct the frozen baseline exactly.
`max(stitched raw grid)` is recorded only as a diagnostic and is not an eighth
candidate.

Area-aware aggregation is excluded from this round.

## Threshold and candidate-selection rules

After a successful Evidence Capture Gate, each A0-A6 threshold will be:

```text
max(candidate score over all 4,721 train_normal originals)
```

No anomaly, validation normal, test normal, or holdout result may influence a
threshold. Candidate ranking is not authorized during the current evidence
capture phase.

When later authorized, development Gate requirements remain:

- Image AUROC ≥ 0.75
- Normal FPR ≤ 0.10
- Anomaly Recall ≥ 0.60

## Evidence Capture Gate

Capture can be declared `RECOVERY_EVIDENCE_READY` only when:

- counts are exactly 4,721 / 590 / 3,333 with zero duplicate IDs;
- holdout inference count is zero;
- every original has seven finite, consistently shaped float32 grids;
- tile scores equal per-grid maxima;
- stitched geometry and mean-overlap reconstruction are deterministic;
- the A0 score reconstructed from raw evidence agrees with the preserved 1.0.0
  checkpoint/train evidence at absolute tolerance `2e-6` and zero relative
  tolerance;
- bank/source/recovery split lineage hashes match; and
- every shard size and SHA256 is recorded in the evidence manifest.

If any reconstruction fails, the status is
`RAW_EVIDENCE_RECONSTRUCTION_FAILED` and no aggregation experiment may begin.

After the Evidence Capture Gate, stop and wait for explicit authorization.
Representation changes, candidate ranking, holdout capture/evaluation,
production promotion, and Optimization 2 are outside this phase.
