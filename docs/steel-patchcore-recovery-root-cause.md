# Steel PatchCore Validity Recovery — Root-Cause Audit

Status: `RECOVERY_EVIDENCE_BLOCKED`

Optimization 1.1 is a post-hoc recovery experiment. This audit preserves the
`steel-patchcore` 1.0.0 baseline conclusion `STEEL_DOMAIN_VALIDATION_FAILED` and
does not reinterpret the baseline as valid.

No model inference, training, memory-bank rebuild, threshold tuning, or frozen
artifact mutation was performed during this audit.

## Immutable baseline

- Baseline commit: `1fca85534e9f007fee49f084853808d0e375de4a`
- Bank SHA256: `291bb2903b6e9a3bc60ca4ae35380bd7cf312bc661366d10f87a75d3f0b8ebda`
- Split SHA256: `64df9f817a995e64c7ee4d00e16962d94f500878fd1404d95227f26e7f913d07`
- Threshold: `0.490039`
- Image AUROC: `0.4817`
- Pixel AUROC (mean per anomaly image): `0.8319`
- AUPRO (mean per anomaly image): `0.5838`
- Formal operating point: TP 0, TN 590, FP 1, FN 6666
- Baseline metrics SHA256: `f873ff2d7fedd26a87154eaf803b434697db583a4530d6e0f30ea88481cf494b`

## Actual image-score path

The path below is traced from the implementation, not inferred from design
documents.

1. A 256×1600 original is cropped into seven 256×256 tiles at x offsets
   `0, 256, 512, 768, 1024, 1280, 1344`. The final two tiles overlap; there is
   no resize, padding, or uncovered column at this stage
   (`model-training/steel_patchcore/tile.py`).
2. Each tile is ImageNet-normalized and passed through frozen
   Wide-ResNet-50-2 (`IMAGENET1K_V1`).
3. `layer2` produces a 32×32×512 feature map. `layer3` produces a
   16×16×1024 map and is bilinearly upsampled to 32×32.
4. The two maps are directly concatenated into 1,024 patch embeddings of
   dimension 1,536. Each concatenated embedding is L2-normalized.
5. Each patch is compared to the 50,000 L2-normalized normal-bank vectors by
   matrix multiplication. The implementation retains the maximum cosine
   similarity and computes one-nearest-neighbour distance as
   `distance = 1 - max_similarity`.
6. The 1,024 distances form a raw 32×32 tile distance grid.
7. Tile score is the maximum raw patch distance.
8. Original-image score is the maximum of the seven tile scores. It is
   therefore the single largest raw distance among 7,168 patch responses.
9. The image is classified anomalous when this score is greater than or equal
   to the train-normal threshold.

In compact form:

```text
original
→ 7 tiles
→ layer2 + bilinear(layer3)
→ direct 1536-d concat
→ per-patch L2 normalization
→ cosine 1-NN distance to 50k bank
→ 32×32 raw distance grid per tile
→ max patch distance per tile
→ max of 7 tile scores
```

## Pixel path is numerically different

The pixel-evidence path branches from the same raw 32×32 distance grid but
does not preserve its absolute distance scale:

1. Each tile grid is independently min-max normalized to `[0, 1]`.
2. It is quantized to uint8 and bilinearly resized to 256×256.
3. The resized tile is min-max normalized again.
4. Seven maps are stitched to 256×1600; overlap uses the mean.
5. Per-image Pixel AUROC and AUPRO are calculated from this stitched relative
   map and then averaged over anomaly images.

Consequently, every non-constant tile has a normalized local maximum near 1
regardless of whether its raw nearest-neighbour distances are globally large
or small. Pixel AUROC measures within-image spatial ordering after per-tile
rescaling; image AUROC measures cross-image ordering of an unnormalized raw
extreme. The observed `0.8319` versus `0.4817` is therefore not contradictory:
useful local relative ordering does not imply useful absolute image ranking.

## PatchCore semantic audit

| Semantic | Actual implementation |
|---|---|
| Feature normalization | ImageNet input normalization plus L2 normalization of each concatenated patch embedding |
| Layer alignment | Bilinear layer3 upsampling followed by direct channel concatenation |
| Nearest neighbour | One neighbour; cosine similarity on unit vectors, converted to `1 - similarity` |
| `num_neighbors` | Constructor field exists but is not used by scoring |
| Patch/tile score | Maximum raw patch distance |
| Original score | Maximum of seven tile maxima |
| Pixel map | Independent per-tile min-max normalization, uint8 quantization, bilinear upsampling |
| Reweighting | None |
| Spatial smoothing | No Gaussian smoothing |
| Bank selection | Uniform reservoir sampling, not PatchCore coreset selection |
| Memory-bank size | 50,000 of 33,840,128 candidate train-normal patches (about 0.148%) |

The 50k reservoir retains about 10.6 patches per train original, or about 1.5
patches per tile in expectation. This is sparse coverage of local steel
textures, but existing artifacts do not contain the nearest-neighbour
distribution needed to quantify whether bank coverage is the dominant cause.
The bank must not be enlarged based on this observation alone.

There is also a reproducibility limitation in resumable training: the feature
checkpoint stores the reservoir, processed count, and completed IDs, but not
the NumPy generator state. On resume the generator is recreated from seed 42
without advancing past earlier draws, so a resumed run is not guaranteed to
produce the same reservoir as an uninterrupted run. The current bank remains
immutable and is identified by its SHA256; this issue does not invalidate the
recorded baseline, but it must be corrected before claiming repeatable bank
reconstruction.

## Empirical failure mechanism supported by current evidence

- The train-normal maximum policy is correctly train-only for the frozen max
  aggregator, but it is an extreme-value threshold over 7,168 responses.
- Test-anomaly median (`0.358937`) is below test-normal median (`0.368732`).
- Every anomaly maximum (`≤ 0.477310`) is below the threshold, while one normal
  reaches `0.490773`.
- Threshold tuning cannot repair Image AUROC below 0.5 because the ranking is
  already failed.
- The weak correlations between image score and Pixel AUROC (`0.0996`) and
  AUPRO (`0.1266`) are consistent with the two paths using different scale
  semantics.

The strongest currently supported explanation is a combination of:

1. **aggregation failure** — a single raw extreme is used for image ranking;
2. **scale-semantics mismatch** — pixel evidence discards absolute tile scale
   while image scoring depends entirely on it; and
3. **possible representation/bank coverage failure** — normal and anomaly raw
   maxima overlap, but patch-distance evidence is missing, so this component is
   not yet quantifiable.

## Existing-evidence inventory

| Artifact | Preserved evidence | Missing evidence needed for recovery |
|---|---|---|
| `steel_eval_ckpt.json` | 590 validation-normal and 591 test-normal `score/pred`; 6,666 anomaly `score/pred/pixel_auc/aupro` | Per-tile scores, raw patch grids, stitched maps, response quantiles, pixel thresholds |
| `train_normal_scores.json` | 4,721 original max scores and IDs | Train-normal responses for any non-max aggregation |
| threshold checkpoints | Original max scores and current maximum | Per-tile or per-patch response distributions |
| `steel_train_ckpt.npz` | 50k reservoir, candidate count, completed train IDs | Source patch ownership and query-to-bank distance evidence |
| `bank.npz` | Frozen 50k×1536 normal feature bank | Query responses for train/dev/holdout images |
| `metrics.json` | Aggregate distributions and formal metrics | Reconstructible per-image response distributions |

No current artifact stores a per-tile score, a raw 32×32 distance grid, a
stitched anomaly map, pixel values, or sufficient map statistics. Pixel AUROC
and AUPRO are two lossy scalars and cannot reconstruct the maps that produced
them. Likewise, an original maximum cannot determine top-k means, percentiles,
top-percentage means, or anomaly-area-aware scores: infinitely many response
vectors share the same maximum.

## Stop condition

`RECOVERY_EVIDENCE_BLOCKED`

The existing evidence is insufficient for aggregation recovery. In particular,
it is impossible to calculate a train-only threshold for any new aggregation,
so candidate evaluation would violate the protocol if it proceeded from the
existing files. No recovery split, candidate grid, development ranking, or
holdout evaluation has been created or executed.

## Minimum necessary re-inference design (not executed)

The next permitted step should be a separately reviewed, preregistered evidence
capture—not a replay of the full 7,257-image formal test:

1. Persist a deterministic 50/50 anomaly dev/holdout manifest before inference.
2. Freeze a finite candidate grid and evidence schema before observing any new
   candidate result.
3. Keep the frozen 1.0.0 bank, backbone, tiling, and distance semantics.
4. Capture raw pre-normalization 32×32 patch-distance grids for development-side
   data only: 4,721 train normals, 590 validation normals, and 3,333 recovery-dev
   anomalies. Do not access recovery holdout at this stage.
5. Store image ID, split role, seven tile IDs/offsets, raw grid dtype/shape,
   bank SHA, original split SHA, recovery-manifest SHA, extractor commit, and
   deterministic configuration. Use atomic checkpointing and unique-ID gates.
6. Derive all preregistered aggregations and every threshold offline. Thresholds
   must use only the 4,721 train-normal grids.
7. Only if one candidate passes the development Gate, freeze it and capture the
   591 test-normal plus 3,333 recovery-holdout anomaly grids once. Do not retune
   after that one-shot holdout.

At float16, the development-side raw grids would contain 61,960,192 values
(about 118 MiB before metadata/compression); the one-shot holdout would add
28,127,232 values (about 54 MiB). This retains the response distribution needed
for every requested bounded aggregator while avoiding repeated GPU inference.

Even if a later post-hoc holdout passes, the maximum allowed claim remains
`RECOVERY_HOLDOUT_PASS`; an independent steel-domain confirmation dataset is
still required before restoring a formal domain-validation claim.
