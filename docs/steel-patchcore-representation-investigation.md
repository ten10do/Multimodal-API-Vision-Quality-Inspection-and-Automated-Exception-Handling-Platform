# Steel PatchCore Validity Recovery — Representation Investigation

Status: `REPRESENTATION_INVESTIGATION` (audit + diagnostics, pre-experiment)

This phase follows `RECOVERY_AGGREGATION_GATE_FAILED` (commit `f2a1de8`). It
treats "representation failure" as a hypothesis to be tested, not a proven
conclusion. `steel-patchcore` 1.0.0 remains `STEEL_DOMAIN_VALIDATION_FAILED`
and is never reinterpreted. The sealed holdout remains untouched
(`HOLDOUT_ACCESS_COUNT = 0`).

## 1. Current implementation semantics (traced from code, not documents)

The path below is the actual runtime path
(`inference-service/inference_app/patchcore_predictor.py`,
`model-training/steel_patchcore/tile.py`, recovery evidence capture).

```text
image (256 x 1600)
→ 7 tiles of 256x256 at x0 = 0,256,512,768,1024,1280,1344  (last two overlap)
→ ImageNet normalize (0.485/0.456/0.406, 0.229/0.224/0.225), no resize (256 input)
→ Wide-ResNet-50-2 IMAGENET1K_V1, eval, no grad
→ conv1 → bn1 → relu → maxpool → layer1
→ layer2: [1, 512, 32, 32]          (native 32x32, stride 8 px / patch)
→ layer3: [1, 1024, 16, 16]         (stride 16 px / patch)
→ layer3 bilinear-upsampled to 32x32, align_corners=False
→ concat(layer2, layer3_up) → [1, 1536, 32, 32]
→ flatten → (1024, 1536) patch embeddings
→ per-patch L2 normalization (zero norm clipped to 1e-8)
→ distance_i = 1 - max_j (emb_i · bank_j)     # cosine 1-NN on unit vectors
→ tile score = max over 1024 patch distances
→ image score (A0) = max over 7 tile scores
```

Confirmed numerics:

| Item | Actual implementation |
|---|---|
| Backbone | `wide_resnet50_2(IMAGENET1K_V1)`, frozen, eval mode |
| Feature layers | `layer2` (512-d) + `layer3` (1024-d) |
| Feature alignment | bilinear upsample `layer3` 16→32 (`align_corners=False`), then channel concat |
| Embedding dim | 1536 |
| Per-patch normalization | **L2 normalization exists** (centralized over all 1536 dims after concat) |
| Distance | cosine on unit vectors → `1 - max_similarity` |
| NN | 1 nearest neighbour (`num_neighbors` field exists but is unused) |
| Bank sampling | reservoir (Algorithm R), `np.random.default_rng(42)`, 50,000 rows |
| Bank source | `train_normal` only; `candidate_patches = 33,840,128`; bank fraction ≈ 0.148% |
| Image score | max over 7 tile maxima (extreme max, unnormalized raw distance) |
| Pixel path | independent per-tile min-max → uint8 → bilinear → re-min-max → mean-stitch |

## 2. Feature-layer configuration risk audit

The frozen R0 concatenates a 512-channel native 32×32 map with a 1024-channel
16×16 map that has been bilinearly upsized to 32×32. Two mechanisms follow
directly and are quantified in the protocol/diagnostics stage:

1. **Dimension/magnitude imbalance.** After concat the joint per-patch L2
   normalization spreads unit mass across 1536 dims, 1024 of which are layer3.
   Unless layer3 activations are systematically ~1/√2 of layer2 magnitude in the
   same directions, cosine similarity is dominated by layer3. A defect that
   perturbs only the finer layer2 channels can therefore be numerically masked.
2. **Effective resolution.** layer3 is native 16×16 and its 32×32 version is
   produced by interpolation, which adds no new spatial information and can
   smear sub-patch detail. Small defects (validated Q1 area ratio ≤ 0.0109,
   ≈ ≤ 4,460 px) are a few layer2 patches or sub-patch at layer3.

Both are testable by the R0/R1/R2 grid (R1 = layer2 only 512-d; R2 = layer3
only 1024-d), holding bank sampler/budget/seed/distance/tiling/aggregation fixed.

## 3. Memory-bank coverage status

- candidate normal patches: `33,840,128`
- frozen bank rows: `50,000`
- fraction: ≈ `0.148%`
- sampler: uniform reservoir (seed 42), **not** a representative coreset.

A low fraction alone does not prove under-coverage. The offline coverage
diagnostic compares train-normal raw nearest-bank distances against
validation-normal distances and per-tile uniformity before any conclusion. If
validation normals are not systematically farther from the bank than train
normals, coverage is treated as adequate for the current scale and the failure
is attributed to representation, not bank size.

## 4. PatchCore reference deviation audit

Classification legend:

- `EXPECTED DESIGN CHOICE` — deliberate, documented, kept for the baseline.
- `POTENTIAL VALIDITY RISK` — plausible mechanism, treated as a hypothesis.
- `CONFIRMED BUG` — numerically wrong relative to the design; none found.

| # | Aspect | Current | PatchCore reference | Class |
|---|---|---|---|---|
| 1 | Feature layers | layer2 + layer3 concat | layer2 + layer3 concat | EXPECTED DESIGN CHOICE |
| 2 | Alignment | bilinear upsample layer3 | average-pool/adaptive-pool alignment | POTENTIAL VALIDITY RISK |
| 3 | Neighborhood aggregation | raw 1×1 patches | local average-pool patch aggregation (adds context, de-noises) | POTENTIAL VALIDITY RISK |
| 4 | Per-patch normalization | L2 over concat | typically L2 or none per implementation; not a spec violation | EXPECTED DESIGN CHOICE |
| 5 | Distance | cosine 1-NN (`1 - max sim`) | cosine k-NN, k=1 default | EXPECTED DESIGN CHOICE |
| 6 | Bank selection | uniform reservoir 50k (0.148%) | greedy/random coreset (typical ~1%+) | EXPECTED DESIGN CHOICE (documented) / POTENTIAL VALIDITY RISK (coverage) |
| 7 | Reweighting | none | optional neighbor-distance reweighting | EXPECTED DESIGN CHOICE |
| 8 | Image score | max patch distance | max patch distance (default) | EXPECTED DESIGN CHOICE |

Confirmed deviations from the reference are **design choices** made for the
1.0.0 baseline (reservoir sampling and no reweighting are explicitly recorded).
The two deviations most likely to matter for representation validity are #2
(interpolation smearing) and #3 (no neighborhood aggregation), plus the #6
coverage fraction. These are investigated, not assumed.

## 5. Existing-evidence diagnostics (no GPU)

A separate offline evaluator consumes only the frozen raw evidence, GT masks,
canonical splits, and bank metadata. It will report:

- A. train-normal vs validation-normal raw nearest-bank distance distribution;
- B. validation-normal vs recovery-dev-anomaly distance distribution;
- C. defect area ratio vs A0 raw score;
- D. max connected-component area vs A0 raw score;
- E. Q1-Q4 score distribution (already in aggregation results, refreshed here);
- F. normal high-response samples (top validation-normal A0);
- G. anomaly low-response samples (bottom recovery-dev-anomaly A0);
- H. score vs tile position (per-tile max and argmax-tile histogram).

These are computed before any re-inference.

## 6. Stop semantics

`steel-patchcore` 1.0.0 remains `FAILED`. Any R/N candidate is `EXPERIMENTAL`
and is never registered as a production candidate or promoted to `steel-patchcore
1.1.0` during this phase. No holdout access is authorized.