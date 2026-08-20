# Steel PatchCore — Canonical Reference Cross-Check Results

Status: **CANONICAL_PATCHCORE_REFERENCE_BLOCKED**
Protocol: `canonical_patchcore_protocol_v1` (`docs/steel-patchcore-canonical-reference-protocol.md`)

---

## 1. Handoff Audit

- Branch: `feat/steel-patchcore-validity-recovery-v1.2`.
- HEAD: `73d9fdf2a81c45b0d32fd60709d3b1925d5b4135` (clean tree at phase start).
- Frozen lineage re-verified by the adapter before any work:
  - frozen bank `bank.npz` SHA256 = `291bb2903b6e9a3bc60ca4ae35380bd7cf312bc661366d10f87a75d3f0b8ebda`
  - diagnostic subset manifest SHA256 = `8d39aec04c35ac7a7edc931c6dfb3494c6ada6a0a6e13e77b89bc894f9919075`
- `HOLDOUT_ACCESS_COUNT` = 0 throughout; recovery holdout not touched.

## 2. Reference Implementation

| Attribute | Value |
|---|---|
| Library | `anomalib` |
| Version | `0.7.0` (pip, exact) |
| Model | `anomalib.models.patchcore.torch_model.PatchcoreModel` |
| Backbone | `wide_resnet50_2` (`timm 0.6.12`), `pretrained=True` (ImageNet-1k, `wide_resnet50_racm-8234f177.pth`) |
| Layers | `["layer2", "layer3"]` |
| Input size | `(256, 256)` — one frozen tile |
| Coreset | greedy k-center (KCenterGreedy) + SparseRandomProjection, `sampling_ratio=0.1` |
| k-NN | `num_neighbors=9` (k=1 patch score + 9-support reweighting) |
| Distance | Euclidean on raw 1536-d embeddings (no L2 normalization) |

## 3. Frozen Diagnostic Protocol (applied verbatim)

- Diagnostic subset reused, never resampled: 1000 train_normal / 300 val_normal / 1000 recovery_dev_anomaly (250×Q1–Q4, seed 42). Subset manifest SHA verified.
- Tiling frozen: `x0 ∈ {0, 256, 512, 768, 1024, 1280, 1344}`, each 256×256 tile fed to the reference as-is (no 256×1600 → 256×256 resize). Original score = max(7 tile scores).
- Threshold = max(train-normal image scores) (train-only calibration, diagnostic only). Primary metric = image AUROC; also normal-vs-Q1..Q4 AUROC.
- C0 = S2 frozen spatial result, AUROC = `0.6029` (read from frozen results, not re-run).
- Gate: C1 ≥ 0.65 AND (C1 − C0) ≥ +0.05 → `SIGNAL_FOUND`; additionally ≥ 0.75 → `STRONG_SIGNAL`; else `GATE_FAILED`; reference cannot run at protocol scale → `REFERENCE_BLOCKED`.

## 4. Canonical PatchCore Semantics (read from installed 0.7.0 source)

1. **Feature extraction** (`TimmFeatureExtractor`): `timm.create_model("wide_resnet50_2", pretrained=True, features_only=True, exportable=True, out_indices=[layer2, layer3])`.
   For 256×256 input: `layer2` → (512, 32, 32), `layer3` → (1024, 16, 16). No internal normalization — caller must supply ImageNet-normalized input.
2. **Local aggregation**: every layer is passed through `AvgPool2d(kernel=3, stride=1, padding=1)` (default `count_include_pad=True`).
3. **Alignment + concat**: `embedding = layer2_pooled`; `layer3_pooled` is upsampled to layer2 spatial size with `F.interpolate(..., mode="nearest")`; concat → 1536-d.
4. **Flatten**: `(B,1536,H,W) → (B,H,W,1536) → (B·H·W, 1536)` — row-major patch order; 32×32 = 1024 patches/tile.
5. **Coreset**: `KCenterGreedy(embedding, 0.1)` = SparseRandomProjection(eps=0.9) to `johnson_lindenstrauss_min_dim(n, eps=0.9)` components (**389 for n = 7,168,000**), then greedy k-center from a random start index (`torch.randint`, not seeded → non-deterministic). The projected features are used only for *selection*; the bank stores **raw 1536-d** embeddings of the selected patches.
6. **Scoring** (`forward`, eval): 1-NN Euclidean (min) distance per patch → `patch_scores`; the image score is the reweighted distance of the most-anomalous patch:
   - `m*` = memory-bank patch nearest to the max-score test patch;
   - find `num_neighbors=9` bank-patch neighbours of `m*`;
   - `weights = 1 − softmax(distances(test_patch, those 9))[0]`;
   - final score = `weights × s*` (isolation-based reweighting).
7. **Anomaly map** (diagnostic only): `patch_scores` bilinearly upsampled to 256×256 then `GaussianBlur2d(σ=4)`.

**Deviation vs our custom PatchCore** (the very thing this cross-check tests):

| Aspect | Custom (prior phases) | Canonical 0.7.0 |
|---|---|---|
| Distance | cosine + per-patch L2-norm | raw Euclidean |
| Layers | layer3-only (S2) / layer2+layer3 bilinear (R2) | layer2+layer3, AvgPool3 + **nearest** align |
| Bank | reservoir 50k | greedy coreset 0.1 (SRP 389 + k-center) |
| Scoring | k=1, no reweighting | k=1 + **9-support reweighting** |

## 5. C0 (frozen)

AUROC = `0.6029` (S2: layer3 + 5×5 avg context, frozen spatial-context result). Quartiles: Q1 0.4790 / Q2 0.5305 / Q3 0.6145 / Q4 0.7876.

## 6. C1 Results (canonical PatchCore, full fidelity)

**Not produced.** The canonical configuration is memory-infeasible on this machine *at the frozen protocol scale*, before any metric can exist:

- Patches: 1000 images × 7 tiles × 1024 patches = **7,168,000**.
- Raw embedding matrix: **44.0 GB** (float32, 1536-d).
- SparseRandomProjection projection matrix: **11.2 GB** (389 components).
- Peak during coreset selection: **≈ 55.2 GB**.
- Total RAM of the machine: **34.2 GB**.
- Resulting coreset bank: **716,800 patches × 1536-d = 4.4 GB**, which also exceeds the 8 GB GPU for the per-tile Euclidean k-NN + reweighting (each tile → 1024 × 716,800 distance matrix ≈ 2.9 GB, ~1.1 TFLOP/tile).

`55.2 GB > 34.2 GB` ⇒ faithful full-fidelity C1 cannot run. The only routes — hand-rolling a streaming coreset, or shrinking the frozen 1000-IDs × 7-tile × 0.1-coreset protocol — are both excluded by this phase's rules (and a hand-rolled version must not be called "canonical").

### Smoke evidence (reference is installed, imports, and runs — the parser passes cleanly)

Run under `.venv-canonical` (Python 3.11.9, torch 2.11.0+cu128, anomalib 0.7.0, CUDA RTX 5060):

```
feature_extractor_backbone      wide_resnet50_2
embedding_shape_one_tile        (1024, 1536)   ← confirms 32x32x1536, 1024 patches/tile
two_tile_embedding_shape        (2048, 1536)
coreset_bank_shape              (204, 1536)    ← 0.1 x 2048 (greedy coreset runs)
sample_score_value              30.053         ← Euclidean image score
anomaly_map_shape               (1, 1, 256, 256)
gpu                            NVIDIA GeForce RTX 5060
CANONICAL_PATCHCORE_SMOKE_OK
```

This proves the reference itself is sound and that its one-tile embedding is exactly (1024 × 1536), which anchors the memory math above.

## 7. C0-vs-C1 Comparison

Not applicable — no C1 metric exists. C0 = 0.6029 stands unchanged.

## 8. Quartile Comparison

Not applicable — no C1 metric exists.

## 9. Root-cause Update

Unchanged from the prior phase. The cross-check that would have isolated our **semantic deviations** (Euclidean + greedy coreset + 9-support reweighting + AvgPool3/nearest alignment + layer2&3) from the **representation itself** (WideResNet-50-2 + ImageNet on steel) could not run because the canonical *greedy coreset* bank construction scales as O(n·d) storage and O(n·k) distance compute, which at 7.168M patches exceeds this machine's 34.2 GB RAM and 8 GB GPU.

> Note: this is itself a meaningful, honest observation about the canonical method — PatchCore's greedy coreset assumes a dataset whose total patch count fits in memory; a 1000-image × 7-tile Severstal patch population (~7.2M) does not on commodity hardware, which is precisely why our custom pipeline used a fixed-size reservoir bank.

## 10. Gate Verdict

`CANONICAL_PATCHCORE_REFERENCE_BLOCKED` — the reference could not be run at the frozen protocol scale (memory), so neither `SIGNAL_FOUND` nor `GATE_FAILED` can be claimed. **STOP.**

## 11. Tests

- `inference-service/tests/test_steel_canonical_reference.py`: **8 passed** (`.venv`, CPU) — frozen config, 7-tile max aggregation, train-only threshold, gate semantics, strong signal, config serialization determinism, manifest reuse + holdout isolation, runtime isolation + baseline immutability.
- `inference-service/scripts/run_steel_canonical_patchcore.py --smoke`: `CANONICAL_PATCHCORE_SMOKE_OK` (`.venv-canonical`, GPU).
- `inference-service/scripts/run_steel_canonical_patchcore.py` (preflight, default): emits `CANONICAL_PATCHCORE_REFERENCE_BLOCKED` with the memory math (`.venv-canonical`).
- Remaining steel suite: each `test_steel_*.py` passes in isolation; the whole-directory collection intermittently trips a **pre-existing** Windows CPU-torch `c10.dll` `OSError [WinError 1114]` import-order issue (numpy/PIL loaded before torch), reproducible **without** this phase's new files and unrelated to these changes.

## 12. Git

Committed precisely (lineage-verification branch, no `git add .`, no reset/clean/stash): protocol doc, `canonical_reference.py`, the reference runner, the adapter tests, and these results docs. No baseline/representation/spatial-context artifact modified; frozen `bank.npz` untouched.

## 13. Limitations

- Full-fidelity C1 could not run (55.2 GB > 34.2 GB RAM) — verdict is `BLOCKED`, not a metric.
- The reference import required environment repairs only (nothing PatchCore-semantic): `anomalib==0.7.0` + `albumentations<1.4`; `requests ftfy regex gdown`; `psutil`; a stub-by-pass of `anomalib.models/anomalib.data/anomalib.pre_processing` package `__init__`s (they eagerly import unrelated models/data that pull `albumentations→imgaug` on NumPy 2 and CLIP); and a one-line `field(default_factory=…)` fix in `timm/models/maxxvit.py` (timm 0.6.12 vs Python 3.11 dataclass rule, unrelated family).
- The anomalous `wide_resnet50_2` pretrained weight used is ImageNet-1k (`wide_resnet50_racm`); canonical anomalib default.
- The full steel test suite has a pre-existing intermittent CPU-torch DLL-load flakiness on Windows (see §11); no test asserts on it and no phase artifact depends on it.