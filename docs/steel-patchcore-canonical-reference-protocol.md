# Steel PatchCore — Canonical Reference Cross-check Protocol

`canonical_patchcore_protocol_v1`

This phase is the Canonical PatchCore Reference Cross-check (Optimization 1.1),
NOT Optimization 2. Objective: determine whether the steel-domain failure is
explained by our simplified custom PatchCore semantics (reservoir / k=1 / no
reweighting / custom alignment / custom scoring), or whether PatchCore +
ImageNet WRN representation itself lacks validity on this domain.

## 1. Frozen lineage (re-verified)

- bank SHA256 `291bb2903b6e9a3bc60ca4ae35380bd7cf312bc661366d10f87a75d3f0b8ebda`
- baseline threshold `0.490039`
- source split SHA256 `64df9f817a995e64c7ee4d00e16962d94f500878fd1404d95227f26e7f913d07`
- recovery split SHA256 `f60ce0a6aa012ad1d9a9636800c1e94f365940500f20b8deb2f7fd9ffbc55448`
- evidence manifest SHA256 `7067bbc6da747ffa90db9532729f579e19ee90fbcdf3867c96f2dd56a0d10303`
- diagnostic subset manifest SHA256 `8d39aec04c35ac7a7edc931c6dfb3494c6ada6a0a6e13e77b89bc894f9919075`
- `steel-patchcore` 1.0.0 permanently `STEEL_DOMAIN_VALIDATION_FAILED`

## 2. Reference implementation identity

- package: **anomalib** (public, open-source; developed by Intel, the canonical
  PatchCore implementation used across the AD literature)
- version: **0.7.0** (PyPI), the last classic PatchCore API
  (`anomalib.models.patchcore.torch_model.PatchcoreModel`)
- environment: dedicated `.venv-canonical` (Python 3.11.9)
  - torch 2.11.0+cu128, torchvision 0.26.0+cu128 (same CUDA family as `.venv-steel`)
  - timm 0.6.12 (pinned by anomalib 0.7.0)
  - `anomalib==0.7.0`
- NOT added to project production dependencies (experimental validation tool)

## 3. Frozen diagnostic subset (REUSED)

train_normal 1000 / validation_normal 300 / recovery_dev_anomaly 1000
(250 × Q1..Q4). No resampling. Holdout access count = 0.

## 4. Frozen candidates

- **C0 — current best custom reference** = S2 (layer3 + 5×5 avg context,
  reservoir 50k, cosine 1-NN k=1, A0 max). AUROC = **0.6029** (read from the
  frozen spatial-context results; NOT re-run).
- **C1 — canonical PatchCore** via anomalib 0.7.0, standard PatchCore
  semantics, WideResNet-50-2, standard layers, coreset bank, k-NN reweighting.

Only C0 and C1. No coreset-ratio / layer / k / neighbor / backbone / resolution
/ implementation search after seeing C1.

## 5. C1 canonical configuration (frozen)

| Setting | Value |
|---|---|
| backbone | wide_resnet50_2 (ImageNet pretrained) |
| feature layers | ["layer2", "layer3"] |
| input | 256×256 tile (frozen seven-tile crop; no full-image resize) |
| feature alignment | reference default (per-source code) |
| memory bank | greedy coreset, ratio 0.1 (reference default) |
| NN | reference default `num_neighbors = 9` |
| image score | reference reweighted anomaly score |
| pixel map | reference anomaly-map generator (recorded, not gated) |

Exact semantics are documented from the installed anomalib 0.7.0 source in the
results report (§4). The candidate is kept as the library's default, NOT
modified to mimic our implementation.

## 6. Tiling & image aggregation (fair)

Original 256×1600 → frozen 7 tiles (x0 = 0/256/512/768/1024/1280/1344).
Each tile is fed to C1 as a 256×256 image. Original image score =
`max(7 tile reference image scores)`. Tiling identical to all prior phases;
only the per-tile algorithm differs.

## 7. Train / eval protocol

- train / memory-bank: 1000 frozen train_normal IDs only (no anomaly training)
- eval: 300 validation_normal + 1000 dev anomaly IDs
- threshold = `max(1000 train_normal reference image scores)` (train-only)
- threshold metrics are diagnostic only; primary metric is Image AUROC

## 8. Core metrics (C1)

Image AUROC; normal score min/p50/p95/p99/max; anomaly score
min/p50/p95/p99/max; anomaly−normal median; TP/TN/FP/FN/Precision/Recall/F1/
Normal-FPR/Anomaly-Recall (diagnostic threshold).

## 9. Quartile metrics (C1 vs C0)

validation-normal vs Q1/Q2/Q3/Q4 AUROC (same 250×4 stratification), compared
against C0 (S2) quartiles. Q1/Q2 movement is inspected.

## 10. Canonical PatchCore Gate (frozen)

- `CANONICAL_PATCHCORE_SIGNAL_FOUND` iff C1 Image AUROC ≥ 0.65 AND
  (C1 − C0) ≥ +0.05.
- `CANONICAL_PATCHCORE_STRONG_SIGNAL` additionally if C1 AUROC ≥ 0.75.
- Otherwise `CANONICAL_PATCHCORE_GATE_FAILED`.
- `CANONICAL_PATCHCORE_REFERENCE_BLOCKED` iff the reference cannot be reliably
  installed/run (never "pretend" the reference was validated).

## 11. Runtime isolation & lifecycle

- C1 model/bank/embeddings under `model-training/runs/steel-canonical-patchcore/`
- never overwrite baseline / representation / spatial-context artifacts
- one reference worker; lifecycle lock; checkpoint/resumable long tasks
- reference env is a separate venv (`.venv-canonical`), not `.venv-steel`

## 12. Interpretation rules

- PASS ≠ "PatchCore validated for Severstal" → only "canonical PatchCore
  semantics show meaningful recovery signal on the frozen development
  diagnostic subset."
- FAILED ≠ "PatchCore never works for steel" → only "both custom and canonical
  PatchCore configurations failed to establish sufficient image-level validity
  under the current WRN-50-2 / Severstal diagnostic protocol."

## 13. Outcome mapping (final verdict)

`CANONICAL_PATCHCORE_SIGNAL_FOUND` | `CANONICAL_PATCHCORE_GATE_FAILED` |
`CANONICAL_PATCHCORE_REFERENCE_BLOCKED`.

STOP after any verdict; do not auto-enter full development / holdout / new
backbone / domain adaption / bank-size search / coreset-ratio search /
Optimization 2.