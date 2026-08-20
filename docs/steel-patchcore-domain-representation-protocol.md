# Steel PatchCore — Domain Representation Investigation Protocol (Optimization 1.1)

Protocol version: `domain_representation_protocol_v1`
Status of prior phase: **canonical PatchCore cross-check remains `CANONICAL_PATCHCORE_REFERENCE_BLOCKED`** (not FAILED — standard PatchCore was not excluded).

## Objective

Under a held-out diagnostic protocol, change **only the feature-representation family**
to test whether the ImageNet-supervised WRN-50-2 representation is the main source of
the steel-domain image-level failure. Everything else is frozen: no holdout, no baseline
mutation, no memory-bank strategy change, no image-aggregation change.

## Frozen candidates

- **D0** = frozen WRN best diagnostic result = **S2** (WRN-50-2 layer3 + 5×5 local average
  context). Image AUROC = `0.6029`; Q1 = `0.4790`, Q2 = `0.5305`, Q3 = `0.6145`,
  Q4 = `0.7876`. **Read from frozen results; never re-run.**
- **D1** = **DINOv2 ViT-S/14** spatial patch-token representation (one new candidate only).

## D1 model identity (frozen)

| Attribute | Value |
|---|---|
| Implementation | `facebookresearch/dinov2` official `torch.hub` |
| Entry point | `torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")` |
| Arch | `vit_small`, `embed_dim=384`, `depth=12`, `num_heads=6`, `patch_size=14`, `img_size=518`, `num_register_tokens=0` |
| Weights | LVD-142M self-supervised; `https://dl.fbaipublicfiles.com/dinov2/vit_small14/vit_small14_pretrain.pth` |
| License | Apache-2.0 |

## D1 feature semantics (frozen)

- Input: existing frozen **256×256 tile** (never resize the 256×1600 original to square).
  Original image → frozen 7 tiles → DINOv2 per tile. DINOv2 `PatchEmbed` asserts H,W are
  exact multiples of `patch_size=14`; 256 is not, so each tile is bilinearly resized
  **256 → 252** (`= 18×14`, ×0.9844, content-preserving, no crop) before the forward pass.
- Patch tokens only: `model.forward_features(x)["x_norm_patchtokens"]`.
  CLS token and any register tokens are **excluded** (plain `dinov2_vits14` has no register
  tokens). Actual geometry for the adapted 252×252 input: **18×18 = 324 patch tokens** × 384-d;
  positional encoding bicubic-interpolated from the 37×37 grid (`img_size=518`).
- Raw DINOv2 spatial patch tokens used directly — **no** extra pooling, no multi-layer
  concatenation, no CLS/token fusion (transformer tokens already carry context).
- Distance: **per-patch L2 normalization + cosine 1-NN** (`distance = 1 - cosine_similarity`),
  matching the validated project semantics. No Euclidean, no PatchCore reweighting, no k>1.

## Frozen non-variable protocol (reused, unchanged)

- Diagnostic subset: **reused, never resampled** — 1000 train_normal / 300 validation_normal /
  1000 recovery_dev_anomaly (250×Q1–Q4, seed 42). Subset manifest SHA verified at runtime.
- Memory bank: reservoir Algorithm R, **budget 50000**, **seed 42**, from the 1000 frozen
  train-normal diagnostic IDs. New D1 experimental bank (never overwrite any old bank);
  runtime path `model-training/runs/steel-domain-representation/D1-dinov2-s14/` (gitignored).
- Tiling: frozen 7 tiles at x0 ∈ {0,256,512,768,1024,1280,1344}. Original image score =
  **max** over 7 tile patch anomaly scores (A0 semantics).
- Threshold: **max(train-normal original-image scores)** — train-only calibration, diagnostic only.
- Primary metric: **Image AUROC** (ranking). Threshold metrics are diagnostic only.

## Evaluation

- D1 eval: 300 validation_normal + 1000 dev anomaly. Report Image AUROC; normal and anomaly
  score min/p50/p95/p99/max; anomaly-median − normal-median; TP/TN/FP/FN/Precision/Recall/F1/
  Normal-FPR/Anomaly-Recall.
- Quartiles: separately `validation_normal vs Q1|Q2|Q3|Q4` AUROC (frozen 250×4).

## Domain Representation Gate (frozen, never lowered after results)

- `DOMAIN_REPRESENTATION_SIGNAL_FOUND`: D1 Image AUROC ≥ **0.70** AND (D1 − D0) ≥ **+0.10**
  (i.e. ≥ ~0.7029).
- `DOMAIN_REPRESENTATION_STRONG_SIGNAL` (extra flag): D1 Image AUROC ≥ **0.80**.
- `SMALL_DEFECT_REPRESENTATION_SIGNAL` (secondary, never substitutes the gate):
  Q1 or Q2 AUROC improvement ≥ **+0.10**.
- Else `DOMAIN_REPRESENTATION_GATE_FAILED`.
- If a credible DINOv2 runtime cannot be established: `DOMAIN_REPRESENTATION_REFERENCE_BLOCKED`.

## Prohibited this phase

No fine-tuning (no LoRA/linear probing/steel SSL/supervised training); no bank-strategy
change (no coreset/k-center/multi-neighbor/bank-size search/ensembling); no second backbone;
no holdout access (`HOLDOUT_ACCESS_COUNT == 0`); no Optimization 2.

## Verdicts (exactly one)

`DOMAIN_REPRESENTATION_SIGNAL_FOUND` · `DOMAIN_REPRESENTATION_GATE_FAILED` · `DOMAIN_REPRESENTATION_REFERENCE_BLOCKED`