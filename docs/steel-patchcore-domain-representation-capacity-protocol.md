# Steel PatchCore — DINOv2 Capacity Cross-Check Protocol (Optimization 1.1)

Protocol version: `domain_representation_capacity_protocol_v1`
Status carry-over: canonical PatchCore cross-check remains **`CANONICAL_PATCHCORE_REFERENCE_BLOCKED`** (not FAILED).

## Question this phase answers (one variable only)

DINOv2 ViT-S/14 produced a clear-but-insufficient recovery signal. Does increasing
**capacity within the same DINOv2 family** — ViT-S/14 → **ViT-B/14** — recover enough
image-level separability under the *identical* frozen diagnostic protocol?

## Frozen references (never re-run)

- **D0** = WRN S2 (layer3 + 5×5 context): Image AUROC **0.6029**; Q1 0.4790 / Q2 0.5305 / Q3 0.6145 / Q4 0.7876.
- **D1** = DINOv2 ViT-S/14 raw patch tokens: Image AUROC **0.6699**; Q1 0.5843 / Q2 0.6086 / Q3 0.6607 / Q4 0.8261.
  D1 verdict = `DOMAIN_REPRESENTATION_GATE_FAILED`, with **`SMALL_DEFECT_REPRESENTATION_SIGNAL = TRUE`** (Q1 +0.1053).

## D2 model identity (frozen, filled from real smoke)

| Attribute | Value |
|---|---|
| Implementation | `facebookresearch/dinov2` official `torch.hub` |
| Entry point | `torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")` |
| Arch | `vit_base`, `embed_dim=768`, `depth=12`, `num_heads=12`, `patch_size=14`, `img_size=518`, `num_register_tokens=0` |
| Weights | LVD-142M self-supervised; `https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth` (SHA256 `0b8b82f8…fad8c73`) |
| License | Apache-2.0 |

## D2 feature semantics (identical to D1)

- Input: frozen **256×256 tile** → bilinear **252×252** (multiple of patch_size 14, no crop).
- Extract `model.forward_features(x)["x_norm_patchtokens"]`; CLS and register tokens excluded
  (plain `dinov2_vitb14` has 0 registers). Geometry: **18×18 = 324 patch tokens** × 768-d.
- Raw spatial patch tokens only — no multi-layer concat, no CLS/token fusion, no extra pooling.
- Distance: **per-patch L2 + cosine 1-NN** (`1 − cos-sim`), k=1, no reweighting, no Euclidean.

## Frozen non-variable protocol (unchanged)

- Diagnostic subset reused, never resampled: 1000 train_normal / 300 validation_normal /
  1000 recovery_dev_anomaly (250×Q1–Q4, seed 42). Manifest SHA verified at runtime.
- Memory bank: reservoir Algorithm R, **budget 50000**, **seed 42**, from the 1000 frozen
  train-normal IDs → isolated **D2 experimental bank** (never overwrite baseline/D1/etc.).
- Tiling: frozen 7 tiles x0 ∈ {0,256,512,768,1024,1280,1344}; original score = **max** over
  7 tiles (A0). Threshold = **max(train-normal scores)** (train-only, diagnostic).
- Primary metric: **Image AUROC**.

## Evaluation

D2 eval: 300 validation_normal + 1000 dev anomaly. Report Image AUROC; normal/anomaly score
min/p50/p95/p99/max; anomaly-median − normal-median; TP/TN/FP/FN/Precision/Recall/F1/
Normal-FPR/Anomaly-Recall; quartiles validation-normal vs Q1|Q2|Q3|Q4.

## Primary Gate (unchanged from D1 — no moving goalposts)

- `DOMAIN_REPRESENTATION_CAPACITY_SIGNAL_FOUND`: D2 Image AUROC ≥ **0.70** AND (D2 − D0) ≥ **+0.10** (i.e. ≥ ~0.7029).
- Strong signal (extra flag): D2 AUROC ≥ **0.80**.
- Else `DOMAIN_REPRESENTATION_CAPACITY_GATE_FAILED`.

## Capacity-specific secondary metric (never substitutes the gate)

- `REPRESENTATION_CAPACITY_GAIN = TRUE` if D2 − D1 ≥ **+0.03**, else FALSE.
- Small-defect deltas (D2−D1 per Q1/Q2, D2−D0 per Q1) are reported as findings only.

## Prohibited this phase

No fine-tuning (LoRA/probing/SSL/supervised); no bank-strategy change (coreset/k-center/larger
bank/k>1; multi-bank); no DINOv2-L/G/registers/CLIP/MAE/ConvNeXt/other backbones; no holdout
(`HOLDOUT_ACCESS_COUNT == 0`); no Optimization 2.

## Verdicts (exactly one)

`DOMAIN_REPRESENTATION_CAPACITY_SIGNAL_FOUND` · `DOMAIN_REPRESENTATION_CAPACITY_GATE_FAILED` · `DOMAIN_REPRESENTATION_CAPACITY_BLOCKED`