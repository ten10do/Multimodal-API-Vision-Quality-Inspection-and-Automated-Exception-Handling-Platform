# Steel PatchCore — DINOv2 Capacity Cross-Check Results (D0 WRN vs D1 S/14 vs D2 B/14)

Verdict: **`DOMAIN_REPRESENTATION_CAPACITY_GATE_FAILED`**

Primary Gate (unchanged from D1): D2 Image AUROC ≥ 0.70 **AND** (D2 − D0) ≥ +0.10.

| Reference | Image AUROC | Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|---|
| **D0** — WRN S2 (frozen) | **0.6029** | 0.4790 | 0.5305 | 0.6145 | 0.7876 |
| **D1** — DINOv2 ViT-S/14 (frozen) | **0.6699** | 0.5843 | 0.6086 | 0.6607 | 0.8261 |
| **D2** — DINOv2 ViT-B/14 | **0.6938** | 0.6043 | 0.6318 | 0.7035 | 0.8358 |

- Δ vs D0 = **+0.0909** (below +0.10).
- D2 AUROC 0.6938 is below 0.70.
- **Both gate conditions failed** → `DOMAIN_REPRESENTATION_CAPACITY_GATE_FAILED`.
- Capacity gain (D2 − D1 ≥ +0.03): **false** (Δ vs D1 = +0.0239).
- Strong signal (≥ 0.80): **false**.

## D2 candidate (frozen)

- Representation: **DINOv2 ViT-B/14 spatial patch tokens only** (raw, CLS/register excluded,
  then per-patch L2). Cosine 1-NN, k=1, no reweighting, no pooling/fusion.
- Geometry: frozen 256×256 tile → bilinear **252×252** → **18×18 = 324 patch tokens** × 768-d.
- Model: `facebookresearch/dinov2` `torch.hub.load(..., "dinov2_vitb14")`; `vit_base`,
  `embed_dim=768`, `depth=12`, `num_heads=12`, `num_register_tokens=0`; weights LVD-142M
  (SHA256 `0b8b82f8…fad8c73`), Apache-2.0.
- Bank: reservoir Algorithm R, budget 50000, seed 42, from the frozen 1000 train-normal
  diagnostic IDs; 2,268,000 candidate patches streamed; bank 50000×768 float32
  (SHA256 `383949c4…f1b292b3`). Isolated D2 bank — no existing bank modified.
- Distance/metrics: per-patch L2 + cosine `1 − max(cos-sim)`; image score = **max** over
  frozen 7 tiles (A0); threshold = max train-normal scores = 0.6678 (train-only, diagnostic).

## Diagnostic metrics (train-only operating point)

- Image AUROC = 0.6938; anomaly-median − normal-median = +0.0545.
- Normal (300): min 0.2902 / p50 0.4202 / p95 0.5701 / p99 0.6218 / max 0.7212.
- Anomaly (1000): min 0.3175 / p50 0.4747 / p95 0.6500 / p99 0.7106 / max 0.7604.
- Operating point (threshold 0.6678): TP=34, FP=1, TN=299, FN=966; precision 0.9714,
  recall 0.0340, F1 0.0657, Normal FPR 0.0033, Anomaly Recall 0.0340.

## Quartile deltas

- D2 vs D1: Q1 +0.0200, Q2 +0.0232, Q3 +0.0428, Q4 +0.0097.
- D2 Q1 vs D0 Q1: +0.1253 (small defects still the strongest recovered band, as with D1).

## Conclusion (frozen scope)

Increasing frozen DINOv2 capacity from ViT-S/14 to ViT-B/14 **did not provide sufficient
development recovery** under the frozen diagnostic protocol: it moved the overall AUROC to
0.6938 (+0.0909 over D0) but did not clear the unchanged Primary Gate, and the capacity gain
over D1 (+0.0239) fell below the +0.03 secondary threshold. This result does **not** rule out
DINOv2 or self-supervised/ViT representations generally, nor fine-tuning, memory-bank
strategy changes, or steel-domain adaptation — those remain out of scope for this phase.

## Prior-phase status (unchanged)

- D1 (DINOv2 ViT-S/14) = `DOMAIN_REPRESENTATION_GATE_FAILED`, with
  **`SMALL_DEFECT_REPRESENTATION_SIGNAL = TRUE`** (meaningful positive signal, not "useless").
- Canonical PatchCore cross-check remains **`CANONICAL_PATCHCORE_REFERENCE_BLOCKED`** (not FAILED).

Holdout access count: **0** (test_normal and recovery_holdout_anomaly never accessed).