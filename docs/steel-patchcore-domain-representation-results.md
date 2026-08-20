# Steel PatchCore — Domain Representation Results (D0 WRN vs D1 DINOv2)

Verdict: **`DOMAIN_REPRESENTATION_GATE_FAILED`**

Frozen gate: D1 Image AUROC ≥ 0.70 **AND** (D1 − D0) ≥ +0.10.

| Reference | Image AUROC | Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|---|
| **D0** — WRN-50-2 layer3 + 5×5 context (frozen, not re-run) | **0.6029** | 0.4790 | 0.5305 | 0.6145 | 0.7876 |
| **D1** — DINOv2 ViT-S/14 patch tokens | **0.6699** | 0.5843 | 0.6086 | 0.6607 | 0.8261 |

- Δ vs D0 = **+0.0670** (below the +0.10 delta requirement).
- D1 AUROC 0.6699 is below the 0.70 minimum.
- **Both gate conditions failed** → `DOMAIN_REPRESENTATION_GATE_FAILED`.
- Strong signal (≥ 0.80): **false**.
- Small-defect secondary signal (Q1 or Q2 lift ≥ +0.10): **true** (Q1 = +0.1053;
  Q2 = +0.0781). This does **not** substitute the overall gate.

## D1 candidate (frozen)

- Representation: **DINOv2 ViT-S/14 spatial patch tokens only** (raw, CLS/register excluded,
  then per-patch L2). Cosine 1-NN, k=1, no reweighting, no pooling/fusion.
- Geometry: frozen 256×256 tile → bilinear **252×252** (multiple of patch_size 14) →
  **18×18 = 324 patch tokens** × 384-d; positional encoding bicubic-interpolated from 37×37.
- Model: `facebookresearch/dinov2` `torch.hub.load(..., "dinov2_vits14")`; `vit_small`,
  `embed_dim=384`, `depth=12`, `num_heads=6`, `num_register_tokens=0`; weights LVD-142M
  (SHA256 `b938bf1bc…ef60cd9`), Apache-2.0.
- Bank: reservoir Algorithm R, budget 50000, seed 42, from the frozen 1000 train-normal
  diagnostic IDs; 2,268,000 candidate patches streamed; bank 50000×384 float32
  (SHA256 `9cc52bf41f0f…18d237`). New experimental bank — no existing bank modified.
- Distance/metrics: per-patch L2 + cosine `1 − max(cos-sim)`; image score = **max** over
  frozen 7 tiles (A0); threshold = max train-normal scores = 0.6451 (train-only, diagnostic).

## Diagnostic metrics (train-only operating point)

- Image AUROC = 0.6699; anomaly-median − normal-median = +0.0422.
- Normal (300): min 0.2249 / p50 0.3342 / p95 0.4970 / p99 0.5679 / max 0.6218.
- Anomaly (1000): min 0.2414 / p50 0.3765 / p95 0.5505 / p99 0.6160 / max 0.6538.
- Operating point (threshold 0.6451): TP=2, FP=0, TN=300, FN=998; precision 1.0000,
  recall 0.0020, F1 0.0040, Normal FPR 0.0000, Anomaly Recall 0.0020.

## Quartiles

Normal vs each dev-anomaly quartile (frozen 250×4): Q1 0.5843, Q2 0.6086, Q3 0.6607, Q4 0.8261.

## Conclusion (frozen scope)

Under the frozen diagnostic protocol, **DINOv2 ViT-S/14 as a drop-in frozen representation
does not provide a sufficient recovery signal** — it improves on D0 (notably Q1 small defects,
+0.105) but does not clear the Domain Representation Gate (0.6699 < 0.70; Δ +0.067 < +0.10).
This result does **not** rule out self-supervised representations in general, nor larger
DINOv2 variants, fine-tuning, nor memory-bank strategy changes — those remain out of scope
for this phase.

## Prior-phase status (unchanged)

Canonical PatchCore cross-check remains **`CANONICAL_PATCHCORE_REFERENCE_BLOCKED`** (memory
infeasible; standard PatchCore was **not** ruled out).

Holdout access count: **0** (test_normal and recovery_holdout_anomaly never accessed).