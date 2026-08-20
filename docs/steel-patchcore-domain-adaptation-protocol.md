# Steel PatchCore — Steel-Domain Representation Adaptation Protocol (Optimization 1.1)

Protocol version: `domain_adaptation_protocol_v1`
Status carry-over: canonical PatchCore cross-check remains **`CANONICAL_PATCHCORE_REFERENCE_BLOCKED`** (not FAILED).

## Question (one variable only)

Does **train-normal-only feature-space domain adaptation** — ZCA covariance whitening
applied to frozen DINOv2 ViT-B/14 patch tokens — recover enough image-level separability?
Only D2 → D3 is evaluated. No bank/search/fine-tune/backbone change is introduced.

## Decision context for this round

Memory-bank algorithm search is deferred: normal-manifold coverage audit found no systematic
bank deficit, and the representation family is the only variable producing consistent positive
signal: D0 WRN S2 = 0.6029 → D1 DINOv2-S/14 = 0.6699 → D2 DINOv2-B/14 = 0.6938
(Q1: 0.4790 → 0.5843 → 0.6043). This round tests statistical train-normal domain adaptation.

## Frozen references (never re-run)

- **D0** = WRN S2: AUROC 0.6029 (Q1 0.4790 / Q2 0.5305 / Q3 0.6145 / Q4 0.7876).
- **D1** = DINOv2 ViT-S/14: AUROC 0.6699 (Q1 0.5843 / Q2 0.6086 / Q3 0.6607 / Q4 0.8261);
  verdict `DOMAIN_REPRESENTATION_GATE_FAILED`, `SMALL_DEFECT_SIGNAL = TRUE`.
- **D2** = DINOv2 ViT-B/14: AUROC **0.6938** (Q1 0.6043 / Q2 0.6318 / Q3 0.7035 / Q4 0.8358);
  verdict `DOMAIN_REPRESENTATION_CAPACITY_GATE_FAILED` — strongest frozen representation so far.

## D3 definition (frozen)

`x` = DINOv2 ViT-B/14 `forward_features(x)["x_norm_patchtokens"]` (CLS/register excluded,
18×18 = 324 patch tokens × 768-d; 256×256 tile bilinearly resized to 252×252).

1. train-normal centering: `x − μ_train`
2. train-normal ZCA whitening: `Σ_train^(−1/2) (x − μ_train)`
3. per-patch L2 normalization
4. cosine 1-NN (`1 − cos-sim`)
5. A0 image aggregation (max patch distance over 7 frozen tiles)

Whitening **changes the metric geometry** by design; cosine is applied afterwards to isolate
the adaptation variable. No Mahalanobis scoring, no Euclidean, no PCA dim reduction (768→768).

## Whitening statistics (train-normal only, HARD)

`μ` and `Σ` come from DINOv2 ViT-B/14 patch tokens of the **1000 frozen train-normal
diagnostic IDs** only. Validation normal / dev anomaly / holdout are forbidden. Enforced by
`WHITENING_TRAIN_ONLY_GATE` in tests.

## Streaming statistics

- Never materialize the full 2,268,000 × 768 token matrix.
- Streaming sufficient statistics in **float64**, Chan et al. (1983) pairwise combine:
  running `(count, mean, M2)`, then covariance = `M2 / n` (population convention, documented).
- 768×768 covariance ≈ 4.7 MB in float64; eigendecomposition `numpy.linalg.eigh`.

## Regularization (fixed rule, not a hyperparameter)

`epsilon = 1e-6 · trace(Σ) / d`, then `Σ_reg = Σ + epsilon·I`. No 1e-4/1e-5/1e-7 grid, no shrinkage
search. If this fixed rule is numerically unusable → `ADAPTATION_NUMERICAL_BLOCKED`.

## Whitening transform

Symmetric eigendecomposition `Σ_reg = Q Λ Q^T`; `W = Q Λ^(−1/2) Q^T` (ZCA). No component
dropping, no PC-count selection.

## Numerical sanity gate (before any bank build)

On a train-normal token sample, verify whitened mean ≈ 0 and covariance ≈ I; report
max-abs-mean, mean-abs-mean, cov diag min/p50/max, off-diag |·| mean/p95/max. BLOCK on:
non-finite values, non-positive eigenvalue after regularization, non-finite/catastrophic
condition number, whitened covariance diag median outside [0.5, 1.5], or off-diagonal |·| max
> 0.25 (bounds only fire on genuine failure; a healthy ZCA shows diag≈1, off-diag max≈0.02).

## D3 memory bank

Reservoir, budget **50000**, seed **42**, over whitened+L2-normalized train tokens → isolated
**D3 experimental bank** (never reuse the unwhitened D2 bank; never overwrite baseline/D1/D2).
Record candidate patches, count, shape, dtype, SHA256, whitening-artifact SHA, weights SHA,
manifest SHA.

## Frozen tiling / aggregation / threshold

7 tiles x0 ∈ {0,256,512,768,1024,1280,1344}; image score = A0 global max patch distance;
threshold = max(train-normal D3 image scores) (train-only, diagnostic). Primary metric = Image AUROC.

## Evaluation

300 validation_normal + 1000 dev anomaly: Image AUROC; normal/anomaly min/p50/p95/p99/max;
anomaly-median − normal-median; TP/TN/FP/FN/Precision/Recall/F1/Normal-FPR/Anomaly-Recall;
validation-normal vs Q1|Q2|Q3|Q4 AUROC.

## Domain Adaptation Gate (frozen, stricter than prior phases)

- `STEEL_DOMAIN_ADAPTATION_SIGNAL_FOUND`: D3 Image AUROC ≥ **0.75** AND D3 − D2 ≥ **+0.05**.
- Strong signal (extra flag): D3 AUROC ≥ **0.80**.
- Else `STEEL_DOMAIN_WHITENING_GATE_FAILED`.
- Secondary (never substitutes): if D3 Q1 − D2 Q1 ≥ **+0.05** → `SMALL_DEFECT_ADAPTATION_SIGNAL = TRUE`.

## Prohibited this phase

No anomaly-supervised training (no classifier/probe/contrastive labels/fine-tuning on defects);
no DINO backprop/LoRA/adapter/SSL fine-tuning; no bank-strategy change (larger/smaller bank,
k-center, coreset, stratified, k>1, multi-bank); no DINO-L/G/backbones; no holdout
(`HOLDOUT_ACCESS_COUNT == 0`); no Optimization 2.

## Verdicts (exactly one)

`STEEL_DOMAIN_ADAPTATION_SIGNAL_FOUND` · `STEEL_DOMAIN_WHITENING_GATE_FAILED` · `ADAPTATION_NUMERICAL_BLOCKED`