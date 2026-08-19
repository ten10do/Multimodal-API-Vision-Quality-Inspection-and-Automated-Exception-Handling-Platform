# Steel PatchCore — Representation Investigation Report

Verdict: `REPRESENTATION_BASE_FEATURE_GATE_FAILED`

This phase follows `RECOVERY_AGGREGATION_GATE_FAILED` (`f2a1de8`). It is
Optimization 1.1 representation investigation, not Optimization 2.
`steel-patchcore` 1.0.0 remains `STEEL_DOMAIN_VALIDATION_FAILED`.

## 1. Handoff Audit

- Branch: `feat/steel-patchcore-validity-recovery-v1.2`
- HEAD at handoff: `f2a1de8481c3a8ee6f958300507676c94dfed415`
- Working tree: clean; `git diff --check` clean
- No running steel Python worker, no lifecycle lock, no GPU process at start
- Frozen lineage re-verified: bank/source-split/recovery-split/evidence-manifest all match
- No Codex partial representation work found to reuse or overwrite

## 2. Current Implementation Semantics (traced from code)

`image (256×1600) → 7 tiles (256×256, offsets 0/256/512/768/1024/1280/1344) →
ImageNet normalize → WRN-50-2 IMAGENET1K_V1 → layer2 (512-d, 32×32) +
bilinear-upsampled layer3 (1024-d, 16→32) → concat (1536-d) → per-patch L2 →
cosine 1-NN (`1 - max sim`) → tile max → image max (A0)`.

Per-patch L2 normalization is present; distance is genuinely cosine on unit
vectors; no Euclidean/cosine mismatch. No per-layer normalization before the
concat. `num_neighbors` is ignored (fixed k=1).

## 3. PatchCore Semantic Deviations

- EXPECTED DESIGN CHOICE: reservoir sampling (not coreset), k=1 no reweighting,
  max image score, layer2+layer3 concat, per-patch L2.
- POTENTIAL VALIDITY RISK (investigated, not assumed):
  1. bilinear upsample (vs average-pool alignment) can smear layer3;
  2. no local neighborhood aggregation (raw 1×1 patches);
  3. 1536-d joint L2 → layer3 (1024-d) dominates layer2 (512-d);
  4. bank fraction ≈ 0.148%.
- CONFIRMED BUG: none in the representation path.

## 4. Existing-evidence Diagnostics

Raw nearest-bank distance distributions over the 8644 frozen originals:

| Role | n | A0 median | patch-mean median |
|---|---|---|---|
| train_normal | 4721 | 0.3649 | 0.1752 |
| validation_normal | 590 | 0.3659 | 0.1766 |
| recovery_dev_anomaly | 3333 | 0.3596 | 0.1816 |

- anomaly − validation-normal A0 median = −0.0063 (anomalies rank below normal).
- area ratio vs A0 Spearman ρ = 0.376; max-component vs A0 ρ = 0.387.
- Q1–Q4 A0 medians: 0.3458 / 0.3509 / 0.3589 / 0.3942.

## 5. Memory-bank Coverage Audit

- validation − train median patch-mean distance gap = **+0.0013** (negligible).
- train vs validation per-tile median raw max track each other closely
  (tile 3 highest ~0.31, tile 0 lowest ~0.27).

Conclusion: normal-manifold coverage shows no obvious systematic deficit (the
0.148% reservoir yields uniform train↔validation patch-distance support), so
bank sampling is deprioritized relative to spatial/context representation. This
is a deprioritization, not a proof that bank sampling is irrelevant.

## 6. Frozen Representation Protocol

`representation_protocol_v1` froze: diagnostic subset (1000/300/1000, area-
stratified 250×4, seed 42); R0/R1/R2 (layer2+layer3 / layer2-only /
layer3-only) and N0/N1/N2 (current / L2-cosine / per-layer-L2-before-concat);
reservoir 50k seed 42; A0 aggregation; frozen tiling. Gates: R — ΔAUROC ≥ 0.10
vs R0 AND AUROC ≥ 0.60; N — ΔAUROC ≥ 0.10 vs N0 AND AUROC ≥ 0.60.

## 7. Diagnostic Dataset Manifest

`representation_diagnostic_manifest.json`,
SHA256 `8d39aec04c35ac7a7edc931c6dfb3494c6ada6a0a6e13e77b89bc894f9919075`.
Quartile boundaries (area ratio) Q1/Q2/Q3 = 0.010889 / 0.026565 / 0.072141.
No duplicates; subset of authorized development roles; holdout = 0.

## 8. Feature-layer Results

| Candidate | Layers | AUROC | normal median | anomaly median | Δmedian | Gate |
|---|---|---|---|---|---|---|
| R0 | l2+l3 (current) | 0.4927 | 0.3628 | 0.3585 | −0.0043 | ref |
| R1 | layer2 | 0.4236 | 0.3330 | 0.3160 | −0.0170 | FAIL |
| R2 | layer3 | 0.5388 | 0.3388 | 0.3449 | +0.0062 | FAIL |

R0 (frozen 1.0.0 bank) reference AUROC on the same subset = 0.4936 — the
subset is representative. `FEATURE_LAYER_GATE_FAILED`: no candidate reaches
0.60, and none improves R0 by ≥ 0.10.

## 9. Normalization Results (Stage N)

| Candidate | Semantics | AUROC | Δ vs N0 |
|---|---|---|---|
| N0 | current (=R0) | 0.4927 | — |
| N1 | per-patch L2 + cosine | 0.4927 | ≡ N0 (code-equivalent) |
| N2 | per-layer L2 before concat | 0.4997 | +0.0071 |

`NORMALIZATION_GATE_FAILED`.

## 10. Small-defect Results

Normal-vs-quartile AUROC (validation_normal vs each anomaly area quartile):

| Candidate | Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|
| R0 | 0.4049 | 0.4282 | 0.4591 | 0.6785 |
| R1 | 0.3487 | 0.3726 | 0.3929 | 0.5801 |
| R2 | 0.4183 | 0.4662 | 0.5304 | 0.7404 |

The monotone defect-size gradient remains: only large defects (Q4) show any
separation, peaking at 0.7404 (R2). Q1–Q3 stay at or below ~0.5 for every
candidate.

## 11. Root-cause Update

1. **Feature-layer configuration does affect ranking** (layer3 0.539 > concat
   0.493 > layer2 0.424), so the layer choice is a real lever — but a weak one,
   topping out at 0.539, far below 0.60.
2. **Layer2 fine-grained detail injects anti-signal**: concatenating layer2 into
   layer3 *lowers* AUROC from 0.539 to 0.493. This supports the imbalance /
   dilution hypothesis (#3 in §3) rather than refuting representation failure.
3. **Normalization does not help**: balancing layer mass (N2) moves AUROC only
   to 0.4997.
4. **Bank coverage shows no obvious systematic deficit** — bank sampling is
   deprioritized relative to spatial/context representation (§5), not proven
   irrelevant.

Conclusion: under the current patch representation (pretrained WRN-50-2
layer2/layer3 + cosine 1-NN), the image-level AUROC stays near chance across
aggregation, feature-layer selection, and normalization. Smaller defects show
substantially weaker separability than larger defects under this patch
representation. Aggregation, feature-layer selection, and normalization have
each shown marginal effect; the observed weak separability is consistent with a
representation/domain-mismatch, but this phase does not assert that small
defects are "sub-resolution" — that requires the spatial/context experiments
in the next phase.

## 12. Gate Verdict

`REPRESENTATION_BASE_FEATURE_GATE_FAILED`

Both `FEATURE_LAYER_GATE_FAILED` and `NORMALIZATION_GATE_FAILED`. No
experimental candidate passes the frozen diagnostic gate.

## 13. Tests

- passed: 66
- failed: 0
- skipped: 1 (`test_registry_candidate_not_production` — MLOps registration out
  of scope; not counted as a pass)
- deselected: 0

## 14. Git

- Stage-1 commit `ddc8944` (audit/protocol/diagnostics/manifest/R-harness/tests).
- Final results commit contains this report + results JSON + normalization
  script. Experimental banks/tensors remain under ignored
  `model-training/runs/steel-representation/`. No `git add .`, no merge to main.

## 15. Limitations

- The recovery split is post-hoc (full baseline test observed in Optimization 1).
- The diagnostic subset thresholds are diagnostic only, never a final threshold.
- No backbone / bank-sampling / patch-scale / tiling change is authorized by
  this phase; those require a separate authorization.
- `steel-patchcore` 1.0.0 conclusion remains `STEEL_DOMAIN_VALIDATION_FAILED`.