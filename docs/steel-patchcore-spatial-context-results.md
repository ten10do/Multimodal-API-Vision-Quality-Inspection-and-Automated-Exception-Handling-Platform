# DEEPSEEK SPATIAL REPRESENTATION INVESTIGATION REPORT

Verdict: `SPATIAL_REPRESENTATION_GATE_FAILED`

This phase continues Optimization 1.1 after `REPRESENTATION_BASE_FEATURE_GATE_FAILED`
(`d5d27ac`). It investigates spatial scale & local context only — not
Optimization 2, no speed tuning, no backbone change, no registry change.

## 1. Handoff Audit

- Branch: `feat/steel-patchcore-validity-recovery-v1.2`
- HEAD at handoff: `d5d27acc5b1a3cafce0bcb73fb1dd6273b82c8c3`
- Working tree clean; `git diff --check` clean
- 0 Python workers, no lifecycle lock, no GPU worker at start
- Frozen hash facts all re-verified against on-disk artifacts

## 2. Wording Corrections (semantic)

Per instruction, prior strong wording in `steel-patchcore-representation-results`
was softened without altering any metric:

- ~~"small defects are sub-resolution"~~ → "smaller defects show substantially
  weaker separability under the current patch representation."
- ~~"memory bank is definitely not a bottleneck"~~ → "normal-manifold coverage
  shows no obvious systematic deficit, so bank sampling is deprioritized
  relative to spatial/context representation."
- Historical metrics untouched.

## 3. Scale Geometry Audit

From the frozen 256×256 tile code path:

| Grid | shape | stride | nominal cell footprint |
|---|---|---|---|
| layer2 | 32×32 | 8 px | 8×8 px |
| layer3 (native) | 16×16 | 16 px | 16×16 px |
| layer3 (as used) | 32×32 (bilinear) | 8 px grid | 16×16 px info source, interpolated |

These are **feature-grid stride / nominal footprint**, explicitly NOT a claim
about theoretical/effective receptive field (not computable from tiling code).

## 4. Defect-to-grid Statistics

Frozen `recovery_dev_anomaly` (3333), largest connected-component bbox:

| Quartile | median bbox W | median bbox H | median side | W (l2 cells) | H (l2 cells) |
|---|---|---|---|---|---|
| Q1 | 29 px | 100 px | 102 px | 3.6 | 12.5 |
| Q2 | 35 px | 216 px | 218 px | 4.4 | 27.0 |
| Q3 | 55 px | 254 px | 255 px | 6.9 | 31.8 |
| Q4 | 314 px | 239 px | 314 px | 39.2 | 29.9 |

- Fraction of largest-component bboxes ≤ 1 cell (8 px): **0.000** overall;
  ≤ 2×2 cells (16 px): **0.000**; ≤ 4×4 cells (32 px): **0.006**.
- Interpretation: Q1 "small-area" defects are **thin, not sub-cell** — median
  29×100 px = ~3.6×12.5 layer2 cells. Weak Q1 separability is therefore a
  thin-morphology / feature-response issue, not a raw sub-cell resolution issue.

## 5. Frozen Protocol

`spatial_context_protocol_v1` frozen BEFORE any result: S0 (layer3),
S1 (layer3+3×3 avg), S2 (layer3+5×5 avg); then (only after S gate fails)
P0 (=S0) and P1 (layer2+3×3 avg). Pooling = `AvgPool2d(k, stride=1,
padding=k//2, count_include_pad=False)`; context first, then per-patch L2,
cosine 1-NN, reservoir 50k seed 42, A0 max, same diagnostic subset.

## 6. Diagnostic Manifest

REUSED (not resampled): `representation_diagnostic_manifest.json`,
SHA256 `8d39aec04c35ac7a7edc931c6dfb3494c6ada6a0a6e13e77b89bc894f9919075`
(1000/300/1000, 250×4). Holdout access count = 0.

## 7. S0/S1/S2 Results

| Candidate | AUROC | Δ vs S0 | normal median | anomaly median | Δmedian | Gate |
|---|---|---|---|---|---|---|
| S0 (layer3) | 0.5388 | — | 0.3341 | 0.3336 | −0.001 (approx) | ref |
| S1 (l3+3×3) | 0.5620 | +0.0232 | 0.2872 | 0.2983 | +0.0111 | FAIL |
| S2 (l3+5×5) | 0.6029 | +0.0641 | 0.2254 | 0.2413 | +0.0159 | FAIL |

Local context monotonically improves separation (5×5 > 3×3 > none).

## 8. Spatial Context Gate

Gate: AUROC ≥ 0.65 AND Δ vs S0 ≥ +0.10. S1 (0.5620, +0.0232) and S2 (0.6029,
+0.0641) both FAIL → `SPATIAL_CONTEXT_GATE_FAILED`.

## 9. P0/P1 Results

| Candidate | AUROC | Δ vs R1 | Q1 | ΔQ1 vs R1 | Gate |
|---|---|---|---|---|---|
| R1 (raw layer2) | 0.4236 | — | 0.3487 | — | ref |
| P0 (=S0, layer3) | 0.5388 | +0.1152 | 0.4183 | +0.0696 | (ref) |
| P1 (l2+3×3) | 0.4882 | +0.0646 | 0.3651 | +0.0164 | FAIL |

Patch Scale Gate: AUROC ≥ 0.60 AND Δ vs R1 ≥ +0.10 AND Q1 Δ ≥ +0.10. P1 fails
all three → `PATCH_SCALE_GATE_FAILED`.

## 10. Small-defect Quartile Results

| Candidate | Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|
| S0 (layer3) | 0.4183 | 0.4662 | 0.5304 | 0.7404 |
| S1 (l3+3×3) | 0.4341 | 0.4948 | 0.5608 | 0.7585 |
| S2 (l3+5×5) | 0.4790 | 0.5305 | 0.6145 | 0.7876 |
| P1 (l2+3×3) | 0.3651 | 0.4117 | 0.4918 | 0.6842 |

Context lifts every quartile on layer3 (S2 Q1 0.479 vs S0 0.418), but Q1–Q3
stay far below large-defect (Q4) separability. layer2+context (P1) improves Q1
only +0.016 over raw layer2.

## 11. Root-cause Update

1. **Local context is a real but weak lever**: on layer3, 3×3 → +0.023 and
   5×5 → +0.064 AUROC, with Q1/Q2 also rising. H1 (local context failure) is
   supported in direction, but the effect is far below the gate.
2. **Higher spatial grid + context does not rescue small defects**: layer2+3×3
   (P1) reaches 0.4882 (vs raw layer2 0.4236), but Q1 improves only +0.016 and
   still underperforms layer3 candidates. H2 is not resolved by grid/context.
3. **Defect-size gradient persists and is amplified by context**: Q4 reaches
   0.7876 (S2) while Q1–Q3 remain weak. Consistent with the §4 finding that Q1
   defects are thin (high aspect ratio), not sub-cell.

Conclusion (levels up per §23): within the current ImageNet WRN-50-2 feature
family, layer selection, normalization, robust aggregation, local context, and
a higher spatial grid together still do not recover steel-defect image-level
validity. Whether memory-bank sampling or a domain-adapted/new backbone is the
decisive next lever is deferred to a later authorized phase.

## 12. Gate Verdict

- Spatial Context Gate: `SPATIAL_CONTEXT_GATE_FAILED`
- Patch Scale Gate: `PATCH_SCALE_GATE_FAILED`
- Final: **`SPATIAL_REPRESENTATION_GATE_FAILED`**

## 13. Tests

- passed: **78**
- failed: **0**
- skipped: **1** (`test_registry_candidate_not_production` — MLOps out of
  scope, not counted as a pass)
- deselected: 0

## 14. Git

- `a3aa6f2` — wording corrections + scale geometry diagnostics + frozen
  protocol + spatial shared module + S/P harness + tests (stage-1).
- Final results commit contains this report + results JSON. Experimental
  S1/S2/P1 banks live under ignored `model-training/runs/steel-spatial-context/`.
  No `git add .`, no merge to main, no registry write. Baseline bank and the
  representation-phase banks were not overwritten (verified by SHA).

## 15. Limitations

- Recovery split is post-hoc (full baseline test observed in Optimization 1).
- Diagnostic-subset thresholds are diagnostic only, never a final threshold.
- "Local context helps" is established only for fixed 3×3/5×5 average pooling
  on layer3; no multi-scale/learned context, no backbone change.
- No bank-sampling / coreset / new-distance / new-tiling / holdout / full
  development authorization in this phase.
- `steel-patchcore` 1.0.0 remains `STEEL_DOMAIN_VALIDATION_FAILED`;
  `HOLDOUT_ACCESS_COUNT = 0`.