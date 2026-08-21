# D3 INDUSTRIALIZATION REPORT

Terminal status: `D3_CANDIDATE_INDUSTRIALIZED`

## 1. Handoff Audit

- Branch remained `feat/steel-patchcore-validity-recovery-v1.2`.
- Starting HEAD was `5cd9d62a2f61b49ad7bed1b485ae708d33c9cc1d` (`RECOVERY_HOLDOUT_PASS`).
- Initial working tree was clean and `git diff --check` had no findings.
- Active Python/steel worker count was zero; no lifecycle lock existed. GPU process output contained only desktop/WDDM processes.
- No reset, clean, stash, main merge, history switch, experiment rerun, training, or GPU model evaluation was performed.

## 2. Artifact Candidate

Created the registered manifest [steel-patchcore-d3-candidate/manifest.json](../model-training/registry/steel-patchcore-d3-candidate/manifest.json):

- status `CANDIDATE`; `production_promotion=false`;
- model `steel-patchcore-d3-candidate@1.2.0-candidate.1`;
- artifact `d3-full-development-9b1ea19`;
- backbone `dinov2_vitb14`, embedding dimension 768;
- frozen ZCA, 50k bank, `cosine-1NN`, A0, and seven-tile configuration;
- full-precision threshold `0.8471092581748962`;
- full SHA256 values for weights, whitening, bank, source/recovery splits, and both committed result JSON files;
- embedded full-development and recovery-holdout evidence.

All seven referenced files were hashed successfully. Evidence lineage was cross-checked in both directions: the candidate manifest matches the hashes, threshold, bank/whitening/weights/splits, metrics, and terminal verdicts recorded by the committed evaluation results. No artifact bytes were changed.

## 3. Registry

Implemented a file-based, fail-closed candidate registry in `steel_patchcore.candidate_registry`:

- `register`: only writes a candidate after schema, artifact, lineage, evidence, and test gates pass;
- `verify_artifact`: resolves controlled URIs, hashes every artifact, and verifies result-file lineage;
- `load_artifact`: blocks on any mismatch, loads NPZ with pickle disabled, checks exact shapes/finiteness, marks arrays read-only, and re-hashes after loading.

The schema accepts exactly `CANDIDATE`. `PRODUCTION`, unknown fields, payload drift, path traversal, absolute project artifact paths, missing files, SHA mismatch, evidence mismatch, threshold drift, and duplicate/different registration all fail closed. This registry has no promotion method.

## 4. Inference Integration

Added `D3CandidatePredictor` to the inference service. The path is explicitly opt-in through `IVQC_D3_CANDIDATE_MANIFEST`; the existing anomaly configuration remains unchanged otherwise. This prevents candidate registration from becoming an implicit deployment or production promotion.

The inference sequence is frozen image tiling → DINOv2-B `x_norm_patchtokens` → existing ZCA → patch L2 → frozen bank cosine 1-NN → A0 score. The predictor loads the exact verified weight file into a `pretrained=False` DINO architecture, rather than permitting a different pretrained checkpoint to be selected.

The candidate payload contains `anomaly_score`, `threshold`, `is_anomaly`, `model_version`, and `artifact_version`. The existing shared `AnomalyResult` retains `is_anomalous` for compatibility and now carries optional `artifact_version` for end-to-end traceability.

## 5. Heatmap

Each tile produces an 18x18 raw patch-distance grid. The raw grids are bilinearly expanded to the tile geometry and mean-stitched across overlaps into a 256x1600 raw anomaly map. A separately normalized 0..1 heatmap is available for review PNG generation.

Image scoring is isolated from localization: A0 is computed directly from the stacked raw tile grids before any resize, stitch, or normalization. Tests pin that heatmap generation cannot change the image score. Raw and normalized arrays are returned read-only.

## 6. Evaluation Pipeline

Added a score-only engineering pipeline that accepts:

- an explicit dataset-role manifest;
- one finite, unique, role-bound score record per image;
- the validated candidate manifest and complete artifact hash set.

It rejects role drift, overlap, duplicates, missing records, foreign IDs, invalid quartiles, non-finite scores, and artifact hash mismatch. Output JSON contains metrics, bootstrap CI, verdict, lineage, timestamp, artifact hashes, model/artifact identity, and a deterministic evaluation fingerprint. It does not load a model, regenerate artifacts, recalibrate the threshold, or rerun an experiment.

## 7. Tests

| Suite | Result |
|---|---:|
| D3 industrialization unit tests | 14 passed |
| D3 real-artifact load/immutability gate | 1 passed |
| All steel tests (17 files) | 184 passed, 1 existing conditional skip |
| Full inference-service tests | 212 passed, 1 existing conditional skip, 4 marker-deselected |
| Full backend tests | 113 passed, 22 marker-deselected |

Coverage includes manifest/schema validation, real and synthetic SHA verification, lineage checks, registry fail-closed behavior, candidate gate, artifact loading, threshold/bank immutability, deterministic D3 inference, required response identity, raw/normalized heatmaps, A0 invariance, evaluation membership, and reproducibility. No GPU test or holdout inference was run.

## 8. Git

- Branch: `feat/steel-patchcore-validity-recovery-v1.2`.
- Only source, tests, registry manifest, and documentation are eligible for precise staging.
- Weights, bank, whitening binary, datasets, checkpoints, caches, run artifacts, virtual environments, and MLflow data remain untracked/ignored and are not committed.
- Required commit message: `feat: industrialize D3 steel anomaly candidate`.
- Main is not merged and the production deployment manifest is unchanged.

## 9. Limitations

- D3 is still a candidate based on one steel dataset and one sealed recovery holdout.
- The frozen threshold operating point has very low anomaly recall; it remains unchanged because this task forbids threshold optimization.
- Q1/small-defect performance is weaker than Q4.
- The heatmap is patch-distance localization and has not passed a separately authorized pixel-level acceptance protocol.
- Real target-line latency, concurrency, memory, fault recovery, drift thresholds, and hardware qualification remain unvalidated.
- Opt-in candidate inference is not production approval.

## 10. Next Recommendation

Keep `steel-patchcore-d3-candidate` at `CANDIDATE`. The next action should require explicit authorization and should be an operational candidate qualification plan: target-hardware latency/load testing, controlled line shadowing, heatmap acceptance criteria, observability, and rollback rehearsal. Do not begin threshold optimization, backbone search, fine-tuning, a new dataset, external validation, production promotion, or Optimization 3 under this task.
